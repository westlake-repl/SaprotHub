import json
import queue
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from easydict import EasyDict

from saprot.data.prosst_labels import (
    RESIDUE_LABEL_IGNORE_INDEX,
    parse_residue_labels,
    validate_residue_labels,
)
from saprot.data.sequence_to_prosst import (
    ESMFOLD_MAX_RESIDUES,
    preparation_artifact_paths,
    prepare_sequence_csv_with_structure_tokens,
)
from saprot.data.structure_to_prosst import (
    prepare_structure_csv_with_structure_tokens,
)
from saprot.scripts.mutation_zeroshot_prosst import score_mutants
from saprot.scripts.saturation_mutagenesis_prosst import (
    score_saturation_mutagenesis,
)
from saprot.scripts.extract_prosst_embeddings import (
    EMBEDDING_LEVELS,
    extract_embeddings as extract_prosst_embeddings,
)
from saprot.scripts.predict_prosst import (
    CLASSIFICATION_TASK_TYPES,
    PAIR_TASK_TYPES,
    SUPPORTED_TASK_TYPES,
    predict_csv,
    validate_adapter_compatibility,
)
from saprot.model.prosst.specs import (
    DEFAULT_PROSST_MODEL,
    MODEL_PROSST_2048,
    PROSST_HUB_NAMESPACE,
    resolve_structure_vocab_size,
)
from saprot.utils.construct_prosst_lmdb import construct_prosst_lmdb
from saprot.utils.prosst_module_loader import (
    load_trainer,
    my_load_dataset,
    my_load_model,
)
from saprot.utils.colab_prosst_templates import (
    INPUT_TEMPLATE_GUIDE,
    get_input_template_name,
)


HF_REPO_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
INPUT_MODE_SEQUENCE = "sequence"
INPUT_MODE_STRUCTURE = "structure"
INPUT_MODES = {INPUT_MODE_SEQUENCE, INPUT_MODE_STRUCTURE}


class ColabProSSTWorkflow:
    """Small Colab-facing workflow wrapper for ProSST tasks.

    ProSST always uses amino-acid tokenizer input ids plus official ProSST
    structure token ids. This helper never builds SaProt-style AA+3Di merged
    sequences.
    """

    def __init__(
        self,
        output_dir: str = "/content/colabprosst_outputs",
        upload_dir: str = "/content/prosst_uploads",
        asset_dir: str = "/content/prosst_structure_assets",
        cache_dir: str = "/content/prosst_structure_cache",
        saprothub_dir: str = "/content/SaprotHub",
    ):
        self.output_dir = Path(output_dir)
        self.upload_dir = Path(upload_dir)
        self.asset_dir = Path(asset_dir)
        self.cache_dir = Path(cache_dir)
        self.saprothub_dir = Path(saprothub_dir)
        self.lmdb_dir = self.saprothub_dir / "LMDB"
        self.weight_dir = self.saprothub_dir / "weights" / "prosst"

        for path in [
            self.output_dir,
            self.upload_dir,
            self.asset_dir,
            self.cache_dir,
            self.lmdb_dir,
            self.weight_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self._pending_downloads = queue.SimpleQueue()
        self._last_preparation_artifacts = {}

    def set_output_dir(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _download(self, path: Path) -> None:
        try:
            import google.colab  # noqa: F401
        except Exception:
            print("saved:", path)
            return

        self._pending_downloads.put(str(path))

    def queue_download(self, path: str) -> None:
        download_path = Path(path)
        if not download_path.is_file():
            raise FileNotFoundError(
                f"Download file does not exist: {download_path}"
            )
        self._download(download_path)

    def pop_pending_download(self) -> Optional[str]:
        try:
            return self._pending_downloads.get_nowait()
        except queue.Empty:
            return None

    def _collect_preparation_artifacts(self, output_csv: str) -> None:
        self._last_preparation_artifacts = {
            name: str(path)
            for name, path in preparation_artifact_paths(output_csv).items()
            if path.is_file()
        }

    def _attach_preparation_artifacts(self, result):
        if isinstance(result, pd.DataFrame):
            result.attrs.update(self._last_preparation_artifacts)
        elif isinstance(result, dict):
            result.update(self._last_preparation_artifacts)
        return result

    @staticmethod
    def normalize_hf_model_repo_id(
        repo_id: str,
        default_namespace: str = PROSST_HUB_NAMESPACE,
    ) -> str:
        repo_id = str(repo_id).strip()
        if "/" not in repo_id and repo_id:
            repo_id = f"{default_namespace}/{repo_id}"

        parts = repo_id.split("/")
        if len(parts) != 2:
            raise ValueError(
                "Hugging Face model ID must be a repository name or look "
                "like `owner/model`."
            )
        for part in parts:
            if (
                not HF_REPO_COMPONENT_PATTERN.fullmatch(part)
                or part.endswith((".", "-"))
                or ".." in part
                or "--" in part
            ):
                raise ValueError(
                    "Invalid Hugging Face namespace or repository name: "
                    f"{part!r}."
                )
        return "/".join(parts)

    @classmethod
    def personal_hf_model_repo_id(cls, repo_name: str) -> str:
        repo_name = str(repo_name).strip()
        if not repo_name or "/" in repo_name or "\\" in repo_name:
            raise ValueError(
                "Enter a Hugging Face repository name without an owner prefix."
            )

        from huggingface_hub import HfApi, get_token

        token = get_token()
        if token is None:
            raise RuntimeError(
                "Log in to Hugging Face before uploading the model."
            )
        try:
            account = HfApi().whoami(token=token)
        except Exception as exc:
            raise RuntimeError(
                "The Hugging Face login could not be verified. Log in again, "
                "then retry the upload."
            ) from exc
        username = str(account.get("name", "")).strip()
        if not username:
            raise RuntimeError(
                "Hugging Face did not return the logged-in user name. Log in "
                "again before uploading."
            )
        return cls.normalize_hf_model_repo_id(f"{username}/{repo_name}")

    @staticmethod
    def list_prossthub_models(token=None) -> list[str]:
        from huggingface_hub import HfApi

        models = HfApi().list_models(
            author=PROSST_HUB_NAMESPACE,
            tags="colabprosst",
            sort="last_modified",
            direction=-1,
            token=token,
        )
        repo_ids = []
        for model in models:
            repo_id = getattr(model, "id", None) or getattr(
                model,
                "modelId",
                None,
            )
            if not repo_id:
                continue
            try:
                repo_id = ColabProSSTWorkflow.normalize_hf_model_repo_id(
                    repo_id
                )
            except ValueError:
                continue
            if repo_id.split("/", 1)[0].lower() != PROSST_HUB_NAMESPACE.lower():
                continue
            if repo_id not in repo_ids:
                repo_ids.append(repo_id)
        return repo_ids

    def maybe_upload_path(self, current_path: str, upload_enabled: bool) -> str:
        current_path = str(current_path).strip()
        if current_path:
            return current_path
        if not upload_enabled:
            raise ValueError("Set a file path or enable upload.")

        try:
            from google.colab import files
        except Exception as exc:
            raise RuntimeError("Colab file upload is only available in Google Colab.") from exc

        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("No file was uploaded.")

        saved_paths = [
            self.save_uploaded_content(filename, content)
            for filename, content in uploaded.items()
        ]

        return str(saved_paths[0])

    def save_uploaded_content(self, filename: str, content: bytes) -> str:
        safe_name = Path(str(filename).replace("\\", "/")).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("Uploaded file must have a valid filename.")

        save_path = self.upload_dir / safe_name
        save_path.write_bytes(bytes(content))
        return str(save_path)

    def maybe_extract_asset_zip(
        self,
        zip_path: str = "",
        upload_enabled: bool = False,
    ) -> Optional[str]:
        zip_path = str(zip_path).strip()
        if not zip_path and not upload_enabled:
            return None
        if not zip_path:
            zip_path = self.maybe_upload_path("", True)

        archive_path = Path(zip_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Structure asset zip does not exist: {archive_path}")
        if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
            raise ValueError(
                f"Structure assets must be uploaded as a valid ZIP file: {archive_path}"
            )

        target_dir = self.asset_dir / archive_path.stem
        shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        self._extract_zip_archive(archive_path, target_dir)

        print("extracted structure assets to", target_dir)
        return str(target_dir)

    @staticmethod
    def _extract_zip_archive(archive_path: Path, target_dir: Path) -> None:
        target_root = target_dir.resolve()

        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_target = (target_dir / member.filename).resolve()
                if target_root not in [member_target, *member_target.parents]:
                    raise ValueError(f"Unsafe zip member path: {member.filename}")
                archive.extract(member, target_dir)

    def resolve_lora_adapter(self, adapter_path: str) -> str:
        adapter = Path(str(adapter_path).strip())
        if adapter.is_file() and adapter.suffix.lower() == ".zip":
            target_dir = self.saprothub_dir / "loaded_lora" / adapter.stem
            shutil.rmtree(target_dir, ignore_errors=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            self._extract_zip_archive(adapter, target_dir)
            candidates = sorted(target_dir.rglob("adapter_config.json"))
            if len(candidates) != 1:
                raise ValueError(
                    "A LoRA ZIP must contain exactly one adapter_config.json; "
                    f"found {len(candidates)}."
                )
            adapter = candidates[0].parent

        if not adapter.is_dir():
            raise FileNotFoundError(
                f"LoRA adapter directory or ZIP does not exist: {adapter}"
            )
        if not (adapter / "adapter_config.json").is_file():
            raise ValueError(
                f"LoRA adapter has no adapter_config.json: {adapter}"
            )
        return str(adapter)

    @staticmethod
    def _normalize_artifact_metadata(metadata: dict) -> dict:
        if not isinstance(metadata, dict):
            raise ValueError("ColabProSST artifact metadata must be a dictionary.")
        model_family = metadata.get("model_family")
        if model_family is not None and str(model_family).lower() != "prosst":
            raise ValueError(
                f"Expected a ProSST artifact, found model_family={model_family!r}."
            )
        normalized = {
            "task_type": metadata.get("task_type", metadata.get("task")),
            "model_path": metadata.get("base_model"),
            "structure_vocab_size": metadata.get("structure_vocab_size"),
            "num_labels": metadata.get("num_labels"),
            "checkpoint_format": metadata.get("checkpoint_format"),
        }
        missing = [
            key
            for key in ["task_type", "model_path", "structure_vocab_size"]
            if normalized[key] in {None, ""}
        ]
        if missing:
            raise ValueError(
                "ColabProSST artifact metadata is missing fields: "
                f"{missing}."
            )
        if normalized["task_type"] not in SUPPORTED_TASK_TYPES:
            raise ValueError(
                "Unsupported community adapter task: "
                f"{normalized['task_type']!r}."
            )
        normalized["structure_vocab_size"] = int(
            normalized["structure_vocab_size"]
        )
        if normalized["num_labels"] is not None:
            normalized["num_labels"] = int(normalized["num_labels"])
        if (
            normalized["task_type"] in CLASSIFICATION_TASK_TYPES
            and normalized["num_labels"] is None
        ):
            raise ValueError(
                "Classification artifact metadata must include num_labels."
            )
        return normalized

    def inspect_model_artifact(self, adapter_path: str) -> dict:
        adapter = Path(self.resolve_lora_adapter(adapter_path))
        metadata_path = adapter / "colabprosst.json"
        if not metadata_path.is_file():
            raise ValueError(
                "ColabProSST adapter is missing colabprosst.json metadata."
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read ColabProSST adapter metadata: {metadata_path}"
            ) from exc

        result = self._normalize_artifact_metadata(metadata)
        validate_adapter_compatibility(
            str(adapter),
            result["task_type"],
            result["model_path"],
            result["structure_vocab_size"],
            result.get("num_labels"),
        )
        result.update(
            {
                "artifact_path": str(adapter),
                "artifact_type": "lora",
            }
        )
        return result

    def download_community_adapter(
        self,
        repo_id: str,
        revision: str = "",
        token=None,
        force_download: bool = False,
    ) -> dict:
        repo_id = self.normalize_hf_model_repo_id(repo_id)
        revision = str(revision or "").strip() or None
        community_root = self.saprothub_dir / "community_models"
        community_root.mkdir(parents=True, exist_ok=True)
        target_dir = community_root / repo_id.replace("/", "__")
        resolved_root = community_root.resolve()
        resolved_target = target_dir.resolve()
        if resolved_root not in resolved_target.parents:
            raise ValueError("Unsafe community model cache path.")
        if force_download and target_dir.exists():
            shutil.rmtree(target_dir)

        from huggingface_hub import snapshot_download

        snapshot_path = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                token=token,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                force_download=force_download,
            )
        )
        adapter_candidates = [
            path.parent
            for path in snapshot_path.rglob("adapter_config.json")
            if ".cache" not in path.parts
        ]
        if len(adapter_candidates) != 1:
            raise ValueError(
                "A community repository must contain exactly one "
                "ColabProSST PEFT adapter; found "
                f"{len(adapter_candidates)} adapters."
            )
        artifact_path = adapter_candidates[0]
        result = self.inspect_model_artifact(str(artifact_path))
        result.update(
            {
                "repo_id": repo_id,
                "revision": revision or "main",
                "snapshot_path": str(snapshot_path),
            }
        )
        return result

    def create_input_templates(
        self,
        template_dir: str = "/content/prosst_input_templates",
        download: bool = False,
    ) -> Path:
        template_home = Path(template_dir)
        template_home.mkdir(parents=True, exist_ok=True)
        for old_template in template_home.glob("prosst_*_template.csv"):
            old_template.unlink()
        for old_name in [
            "README.txt",
            "00_README_FIRST.txt",
            "prosst_input_templates.zip",
        ]:
            old_path = template_home / old_name
            if old_path.is_file():
                old_path.unlink()

        def template_path(group, task, input_mode):
            return template_home / get_input_template_name(
                group,
                task,
                input_mode,
            )

        pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "mutant": "D3A",
                    "structure_file": "protein_1.pdb",
                },
                {
                    "sequence": "ACDE",
                    "mutant": "D3A:E4A",
                    "structure_file": "protein_2.cif",
                },
            ]
        ).to_csv(
            template_path("zero_shot", "single", "structure"), index=False
        )

        pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "mutant": "D3A",
                }
            ]
        ).to_csv(
            template_path("zero_shot", "single", "sequence"), index=False
        )

        pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "label": 1,
                    "stage": "train",
                    "structure_file": "train_protein.pdb",
                },
                {
                    "sequence": "ACE",
                    "label": 0,
                    "stage": "valid",
                    "structure_file": "valid_protein.pdb",
                },
                {
                    "sequence": "ACF",
                    "label": 1,
                    "stage": "test",
                    "structure_file": "test_protein.pdb",
                },
            ]
        ).to_csv(
            template_path("training", "classification", "structure"),
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "label": 1,
                    "stage": "train",
                },
                {
                    "sequence": "ACE",
                    "label": 0,
                    "stage": "valid",
                },
                {
                    "sequence": "ACF",
                    "label": 1,
                    "stage": "test",
                },
            ]
        ).to_csv(
            template_path("training", "classification", "sequence"),
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "residue_labels": "0 1 0",
                    "stage": "train",
                    "structure_file": "train_protein.pdb",
                },
                {
                    "sequence": "ACE",
                    "residue_labels": "1 -100 0",
                    "stage": "valid",
                    "structure_file": "valid_protein.pdb",
                },
                {
                    "sequence": "ACF",
                    "residue_labels": "0 1 1",
                    "stage": "test",
                    "structure_file": "test_protein.pdb",
                },
            ]
        ).to_csv(
            template_path("training", "token_classification", "structure"),
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "residue_labels": "0 1 0",
                    "stage": "train",
                },
                {
                    "sequence": "ACE",
                    "residue_labels": "1 -100 0",
                    "stage": "valid",
                },
                {
                    "sequence": "ACF",
                    "residue_labels": "0 1 1",
                    "stage": "test",
                },
            ]
        ).to_csv(
            template_path("training", "token_classification", "sequence"),
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "label": 0.5,
                    "stage": "train",
                    "structure_file": "train_protein.pdb",
                },
                {
                    "sequence": "ACE",
                    "label": 0.2,
                    "stage": "valid",
                    "structure_file": "valid_protein.pdb",
                },
                {
                    "sequence": "ACF",
                    "label": 0.8,
                    "stage": "test",
                    "structure_file": "test_protein.pdb",
                },
            ]
        ).to_csv(
            template_path("training", "regression", "structure"), index=False
        )

        pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "label": 0.5,
                    "stage": "train",
                },
                {
                    "sequence": "ACE",
                    "label": 0.2,
                    "stage": "valid",
                },
                {
                    "sequence": "ACF",
                    "label": 0.8,
                    "stage": "test",
                },
            ]
        ).to_csv(
            template_path("training", "regression", "sequence"), index=False
        )

        pair_examples = [
            {
                "sequence_1": "ACD",
                "sequence_2": "AC",
                "stage": "train",
                "structure_file_1": "train_protein_1.pdb",
                "structure_file_2": "train_protein_2.pdb",
            },
            {
                "sequence_1": "ACE",
                "sequence_2": "AD",
                "stage": "valid",
                "structure_file_1": "valid_protein_1.pdb",
                "structure_file_2": "valid_protein_2.pdb",
            },
            {
                "sequence_1": "ACF",
                "sequence_2": "AE",
                "stage": "test",
                "structure_file_1": "test_protein_1.pdb",
                "structure_file_2": "test_protein_2.pdb",
            },
        ]
        pair_labels = {
            "pair_classification": [0, 1, 0],
            "pair_regression": [0.5, 0.2, 0.8],
        }
        for task_type, labels in pair_labels.items():
            token_rows = []
            sequence_rows = []
            for example, label in zip(pair_examples, labels):
                common = {
                    "sequence_1": example["sequence_1"],
                    "sequence_2": example["sequence_2"],
                    "label": label,
                    "stage": example["stage"],
                }
                token_rows.append(
                    {
                        **common,
                        "structure_file_1": example["structure_file_1"],
                        "structure_file_2": example["structure_file_2"],
                    }
                )
                sequence_rows.append(common)

            pd.DataFrame(token_rows).to_csv(
                template_path("training", task_type, "structure"),
                index=False,
            )
            pd.DataFrame(sequence_rows).to_csv(
                template_path("training", task_type, "sequence"),
                index=False,
            )

        single_input_structures = pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "structure_file": "protein_1.pdb",
                },
                {
                    "sequence": "ACE",
                    "structure_file": "protein_2.pdb",
                },
            ]
        )
        for group in ["prediction", "embedding"]:
            single_input_structures.to_csv(
                template_path(group, "single", "structure"),
                index=False,
            )
        single_input_structures.iloc[[0]].to_csv(
            template_path("saturation", "single", "structure"),
            index=False,
        )

        single_input_sequences = pd.DataFrame(
            [{"sequence": "ACD"}, {"sequence": "ACE"}]
        )
        for group in ["prediction", "embedding"]:
            single_input_sequences.to_csv(
                template_path(group, "single", "sequence"),
                index=False,
            )
        single_input_sequences.iloc[[0]].to_csv(
            template_path("saturation", "single", "sequence"),
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "sequence_1": example["sequence_1"],
                    "sequence_2": example["sequence_2"],
                    "structure_file_1": example["structure_file_1"],
                    "structure_file_2": example["structure_file_2"],
                }
                for example in pair_examples[:2]
            ]
        ).to_csv(
            template_path("prediction", "pair", "structure"),
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "sequence_1": example["sequence_1"],
                    "sequence_2": example["sequence_2"],
                }
                for example in pair_examples[:2]
            ]
        ).to_csv(
            template_path("prediction", "pair", "sequence"),
            index=False,
        )

        instructions_path = template_home / "00_README_FIRST.txt"
        instructions = [
            "ColabProSST input templates\n\n"
            "IMPORTANT: first choose the task in ColabProSST, then use the "
            "matching template below. Do not use a protein-pair template for "
            "a single-protein task, or a training template for prediction.\n\n"
        ]
        for section, entries in INPUT_TEMPLATE_GUIDE:
            instructions.append(f"{section}\n")
            for group, task, label in entries:
                instructions.extend(
                    [
                        f"{label}:\n",
                        "  "
                        f"{get_input_template_name(group, task, 'structure')}\n",
                        "  "
                        f"{get_input_template_name(group, task, 'sequence')}\n",
                    ]
                )
            instructions.append("\n")
        instructions.append(
            "For each task, choose exactly one input method:\n"
            "1. Sequence + structure files (recommended): use the filename "
            "containing _structure_, replace its example structure filenames, "
            "and upload the referenced PDB/mmCIF files together as one "
            "Structure ZIP.\n"
            "2. Sequence only: use the filename containing _sequence_ only "
            "when no experimental or predicted structure file is available. "
            "ColabProSST runs ESMFold v1 locally; a GPU is strongly recommended, "
            f"and each sequence may contain at most {ESMFOLD_MAX_RESIDUES} "
            "residues. The model is downloaded on first use.\n\n"
            "Protein-pair files use sequence_1 and sequence_2. Their structure "
            "input also uses structure_file_1 and structure_file_2. Optional "
            "chain, chain_1, and chain_2 columns select chains.\n\n"
            "Unknown residues:\n"
            "- In sequence-only input, X residues are predicted automatically "
            "with ESMC-600M before structure prediction. ColabProSST logs each "
            "predicted residue and confidence and saves a complete audit report. "
            "Low-confidence predictions are marked for review.\n"
            "- In sequence + structure-file input, X residues are restored from "
            "matching residue identities in the supplied structure when possible.\n"
            "- Other non-standard sequence characters are rejected.\n\n"
            "Downloads after a successful task:\n"
            "- Sequence-only runs provide a generated structure ZIP and a "
            "matching reusable structure-input CSV. Upload both with "
            "Sequence + structure files next time.\n"
            "- If either input method handles X residues, a completed sequence "
            "CSV is provided. ESMC predictions also include a confidence report.\n"
        )
        instructions_path.write_text(
            "".join(instructions),
            encoding="utf-8",
        )
        template_zip = template_home / "prosst_input_templates.zip"
        with zipfile.ZipFile(template_zip, "w") as archive:
            for csv_path in sorted(template_home.glob("*.csv")):
                archive.write(csv_path, arcname=csv_path.name)
            archive.write(instructions_path, arcname=instructions_path.name)

        print("input template directory:", template_home)
        print("input template package:", template_zip)
        if download:
            self._download(template_zip)

        return template_zip

    def _prepare_input_csv(
        self,
        input_csv: str,
        upload_csv: bool,
        suffix: str,
        structure_vocab_size: Optional[int] = None,
        input_mode: str = INPUT_MODE_SEQUENCE,
        pair_mode: bool = False,
        structure_zip: str = "",
    ) -> str:
        input_csv = self.maybe_upload_path(input_csv, upload_csv)
        self._last_preparation_artifacts = {}
        input_mode = str(input_mode).strip().lower()
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {sorted(INPUT_MODES)}.")
        if input_mode == INPUT_MODE_SEQUENCE:
            output_path = self.output_dir / f"prosst_{suffix}_prepared.csv"
            prepared_csv = prepare_sequence_csv_with_structure_tokens(
                input_csv=input_csv,
                output_csv=str(output_path),
                cache_dir=str(self.cache_dir),
                structure_vocab_size=int(structure_vocab_size),
                pair_mode=pair_mode,
            )
            self._collect_preparation_artifacts(str(output_path))
            return prepared_csv
        if input_mode == INPUT_MODE_STRUCTURE:
            structure_dir = self.maybe_extract_asset_zip(structure_zip)
            if structure_dir is None:
                raise ValueError(
                    "Sequence + structure files input requires a structure ZIP."
                )
            output_path = self.output_dir / f"prosst_{suffix}_prepared.csv"
            prepared_csv = prepare_structure_csv_with_structure_tokens(
                input_csv=input_csv,
                structure_dir=structure_dir,
                output_csv=str(output_path),
                cache_dir=str(self.cache_dir),
                structure_vocab_size=int(structure_vocab_size),
                pair_mode=pair_mode,
            )
            self._collect_preparation_artifacts(str(output_path))
            return prepared_csv
        raise AssertionError(f"Unhandled input mode: {input_mode}")

    @staticmethod
    def _validate_category_ids(labels, num_labels: int, task_name: str) -> None:
        if int(num_labels) < 2:
            raise ValueError(f"{task_name} num_labels must be at least 2.")
        unique_labels = sorted(set(int(label) for label in labels))
        if len(unique_labels) != int(num_labels):
            hint = (
                " If these labels are continuous scores, choose the regression "
                "workflow instead."
                if "classification" in task_name.lower()
                else ""
            )
            raise ValueError(
                f"{task_name} NUM_LABELS does not match the uploaded dataset: "
                f"NUM_LABELS={num_labels}, observed_categories={len(unique_labels)}, "
                f"labels={unique_labels}.{hint}"
            )
        expected_labels = list(range(int(num_labels)))
        if unique_labels != expected_labels:
            raise ValueError(
                f"{task_name} labels must be contiguous category IDs starting "
                f"at 0: expected={expected_labels}, observed={unique_labels}."
            )

    @classmethod
    def _validate_training_labels(
        cls,
        input_csv: str,
        task_type: str,
        num_labels: int,
    ) -> None:
        df = pd.read_csv(input_csv)
        lower_columns = {column.lower(): column for column in df.columns}
        if task_type == "token_classification":
            label_column = lower_columns.get(
                "residue_labels",
                lower_columns.get("label"),
            )
        else:
            label_column = lower_columns.get("label", lower_columns.get("fitness"))
        if label_column is None:
            expected = (
                "residue_labels"
                if task_type == "token_classification"
                else "label"
            )
            raise ValueError(f"Training CSV must contain a {expected} column.")

        if task_type in {"classification", "pair_classification"}:
            labels = pd.to_numeric(df[label_column].dropna(), errors="raise")
            integer_labels = labels.astype(int)
            if not labels.equals(integer_labels.astype(labels.dtype)):
                raise ValueError(
                    "Classification labels must be integer category IDs in the "
                    "range 0..NUM_LABELS-1."
                )
            task_name = (
                "Protein-pair classification"
                if task_type == "pair_classification"
                else "Classification"
            )
            cls._validate_category_ids(
                integer_labels.tolist(),
                num_labels,
                task_name,
            )
        elif task_type in {"regression", "pair_regression"}:
            pd.to_numeric(df[label_column], errors="raise")
        elif task_type == "token_classification":
            sequence_column = lower_columns.get(
                "sequence",
                lower_columns.get("protein"),
            )
            if sequence_column is None:
                raise ValueError("Training CSV must contain a sequence column.")

            category_ids = []
            for row_idx, row in df.iterrows():
                sequence = str(row[sequence_column]).strip().upper()
                labels = parse_residue_labels(row[label_column])
                validate_residue_labels(
                    sequence,
                    labels,
                    context=f"row {row_idx}",
                )
                category_ids.extend(
                    label
                    for label in labels
                    if label != RESIDUE_LABEL_IGNORE_INDEX
                )
            cls._validate_category_ids(
                category_ids,
                num_labels,
                "Residue-level classification",
            )

    @staticmethod
    def _validate_task_name(task_name: str) -> str:
        task_name = str(task_name).strip()
        if (
            not task_name
            or task_name in {".", ".."}
            or "/" in task_name
            or "\\" in task_name
        ):
            raise ValueError(
                "task_name must be a non-empty file-name-safe value without path separators."
            )
        return task_name

    @staticmethod
    def _close_lmdb_datamodule(data_module) -> None:
        for name in ["train_dataset", "valid_dataset", "test_dataset"]:
            dataset = getattr(data_module, name, None)
            if dataset is not None and hasattr(dataset, "_close_lmdb"):
                dataset._close_lmdb()
        if hasattr(data_module, "_close_lmdb"):
            data_module._close_lmdb()

    def run_zero_shot(
        self,
        input_csv: str,
        upload_csv: bool = False,
        model_path: str = MODEL_PROSST_2048,
        structure_vocab_size: Optional[int] = None,
        output_csv: Optional[str] = None,
        download: bool = True,
        input_mode: str = INPUT_MODE_SEQUENCE,
        structure_zip: str = "",
    ) -> pd.DataFrame:
        structure_vocab_size = resolve_structure_vocab_size(
            model_path,
            structure_vocab_size,
        )
        input_csv = self._prepare_input_csv(
            input_csv,
            upload_csv,
            "mutation",
            structure_vocab_size,
            input_mode=input_mode,
            structure_zip=structure_zip,
        )
        output_path = Path(output_csv) if output_csv else self.output_dir / "prosst_mutation_scores.csv"

        df = score_mutants(
            input_csv=input_csv,
            output_csv=str(output_path),
            model_path=model_path,
            cache_dir=str(self.cache_dir),
            structure_vocab_size=structure_vocab_size,
            structure_base_dir=None,
        )
        df.attrs["output_csv"] = str(output_path)
        print("saved mutation scores:", output_path)
        if download:
            self._download(output_path)
        return self._attach_preparation_artifacts(df)

    def run_saturation_mutagenesis(
        self,
        input_csv: str,
        upload_csv: bool = False,
        model_path: str = MODEL_PROSST_2048,
        structure_vocab_size: Optional[int] = None,
        output_csv: Optional[str] = None,
        output_matrix_csv: Optional[str] = None,
        output_heatmap_png: Optional[str] = None,
        download: bool = True,
        input_mode: str = INPUT_MODE_SEQUENCE,
        structure_zip: str = "",
    ) -> dict:
        structure_vocab_size = resolve_structure_vocab_size(
            model_path,
            structure_vocab_size,
        )
        input_csv = self._prepare_input_csv(
            input_csv,
            upload_csv,
            "saturation",
            structure_vocab_size,
            input_mode=input_mode,
            structure_zip=structure_zip,
        )
        score_path = (
            Path(output_csv)
            if output_csv
            else self.output_dir / "prosst_saturation_scores.csv"
        )
        matrix_path = (
            Path(output_matrix_csv)
            if output_matrix_csv
            else self.output_dir / "prosst_saturation_matrix.csv"
        )
        heatmap_path = (
            Path(output_heatmap_png)
            if output_heatmap_png
            else self.output_dir / "prosst_saturation_heatmap.png"
        )
        result = score_saturation_mutagenesis(
            input_csv=input_csv,
            output_csv=str(score_path),
            output_matrix_csv=str(matrix_path),
            output_heatmap_png=str(heatmap_path),
            model_path=model_path,
            cache_dir=str(self.cache_dir),
            structure_vocab_size=structure_vocab_size,
            structure_base_dir=None,
        )

        archive_path = self.output_dir / "prosst_saturation_mutagenesis.zip"
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path in [score_path, matrix_path, heatmap_path]:
                archive.write(path, arcname=path.name)
        result["archive_path"] = str(archive_path)

        print("saved saturation scores:", score_path)
        print("saved saturation matrix:", matrix_path)
        print("saved saturation heatmap:", heatmap_path)
        print("saved saturation package:", archive_path)
        if download:
            self._download(archive_path)
        return self._attach_preparation_artifacts(result)

    def extract_embeddings(
        self,
        input_csv: str,
        upload_csv: bool = False,
        model_path: str = MODEL_PROSST_2048,
        structure_vocab_size: Optional[int] = None,
        level: str = "protein",
        batch_size: int = 1,
        max_length: int = 2046,
        output_pt: Optional[str] = None,
        output_index_csv: Optional[str] = None,
        adapter_path: str = "",
        download: bool = True,
        input_mode: str = INPUT_MODE_SEQUENCE,
        structure_zip: str = "",
    ) -> dict:
        level = str(level).strip().lower()
        if level not in EMBEDDING_LEVELS:
            raise ValueError(
                f"Embedding level must be one of {sorted(EMBEDDING_LEVELS)}."
            )
        structure_vocab_size = resolve_structure_vocab_size(
            model_path,
            structure_vocab_size,
        )
        adapter_path = str(adapter_path or "").strip()
        artifact_metadata = None
        if adapter_path:
            artifact_metadata = self.inspect_model_artifact(adapter_path)
            adapter_path = artifact_metadata["artifact_path"]
            if artifact_metadata["model_path"] != model_path:
                raise ValueError(
                    "Embedding artifact base model does not match the selected "
                    f"model: {artifact_metadata['model_path']} != {model_path}."
                )
            if (
                artifact_metadata["structure_vocab_size"]
                != structure_vocab_size
            ):
                raise ValueError(
                    "Embedding artifact structure vocabulary does not match "
                    "the selected model."
                )
        input_csv = self._prepare_input_csv(
            input_csv,
            upload_csv,
            "embedding",
            structure_vocab_size,
            input_mode=input_mode,
            structure_zip=structure_zip,
        )

        embedding_path = (
            Path(output_pt)
            if output_pt
            else self.output_dir / f"prosst_{level}_embeddings.pt"
        )
        index_path = (
            Path(output_index_csv)
            if output_index_csv
            else self.output_dir / f"prosst_{level}_embeddings_index.csv"
        )
        result = extract_prosst_embeddings(
            input_csv=input_csv,
            output_pt=str(embedding_path),
            output_index_csv=str(index_path),
            model_path=model_path,
            level=level,
            cache_dir=str(self.cache_dir),
            structure_vocab_size=structure_vocab_size,
            batch_size=batch_size,
            max_length=max_length,
            structure_base_dir=None,
            adapter_path=adapter_path,
            adapter_task_type=(
                artifact_metadata["task_type"] if artifact_metadata else None
            ),
            adapter_num_labels=(
                artifact_metadata.get("num_labels") or 2
                if artifact_metadata
                else 2
            ),
        )

        archive_path = self.output_dir / f"prosst_{level}_embeddings.zip"
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.write(embedding_path, arcname=embedding_path.name)
            archive.write(index_path, arcname=index_path.name)
        result["archive_path"] = str(archive_path)

        print("saved embeddings:", embedding_path)
        print("saved embedding index:", index_path)
        print("saved embedding package:", archive_path)
        if download:
            self._download(archive_path)
        return self._attach_preparation_artifacts(result)

    def train_downstream(
        self,
        task_type: str,
        input_csv: str,
        upload_csv: bool = False,
        task_name: str = "ProSSTUserTask",
        num_labels: int = 2,
        max_epochs: int = 2,
        batch_size: int = 1,
        model_path: str = MODEL_PROSST_2048,
        structure_vocab_size: Optional[int] = None,
        gradient_checkpointing: bool = True,
        initial_adapter: str = "",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = 2.0e-5,
        download: bool = True,
        input_mode: str = INPUT_MODE_SEQUENCE,
        structure_zip: str = "",
    ) -> dict:
        if task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported ProSST task_type: {task_type}.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if lora_rank < 1:
            raise ValueError("LoRA rank must be at least 1.")
        if lora_alpha < 1:
            raise ValueError("LoRA alpha must be at least 1.")
        if not 0 <= lora_dropout < 1:
            raise ValueError("LoRA dropout must be in the range [0, 1).")
        task_name = self._validate_task_name(task_name)
        structure_vocab_size = resolve_structure_vocab_size(
            model_path,
            structure_vocab_size,
        )
        initial_adapter = str(initial_adapter or "").strip()
        if initial_adapter:
            initial_adapter = self.resolve_lora_adapter(initial_adapter)
            validate_adapter_compatibility(
                initial_adapter,
                task_type,
                model_path,
                structure_vocab_size,
                num_labels,
            )

        input_csv = self._prepare_input_csv(
            input_csv,
            upload_csv,
            f"{task_type}_train",
            structure_vocab_size,
            input_mode=input_mode,
            pair_mode=task_type in PAIR_TASK_TYPES,
            structure_zip=structure_zip,
        )
        self._validate_training_labels(input_csv, task_type, num_labels)

        construct_prosst_lmdb(
            input_csv,
            str(self.lmdb_dir),
            task_name,
            task_type,
            cache_dir=str(self.cache_dir),
            structure_vocab_size=structure_vocab_size,
            structure_base_dir=None,
        )

        model_py = {
            "classification": "prosst/prosst_classification_model",
            "regression": "prosst/prosst_regression_model",
            "token_classification": "prosst/prosst_token_classification_model",
            "pair_classification": (
                "prosst/prosst_pair_classification_model"
            ),
            "pair_regression": "prosst/prosst_pair_regression_model",
        }[task_type]
        dataset_py = {
            "classification": "prosst/prosst_classification_dataset",
            "regression": "prosst/prosst_regression_dataset",
            "token_classification": (
                "prosst/prosst_token_classification_dataset"
            ),
            "pair_classification": (
                "prosst/prosst_pair_classification_dataset"
            ),
            "pair_regression": "prosst/prosst_pair_regression_dataset",
        }[task_type]

        adapter_path = self.weight_dir / f"{task_name}_lora"
        test_result_csv = self.output_dir / f"{task_name}_{task_type}_test_predictions.csv"
        model_kwargs = {
            "config_path": model_path,
            "structure_vocab_size": structure_vocab_size,
            "load_pretrained": True,
            "gradient_checkpointing": gradient_checkpointing,
            "save_path": str(adapter_path),
            "save_weights_only": True,
            "test_result_path": str(test_result_csv),
            "lr_scheduler_kwargs": {
                "class": "ConstantLRScheduler",
                "init_lr": float(learning_rate),
            },
            "optimizer_kwargs": {"class": "AdamW", "betas": [0.9, 0.98], "weight_decay": 0.01},
        }
        if task_type in CLASSIFICATION_TASK_TYPES:
            model_kwargs["num_labels"] = num_labels
        model_kwargs["lora_kwargs"] = {
            "is_trainable": True,
            "num_lora": 1,
            "config_list": (
                [{"lora_config_path": initial_adapter}]
                if initial_adapter
                else []
            ),
            "r": int(lora_rank),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
        }

        config = EasyDict(
            {
                "model": {"model_py_path": model_py, "kwargs": model_kwargs},
                "dataset": {
                    "dataset_py_path": dataset_py,
                    "train_lmdb": str(self.lmdb_dir / task_name / "train"),
                    "valid_lmdb": str(self.lmdb_dir / task_name / "valid"),
                    "test_lmdb": str(self.lmdb_dir / task_name / "test"),
                    "dataloader_kwargs": {"batch_size": batch_size, "num_workers": 0},
                    "kwargs": {
                        "tokenizer": model_path,
                        "structure_vocab_size": structure_vocab_size,
                    },
                },
                "Trainer": {
                    "max_epochs": max_epochs,
                    "log_every_n_steps": 1,
                    "strategy": {"class": "auto"},
                    "logger": False,
                    "enable_checkpointing": False,
                    "val_check_interval": 1.0,
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1,
                    "num_nodes": 1,
                    "accumulate_grad_batches": 1,
                    "precision": 16 if torch.cuda.is_available() else 32,
                    "num_sanity_val_steps": 0,
                },
            }
        )

        model = my_load_model(config.model)
        data_module = my_load_dataset(config.dataset)
        trainer = load_trainer(config)

        # The initial adapter is already loaded into memory at this point.
        # Remove same-name outputs so a previous run cannot be mistaken for the
        # best adapter or test predictions from this run.
        if adapter_path.exists():
            shutil.rmtree(adapter_path)
        if test_result_csv.exists():
            test_result_csv.unlink()

        try:
            trainer.fit(model=model, datamodule=data_module)
            if not adapter_path.exists():
                print(
                    "validation did not save an adapter; saving the final "
                    "training state to",
                    adapter_path,
                )
                model.save_checkpoint(str(adapter_path), save_weights_only=True)
            print("loading best LoRA adapter from", adapter_path)
            model.load_lora_adapter(str(adapter_path))

            trainer.test(model=model, datamodule=data_module)
        finally:
            self._close_lmdb_datamodule(data_module)

        print("test predictions:", test_result_csv)
        print("LoRA adapter:", adapter_path)

        adapter_download_path = Path(
            shutil.make_archive(
                str(adapter_path),
                "zip",
                root_dir=adapter_path,
            )
        )

        if download:
            if test_result_csv.exists():
                self._download(test_result_csv)
            if adapter_download_path.exists():
                self._download(adapter_download_path)

        result = {
            "adapter_path": str(adapter_path),
            "adapter_download_path": str(adapter_download_path),
            "test_result_csv": str(test_result_csv),
            "task_type": task_type,
            "model_path": model_path,
            "structure_vocab_size": structure_vocab_size,
            "initial_adapter": initial_adapter,
        }
        return self._attach_preparation_artifacts(result)

    def predict_downstream(
        self,
        task_type: str,
        input_csv: str,
        adapter_path: str,
        upload_csv: bool = False,
        num_labels: int = 2,
        batch_size: int = 1,
        model_path: str = MODEL_PROSST_2048,
        structure_vocab_size: Optional[int] = None,
        output_csv: Optional[str] = None,
        download: bool = True,
        input_mode: str = INPUT_MODE_SEQUENCE,
        structure_zip: str = "",
    ) -> pd.DataFrame:
        if task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported ProSST task_type: {task_type}.")
        structure_vocab_size = resolve_structure_vocab_size(
            model_path,
            structure_vocab_size,
        )
        adapter_path = self.resolve_lora_adapter(adapter_path)

        input_csv = self._prepare_input_csv(
            input_csv,
            upload_csv,
            f"{task_type}_predict",
            structure_vocab_size,
            input_mode=input_mode,
            pair_mode=task_type in PAIR_TASK_TYPES,
            structure_zip=structure_zip,
        )
        output_path = Path(output_csv) if output_csv else self.output_dir / f"prosst_{task_type}_predictions.csv"

        df = predict_csv(
            input_csv=input_csv,
            output_csv=str(output_path),
            task_type=task_type,
            adapter_path=adapter_path,
            model_path=model_path,
            num_labels=num_labels,
            batch_size=batch_size,
            cache_dir=str(self.cache_dir),
            structure_vocab_size=structure_vocab_size,
            structure_base_dir=None,
        )
        df.attrs["output_csv"] = str(output_path)

        print("saved predictions:", output_path)
        if download:
            self._download(output_path)
        return self._attach_preparation_artifacts(df)

    def upload_adapter_to_hf(
        self,
        repo_id: str,
        adapter_path: str,
        task_type: str,
        num_labels: int = 2,
        model_path: str = MODEL_PROSST_2048,
        structure_vocab_size: Optional[int] = None,
        private: bool = False,
        run_login: bool = True,
        title: str = "ColabProSST model",
        description: str = "A ProSST adapter trained with ColabProSST.",
        download_package: bool = False,
        allow_update: bool = False,
    ) -> Path:
        repo_id = self.normalize_hf_model_repo_id(repo_id)
        if task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported ProSST task_type: {task_type}.")
        structure_vocab_size = resolve_structure_vocab_size(
            model_path,
            structure_vocab_size,
        )

        adapter = Path(self.resolve_lora_adapter(adapter_path))
        validate_adapter_compatibility(
            str(adapter),
            task_type,
            model_path,
            structure_vocab_size,
            num_labels,
        )

        if run_login:
            from huggingface_hub import get_token, notebook_login

            notebook_login(new_session=False, write_permission=True)
            if get_token() is None:
                raise RuntimeError(
                    "Complete the Hugging Face login widget, then run the "
                    "upload again. A write token is required."
                )

        package_root = self.saprothub_dir / "model_to_push" / "prosst"
        package_dir = package_root / repo_id.replace("/", "__")
        shutil.rmtree(package_dir, ignore_errors=True)
        package_dir.mkdir(parents=True, exist_ok=True)

        for source in adapter.iterdir():
            destination = package_dir / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        input_format = "amino-acid input_ids + ProSST ss_input_ids"
        if task_type in PAIR_TASK_TYPES:
            input_format = f"two sets of {input_format}"
        metadata = {
            "schema_version": 1,
            "model_family": "ProSST",
            "base_model": model_path,
            "artifact_type": "lora",
            "checkpoint_format": "SaprotHub/ColabProSST PEFT adapter",
            "task_type": task_type,
            "input_format": input_format,
            "structure_vocab_size": structure_vocab_size,
            "colab_tool": "ColabProSST",
            "hub_namespace": repo_id.split("/", 1)[0],
        }
        if task_type in CLASSIFICATION_TASK_TYPES:
            metadata["num_labels"] = int(num_labels)
        (package_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        input_description = (
            "Each pair member has its own amino-acid tokenizer `input_ids` "
            "and matching ProSST structure token `ss_input_ids`."
            if task_type in PAIR_TASK_TYPES
            else "ColabProSST uses amino-acid tokenizer `input_ids` together "
            "with ProSST structure token `ss_input_ids`."
        )
        readme = f"""---
library_name: pytorch
base_model: {model_path}
tags:
- protein-language-model
- prosst
- colabprosst
- prossthub
- peft
---

# {title}

{description}

This repository contains a SaprotHub/ColabProSST PEFT adapter and metadata for a ProSST downstream model. The task head is saved with the adapter, while the official ProSST backbone is loaded from `{model_path}`.

## Input Format

{input_description} ColabProSST does not use SaProt AA+3Di merged tokens.

## Task

- Task type: `{task_type}`
- Base model: `{model_path}`
- Structure vocabulary: `{structure_vocab_size}`
- Artifact type: `LoRA / PEFT adapter`

In ColabProSST, choose **Hugging Face repository** as the model source and
enter `{repo_id}`. The interface reads `metadata.json`, selects the matching
official ProSST base model, and validates the task and structure vocabulary.

Use `saprot/scripts/predict_prosst.py` from SaprotHub
to run prediction with this artifact outside the notebook.
"""
        (package_dir / "README.md").write_text(readme, encoding="utf-8")

        from huggingface_hub import HfApi

        api = HfApi()
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=allow_update,
            )
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(package_dir),
                commit_message="Upload ColabProSST model artifact",
            )
        except Exception as exc:
            if repo_id.split("/", 1)[0].lower() == PROSST_HUB_NAMESPACE.lower():
                raise RuntimeError(
                    f"Could not upload to {repo_id}. Confirm that the logged-in "
                    f"Hugging Face account has write access to "
                    f"{PROSST_HUB_NAMESPACE}. Also use a new repository name, "
                    "or enable updating an existing repository."
                ) from exc
            raise
        print("uploaded to", f"https://huggingface.co/{repo_id}")

        if download_package:
            archive_path = Path(shutil.make_archive(str(package_dir), "zip", package_dir))
            self._download(archive_path)

        return package_dir
