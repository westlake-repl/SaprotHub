import gc
import json
import platform
import shutil
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch

from saprot.data.pdb2prosst import parse_structure_tokens
from saprot.data.sequence_to_prosst import predict_structure_with_esmfold
from saprot.model.prosst.specs import (
    DEFAULT_PROSST_MODEL,
    PROSST_MODEL_SPECS,
    get_prosst_model_spec,
)
from saprot.utils.colab_prosst_workflow import ColabProSSTWorkflow


ACCEPTANCE_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
ACCEPTANCE_PROFILES = {"core", "full"}


@dataclass
class AcceptanceResult:
    name: str
    status: str
    required: bool
    duration_seconds: float
    details: dict
    error: str = ""


class ColabProSSTAcceptanceRunner:
    """Run release checks through the real ColabProSST workflow APIs."""

    def __init__(
        self,
        profile: str = "full",
        output_root: str = "/content/colabprosst_acceptance",
        saprothub_dir: str = "/content/SaprotHub",
        model_path: str = DEFAULT_PROSST_MODEL.model_path,
        require_gpu: bool = True,
        run_hf_upload: bool = False,
        hf_repo_id: str = "",
        hf_private: bool = True,
        workflow=None,
    ):
        profile = str(profile).strip().lower()
        if profile not in ACCEPTANCE_PROFILES:
            raise ValueError(
                f"profile must be one of {sorted(ACCEPTANCE_PROFILES)}."
            )
        self.profile = profile
        self.model_spec = get_prosst_model_spec(model_path)
        self.require_gpu = bool(require_gpu)
        self.run_hf_upload = bool(run_hf_upload)
        self.hf_repo_id = str(hf_repo_id).strip()
        self.hf_private = bool(hf_private)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(output_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.results = []
        self.assets = {}
        self.task_prefix = f"ColabProSSTAcceptance_{self.run_id}"
        self.saprothub_dir = Path(saprothub_dir)
        self.workflow = workflow or ColabProSSTWorkflow(
            output_dir=str(self.run_dir / "outputs"),
            upload_dir=str(self.run_dir / "uploads"),
            asset_dir=str(self.run_dir / "structures"),
            cache_dir=str(Path(output_root) / "shared_cache"),
            saprothub_dir=str(self.saprothub_dir),
        )
        self.report_csv = self.run_dir / "acceptance_report.csv"
        self.report_json = self.run_dir / "acceptance_report.json"
        self.report_markdown = self.run_dir / "ACCEPTANCE_REPORT.md"
        self.report_zip = self.run_dir / "colabprosst_acceptance_report.zip"

    @property
    def result_by_name(self):
        return {result.name: result for result in self.results}

    def _dependencies_passed(self, dependencies) -> bool:
        results = self.result_by_name
        return all(
            dependency in results and results[dependency].status == "PASS"
            for dependency in dependencies
        )

    @staticmethod
    def _json_safe(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): ColabProSSTAcceptanceRunner._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [ColabProSSTAcceptanceRunner._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def run_step(
        self,
        name: str,
        action: Callable,
        required: bool = True,
        dependencies=(),
    ) -> Optional[dict]:
        if not self._dependencies_passed(dependencies):
            missing = [
                dependency
                for dependency in dependencies
                if self.result_by_name.get(dependency, None) is None
                or self.result_by_name[dependency].status != "PASS"
            ]
            self.results.append(
                AcceptanceResult(
                    name=name,
                    status="SKIP",
                    required=required,
                    duration_seconds=0.0,
                    details={"blocked_by": missing},
                )
            )
            print(f"[SKIP] {name}: blocked by {missing}")
            self.write_reports()
            return None

        print(f"\n===== {name} =====", flush=True)
        started = time.perf_counter()
        try:
            details = self._json_safe(action() or {})
            result = AcceptanceResult(
                name=name,
                status="PASS",
                required=required,
                duration_seconds=round(time.perf_counter() - started, 3),
                details=details,
            )
            print(f"[PASS] {name}", flush=True)
            return details
        except Exception as exc:
            result = AcceptanceResult(
                name=name,
                status="FAIL",
                required=required,
                duration_seconds=round(time.perf_counter() - started, 3),
                details={},
                error="".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            )
            print(f"[FAIL] {name}: {exc}", flush=True)
            return None
        finally:
            self.results.append(result)
            self._release_memory()
            self.write_reports()

    @staticmethod
    def _release_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def runtime_check(self) -> dict:
        cuda_available = torch.cuda.is_available()
        if self.require_gpu and not cuda_available:
            raise RuntimeError(
                "A GPU runtime is required for full Colab acceptance. Select "
                "Runtime > Change runtime type > GPU and run again."
            )
        details = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "profile": self.profile,
            "model_path": self.model_spec.model_path,
        }
        if cuda_available:
            details.update(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "gpu_memory_gib": round(
                        torch.cuda.get_device_properties(0).total_memory
                        / (1024**3),
                        2,
                    ),
                }
            )
        return details

    def family_metadata_check(self) -> dict:
        from transformers import AutoConfig, AutoTokenizer

        checked = []
        for spec in PROSST_MODEL_SPECS:
            AutoConfig.from_pretrained(spec.model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(
                spec.model_path,
                trust_remote_code=True,
            )
            encoded = tokenizer(ACCEPTANCE_SEQUENCE)
            if len(encoded["input_ids"]) != len(ACCEPTANCE_SEQUENCE) + 2:
                raise ValueError(
                    f"Unexpected tokenizer length for {spec.model_path}."
                )
            checked.append(
                {
                    "model_path": spec.model_path,
                    "structure_vocab_size": spec.structure_vocab_size,
                    "encoded_length": len(encoded["input_ids"]),
                }
            )
        return {"models": checked}

    def prepare_live_sequence_input(self) -> dict:
        source_csv = self.run_dir / "sequence_only_input.csv"
        pd.DataFrame([{"sequence": ACCEPTANCE_SEQUENCE}]).to_csv(
            source_csv,
            index=False,
        )
        prepared = self.workflow.prepare_sequence_input_csv(
            input_csv=str(source_csv),
            structure_vocab_size=self.model_spec.structure_vocab_size,
            output_csv=str(self.run_dir / "prepared_input.csv"),
            download=False,
        )
        prepared_csv = Path(prepared.attrs["output_csv"])
        row = pd.read_csv(prepared_csv).iloc[0]
        tokens = parse_structure_tokens(row["structure_tokens"])
        if len(tokens) != len(ACCEPTANCE_SEQUENCE):
            raise ValueError(
                "Prepared structure token length does not match the sequence."
            )
        if int(row["structure_vocab_size"]) != self.model_spec.structure_vocab_size:
            raise ValueError("Prepared structure vocabulary does not match the model.")

        pdb_path = Path(
            predict_structure_with_esmfold(
                ACCEPTANCE_SEQUENCE,
                cache_dir=str(self.workflow.cache_dir),
            )
        )
        converted = self.workflow.convert_structure(
            structure_path=str(pdb_path),
            structure_vocab_size=self.model_spec.structure_vocab_size,
            output_csv=str(self.run_dir / "converted_structure.csv"),
            download=False,
        )
        converted_tokens = parse_structure_tokens(
            converted.iloc[0]["structure_tokens"]
        )
        if converted_tokens != tokens:
            raise ValueError(
                "Direct structure conversion and sequence-only preparation "
                "produced different tokens."
            )

        self.assets.update(
            {
                "source_csv": source_csv,
                "prepared_csv": prepared_csv,
                "pdb_path": pdb_path,
                "structure_tokens": row["structure_tokens"],
            }
        )
        return {
            "sequence_length": len(ACCEPTANCE_SEQUENCE),
            "structure_token_count": len(tokens),
            "structure_vocab_size": self.model_spec.structure_vocab_size,
            "prepared_csv": prepared_csv,
            "predicted_pdb": pdb_path,
        }

    def build_task_inputs(self) -> dict:
        token_text = str(self.assets["structure_tokens"])
        vocab_size = self.model_spec.structure_vocab_size
        sequence = ACCEPTANCE_SEQUENCE
        stages = ["train", "train", "valid", "valid", "test", "test"]
        labels = [0, 1, 0, 1, 0, 1]

        def write(name, rows):
            path = self.run_dir / name
            pd.DataFrame(rows).to_csv(path, index=False)
            self.assets[path.stem] = path
            return path

        common = {
            "sequence": sequence,
            "structure_tokens": token_text,
            "structure_vocab_size": vocab_size,
        }
        classification = write(
            "classification_prepared.csv",
            [
                {**common, "label": label, "stage": stage}
                for label, stage in zip(labels, stages)
            ],
        )
        regression = write(
            "regression_prepared.csv",
            [
                {**common, "label": value, "stage": stage}
                for value, stage in zip(
                    [0.1, 0.9, 0.2, 0.8, 0.3, 0.7], stages
                )
            ],
        )
        residue_labels = " ".join(
            str(index % 2) for index in range(len(sequence))
        )
        token_classification = write(
            "token_classification_prepared.csv",
            [
                {**common, "residue_labels": residue_labels, "stage": stage}
                for stage in stages
            ],
        )
        pair_common = {
            "sequence_1": sequence,
            "sequence_2": sequence,
            "structure_tokens_1": token_text,
            "structure_tokens_2": token_text,
            "structure_vocab_size": vocab_size,
        }
        pair_classification = write(
            "pair_classification_prepared.csv",
            [
                {**pair_common, "label": label, "stage": stage}
                for label, stage in zip(labels, stages)
            ],
        )
        pair_regression = write(
            "pair_regression_prepared.csv",
            [
                {**pair_common, "label": value, "stage": stage}
                for value, stage in zip(
                    [0.1, 0.9, 0.2, 0.8, 0.3, 0.7], stages
                )
            ],
        )
        mutation_sequence = write(
            "mutation_sequence_only.csv",
            [
                {"sequence": sequence, "mutant": "A1G"},
                {"sequence": sequence, "mutant": "C2A:D3E"},
            ],
        )
        saturation = write("saturation_prepared.csv", [common])
        embedding = write("embedding_prepared.csv", [common, common])
        return {
            "classification": classification,
            "regression": regression,
            "token_classification": token_classification,
            "pair_classification": pair_classification,
            "pair_regression": pair_regression,
            "mutation_sequence": mutation_sequence,
            "saturation": saturation,
            "embedding": embedding,
        }

    def family_model_forward_check(self, spec) -> dict:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_path,
            trust_remote_code=True,
        )
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if device.type == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForMaskedLM.from_pretrained(
            spec.model_path,
            **load_kwargs,
        )
        try:
            model = model.to(device).eval()
            inputs = tokenizer(ACCEPTANCE_SEQUENCE, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            ss_input_ids = torch.full_like(inputs["input_ids"], 3)
            ss_input_ids[:, 0] = 1
            ss_input_ids[:, -1] = 2
            with torch.inference_mode():
                output = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    ss_input_ids=ss_input_ids,
                )
            logits = output.logits
            if logits.shape[:2] != inputs["input_ids"].shape:
                raise ValueError(
                    f"Unexpected logits shape for {spec.model_path}: "
                    f"{tuple(logits.shape)}."
                )
            return {
                "model_path": spec.model_path,
                "structure_vocab_size": spec.structure_vocab_size,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "input_shape": list(inputs["input_ids"].shape),
                "ss_input_shape": list(ss_input_ids.shape),
                "logits_shape": list(logits.shape),
                "device": str(device),
            }
        finally:
            del model

    @staticmethod
    def _require_file(path, label: str) -> Path:
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label} was not created: {path}")
        return path

    def mutation_check(self) -> dict:
        output_csv = self.run_dir / "mutation_scores.csv"
        result = self.workflow.run_zero_shot(
            input_csv=str(self.assets["mutation_sequence_only"]),
            model_path=self.model_spec.model_path,
            structure_vocab_size=self.model_spec.structure_vocab_size,
            output_csv=str(output_csv),
            download=False,
            input_mode="sequence",
        )
        self._require_file(output_csv, "Mutation score CSV")
        if len(result) != 2 or "score" not in result.columns:
            raise ValueError("Mutation output does not contain two score rows.")
        prepared = self._require_file(
            result.attrs["prepared_input_csv"],
            "Mutation prepared-input CSV",
        )
        return {
            "rows": len(result),
            "output_csv": output_csv,
            "prepared_input_csv": prepared,
        }

    def saturation_check(self) -> dict:
        result = self.workflow.run_saturation_mutagenesis(
            input_csv=str(self.assets["saturation_prepared"]),
            model_path=self.model_spec.model_path,
            structure_vocab_size=self.model_spec.structure_vocab_size,
            output_csv=str(self.run_dir / "saturation_scores.csv"),
            output_matrix_csv=str(self.run_dir / "saturation_matrix.csv"),
            output_heatmap_png=str(self.run_dir / "saturation_heatmap.png"),
            download=False,
            input_mode="tokens",
        )
        paths = {
            "scores": self._require_file(result["output_csv"], "Saturation scores"),
            "matrix": self._require_file(
                result["output_matrix_csv"], "Saturation matrix"
            ),
            "heatmap": self._require_file(
                result["output_heatmap_png"], "Saturation heatmap"
            ),
            "archive": self._require_file(
                result["archive_path"], "Saturation archive"
            ),
        }
        scores = pd.read_csv(paths["scores"])
        expected_rows = len(ACCEPTANCE_SEQUENCE) * 20
        if len(scores) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} saturation scores, got {len(scores)}."
            )
        return {**paths, "score_rows": len(scores)}

    def embedding_check(self) -> dict:
        result = self.workflow.extract_embeddings(
            input_csv=str(self.assets["embedding_prepared"]),
            model_path=self.model_spec.model_path,
            structure_vocab_size=self.model_spec.structure_vocab_size,
            level="both",
            batch_size=1,
            max_length=2046,
            output_pt=str(self.run_dir / "both_embeddings.pt"),
            output_index_csv=str(self.run_dir / "both_embeddings_index.csv"),
            download=False,
            input_mode="tokens",
        )
        embedding_path = self._require_file(
            result["output_pt"], "Embedding tensor file"
        )
        index_path = self._require_file(
            result["output_index_csv"], "Embedding index CSV"
        )
        archive_path = self._require_file(
            result["archive_path"], "Embedding archive"
        )
        index = pd.read_csv(index_path)
        if len(index) != 2:
            raise ValueError(f"Expected two embedding index rows, got {len(index)}.")
        saved = torch.load(embedding_path, map_location="cpu")
        if not isinstance(saved, dict) or not saved:
            raise ValueError("Embedding tensor file is empty or malformed.")
        del saved
        return {
            "output_pt": embedding_path,
            "output_index_csv": index_path,
            "archive_path": archive_path,
            "index_rows": len(index),
        }

    def train_task_check(
        self,
        task_type: str,
        asset_key: str,
        suffix: str,
        training_method: str = "full",
        save_training_state: bool = False,
        initial_checkpoint: str = "",
        resume_optimizer_state: bool = False,
        keep_checkpoint: bool = False,
    ) -> dict:
        result = self.workflow.train_downstream(
            task_type=task_type,
            input_csv=str(self.assets[asset_key]),
            task_name=f"{self.task_prefix}_{suffix}",
            num_labels=2,
            max_epochs=1,
            batch_size=1,
            learning_rate=2.0e-5,
            model_path=self.model_spec.model_path,
            structure_vocab_size=self.model_spec.structure_vocab_size,
            freeze_backbone=True,
            gradient_checkpointing=True,
            initial_checkpoint=initial_checkpoint,
            resume_optimizer_state=resume_optimizer_state,
            save_training_state=save_training_state,
            training_method=training_method,
            download=False,
            input_mode="tokens",
        )
        checkpoint = Path(result["checkpoint_path"])
        if checkpoint.is_dir():
            if not (checkpoint / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    f"LoRA adapter config was not created: {checkpoint}"
                )
        else:
            self._require_file(checkpoint, "Training checkpoint")
        test_csv = self._require_file(
            result["test_result_csv"],
            "Training test-prediction CSV",
        )
        test_rows = len(pd.read_csv(test_csv))
        if test_rows != 2:
            raise ValueError(f"Expected two test predictions, got {test_rows}.")
        if save_training_state and training_method == "full":
            state = torch.load(checkpoint, map_location="cpu")
            missing = sorted(
                {
                    "model",
                    "global_step",
                    "epoch",
                    "best_value",
                    "lr_scheduler",
                    "optimizer",
                }
                - set(state)
            )
            del state
            if missing:
                raise ValueError(
                    f"Exact-resume checkpoint is missing fields: {missing}."
                )
        details = {
            **result,
            "checkpoint_exists": checkpoint.exists(),
            "test_rows": test_rows,
        }
        self.assets[f"{suffix}_result"] = result
        if initial_checkpoint and resume_optimizer_state:
            initial_path = Path(initial_checkpoint)
            if initial_path != checkpoint:
                initial_path.unlink(missing_ok=True)
        if not keep_checkpoint:
            if checkpoint.is_dir():
                shutil.rmtree(checkpoint, ignore_errors=True)
            else:
                checkpoint.unlink(missing_ok=True)
            download_path = Path(result["checkpoint_download_path"])
            if download_path != checkpoint:
                download_path.unlink(missing_ok=True)
        return details

    def prediction_check(self) -> dict:
        training = self.assets["classification_full_result"]
        output_csv = self.run_dir / "classification_predictions.csv"
        result = self.workflow.predict_downstream(
            task_type="classification",
            input_csv=str(self.assets["classification_prepared"]),
            checkpoint_path=training["checkpoint_path"],
            num_labels=2,
            batch_size=1,
            model_path=self.model_spec.model_path,
            structure_vocab_size=self.model_spec.structure_vocab_size,
            output_csv=str(output_csv),
            download=False,
            input_mode="tokens",
        )
        self._require_file(output_csv, "Classification prediction CSV")
        if len(result) != 6:
            raise ValueError(f"Expected six predictions, got {len(result)}.")
        return {"output_csv": output_csv, "rows": len(result)}

    def hf_upload_check(self) -> dict:
        if not self.hf_repo_id:
            raise ValueError(
                "Set HF_REPO_ID to a new model repository before enabling upload."
            )
        from huggingface_hub import get_token

        if get_token() is None:
            raise RuntimeError(
                "No Hugging Face token is available. Add HF_TOKEN to Colab "
                "Secrets or log in before running acceptance."
            )
        training = self.assets.get(
            "classification_resume_result",
            self.assets["classification_full_result"],
        )
        package = self.workflow.upload_checkpoint_to_hf(
            repo_id=self.hf_repo_id,
            checkpoint_path=training["checkpoint_path"],
            task_type="classification",
            num_labels=2,
            model_path=self.model_spec.model_path,
            structure_vocab_size=self.model_spec.structure_vocab_size,
            private=self.hf_private,
            run_login=False,
            title="ColabProSST acceptance model",
            description=(
                "Temporary model created by the automated ColabProSST "
                "release acceptance suite."
            ),
            download_package=False,
            allow_update=False,
        )
        self._require_file(package / "metadata.json", "Uploaded package metadata")
        return {
            "repo_id": self.hf_repo_id,
            "url": f"https://huggingface.co/{self.hf_repo_id}",
            "package_dir": package,
        }

    def skip_step(self, name: str, reason: str, required: bool = False):
        self.results.append(
            AcceptanceResult(
                name=name,
                status="SKIP",
                required=required,
                duration_seconds=0.0,
                details={"reason": reason},
            )
        )
        print(f"[SKIP] {name}: {reason}")
        self.write_reports()

    def run(self) -> dict:
        runtime_name = "Runtime and GPU"
        metadata_name = "Official family metadata and tokenizers"
        preparation_name = "Sequence-only preparation and direct conversion"
        assets_name = "Prepared-token task inputs"

        try:
            self.run_step(runtime_name, self.runtime_check)
            self.run_step(
                metadata_name,
                self.family_metadata_check,
                dependencies=(runtime_name,),
            )
            self.run_step(
                preparation_name,
                self.prepare_live_sequence_input,
                dependencies=(runtime_name,),
            )
            self.run_step(
                assets_name,
                self.build_task_inputs,
                dependencies=(preparation_name,),
            )

            workflow_dependency = (assets_name,)
            self.run_step(
                "Sequence-only zero-shot mutation",
                self.mutation_check,
                dependencies=workflow_dependency,
            )
            self.run_step(
                "Prepared-token saturation mutagenesis",
                self.saturation_check,
                dependencies=workflow_dependency,
            )
            self.run_step(
                "Prepared-token protein and residue embeddings",
                self.embedding_check,
                dependencies=workflow_dependency,
            )
            self.run_step(
                "Full classification training with resume state",
                lambda: self.train_task_check(
                    "classification",
                    "classification_prepared",
                    "classification_full",
                    save_training_state=True,
                    keep_checkpoint=True,
                ),
                dependencies=workflow_dependency,
            )
            self.run_step(
                "Checkpoint classification prediction",
                self.prediction_check,
                dependencies=("Full classification training with resume state",),
            )
            self.run_step(
                "Exact checkpoint resume",
                lambda: self.train_task_check(
                    "classification",
                    "classification_prepared",
                    "classification_resume",
                    save_training_state=True,
                    initial_checkpoint=self.assets[
                        "classification_full_result"
                    ]["checkpoint_path"],
                    resume_optimizer_state=True,
                    keep_checkpoint=True,
                ),
                dependencies=("Full classification training with resume state",),
            )
            self.run_step(
                "LoRA classification training",
                lambda: self.train_task_check(
                    "classification",
                    "classification_prepared",
                    "classification_lora",
                    training_method="lora",
                ),
                dependencies=workflow_dependency,
            )
            for task_type, asset_key, label in [
                ("regression", "regression_prepared", "Protein regression training"),
                (
                    "token_classification",
                    "token_classification_prepared",
                    "Residue classification training",
                ),
                (
                    "pair_classification",
                    "pair_classification_prepared",
                    "Protein-pair classification training",
                ),
                (
                    "pair_regression",
                    "pair_regression_prepared",
                    "Protein-pair regression training",
                ),
            ]:
                self.run_step(
                    label,
                    lambda task_type=task_type, asset_key=asset_key: (
                        self.train_task_check(
                            task_type,
                            asset_key,
                            task_type,
                        )
                    ),
                    dependencies=workflow_dependency,
                )

            if self.profile == "full":
                for spec in PROSST_MODEL_SPECS:
                    self.run_step(
                        f"Official model forward ProSST-{spec.structure_vocab_size}",
                        lambda spec=spec: self.family_model_forward_check(spec),
                        dependencies=(runtime_name, metadata_name),
                    )
            else:
                self.skip_step(
                    "Six official model weight forwards",
                    "Core profile checks metadata/tokenizers but skips six "
                    "sequential weight downloads and forwards.",
                )

            if self.run_hf_upload:
                self.run_step(
                    "Hugging Face model upload",
                    self.hf_upload_check,
                    required=False,
                    dependencies=("Full classification training with resume state",),
                )
            else:
                self.skip_step(
                    "Hugging Face model upload",
                    "Disabled. Enable RUN_HF_UPLOAD and provide a new HF_REPO_ID.",
                )
        finally:
            self.cleanup_task_artifacts()
            package = self.package_report()

        required_failures = [
            result.name
            for result in self.results
            if result.required and result.status != "PASS"
        ]
        return {
            "success": not required_failures,
            "required_failures": required_failures,
            "report_csv": str(self.report_csv),
            "report_json": str(self.report_json),
            "report_markdown": str(self.report_markdown),
            "report_zip": str(package),
            "results": [asdict(result) for result in self.results],
        }

    def write_reports(self):
        records = [asdict(result) for result in self.results]
        flat_records = [
            {
                "name": record["name"],
                "status": record["status"],
                "required": record["required"],
                "duration_seconds": record["duration_seconds"],
                "details": json.dumps(record["details"], ensure_ascii=False),
                "error": record["error"],
            }
            for record in records
        ]
        pd.DataFrame(
            flat_records,
            columns=[
                "name",
                "status",
                "required",
                "duration_seconds",
                "details",
                "error",
            ],
        ).to_csv(self.report_csv, index=False)
        self.report_json.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "profile": self.profile,
                    "model_path": self.model_spec.model_path,
                    "results": records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        passed = sum(result.status == "PASS" for result in self.results)
        failed = sum(result.status == "FAIL" for result in self.results)
        skipped = sum(result.status == "SKIP" for result in self.results)
        lines = [
            "# ColabProSST Acceptance Report",
            "",
            f"- Run ID: `{self.run_id}`",
            f"- Profile: `{self.profile}`",
            f"- Base model: `{self.model_spec.model_path}`",
            f"- Results: **{passed} passed, {failed} failed, {skipped} skipped**",
            "",
            "| Check | Status | Required | Seconds |",
            "|---|---:|---:|---:|",
        ]
        lines.extend(
            f"| {result.name} | {result.status} | {result.required} | "
            f"{result.duration_seconds:.3f} |"
            for result in self.results
        )
        lines.extend(
            [
                "",
                "## Manual Checks",
                "",
                "- Confirm each browser download button starts exactly one download.",
                "- Confirm the Colab interface remains visible after long tasks.",
                "- If Hugging Face upload is disabled, test it separately with a write token.",
                "",
            ]
        )
        self.report_markdown.write_text("\n".join(lines), encoding="utf-8")

    def package_report(self) -> Path:
        self.write_reports()
        with zipfile.ZipFile(
            self.report_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path in sorted(self.run_dir.rglob("*")):
                if not path.is_file() or path == self.report_zip:
                    continue
                if path.suffix.lower() in {".pt", ".bin", ".safetensors"}:
                    continue
                archive.write(path, arcname=path.relative_to(self.run_dir))
        return self.report_zip

    def cleanup_task_artifacts(self):
        weight_dir = self.saprothub_dir / "weights" / "prosst"
        lmdb_dir = self.saprothub_dir / "LMDB"
        for path in weight_dir.glob(f"{self.task_prefix}*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        for path in lmdb_dir.glob(f"{self.task_prefix}*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
