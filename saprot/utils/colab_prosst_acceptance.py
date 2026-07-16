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

