import json
import tempfile
import unittest
from pathlib import Path

from saprot.utils.colab_prosst_acceptance import (
    ACCEPTANCE_SEQUENCE,
    ColabProSSTAcceptanceRunner,
)


class FakeWorkflow:
    def __init__(self, root):
        self.cache_dir = Path(root) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class ColabProSSTAcceptanceTest(unittest.TestCase):
    def test_steps_record_pass_fail_and_dependency_skip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ColabProSSTAcceptanceRunner(
                profile="core",
                output_root=tmpdir,
                saprothub_dir=tmpdir,
                require_gpu=False,
                workflow=FakeWorkflow(tmpdir),
            )
            runner.run_step("pass", lambda: {"value": Path("result.csv")})
            runner.run_step("fail", lambda: 1 / 0)
            runner.run_step(
                "skip",
                lambda: {},
                dependencies=("fail",),
            )

            self.assertEqual(
                [result.status for result in runner.results],
                ["PASS", "FAIL", "SKIP"],
            )
            report = json.loads(runner.report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["results"][0]["details"]["value"], "result.csv")
            self.assertIn("ZeroDivisionError", report["results"][1]["error"])

    def test_task_inputs_cover_all_supported_downstream_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ColabProSSTAcceptanceRunner(
                profile="core",
                output_root=tmpdir,
                saprothub_dir=tmpdir,
                require_gpu=False,
                workflow=FakeWorkflow(tmpdir),
            )
            runner.assets["structure_tokens"] = " ".join(
                "0" for _ in ACCEPTANCE_SEQUENCE
            )
            paths = runner.build_task_inputs()

            self.assertEqual(
                set(paths),
                {
                    "classification",
                    "regression",
                    "token_classification",
                    "pair_classification",
                    "pair_regression",
                    "mutation_sequence",
                    "saturation",
                    "embedding",
                },
            )
            for path in paths.values():
                self.assertTrue(path.is_file())

    def test_report_package_excludes_weight_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ColabProSSTAcceptanceRunner(
                profile="core",
                output_root=tmpdir,
                saprothub_dir=tmpdir,
                require_gpu=False,
                workflow=FakeWorkflow(tmpdir),
            )
            (runner.run_dir / "small.csv").write_text("x\n1\n", encoding="utf-8")
            (runner.run_dir / "large.pt").write_bytes(b"weights")
            runner.run_step("pass", lambda: {})
            package = runner.package_report()

            import zipfile

            with zipfile.ZipFile(package) as archive:
                names = set(archive.namelist())
            self.assertIn("small.csv", names)
            self.assertNotIn("large.pt", names)

    def test_core_run_orchestrates_every_workflow_and_returns_report(self):
        class SimulatedRunner(ColabProSSTAcceptanceRunner):
            def runtime_check(self):
                return {"cuda_available": True}

            def family_metadata_check(self):
                return {"models": ["simulated"]}

            def prepare_live_sequence_input(self):
                self.assets["structure_tokens"] = " ".join(
                    "0" for _ in ACCEPTANCE_SEQUENCE
                )
                return {"sequence_length": len(ACCEPTANCE_SEQUENCE)}

            def mutation_check(self):
                return {"rows": 2}

            def saturation_check(self):
                return {"score_rows": len(ACCEPTANCE_SEQUENCE) * 20}

            def embedding_check(self):
                return {"index_rows": 2}

            def train_task_check(self, task_type, asset_key, suffix, **kwargs):
                result = {
                    "checkpoint_path": f"{suffix}.pt",
                    "task_type": task_type,
                }
                self.assets[f"{suffix}_result"] = result
                return result

            def prediction_check(self):
                return {"rows": 6}

            def cleanup_task_artifacts(self):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = SimulatedRunner(
                profile="core",
                output_root=tmpdir,
                saprothub_dir=tmpdir,
                require_gpu=False,
                workflow=FakeWorkflow(tmpdir),
            )
            result = runner.run()

            self.assertTrue(result["success"])
            self.assertEqual(result["required_failures"], [])
            statuses = {item["name"]: item["status"] for item in result["results"]}
            self.assertEqual(statuses["Exact checkpoint resume"], "PASS")
            self.assertEqual(statuses["LoRA classification training"], "PASS")
            self.assertEqual(statuses["Protein-pair regression training"], "PASS")
            self.assertEqual(statuses["Six official model weight forwards"], "SKIP")
            self.assertTrue(Path(result["report_zip"]).is_file())


if __name__ == "__main__":
    unittest.main()
