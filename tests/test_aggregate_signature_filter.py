import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_paper_results import command_signature, load_rows


class AggregateSignatureFilterTests(unittest.TestCase):
    def test_stale_and_legacy_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_root = root / "results" / "suite"
            (result_root / "jobs").mkdir(parents=True)
            (result_root / "status").mkdir()
            rows = [
                {
                    "tag": "claimarc_canonical", "_suite_job": "canonical_s0",
                    "evidence_policy": "args_only", "acc": 0.8,
                },
                {
                    "tag": "old_ablation", "_suite_job": "ablation_s0",
                    "evidence_policy": "args_only", "acc": 0.7,
                },
                {
                    "tag": "qwen2p5_7b_qlora_sft",
                    "_suite_job": "qwen2p5_7b_qlora_s0",
                    "evidence_policy": "args_only", "acc": 0.7,
                },
            ]
            (result_root / "jobs" / "rows.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            for name, command in (
                ("canonical_s0", ("/remote/venv/python", "-m", "model", "--x", "1")),
                ("ablation_s0", ("/remote/venv/python", "-m", "model", "--x", "0")),
            ):
                (result_root / "status" / f"{name}.json").write_text(
                    json.dumps({
                        "returncode": 0,
                        "command": command,
                        "signature": command_signature(name, command),
                    }),
                    encoding="utf-8")

            accepted, rejected = load_rows(
                root, result_root, "args_only",
                expected_commands={
                    "canonical_s0": (
                        "/local/python", "-m", "model", "--x", "1"),
                    "ablation_s0": (
                        "/local/python", "-m", "model", "--x", "1"),
                },
                excluded_tags={"qwen2p5_7b_qlora_sft"},
            )

            self.assertEqual(
                [row["tag"] for row in accepted], ["claimarc_canonical"])
            reasons = {
                item["tag"]: item["reasons"] for item in rejected
            }
            self.assertIn("stale_job_command", reasons["old_ablation"])
            self.assertIn(
                "explicitly_excluded_legacy_result",
                reasons["qwen2p5_7b_qlora_sft"],
            )


if __name__ == "__main__":
    unittest.main()
