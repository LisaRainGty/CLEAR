import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_paper_suite_fusion_hard_confirmation",
    ROOT / "scripts/run_paper_suite.py",
)
suite = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = suite
RUNNER_SPEC.loader.exec_module(suite)

SELECTOR_SPEC = importlib.util.spec_from_file_location(
    "select_fusion_racl_combo",
    ROOT / "scripts/select_fusion_racl_combo.py",
)
selector = importlib.util.module_from_spec(SELECTOR_SPEC)
sys.modules[SELECTOR_SPEC.name] = selector
SELECTOR_SPEC.loader.exec_module(selector)


class FusionRaclComboSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ROOT / "configs/racl_locked_fusion_hard_positive_confirmation.json"
        suite.configure(cls.config)
        cls.jobs = suite.build_jobs({"racl_tune"})
        suite.verify_frozen_contract(cls.jobs)
        cls.by_name = {job.name: job for job in cls.jobs}

    def test_only_three_new_fusion_hard_positive_runs_plus_reused_control(self):
        self.assertEqual(len(self.jobs), 7)
        expected = {
            f"racl_fg_global_hardpos_l050_t007_k3_5_w3c6_s{seed}"
            for seed in (0, 1, 2)
        }
        self.assertEqual(
            {name for name in self.by_name if "global_hardpos" in name},
            expected,
        )

    def test_candidate_locks_fusion_and_hard_positive(self):
        for seed in (0, 1, 2):
            command = self.by_name[
                f"racl_fg_global_hardpos_l050_t007_k3_5_w3c6_s{seed}"
            ].command
            self.assertNotIn("--no_fusion", command)
            for flag in (
                "--validation_only", "--cl_hard_pos",
                "--cl_no_attr_block", "--cl_class_balanced",
            ):
                self.assertIn(flag, command)
            for flag, value in (
                ("--n_fusion", "1"), ("--heads", "8"),
                ("--fusion_dropout", "0.2"), ("--lr_fusion", "5e-05"),
                ("--lambda_cl", "0.5"), ("--tau", "0.07"),
                ("--Kp", "3"), ("--Kn", "5"),
            ):
                self.assertEqual(command[command.index(flag) + 1], value)

    def test_selector_accepts_parent_phase_candidate(self):
        manifest = {
            "aggregates": {},
            "parent_phase_summary": {
                "candidate": "parent",
                "aggregate": {"seeds": [0, 1, 2]},
            },
        }
        self.assertEqual(
            selector.extract_candidate(manifest, "parent"),
            {"seeds": [0, 1, 2]},
        )


if __name__ == "__main__":
    unittest.main()
