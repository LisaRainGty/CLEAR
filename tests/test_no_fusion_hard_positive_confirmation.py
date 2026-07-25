import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_paper_suite_hard_positive_confirmation",
    ROOT / "scripts/run_paper_suite.py",
)
suite = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = suite
SPEC.loader.exec_module(suite)


class NoFusionHardPositiveConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ROOT / "configs/racl_no_fusion_hard_positive_confirmation.json"
        suite.configure(cls.config)
        cls.jobs = suite.build_jobs({"racl_tune"})
        suite.verify_frozen_contract(cls.jobs)
        cls.by_name = {job.name: job for job in cls.jobs}

    def test_only_three_new_hard_positive_runs_plus_reused_control(self):
        self.assertEqual(len(self.jobs), 7)
        self.assertEqual(
            {
                name for name in self.by_name
                if name.startswith("racl_nf_racl_hardpos_l050_t007_s")
            },
            {
                f"racl_nf_racl_hardpos_l050_t007_s{seed}"
                for seed in (0, 1, 2)
            },
        )

    def test_hard_positive_candidate_is_validation_only_and_no_fusion(self):
        for seed in (0, 1, 2):
            command = self.by_name[
                f"racl_nf_racl_hardpos_l050_t007_s{seed}"
            ].command
            for flag in (
                "--validation_only", "--no_fusion", "--cl_hard_pos",
                "--cl_no_attr_block", "--cl_class_balanced",
            ):
                self.assertIn(flag, command)
            self.assertNotIn("--cl_exclude_self", command)
            for flag, value in (
                ("--lambda_cl", "0.5"), ("--tau", "0.07"),
                ("--Kp", "3"), ("--Kn", "5"),
                ("--warmup", "3"), ("--cl_epochs", "6"),
            ):
                self.assertEqual(command[command.index(flag) + 1], value)

    def test_control_signature_remains_reusable(self):
        command = self.by_name["racl_nf_no_racl_s0"].command
        self.assertIn("--validation_only", command)
        self.assertIn("--no_fusion", command)
        self.assertIn("--no_cl", command)
        self.assertNotIn("--cl_hard_pos", command)


if __name__ == "__main__":
    unittest.main()
