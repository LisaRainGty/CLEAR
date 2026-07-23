import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_paper_suite as suite  # noqa: E402


class LockedFusionGlobalRaclTuningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suite.configure(ROOT / "configs/racl_locked_fusion_tuning.json")
        cls.jobs = suite.build_jobs({"racl_tune"})
        cls.commands = {job.name: list(job.command) for job in cls.jobs}

    def test_minimal_nonredundant_job_budget(self):
        self.assertEqual(len(self.jobs), 13)
        self.assertNotIn(
            "racl_fg_global_l050_t007_k3_5_w3c6_reused_s0",
            self.commands,
        )

    def test_every_training_job_uses_locked_fusion_and_global_retrieval(self):
        training = [job for job in self.jobs if job.name != "select_racl_tuning"]
        self.assertEqual(len(training), 12)
        for job in training:
            command = list(job.command)
            self.assertNotIn("--no_fusion", command)
            self.assertIn("--cl_no_attr_block", command)
            self.assertIn("--cl_class_balanced", command)
            for flag, value in (
                ("--n_fusion", "1"),
                ("--heads", "8"),
                ("--fusion_dropout", "0.2"),
                ("--lr_fusion", "5e-05"),
            ):
                self.assertEqual(command[command.index(flag) + 1], value)
                self.assertEqual(command.count(flag), 1)

    def test_no_racl_is_diagnostic_only(self):
        for seed in (0, 1, 2):
            self.assertIn("--no_cl", self.commands[f"racl_fg_no_racl_w3c6_s{seed}"])
        for name, command in self.commands.items():
            if name.startswith("racl_fg_global_"):
                self.assertNotIn("--no_cl", command)


if __name__ == "__main__":
    unittest.main()
