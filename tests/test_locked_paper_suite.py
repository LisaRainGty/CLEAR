import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_paper_suite as suite  # noqa: E402


class LockedNoFusionRaclSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suite.configure(ROOT / "configs/paper_fair_locked_no_fusion_racl.json")
        cls.jobs = suite.build_jobs({
            "audit", "table3", "ablation", "hparams", "xdom", "analysis",
        })
        cls.commands = {job.name: list(job.command) for job in cls.jobs}

    def test_expected_job_budget(self):
        self.assertEqual(len(self.jobs), 184)

    def test_canonical_is_locked_no_fusion_attribute_racl(self):
        command = self.commands["claimarc_canonical_s0"]
        self.assertIn("--no_fusion", command)
        self.assertIn("--cl_exclude_self", command)
        self.assertIn("--cl_class_balanced", command)
        self.assertNotIn("--cl_no_attr_block", command)
        for flag, value in (
            ("--warmup", "2"), ("--cl_epochs", "4"),
            ("--lambda_cl", "0.1"), ("--tau", "0.1"),
            ("--Kp", "3"), ("--Kn", "5"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
            self.assertEqual(command.count(flag), 1)

    def test_fusion_and_global_retrieval_are_explicit_ablations(self):
        with_fusion = self.commands["with_fusion_s0"]
        self.assertNotIn("--no_fusion", with_fusion)
        self.assertEqual(with_fusion[with_fusion.index("--n_fusion") + 1], "1")
        global_retrieval = self.commands["global_racl_retrieval_s0"]
        self.assertIn("--no_fusion", global_retrieval)
        self.assertIn("--cl_no_attr_block", global_retrieval)

    def test_cross_domain_claimarc_uses_same_lock(self):
        command = self.commands["xdom_rooms_s0_clarc"]
        self.assertIn("--no_fusion", command)
        self.assertIn("--cl_exclude_self", command)
        self.assertIn("--cl_class_balanced", command)
        self.assertNotIn("--cl_no_attr_block", command)


if __name__ == "__main__":
    unittest.main()
