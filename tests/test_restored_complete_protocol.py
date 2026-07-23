import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_paper_suite as suite  # noqa: E402
from scripts.aggregate_paper_results import aggregate  # noqa: E402


class RestoredCompleteProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suite.configure(ROOT / "configs/paper_fair.json")
        cls.jobs = suite.build_jobs({"table3", "ablation", "xdom"})
        cls.commands = {job.name: list(job.command) for job in cls.jobs}

    def test_complete_model_matches_recorded_three_seed_command(self):
        for seed in (0, 1, 2):
            command = self.commands[f"claimarc_canonical_s{seed}"]
            self.assertNotIn("--no_fusion", command)
            self.assertNotIn("--cl_exclude_self", command)
            self.assertNotIn("--validation_only", command)
            self.assertIn("--cl_no_attr_block", command)
            self.assertIn("--cl_class_balanced", command)
            for flag, value in (
                ("--warmup", "3"),
                ("--cl_epochs", "6"),
                ("--lambda_cl", "0.5"),
                ("--tau", "0.07"),
                ("--Kp", "3"),
                ("--Kn", "5"),
            ):
                self.assertEqual(command[command.index(flag) + 1], value)
                self.assertEqual(command.count(flag), 1)

    def test_no_fusion_and_no_racl_remain_ablations(self):
        canonical = self.commands["claimarc_canonical_s0"]
        no_fusion = self.commands["no_fusion_s0"]
        no_racl = self.commands["no_racl_s0"]
        self.assertNotIn("--no_fusion", canonical)
        self.assertIn("--no_fusion", no_fusion)
        self.assertIn("--no_cl", no_racl)

    def test_cross_domain_claimarc_uses_complete_protocol(self):
        command = self.commands["xdom_rooms_s0_clarc"]
        self.assertNotIn("--no_fusion", command)
        self.assertNotIn("--cl_exclude_self", command)
        self.assertIn("--cl_no_attr_block", command)
        self.assertIn("--cl_class_balanced", command)
        for flag, value in (
            ("--warmup", "3"),
            ("--cl_epochs", "6"),
            ("--lambda_cl", "0.5"),
            ("--tau", "0.07"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_paper_aggregation_reproduces_archived_sample_sd(self):
        rows = [
            {"tag": "claimarc_canonical", "seed": 0, "macro_f1": 0.8100},
            {"tag": "claimarc_canonical", "seed": 1, "macro_f1": 0.8072},
            {"tag": "claimarc_canonical", "seed": 2, "macro_f1": 0.8017},
        ]
        metric = aggregate(rows)["claimarc_canonical"]["macro_f1"]
        self.assertAlmostEqual(metric["mean"], 0.8063, places=7)
        self.assertAlmostEqual(metric["std"], 0.0042225585, places=7)
        self.assertEqual(metric["std_ddof"], 1)


if __name__ == "__main__":
    unittest.main()
