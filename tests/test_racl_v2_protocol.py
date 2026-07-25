import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_paper_suite as suite  # noqa: E402


class RaclV2ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suite.configure(ROOT / "configs/racl_v2_shared_decision_tuning.json")
        cls.jobs = suite.build_jobs({"racl_tune"})
        suite.verify_frozen_contract(cls.jobs)
        cls.commands = {job.name: list(job.command) for job in cls.jobs}

    def test_budget_is_four_candidates_by_three_seeds_plus_selector(self):
        self.assertEqual(len(self.jobs), 13)
        self.assertEqual(
            len([job for job in self.jobs if job.name != "select_racl_tuning"]),
            12,
        )

    def test_every_candidate_uses_same_shared_classifier_and_locked_fusion(self):
        for name, command in self.commands.items():
            if name == "select_racl_tuning":
                continue
            self.assertIn("--validation_only", command)
            self.assertIn("--racl_logit_alpha", command)
            self.assertEqual(
                command[command.index("--racl_logit_alpha") + 1],
                "1.0",
            )
            self.assertIn("--cl_no_attr_block", command)
            self.assertIn("--cl_class_balanced", command)
            for flag, value in (
                ("--n_fusion", "1"),
                ("--heads", "8"),
                ("--fusion_dropout", "0.2"),
                ("--lr_fusion", "5e-05"),
            ):
                self.assertEqual(command[command.index(flag) + 1], value)

    def test_set_nce_is_preregistered_only_for_two_racl_candidates(self):
        set_jobs = {
            name
            for name, command in self.commands.items()
            if "--cl_set_nce" in command
        }
        self.assertEqual(
            set_jobs,
            {
                f"racl_v2_shared_set_l005_t010_s{seed}"
                for seed in (0, 1, 2)
            }
            | {
                f"racl_v2_shared_set_l010_t010_s{seed}"
                for seed in (0, 1, 2)
            },
        )

    def test_no_racl_control_keeps_identical_shared_classifier(self):
        for seed in (0, 1, 2):
            command = self.commands[f"racl_v2_no_racl_shared_s{seed}"]
            self.assertIn("--no_cl", command)
            self.assertIn("--racl_logit_alpha", command)
            self.assertNotIn("--cl_set_nce", command)

    def test_racl_candidates_exclude_self_and_keep_hard_negatives(self):
        for name, command in self.commands.items():
            if not name.startswith("racl_v2_shared_"):
                continue
            self.assertIn("--cl_exclude_self", command)
            self.assertEqual(command[command.index("--Kn") + 1], "5")
            self.assertNotIn("--cl_hard_pos", command)


if __name__ == "__main__":
    unittest.main()
