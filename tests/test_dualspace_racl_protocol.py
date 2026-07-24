import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_paper_suite as suite  # noqa: E402


class DualSpaceRaclProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suite.configure(ROOT / "configs/racl_dualspace_boundary_tuning.json")
        cls.jobs = suite.build_jobs({"racl_tune"})
        suite.verify_frozen_contract(cls.jobs)
        cls.commands = {job.name: list(job.command) for job in cls.jobs}

    def test_budget_is_three_candidates_by_three_seeds_plus_selector(self):
        self.assertEqual(len(self.jobs), 10)
        self.assertEqual(
            len([job for job in self.jobs if job.name != "select_racl_tuning"]),
            9,
        )

    def test_all_training_jobs_are_validation_only_and_capacity_matched(self):
        for name, command in self.commands.items():
            if name == "select_racl_tuning":
                self.assertTrue(command[1].endswith("select_dualspace_racl.py"))
                continue
            self.assertIn("--validation_only", command)
            self.assertIn("--save_val_pred", command)
            self.assertNotIn("--cl_class_balanced", command)
            self.assertIn("--cl_no_attr_block", command)
            self.assertEqual(
                command[command.index("--racl_logit_alpha") + 1], "1.0"
            )
            self.assertEqual(
                command[command.index("--racl_memory_head_alpha") + 1], "1.0"
            )
            for flag, value in (
                ("--n_fusion", "1"),
                ("--heads", "8"),
                ("--fusion_dropout", "0.2"),
                ("--lr_fusion", "5e-05"),
            ):
                self.assertEqual(command[command.index(flag) + 1], value)

    def test_control_has_matched_heads_but_no_racl_or_memory_context(self):
        for seed in (0, 1, 2):
            command = self.commands[f"dualspace_racl_matched_no_racl_s{seed}"]
            self.assertIn("--no_cl", command)
            self.assertNotIn("--racl_dual_space", command)
            self.assertNotIn("--racl_memory_context", command)

    def test_two_racl_candidates_share_frozen_semantic_boundary_design(self):
        for candidate in ("dualspace_margin", "dualspace_margin_memory"):
            for seed in (0, 1, 2):
                command = self.commands[f"dualspace_racl_{candidate}_s{seed}"]
                for flag in (
                    "--racl_dual_space",
                    "--racl_all_samples",
                    "--racl_local_margin",
                    "--cl_exclude_self",
                ):
                    self.assertIn(flag, command)
                self.assertNotIn("--cl_hard_pos", command)
                self.assertNotIn("--cl_set_nce", command)
                self.assertEqual(command[command.index("--Kp") + 1], "3")
                self.assertEqual(command[command.index("--Kn") + 1], "5")
                self.assertEqual(command[command.index("--racl_margin") + 1], "0.15")
                self.assertEqual(
                    command[command.index("--racl_semantic_revision") + 1],
                    "dualspace_bge_joint_v1",
                )
                self.assertEqual(
                    command[command.index("--racl_semantic_cache_sha256") + 1],
                    "8688bfeffcb62d13914c3c8e1a047039de826e4b6224653e66c404890f67dba7",
                )

    def test_memory_context_is_only_enabled_for_the_third_candidate(self):
        enabled = {
            name for name, command in self.commands.items()
            if "--racl_memory_context" in command
        }
        self.assertEqual(enabled, {
            f"dualspace_racl_dualspace_margin_memory_s{seed}"
            for seed in (0, 1, 2)
        })


if __name__ == "__main__":
    unittest.main()
