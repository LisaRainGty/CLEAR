import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_paper_suite_final", ROOT / "scripts/run_paper_suite.py")
suite = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = suite
SPEC.loader.exec_module(suite)


class FinalLockedFusionGlobalRaclSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ROOT / "configs/paper_fair_locked_fusion_v2.json"
        suite.configure(cls.config)
        cls.jobs = suite.build_jobs({"table3", "ablation"})
        suite.verify_frozen_contract(cls.jobs)
        cls.by_name = {job.name: job for job in cls.jobs}

    def test_canonical_locks_selected_fusion_and_global_racl(self):
        command = self.by_name["claimarc_canonical_s0"].command
        self.assertNotIn("--no_fusion", command)
        self.assertIn("--cl_no_attr_block", command)
        self.assertNotIn("--cl_exclude_self", command)
        for flag, value in (
            ("--n_fusion", "1"), ("--heads", "8"),
            ("--fusion_dropout", "0.2"), ("--lr_fusion", "5e-05"),
            ("--warmup", "3"), ("--cl_epochs", "6"),
            ("--lambda_cl", "0.5"), ("--tau", "0.07"),
            ("--Kp", "3"), ("--Kn", "5"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_only_two_additional_evidence_views_are_trained(self):
        expected = {
            f"input_{view}_s{seed}"
            for view in ("sources_only", "sources_plus_arguments")
            for seed in (0, 1, 2)
        }
        actual = {name for name in self.by_name if name.startswith("input_")}
        self.assertEqual(actual, expected)
        for name in expected:
            command = self.by_name[name].command
            policy = command[command.index("--evidence_policy") + 1]
            expected_policy = (
                "sources_only" if "input_sources_only" in name else "source_first")
            self.assertEqual(policy, expected_policy)

    def test_arguments_only_is_reused_as_canonical_not_retrained(self):
        self.assertFalse(any(
            name.startswith("input_arguments_only")
            for name in self.by_name))

    def test_cross_domain_claimarc_uses_locked_fusion_only(self):
        jobs = suite.build_jobs({"xdom"})
        commands = {job.name: job.command for job in jobs}
        clarc = commands["xdom_rooms_s0_clarc"]
        bert = commands["xdom_rooms_s0_bert_cls"]
        for flag, value in (
            ("--n_fusion", "1"), ("--heads", "8"),
            ("--fusion_dropout", "0.2"), ("--lr_fusion", "5e-05"),
        ):
            self.assertEqual(clarc[clarc.index(flag) + 1], value)
            self.assertNotIn(flag, bert)


if __name__ == "__main__":
    unittest.main()
