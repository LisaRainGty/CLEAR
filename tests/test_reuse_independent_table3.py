import unittest

from scripts.reuse_independent_table3 import module_name, normalized_command


class ReuseIndependentTable3Tests(unittest.TestCase):
    def test_only_explicit_model_modules_are_detected(self):
        command = ("python", "-m", "models.baselines_ft", "--seed", "0")
        self.assertEqual(module_name(command), "models.baselines_ft")
        self.assertEqual(module_name(("python", "script.py")), "")

    def test_namespace_paths_are_normalized_without_changing_other_flags(self):
        source = (
            "python",
            "--save_pred",
            "/root/CLEAR/embeddings/old_namespace/pred.pt",
            "--evidence_policy",
            "args_only",
        )
        target = (
            "python",
            "--save_pred",
            "/root/CLEAR/embeddings/new_namespace/pred.pt",
            "--evidence_policy",
            "args_only",
        )
        self.assertEqual(
            normalized_command(source, "old_namespace"),
            normalized_command(target, "new_namespace"),
        )


if __name__ == "__main__":
    unittest.main()
