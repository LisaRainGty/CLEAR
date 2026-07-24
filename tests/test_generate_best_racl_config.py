import json
import unittest
from pathlib import Path

from scripts.generate_best_racl_config import build_config


ROOT = Path(__file__).resolve().parents[1]


def selection(no_fusion: bool, hard_positive: bool) -> dict:
    return {
        "dataset_sha256": "b6bc9a91f87da3a489af216a036e4d11bd02d7eb8895e9d7f2cd9d78e26bd618",
        "evidence_policy": "args_only",
        "test_metrics_accessed": False,
        "racl_mandatory": True,
        "selected_candidate": (
            f"{'no_fusion' if no_fusion else 'fusion'}_"
            f"{'hard' if hard_positive else 'easy'}"
        ),
        "selected_spec": {
            "no_fusion": no_fusion,
            "hard_positive": hard_positive,
            "racl_mandatory": True,
            "attribute_blocked": False,
            "class_balanced": True,
            "warmup_epochs": 3,
            "contrastive_epochs": 6,
            "lambda_cl": 0.5,
            "tau": 0.07,
            "kp": 3,
            "kn": 5,
        },
    }


class GenerateBestRaclConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads(
            (ROOT / "configs/paper_fair_locked_fusion_v2.json").read_text(
                encoding="utf-8"
            )
        )

    def test_no_fusion_hard_positive_config(self):
        cfg = build_config(
            self.base,
            selection(True, True),
            selection_path="results/selection.json",
            selection_sha256="a" * 64,
        )
        claimarc = cfg["claimarc"]
        self.assertTrue(claimarc["no_fusion_main"])
        self.assertNotIn("locked_fusion", claimarc)
        self.assertIn("with_fusion_reference", claimarc)
        self.assertTrue(claimarc["locked_racl"]["hard_positive"])

    def test_fusion_easy_positive_config(self):
        cfg = build_config(
            self.base,
            selection(False, False),
            selection_path="results/selection.json",
            selection_sha256="b" * 64,
        )
        claimarc = cfg["claimarc"]
        self.assertFalse(claimarc["no_fusion_main"])
        self.assertIn("locked_fusion", claimarc)
        self.assertNotIn("with_fusion_reference", claimarc)
        self.assertFalse(claimarc["locked_racl"]["hard_positive"])


if __name__ == "__main__":
    unittest.main()
