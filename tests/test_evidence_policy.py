"""Regression tests proving that the two frozen evidence views cannot mix."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.baselines import evidence_text  # noqa: E402
from models.data import apply_evidence_policy, build_evidence_ids  # noqa: E402


class FakeTokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def __init__(self):
        self.special = {
            "[ATTR]": 10, "[EVD]": 11, "[PARAM]": 12, "[OCR]": 13,
            "[VLM]": 14, "[ARG_SUP]": 15, "[ARG_REF]": 16,
            "[ARG_GAP]": 17, "[SEP_E]": 18,
        }
        self.text = {
            "ATTRIBUTE": 100, "RAW_PARAM": 201, "RAW_OCR": 202,
            "RAW_VLM": 203, "ARG_SUPPORT": 301, "ARG_REFUTE": 302,
            "ARG_GAP": 303,
        }

    def convert_tokens_to_ids(self, token):
        return self.special[token]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self.text[text]]


def record():
    return {
        "attribute_name": "ATTRIBUTE",
        "evidence_params": [{"raw_text": "RAW_PARAM"}],
        "evidence_ocr": [{"raw_text": "RAW_OCR"}],
        "evidence_vlm": [{"raw_quote": "RAW_VLM"}],
        "arguments": {
            "supporting_argument": "ARG_SUPPORT",
            "refuting_argument": "ARG_REFUTE",
            "evidence_gap": "ARG_GAP",
        },
    }


class EvidencePolicyIsolationTest(unittest.TestCase):
    def test_args_only_hides_every_raw_source(self):
        ids = build_evidence_ids(FakeTokenizer(), record(), "args_only")
        self.assertTrue({301, 302, 303}.issubset(ids))
        self.assertTrue({201, 202, 203}.isdisjoint(ids))
        text = evidence_text(record(), "args_only")
        self.assertIn("ARG_SUPPORT", text)
        self.assertNotIn("RAW_PARAM", text)
        self.assertNotIn("RAW_OCR", text)
        self.assertNotIn("RAW_VLM", text)

    def test_sources_only_hides_every_argument(self):
        ids = build_evidence_ids(FakeTokenizer(), record(), "sources_only")
        self.assertTrue({201, 202, 203}.issubset(ids))
        self.assertTrue({301, 302, 303}.isdisjoint(ids))
        text = evidence_text(record(), "sources_only")
        self.assertIn("RAW_PARAM", text)
        self.assertNotIn("ARG_SUPPORT", text)

    def test_policy_is_applied_to_all_splits(self):
        splits = {"train": [record()], "val": [record()], "test": [record()]}
        apply_evidence_policy(splits, "args_only")
        self.assertEqual(
            [row["_evidence_policy"] for rows in splits.values() for row in rows],
            ["args_only", "args_only", "args_only"],
        )


if __name__ == "__main__":
    unittest.main()
