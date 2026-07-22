"""Unit tests for label-blind argument faithfulness checks."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_arguments_dataset import build_prompt, validate_grounding  # noqa: E402


def payload():
    return {
        "attribute_name": "鞋跟高度",
        "claim": "增高5公分",
        "PARAM": ["中跟"],
        "OCR": ["跟高5CM"],
        "VLM": [],
    }


class ArgumentGroundingTest(unittest.TestCase):
    def test_direct_source_anchor_passes(self):
        result = validate_grounding({
            "supporting_argument": "OCR标注跟高5CM",
            "refuting_argument": "",
            "evidence_gap": "是否为穿着后的实际增高高度",
        }, payload())
        self.assertEqual(result["supporting_argument"], "OCR标注跟高5CM")

    def test_claim_copy_without_source_anchor_fails(self):
        with self.assertRaisesRegex(ValueError, "no direct"):
            validate_grounding({
                "supporting_argument": "穿上能够增高五公分",
                "refuting_argument": "",
                "evidence_gap": "",
            }, payload())

    def test_unsupported_number_fails_even_with_source_words(self):
        with self.assertRaisesRegex(ValueError, "numbers absent"):
            validate_grounding({
                "supporting_argument": "OCR标注跟高6CM",
                "refuting_argument": "",
                "evidence_gap": "",
            }, payload())

    def test_speculative_support_fails(self):
        with self.assertRaisesRegex(ValueError, "speculative"):
            validate_grounding({
                "supporting_argument": "跟高5CM可能有助于增高",
                "refuting_argument": "",
                "evidence_gap": "",
            }, payload())

    def test_gap_may_name_claim_number(self):
        validate_grounding({
            "supporting_argument": "",
            "refuting_argument": "",
            "evidence_gap": "缺少实际增高5公分的测量",
        }, payload())

    def test_gap_cannot_assert_a_contradiction(self):
        with self.assertRaisesRegex(ValueError, "asserts a conclusion"):
            validate_grounding({
                "supporting_argument": "",
                "refuting_argument": "",
                "evidence_gap": "尺寸声称与实际测量值不符",
            }, payload())

    def test_support_and_refute_cannot_be_identical(self):
        with self.assertRaisesRegex(ValueError, "are identical"):
            validate_grounding({
                "supporting_argument": "OCR标注跟高5CM",
                "refuting_argument": "OCR标注跟高5CM",
                "evidence_gap": "",
            }, payload())

    def test_retry_prompt_is_attempt_and_failure_specific(self):
        prompt = build_prompt(
            payload(), attempt=2,
            failure_hint="ValueError('supporting_argument and refuting_argument are identical')",
        )
        self.assertIn("第2次", prompt)
        self.assertIn("supporting_argument 必须置空", prompt)


if __name__ == "__main__":
    unittest.main()
