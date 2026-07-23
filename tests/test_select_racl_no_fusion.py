"""Unit tests for the predeclared no-fusion RACL parsimony gate."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_racl_no_fusion", ROOT / "scripts/select_racl_no_fusion.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def summary(ap, f1):
    return {
        "auprc": {"mean": sum(ap) / len(ap), "values": ap},
        "pos_f1": {"mean": sum(f1) / len(f1), "values": f1},
    }


SELECTION = {
    "diagnostic_control": "no_racl",
    "primary_metric": "auprc",
    "tie_breaker": "pos_f1",
}


class RaclParsimonyGateTest(unittest.TestCase):
    def test_selects_best_racl_by_ap(self):
        aggregates = {
            "no_racl": summary([0.60, 0.62, 0.61], [0.66, 0.65, 0.67]),
            "racl_good": summary([0.63, 0.64, 0.60], [0.67, 0.66, 0.66]),
            "racl_lower_ap": summary([0.59, 0.61, 0.60], [0.70, 0.70, 0.70]),
        }
        selected, ranked, diagnostics = MODULE.select_candidate(
            aggregates, SELECTION
        )
        self.assertEqual(selected, "racl_good")
        self.assertEqual(ranked[0], "racl_good")
        self.assertTrue(diagnostics["racl_good"]["selectable"])
        self.assertFalse(diagnostics["no_racl"]["selectable"])

    def test_no_racl_is_never_selectable(self):
        aggregates = {
            "no_racl": summary([0.60, 0.62, 0.61], [0.66, 0.65, 0.67]),
            "racl_weaker": summary([0.53, 0.54, 0.50], [0.62, 0.63, 0.61]),
        }
        selected, _, diagnostics = MODULE.select_candidate(
            aggregates, SELECTION
        )
        self.assertEqual(selected, "racl_weaker")
        self.assertFalse(diagnostics["no_racl"]["selectable"])


if __name__ == "__main__":
    unittest.main()
