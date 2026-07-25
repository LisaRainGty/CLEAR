import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.train import (  # noqa: E402
    FixedSemanticRACL,
    MemoryBank,
    _memory_features,
    dualspace_boundary_loss,
)


class DualSpaceRaclCoreTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "pair_id": f"p{i}",
                "room_id": f"room{i}",
                "y": i % 2,
                "c": 0.2 + 0.1 * i,
            }
            for i in range(8)
        ]
        rng = np.random.RandomState(7)
        q = rng.normal(size=(8, 6)).astype(np.float32)
        self.q = q / np.linalg.norm(q, axis=1, keepdims=True)

    def test_fixed_index_has_positive_and_negative_coverage(self):
        index = FixedSemanticRACL(
            self.q, self.rows, kp=2, kn=2, device="cpu"
        )
        self.assertEqual(index.stats["positive_coverage"], 1.0)
        self.assertEqual(index.stats["negative_coverage"], 1.0)
        for row in self.rows:
            pos, neg, _, _, _ = index.get(row["pair_id"])
            anchor_y = row["y"]
            self.assertTrue(all(self.rows[j]["y"] == anchor_y for j in pos))
            self.assertTrue(all(self.rows[j]["y"] != anchor_y for j in neg))

    def test_local_boundary_loss_is_finite_and_differentiable(self):
        index = FixedSemanticRACL(
            self.q, self.rows, kp=2, kn=2, device="cpu"
        )
        bank_g = F.normalize(torch.randn(8, 5), dim=-1)
        bank = MemoryBank(
            bank_g.detach(),
            attrs=[""] * 8,
            y=torch.tensor([row["y"] for row in self.rows]),
            c=np.asarray([row["c"] for row in self.rows]),
            p=np.linspace(0.15, 0.85, 8),
        )
        anchor_g = torch.randn(3, 5, requires_grad=True)
        anchor_logit = torch.randn(3, requires_grad=True)
        batch = SimpleNamespace(
            pair_id=["p0", "p1", "p2"],
            y=torch.tensor([0.0, 1.0, 0.0]),
            c=torch.tensor([0.2, 0.3, 0.4]),
        )
        args = SimpleNamespace(
            tau=0.1,
            racl_margin=0.15,
            racl_geom_weight=0.05,
            racl_rank_weight=0.25,
        )
        loss = dualspace_boundary_loss(
            anchor_logit, anchor_g, batch, bank, index, args
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(anchor_g.grad).all())
        self.assertTrue(torch.isfinite(anchor_logit.grad).all())

    def test_memory_features_exclude_same_group_and_pair(self):
        features = _memory_features(
            self.q[:2], self.rows[:2], self.q, self.rows, k=3, device="cpu"
        )
        self.assertEqual(features.shape, (2, 3))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features[:, 1] >= 0))
        self.assertTrue(np.all(features[:, 1] <= 1))


if __name__ == "__main__":
    unittest.main()
