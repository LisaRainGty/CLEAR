#!/usr/bin/env python3
"""Recompute Figure 6 risk-coverage curves from the fair canonical bundles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from models.train import best_threshold_macroF1, rkc_attr_predict


COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.65)


def arr(block, key):
    value = block[key]
    return value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def one(path: Path, random_repetitions: int):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    train, val, test = bundle["train"], bundle["val"], bundle["test"]
    gtr = torch.as_tensor(arr(train, "g"), dtype=torch.float32)
    gva = torch.as_tensor(arr(val, "g"), dtype=torch.float32)
    gte = torch.as_tensor(arr(test, "g"), dtype=torch.float32)
    ytr, ctr, atr = arr(train, "y"), arr(train, "c"), arr(train, "attr")
    yv, pv, ava = arr(val, "y"), arr(val, "p"), arr(val, "attr")
    y, p, ate = arr(test, "y"), arr(test, "p"), arr(test, "attr")
    rva = rkc_attr_predict(gtr, ytr, ctr, atr, gva, ava)
    rte = rkc_attr_predict(gtr, ytr, ctr, atr, gte, ate)
    # Section 3.2.9 freezes the accepted prediction to the arithmetic mean of
    # the forward and retrieval estimates; only the operating threshold is fit on val.
    best_alpha = 0.5
    score_val = 0.5 * (pv + rva)
    score = 0.5 * (p + rte)
    threshold = best_threshold_macroF1(yv, score_val)
    disagreement = np.abs(p - rte)
    order = np.argsort(disagreement)
    rng = np.random.RandomState(20260721)
    result = {}
    for coverage in COVERAGES:
        keep_n = max(2, int(round(coverage * len(y))))
        keep = order[:keep_n]
        acc = float(np.mean((score[keep] >= threshold) == y[keep]))
        ap = float(average_precision_score(y[keep], score[keep]))
        random_acc, random_ap = [], []
        for _ in range(random_repetitions):
            idx = rng.choice(len(y), keep_n, replace=False)
            random_acc.append(float(np.mean((score[idx] >= threshold) == y[idx])))
            random_ap.append(float(average_precision_score(y[idx], score[idx])))
        result[str(coverage)] = {
            "acc": acc, "ap": ap,
            "racc": float(np.mean(random_acc)), "rap": float(np.mean(random_ap)),
            "n": keep_n, "alpha": float(best_alpha), "threshold": float(threshold),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-repetitions", type=int, default=200)
    args = parser.parse_args()
    per_seed = [one(args.emb_dir / f"emb_geom_racl_s{seed}.pt", args.random_repetitions)
                for seed in range(3)]
    result = {}
    for coverage in map(str, COVERAGES):
        result[coverage] = {}
        for key in ("acc", "ap", "racc", "rap"):
            values = np.asarray([row[coverage][key] for row in per_seed])
            result[coverage][key] = [float(values.mean()), float(values.std())]
        result[coverage]["n"] = per_seed[0][coverage]["n"]
        result[coverage]["per_seed"] = [row[coverage] for row in per_seed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"[written] {args.output}")


if __name__ == "__main__":
    main()
