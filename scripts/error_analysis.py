#!/usr/bin/env python3
"""Recompute the manuscript's confusion and evidence-coverage error analysis."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--emb-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundles = [load(args.emb_dir / f"emb_geom_racl_s{s}.pt") for s in range(3)]
    val_p = np.mean([np.asarray(b["val"]["p"]) for b in bundles], axis=0)
    test_p = np.mean([np.asarray(b["test"]["p"]) for b in bundles], axis=0)
    val_y = np.asarray(bundles[0]["val"]["y"], dtype=int)
    test_y = np.asarray(bundles[0]["test"]["y"], dtype=int)
    pair_ids = list(bundles[0]["test"]["pair_id"])
    threshold = float(max(np.linspace(0.01, 0.99, 99), key=lambda t:
                          f1_score(val_y, val_p >= t, average="macro", zero_division=0)))
    pred = (test_p >= threshold).astype(int)
    records = {}
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "test":
                records[row.get("pair_id", "")] = row
    if set(pair_ids) != set(records):
        raise ValueError("prediction and dataset test pair_id sets differ")
    confusion = {
        "tp": int(np.sum((test_y == 1) & (pred == 1))),
        "fn": int(np.sum((test_y == 1) & (pred == 0))),
        "fp": int(np.sum((test_y == 0) & (pred == 1))),
        "tn": int(np.sum((test_y == 0) & (pred == 0))),
    }
    groups = defaultdict(list)
    categories = defaultdict(list)
    for index, pair_id in enumerate(pair_ids):
        row = records[pair_id]
        count = sum(bool(row.get(key)) for key in (
            "evidence_params", "evidence_ocr", "evidence_vlm"))
        groups[str(count)].append(index)
        categories[str(row.get("category", ""))].append(index)

    def summarize(mapping):
        result = {}
        for name, indices in sorted(mapping.items()):
            idx = np.asarray(indices, dtype=int)
            result[name] = {
                "n": int(len(idx)), "positive": int(test_y[idx].sum()),
                "errors": int(np.sum(pred[idx] != test_y[idx])),
                "error_rate": float(np.mean(pred[idx] != test_y[idx])),
            }
        return result

    margin = np.abs(test_p - threshold)
    error_idx = np.where(pred != test_y)[0]
    hard = error_idx[np.argsort(margin[error_idx])[:50]]
    result = {
        "threshold": threshold, "confusion": confusion,
        "by_available_source_types": summarize(groups),
        "by_category": summarize(categories),
        "closest_50_errors": [{
            "pair_id": pair_ids[i], "y": int(test_y[i]), "prediction": int(pred[i]),
            "probability": float(test_p[i]), "category": records[pair_ids[i]].get("category", ""),
            "attribute_id": records[pair_ids[i]].get("attribute_id", ""),
        } for i in hard],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"[written] {args.output}")


if __name__ == "__main__":
    main()
