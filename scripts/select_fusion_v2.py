#!/usr/bin/env python3
"""Select a fusion-v2 candidate using validation-only three-seed results."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = ("acc", "pos_f1", "macro_f1", "auprc", "auroc", "wF1", "ece")


def load_one(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one result in {path}, found {len(rows)}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = cfg["fusion_v2"]
    seeds = [int(seed) for seed in protocol["seeds"]]
    expected_hash = cfg["dataset"]["sha256"]
    expected_policy = cfg["fair_comparison"]["evidence_policy"]
    result_root = Path(args.result_root).resolve()

    aggregates = {}
    for candidate in protocol["candidates"]:
        name = str(candidate["name"])
        rows = []
        for seed in seeds:
            path = result_root / "jobs" / f"fusion_v2_{name}_s{seed}.jsonl"
            if not path.exists():
                raise RuntimeError(f"missing candidate result: {path}")
            row = load_one(path)
            if row.get("evaluation_split") != "validation" or not row.get("validation_only"):
                raise RuntimeError(f"non-validation result forbidden in tuning: {path}")
            if "n_test" in row or "pos_test" in row:
                raise RuntimeError(f"test metric leaked into tuning result: {path}")
            if row.get("dataset_sha256") != expected_hash:
                raise RuntimeError(f"dataset hash mismatch in {path}")
            if row.get("evidence_policy") != expected_policy:
                raise RuntimeError(f"evidence-policy mismatch in {path}")
            if int(row.get("seed")) != seed:
                raise RuntimeError(f"seed mismatch in {path}")
            rows.append(row)
        summary = {"seeds": seeds, "runs": rows}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
            }
        aggregates[name] = summary

    selection = protocol["selection"]
    primary = str(selection["primary_metric"])
    tie_breaker = str(selection["tie_breaker"])
    ranked = sorted(
        aggregates,
        key=lambda name: (
            -aggregates[name][primary]["mean"],
            -aggregates[name][tie_breaker]["mean"],
            name,
        ),
    )
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset_sha256": expected_hash,
        "evidence_policy": expected_policy,
        "test_metrics_accessed": False,
        "selection": selection,
        "ranked_candidates": ranked,
        "selected_candidate": ranked[0],
        "aggregates": aggregates,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_candidate": ranked[0],
        "primary_metric": primary,
        "validation_mean": aggregates[ranked[0]][primary]["mean"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
