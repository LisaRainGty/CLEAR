#!/usr/bin/env python3
"""Compare locked fusion with no-fusion using validation evidence only."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from models.train import best_threshold_macroF1, ece, macro_f1


METRICS = ("acc", "pos_f1", "macro_f1", "wF1", "auprc", "auroc", "ece")


def summarize(rows: list[dict]) -> dict:
    out = {"n_runs": len(rows), "seeds": [int(row["seed"]) for row in rows]}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        out[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "values": values,
        }
    return out


def validation_metrics(bundle_path: Path) -> tuple[dict, dict]:
    bundle = torch.load(bundle_path, map_location="cpu")
    provenance = dict(bundle.get("provenance", {}))
    val = bundle["val"]
    p = np.asarray(val["p"], dtype=float)
    y = np.asarray(val["y"], dtype=int)
    c = np.asarray(val["c"], dtype=float)
    threshold = best_threshold_macroF1(y, p)
    pred = (p >= threshold).astype(int)
    row = {
        "tag": "no_fusion",
        "seed": int(provenance["seed"]),
        "evaluation_split": "validation",
        "validation_only_reconstruction": True,
        "thr": float(threshold),
        "acc": float((pred == y).mean()),
        "pos_f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(macro_f1(y, pred)),
        "wF1": float(macro_f1(y, pred, w=np.clip(c, 0.05, None))),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "ece": float(ece(y, p)),
        "n_val": int(len(y)),
        "pos_val": int(y.sum()),
    }
    return row, provenance


def load_result(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one result in {path}, found {len(rows)}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-selection", required=True)
    parser.add_argument("--no-fusion-glob", required=True)
    parser.add_argument("--historical-test-root", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selection_path = Path(args.fusion_selection).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_metrics_accessed") is not False:
        raise RuntimeError("fusion selector did not prove test isolation")
    selected = str(selection["selected_candidate"])
    fusion = selection["aggregates"][selected]
    expected_hash = str(selection["dataset_sha256"])
    expected_policy = str(selection["evidence_policy"])

    bundle_paths = [Path(path) for path in sorted(glob.glob(args.no_fusion_glob))]
    if len(bundle_paths) != 3:
        raise RuntimeError(f"expected three no-fusion bundles, found {len(bundle_paths)}")
    no_fusion_rows = []
    for path in bundle_paths:
        row, provenance = validation_metrics(path)
        if provenance.get("dataset_sha256") != expected_hash:
            raise RuntimeError(f"dataset mismatch in {path}")
        if provenance.get("evidence_policy") != expected_policy:
            raise RuntimeError(f"evidence-policy mismatch in {path}")
        if provenance.get("tag") != "no_fusion":
            raise RuntimeError(f"wrong bundle tag in {path}")
        no_fusion_rows.append(row)
    no_fusion_rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in no_fusion_rows] != [0, 1, 2]:
        raise RuntimeError("no-fusion bundles do not cover seeds 0,1,2 exactly")
    no_fusion = summarize(no_fusion_rows)

    validation_delta = {}
    for metric in METRICS:
        validation_delta[metric] = (
            float(fusion[metric]["mean"]) - float(no_fusion[metric]["mean"])
        )

    historical_test = None
    if args.historical_test_root:
        root = Path(args.historical_test_root).resolve()
        variants = {}
        for tag in ("claimarc_canonical", "no_fusion"):
            rows = [load_result(root / "jobs" / f"{tag}_s{seed}.jsonl")
                    for seed in (0, 1, 2)]
            for row in rows:
                if row.get("dataset_sha256") != expected_hash:
                    raise RuntimeError(f"historical {tag} dataset mismatch")
                if row.get("evidence_policy") != expected_policy:
                    raise RuntimeError(f"historical {tag} policy mismatch")
            variants[tag] = summarize(rows)
        historical_test = {
            "role": "context_only_not_used_for_architecture_selection",
            "fusion_variant": "legacy default n_fusion=2, not locked fusion_v2",
            "variants": variants,
            "fusion_minus_no_fusion": {
                metric: (
                    variants["claimarc_canonical"][metric]["mean"]
                    - variants["no_fusion"][metric]["mean"]
                )
                for metric in METRICS
            },
        }

    report = {
        "schema_version": 1,
        "decision_evidence": "validation only",
        "dataset_sha256": expected_hash,
        "evidence_policy": expected_policy,
        "selected_fusion_candidate": selected,
        "fusion_selection_manifest": str(selection_path),
        "fusion": fusion,
        "no_fusion": {**no_fusion, "runs": no_fusion_rows},
        "fusion_minus_no_fusion_validation": validation_delta,
        "historical_test_context": historical_test,
        "standard_deviation": "population ddof=0",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_fusion_candidate": selected,
        "validation_delta": validation_delta,
        "fusion_auprc": fusion["auprc"],
        "no_fusion_auprc": no_fusion["auprc"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
