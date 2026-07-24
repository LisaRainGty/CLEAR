#!/usr/bin/env python3
"""Rank the four mandatory-RACL fusion/positive-mining combinations on validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_DATASET = "b6bc9a91f87da3a489af216a036e4d11bd02d7eb8895e9d7f2cd9d78e26bd618"
EXPECTED_POLICY = "args_only"


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_sha256") != EXPECTED_DATASET:
        raise RuntimeError(f"dataset hash mismatch: {path}")
    if payload.get("evidence_policy") != EXPECTED_POLICY:
        raise RuntimeError(f"evidence policy mismatch: {path}")
    if payload.get("test_metrics_accessed") is not False:
        raise RuntimeError(f"test metrics were accessed: {path}")
    if payload.get("racl_mandatory") is not True:
        raise RuntimeError(f"RACL was not mandatory: {path}")
    return payload


def extract_candidate(manifest: dict, candidate: str) -> dict:
    if candidate in manifest.get("aggregates", {}):
        return manifest["aggregates"][candidate]
    parent = manifest.get("parent_phase_summary") or {}
    if parent.get("candidate") == candidate:
        return parent["aggregate"]
    raise RuntimeError(f"candidate {candidate!r} is absent from its manifest")


def normalized_metrics(summary: dict) -> dict:
    metrics = {}
    for name in ("acc", "pos_f1", "macro_f1", "wF1", "auprc", "auroc", "ece"):
        item = summary.get(name, {})
        if "mean" not in item:
            raise RuntimeError(f"missing validation metric {name}")
        metrics[name] = {
            "mean": float(item["mean"]),
            "std": float(item.get("std", 0.0)),
        }
    seeds = [int(seed) for seed in summary.get("seeds", [])]
    if seeds != [0, 1, 2]:
        raise RuntimeError(f"unexpected validation seeds: {seeds}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-easy-manifest", type=Path, required=True)
    parser.add_argument("--fusion-hard-manifest", type=Path, required=True)
    parser.add_argument("--no-fusion-easy-manifest", type=Path, required=True)
    parser.add_argument("--no-fusion-hard-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_specs = {
        "fusion_easy": (
            args.fusion_easy_manifest,
            "global_l050_t007_k3_5_w3c6_reused",
            {"no_fusion": False, "hard_positive": False},
        ),
        "fusion_hard": (
            args.fusion_hard_manifest,
            "global_hardpos_l050_t007_k3_5_w3c6",
            {"no_fusion": False, "hard_positive": True},
        ),
        "no_fusion_easy": (
            args.no_fusion_easy_manifest,
            "racl_legacy_l050_t007",
            {"no_fusion": True, "hard_positive": False},
        ),
        "no_fusion_hard": (
            args.no_fusion_hard_manifest,
            "racl_hardpos_l050_t007",
            {"no_fusion": True, "hard_positive": True},
        ),
    }
    candidates = {}
    for alias, (path, candidate, architecture) in source_specs.items():
        resolved = path.resolve()
        manifest = load_manifest(resolved)
        expected_architecture = (
            "no_fusion" if architecture["no_fusion"] else "locked_fusion_global"
        )
        if manifest.get("architecture") != expected_architecture:
            raise RuntimeError(f"architecture mismatch for {alias}")
        summary = extract_candidate(manifest, candidate)
        spec = summary.get("spec", {})
        observed_hard = bool(spec.get("hard_positive", False))
        if observed_hard != architecture["hard_positive"]:
            raise RuntimeError(f"hard-positive mismatch for {alias}")
        candidates[alias] = {
            "source_manifest": str(resolved),
            "source_manifest_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            "source_candidate": candidate,
            **architecture,
            "racl_mandatory": True,
            "attribute_blocked": False,
            "class_balanced": True,
            "warmup_epochs": 3,
            "contrastive_epochs": 6,
            "lambda_cl": 0.5,
            "tau": 0.07,
            "kp": 3,
            "kn": 5,
            "metrics": normalized_metrics(summary),
        }

    ranked = sorted(
        candidates,
        key=lambda name: (
            -candidates[name]["metrics"]["auprc"]["mean"],
            -candidates[name]["metrics"]["pos_f1"]["mean"],
            name,
        ),
    )
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": EXPECTED_DATASET,
        "evidence_policy": EXPECTED_POLICY,
        "selection_split": "validation",
        "selection_primary_metric": "auprc",
        "selection_tie_breaker": "pos_f1",
        "selection_rule": "highest three-seed validation AUPRC mean, then positive-F1 mean, then lexical alias",
        "test_metrics_accessed": False,
        "prior_test_metrics_informed_high_level_search_scope": True,
        "clean_confirmatory_claim_requires_new_untouched_holdout": True,
        "racl_mandatory": True,
        "ranked_candidates": ranked,
        "selected_candidate": ranked[0],
        "selected_spec": candidates[ranked[0]],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_candidate": ranked[0],
        "validation_auprc": candidates[ranked[0]]["metrics"]["auprc"]["mean"],
        "validation_pos_f1": candidates[ranked[0]]["metrics"]["pos_f1"]["mean"],
        "test_metrics_accessed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
