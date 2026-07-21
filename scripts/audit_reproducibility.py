#!/usr/bin/env python3
"""Audit the frozen paper dataset and historical result provenance using stdlib only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc


def claim_length(rec: dict) -> int:
    claim = rec.get("claim", {}) or {}
    passage = str(claim.get("passage", "") or "")
    if passage:
        return len(passage)
    segments = claim.get("segments", []) or []
    text = "".join(str(item.get("text", "") or "") for item in segments)
    return len(text)


def evidence_length(rec: dict) -> int:
    total = 0
    for key, field in (("evidence_params", "raw_text"),
                       ("evidence_ocr", "raw_text"),
                       ("evidence_vlm", "raw_quote")):
        total += sum(len(str(item.get(field, "") or "")) for item in rec.get(key, []) or [])
    return total


def has_arguments(rec: dict) -> bool:
    args = rec.get("arguments", {}) or {}
    return any(str(args.get(key, "") or "").strip() for key in
               ("supporting_argument", "refuting_argument", "evidence_gap"))


def source_coverage(rec: dict) -> int:
    return sum(bool(rec.get(key)) for key in
               ("evidence_params", "evidence_ocr", "evidence_vlm"))


def aggregate_results(path: Path) -> dict:
    groups = defaultdict(list)
    if not path.exists():
        return {}
    for rec in read_jsonl(path):
        if not rec.get("error"):
            groups[rec.get("tag", "unknown")].append(rec)
    out = {}
    for tag, rows in sorted(groups.items()):
        block = {"runs": len(rows), "seeds": sorted({row.get("seed") for row in rows})}
        for metric in ("acc", "pos_f1", "macro_f1", "auprc", "auroc", "wF1", "ece"):
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            if values:
                block[metric] = {
                    "mean": round(statistics.fmean(values), 6),
                    "sd_population": round(statistics.pstdev(values), 6),
                }
        block["has_dataset_hash"] = all(bool(row.get("dataset_sha256")) for row in rows)
        block["recorded_evidence_policies"] = sorted({
            row.get("evidence_policy", "MISSING") for row in rows
        })
        out[tag] = block
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/paper_fair.json"))
    parser.add_argument("--out", default=str(ROOT / "results/audit/reproducibility_report.json"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    spec = cfg["dataset"]
    dataset = ROOT / spec["path"]
    rows = list(read_jsonl(dataset))

    split_rows = Counter(str(r.get("split") or "train") for r in rows)
    split_pos = Counter()
    rooms = defaultdict(set)
    categories = Counter()
    sample_roles = Counter()
    stage_sources = Counter()
    coverage = Counter()
    c_values = []
    argument_count = 0
    claim_lengths = []
    evidence_lengths = []
    aligned_pair_count = 0
    mention_total = 0
    mention_neg = 0
    for rec in rows:
        split = str(rec.get("split") or "train")
        split_pos[split] += int(rec.get("y", 0))
        rooms[split].add(str(rec.get("room_id", "")))
        categories[str(rec.get("category", ""))] += 1
        sample_roles[str(rec.get("sample_role", ""))] += 1
        stage_sources[str(rec.get("_stagec_source", ""))] += 1
        coverage[source_coverage(rec)] += 1
        c_values.append(float(rec.get("c", 0.0)))
        argument_count += int(has_arguments(rec))
        claim_lengths.append(claim_length(rec))
        evidence_lengths.append(evidence_length(rec))
        aligned = rec.get("_aligned_consumer_mentions") or []
        aligned_pair_count += int(bool(aligned))
        mention_total += int(rec.get("_consumer_mentions_total", 0) or 0)
        mention_neg += int(rec.get("_consumer_mentions_neg", 0) or 0)

    overlap = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{a}_{b}"] = sorted(rooms[a] & rooms[b])

    digest = sha256_file(dataset)
    checks = {
        "dataset_exists": dataset.is_file(),
        "dataset_sha256": digest == spec["sha256"],
        "row_count": len(rows) == spec["expected_rows"],
        "split_rows": dict(split_rows) == spec["expected_split_rows"],
        "split_positives": dict(split_pos) == spec["expected_split_positives"],
        "room_group_leakage_absent": all(not values for values in overlap.values()),
        "shared_sources_nonempty": sum(coverage[k] for k in (1, 2, 3)) > 0,
        "input_policy_is_sources_only": (
            cfg.get("fair_comparison", {}).get("evidence_policy") == "sources_only"
        ),
        "generated_arguments_excluded": argument_count == 0,
        "argument_record_count": argument_count == spec.get("expected_argument_records", 0),
        "source_availability": (
            {str(k): coverage[k] for k in range(4)}
            == spec.get("expected_source_availability", {})
        ),
        "arguments_forbidden_by_config": (
            cfg.get("fair_comparison", {}).get("arguments_allowed") is False
        ),
        "single_source_experiments_forbidden": (
            cfg.get("fair_comparison", {}).get("single_source_experiments_allowed") is False
        ),
    }

    results = aggregate_results(ROOT / "results/campaign6_results.jsonl")
    c6_args = results.get("c6_args", {})
    c6_canon = results.get("c6_canon", {})
    headline_matches_args = all(
        math.isclose(c6_args.get(metric, {}).get("mean", -1), value, abs_tol=0.00015)
        for metric, value in (("acc", 0.8261), ("pos_f1", 0.7335),
                              ("auprc", 0.7541), ("auroc", 0.9042))
    )
    report = {
        "schema_version": 2,
        "config": str(config_path.relative_to(ROOT)),
        "dataset": {
            "path": str(dataset.relative_to(ROOT)),
            "sha256": digest,
            "rows": len(rows),
            "split_rows": dict(split_rows),
            "split_positives": dict(split_pos),
            "split_positive_rates": {k: split_pos[k] / split_rows[k] for k in split_rows},
            "rooms_by_split": {k: len(v) for k, v in rooms.items()},
            "room_overlap": overlap,
            "categories": dict(categories),
            "sample_roles": dict(sample_roles),
            "construction_sources": dict(stage_sources),
            "source_coverage_0_1_2_3": {str(k): coverage[k] for k in range(4)},
            "argument_records": argument_count,
            "argument_coverage": argument_count / len(rows) if rows else 0.0,
            "c_mean": statistics.fmean(c_values),
            "c_median": statistics.median(c_values),
            "c_min": min(c_values),
            "c_max": max(c_values),
            "claim_length_mean_chars": statistics.fmean(claim_lengths),
            "raw_evidence_length_mean_chars": statistics.fmean(evidence_lengths),
            "pairs_with_aligned_comments": aligned_pair_count,
            "pairs_with_aligned_comments_rate": aligned_pair_count / len(rows),
            "consumer_mentions_total_mean": mention_total / len(rows),
            "aligned_negative_mentions_mean": mention_neg / len(rows),
        },
        "checks": checks,
        "historical_results": results,
        "interpretation": {
            "paper_headline_matches_c6_args": headline_matches_args,
            "c6_args_view_is_empty": argument_count == 0,
            "c6_canon_is_current_shared_source_view_reference": bool(c6_canon),
            "fair_rerun_required": True,
            "reason": (
                "The paper headline was selected from args_only although the frozen dataset has "
                "no generated arguments. Historical baselines and most ablations used source evidence."
            ),
        },
        "overall": "FAIL_NEEDS_FAIR_RERUN" if not all(checks.values()) else "PASS",
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nAUDIT_REPORT={out}")
    return 2 if report["overall"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
