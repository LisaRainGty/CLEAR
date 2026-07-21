"""Build proposal-faithful stateful data from full-pair LLM/VLM reviews.

Version 1 promotion produced a very strict "main" supervised candidate.  That
view is useful for audits, but it is too narrow for the proposal's target:
consumer-perceived deception conditioned on a livestream claim.  This builder
keeps every reviewed product-attribute pair and separates:

- `y_perception`: whether aligned consumers refute the repaired streamer claim;
- `promotion_state`: what is complete, missing, or ambiguous in the triplet;
- `c_reliability`: how much the row should influence supervised losses;
- `contrastive_mask`: whether the row is clean enough for hard contrastive
  retrieval/evidence objectives.

Objective product evidence contradiction never creates a positive label by
itself.  Missing evidence/claim becomes repair state or low-reliability silver,
not a forced negative.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl
from data_quality.build_full_pair_promoted_dataset_v1 import (
    aligned_comments,
    clean,
    claim_text_norm,
    evidence_payload,
    pair_id,
    promotion_state,
    relation_counts,
    reliability,
)
from data_quality.rebuild_repaired_datasets_v1 import assign_room_splits, split_leakage


REFUTE_RELATIONS = {"refute"}
SUPPORT_RELATIONS = {"support"}
AMBIGUOUS_STATES = {
    "silver_mixed_comment_relation",
    "silver_conflicting_comment_relation",
    "silver_conflicting_claim_family",
}
REPAIR_STATES = {
    "repair_missing_claim",
    "repair_missing_evidence",
    "repair_insufficient_product_evidence",
    "repair_identity_claim_value",
    "repair_numeric_value_judgment",
}
EVIDENCE_INCOMPLETE_STATES = {
    "silver_refute_missing_product_evidence",
    "silver_refute_insufficient_product_evidence",
}
GUARDED_STATES = {
    "silver_schema_meta_attribute",
    "silver_subjective_eval_attribute",
    "silver_commercial_promise_attribute",
    "silver_attribute_semantic_drift",
    "silver_enumeration_evidence_extra_values",
}


def read_queue(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        pid = pair_id(row)
        if pid:
            out[pid] = row
    return out


def read_reviews(paths: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read review JSONL files, keeping the last review per pair.

    Later review files intentionally override earlier ones.  This lets a VLM
    evidence repair replace a no-image review without mutating provenance.
    """
    out: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[str]] = defaultdict(list)
    rows = 0
    for path in paths:
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        for row in read_jsonl(p):
            rows += 1
            pid = pair_id(row)
            if not pid:
                continue
            out[pid] = row
            sources[pid].append(str(p))
    return out, {
        "review_input_files": [p for p in paths if p and Path(p).exists()],
        "review_rows_read": rows,
        "review_pairs": len(out),
        "review_pairs_overwritten": sum(1 for vals in sources.values() if len(vals) > 1),
    }


def y_perception_from_relations(review: dict[str, Any], rel: Counter) -> int | None:
    if not review.get("claim_found"):
        return None
    if rel.get("refute", 0) > 0:
        return 1
    if rel.get("support", 0) > 0 or rel.get("mixed", 0) > 0:
        return 0
    return None


def sample_role(state: str, y_perception: int | None, rel: Counter) -> str:
    if y_perception is None:
        if state in REPAIR_STATES:
            return "repair_unlabeled"
        return "lowinfo_unlabeled"
    if state in {"main_positive_refute", "main_negative_support"}:
        return "supervised_main"
    if state in EVIDENCE_INCOMPLETE_STATES:
        return "supervised_silver_evidence_incomplete"
    if state in AMBIGUOUS_STATES or rel.get("support", 0) > 0 and rel.get("refute", 0) > 0:
        return "supervised_silver_ambiguous"
    if state in REPAIR_STATES:
        return "supervised_silver_repair_needed"
    return "supervised_silver_guarded"


def contrastive_mask(
    state: str,
    y_perception: int | None,
    review: dict[str, Any],
    rel: Counter,
) -> bool:
    if y_perception is None:
        return False
    if not review.get("claim_found") or not review.get("product_evidence_found"):
        return False
    if state not in {"main_positive_refute", "main_negative_support"}:
        return False
    if rel.get("support", 0) > 0 and rel.get("refute", 0) > 0:
        return False
    return True


def reliability_v2(
    queue_row: dict[str, Any],
    review: dict[str, Any],
    aligned: list[dict[str, Any]],
    state: str,
    y_perception: int | None,
) -> float:
    c = reliability(queue_row, review, aligned)
    if y_perception is None:
        c = min(c, 0.22)
    if state in EVIDENCE_INCOMPLETE_STATES:
        c = min(c, 0.48)
    if state in REPAIR_STATES:
        c = min(c, 0.35)
    if state in AMBIGUOUS_STATES:
        c = min(c, 0.42)
    if state in GUARDED_STATES:
        c = min(c, 0.45)
    return round(max(0.03, c), 4)


def claim_family_id(row: dict[str, Any]) -> str:
    return "|".join([
        clean(row.get("product_id")),
        clean(row.get("room_id")),
        claim_text_norm(row),
    ])


def annotate_claim_families(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = claim_family_id(row)
        if key.count("|") == 2 and key.split("|")[-1]:
            groups[key].append(row)

    duplicate_groups = 0
    conflicting_groups = 0
    masked = 0
    examples: list[dict[str, Any]] = []
    for gid, vals in groups.items():
        if len(vals) < 2:
            continue
        duplicate_groups += 1
        labels = {row.get("y_perception") for row in vals if row.get("y_perception") is not None}
        conflict = len(labels) > 1
        if conflict:
            conflicting_groups += 1
        vals_sorted = sorted(
            vals,
            key=lambda r: (
                int(r.get("contrastive_mask")),
                float(r.get("c_reliability") or 0),
                int((r.get("proposal_label_audit") or {}).get("aligned_comment_count") or 0),
                clean(r.get("pair_id")),
            ),
            reverse=True,
        )
        kept = vals_sorted[0]
        for row in vals:
            audit = row.setdefault("proposal_label_audit", {})
            audit["claim_family_group_id"] = gid
            audit["claim_family_size"] = len(vals)
            audit["claim_family_kept_pair_id"] = kept.get("pair_id")
            audit["claim_family_conflicting_labels"] = conflict
            if row is not kept or conflict:
                if row.get("contrastive_mask"):
                    masked += 1
                row["contrastive_mask"] = False
                row["contrastive_mask_reason"] = (
                    "conflicting_claim_family" if conflict else "duplicate_claim_family"
                )
        if len(examples) < 20:
            examples.append({
                "group_id": gid,
                "size": len(vals),
                "kept_pair_id": kept.get("pair_id"),
                "conflicting_labels": conflict,
                "pair_ids": [r.get("pair_id") for r in vals[:8]],
            })
    return {
        "duplicate_claim_family_groups": duplicate_groups,
        "conflicting_claim_family_groups": conflicting_groups,
        "contrastive_masked_by_claim_family": masked,
        "claim_family_examples": examples,
    }


def build_row(queue_row: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    aligned = aligned_comments(queue_row, review)
    if not review.get("claim_found"):
        # Without a recovered livestream claim, comment relations cannot be
        # proposal-valid alignments even if the raw LLM judgment named refute.
        aligned = []
    rel = relation_counts(aligned)
    state = promotion_state(queue_row, review, rel)
    y = y_perception_from_relations(review, rel)
    c = reliability_v2(queue_row, review, aligned, state, y)
    ev_params, ev_ocr, ev_vlm = evidence_payload(review)
    claim_text = clean(review.get("claim_text"))
    mask = contrastive_mask(state, y, review, rel)
    role = sample_role(state, y, rel)
    label_observed = y is not None
    row = {
        "pair_id": pair_id(queue_row),
        "product_id": queue_row.get("product_id"),
        "room_id": queue_row.get("room_id"),
        "category": queue_row.get("category"),
        "subcategory": queue_row.get("subcategory"),
        "attribute_id": queue_row.get("attribute_id"),
        "attribute_name": queue_row.get("attribute_name"),
        "product_title": queue_row.get("product_title"),
        "y": y,
        "y_perception": y,
        "label_observed": label_observed,
        "c": c,
        "c_reliability": c,
        "sample_role": role,
        "contrastive_mask": mask,
        "contrastive_mask_reason": "" if mask else "not_strict_triplet",
        "confidence": clean(review.get("confidence")) or "low",
        "claim": {
            "has_claim_srt": bool(review.get("claim_found")),
            "passage": claim_text,
            "segments": [
                {
                    "claim_id": f"{pair_id(queue_row)}__fullpair_llm_v2",
                    "clip_id": clean(review.get("claim_source")),
                    "start_ts": clean(review.get("claim_timestamp")),
                    "end_ts": "",
                    "text": claim_text,
                    "_reconstructed": True,
                }
            ] if claim_text else [],
        },
        "evidence_params": ev_params,
        "evidence_ocr": ev_ocr,
        "evidence_vlm": ev_vlm,
        "coverage": int(bool(ev_params)) + int(bool(ev_ocr)) + int(bool(ev_vlm)),
        "evidence_count": len(ev_params) + len(ev_ocr) + len(ev_vlm),
        "proposal_label_audit": {
            "policy": "stateful_proposal_dataset_v2",
            "promotion_state": state,
            "comment_relation_counts": dict(rel),
            "aligned_comment_count": len(aligned),
            "label_basis": clean(review.get("label_basis")),
            "llm_action": clean(review.get("action")),
            "claim_evidence_relation": clean(review.get("claim_evidence_relation")),
            "old_y": queue_row.get("old_y"),
            "old_c": queue_row.get("old_c"),
            "old_label_state": queue_row.get("old_label_state"),
            "positive_rule": "aligned consumer refutation of the repaired livestream claim",
            "no_shortcut": "product evidence contradiction alone never sets y_perception=1",
        },
        "label_audit": {
            "policy": "stateful_proposal_dataset_v2",
            "promotion_state": state,
            "comment_relation_counts": dict(rel),
            "aligned_comment_count": len(aligned),
        },
        "_full_pair_queue_type": queue_row.get("queue_type"),
        "_full_pair_priority": queue_row.get("priority"),
        "_full_pair_reconstruction_flags": queue_row.get("reconstruction_flags"),
        "_llm_review": {
            "model": review.get("model"),
            "claim_found": review.get("claim_found"),
            "product_evidence_found": review.get("product_evidence_found"),
            "raw_new_y": review.get("raw_new_y"),
            "clean_new_y": review.get("new_y"),
        },
        "_aligned_consumer_mentions": aligned,
        "_consumer_mentions_total": queue_row.get("consumer_mentions_total"),
        "_consumer_mentions_neg": queue_row.get("consumer_mentions_neg"),
        "_attribute_scope": "product_attribute",
    }
    return row


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stateful Proposal Dataset v2",
        "",
        "This report separates consumer-perception labels from triplet completion status.",
        "Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.",
        "",
        "## Inputs",
        "",
        f"- queue: `{report['queue']}`",
        f"- reviews: `{report['reviews']}`",
        "",
        "## Outputs",
        "",
        f"- all stateful rows: `{report['out_all']}`",
        f"- observed supervised rows: `{report['out_supervised']}`",
        f"- contrastive-eligible rows: `{report['out_contrastive']}`",
        f"- repair/unobserved rows: `{report['out_repair']}`",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "reviewed_rows",
        "label_observed_rows",
        "contrastive_rows",
        "repair_rows",
        "missing_queue_rows",
        "y_perception",
        "sample_role",
        "promotion_state",
        "claim_found",
        "product_evidence_found",
        "category_observed",
        "duplicate_claim_family_groups",
        "conflicting_claim_family_groups",
        "contrastive_masked_by_claim_family",
        "split",
        "split_leakage",
    ]:
        lines.append(f"- `{key}`: `{report.get(key)}`")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def summarize(
    rows: list[dict[str, Any]],
    supervised: list[dict[str, Any]],
    contrastive: list[dict[str, Any]],
    repair: list[dict[str, Any]],
    missing_queue_rows: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "reviewed_rows": len(rows),
        "label_observed_rows": len(supervised),
        "contrastive_rows": len(contrastive),
        "repair_rows": len(repair),
        "missing_queue_rows": missing_queue_rows,
        "y_perception": dict(Counter(str(r.get("y_perception")) for r in rows)),
        "sample_role": dict(Counter(clean(r.get("sample_role")) for r in rows)),
        "promotion_state": dict(Counter(clean((r.get("proposal_label_audit") or {}).get("promotion_state")) for r in rows)),
        "claim_found": dict(Counter(bool((r.get("_llm_review") or {}).get("claim_found")) for r in rows)),
        "product_evidence_found": dict(Counter(bool((r.get("_llm_review") or {}).get("product_evidence_found")) for r in rows)),
        "category_observed": dict(Counter(clean(r.get("category")) for r in supervised)),
        "split": dict(Counter(clean(r.get("split")) for r in supervised)),
        "split_leakage": split_leakage(supervised) if supervised else {},
    }
    report.update(extra)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl")
    ap.add_argument("--reviews", nargs="+", required=True)
    ap.add_argument("--out_all", default="data/final/repaired_v1/stateful_proposal_dataset_v2_all_20260614.jsonl")
    ap.add_argument("--out_supervised", default="data/final/repaired_v1/stateful_proposal_dataset_v2_supervised_20260614.jsonl")
    ap.add_argument("--out_contrastive", default="data/final/repaired_v1/stateful_proposal_dataset_v2_contrastive_20260614.jsonl")
    ap.add_argument("--out_repair", default="data/final/repaired_v1/stateful_proposal_dataset_v2_repair_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/stateful_proposal_dataset_v2_20260614.report.json")
    ap.add_argument("--markdown", default="docs/STATEFUL_PROPOSAL_DATASET_V2_20260614.md")
    args = ap.parse_args()

    queue = read_queue(args.queue)
    reviews, review_report = read_reviews(args.reviews)
    rows: list[dict[str, Any]] = []
    missing_queue_rows = 0
    for pid, review in reviews.items():
        qrow = queue.get(pid)
        if not qrow:
            missing_queue_rows += 1
            continue
        rows.append(build_row(qrow, review))

    family_report = annotate_claim_families(rows)
    supervised = assign_room_splits([r for r in rows if r.get("label_observed")]) if rows else []
    split_by_pair = {pair_id(r): r.get("split") for r in supervised}
    for row in rows:
        row["split"] = split_by_pair.get(pair_id(row), "")
    contrastive = [r for r in rows if r.get("contrastive_mask")]
    repair = [r for r in rows if not r.get("label_observed")]

    write_jsonl(args.out_all, rows)
    write_jsonl(args.out_supervised, supervised)
    write_jsonl(args.out_contrastive, contrastive)
    write_jsonl(args.out_repair, repair)
    report = summarize(rows, supervised, contrastive, repair, missing_queue_rows, {**review_report, **family_report})
    report.update({
        "queue": args.queue,
        "reviews": args.reviews,
        "out_all": args.out_all,
        "out_supervised": args.out_supervised,
        "out_contrastive": args.out_contrastive,
        "out_repair": args.out_repair,
        "report": args.report,
        "markdown": args.markdown,
    })
    write_json(args.report, report)
    write_markdown(args.markdown, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
