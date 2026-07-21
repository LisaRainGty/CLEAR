"""Rebuild versioned CLAIMARC datasets after deterministic pipeline repairs.

This script does not call an external model and does not overwrite the
canonical Stage B/C/final artifacts. It repairs the most consequential label
bug found in the audit: B4/B5 stored global review polarity, while the method
requires attribute-level polarity from Stage A.

Outputs:
- repaired pair_records with attribute-level review polarity
- repaired labels
- versioned final datasets with clean room-level splits
- a compact JSON/Markdown audit report
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import config
from common import product_index as pidx
from common.io_utils import read_jsonl, write_json, write_jsonl
from data_quality.audit_dataset_quality import has_claim, quality_bucket, source_count
from labels.build_labels import compute_label


PROTECTED_AUX_FIELDS = {
    "pair_id",
    "product_id",
    "attribute_id",
    "room_id",
    "split",
    "y",
    "c",
    "label_audit",
}

SAFE_AUX_FIELDS = {
    "arguments",
    "_evidence_policy",
    "_dropped_args_without_source",
}

BAD_CLAIM_QUALITY = {"no_claim", "garbled"}
SUPPORTED_STATES = {"supported"}
LOW_RISK = {"none", "low", ""}

SERVICE_OR_PROCESS_TERMS = {
    "客服",
    "售后",
    "物流",
    "发货",
    "快递",
    "配送",
    "退货",
    "退款",
    "退换",
    "下单",
    "店铺",
    "包装服务",
}

SUBJECTIVE_EVAL_TERMS = set(getattr(config, "EVAL_LEAKAGE_KEYWORDS", [])) | {
    "体验",
    "感受",
    "评价",
    "满意",
    "推荐",
    "回购",
    "复购",
    "性价比",
    "划算",
    "喜欢",
    "购买意愿",
    "真实性评价",
}


def key_from_ids(pid: Any, aid: Any) -> tuple[str, str]:
    return str(pid), str(aid)


def pair_id(rec: dict[str, Any]) -> str:
    return str(rec.get("pair_id") or f"p{rec.get('product_id')}__{rec.get('attribute_id')}")


def read_jsonl_list(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def mention_score(row: dict[str, Any]) -> tuple[int, int, int, int]:
    explicit = 1 if row.get("explicit_fact_hit") else 0
    strong = 1 if row.get("mention_strength") == "strong" else 0
    negative = 1 if row.get("polarity") == "neg" else 0
    span_len = len(str(row.get("evidence_span", "") or ""))
    return explicit, strong, negative, span_len


def load_stagea_mentions(path: str | Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Best Stage A mention per (product, attribute, review_id)."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        rid = str(row.get("review_id", ""))
        key = (str(row.get("product_id", "")), str(row.get("attribute_id", "")), rid)
        if not all(key):
            continue
        prev = out.get(key)
        if prev is None or mention_score(row) > mention_score(prev):
            out[key] = row
    return out


def load_pair_stagea_meta(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    pairs: dict[tuple[str, str], Counter] = defaultdict(Counter)
    examples: dict[tuple[str, str], str] = {}
    for row in read_jsonl(path):
        key = (str(row.get("product_id", "")), str(row.get("attribute_id", "")))
        if not all(key):
            continue
        pairs[key][str(row.get("type", "") or "")] += 1
        examples.setdefault(key, str(row.get("category", "")))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, counts in pairs.items():
        dominant = counts.most_common(1)[0][0] if counts else ""
        out[key] = {
            "stagea_type": dominant,
            "stagea_type_counts": dict(counts),
            "category": examples.get(key, ""),
        }
    return out


def attribute_scope(rec: dict[str, Any], meta: dict[str, Any] | None = None) -> str:
    name = str(rec.get("attribute_name") or rec.get("attribute_canonical") or rec.get("attribute_id") or "")
    aid = str(rec.get("attribute_id", ""))
    text = f"{name} {aid}"
    stagea_type = str((meta or {}).get("stagea_type") or rec.get("_stagea_type") or "")
    if stagea_type == "service" or any(term in text for term in SERVICE_OR_PROCESS_TERMS):
        return "service_or_process"
    if any(term in text for term in SUBJECTIVE_EVAL_TERMS):
        return "subjective_or_personal_eval"
    return "product_attribute"


def repair_pair_records(
    pair_records_path: str | Path,
    stagea_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stagea = load_stagea_mentions(stagea_path)
    old_rows = read_jsonl_list(pair_records_path)
    repaired_rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    stats = Counter()
    flip_examples: list[dict[str, Any]] = []

    for old in old_rows:
        rec = copy.deepcopy(old)
        pid = str(rec.get("product_id", ""))
        aid = str(rec.get("attribute_id", ""))
        before_label = compute_label(old)
        comments = rec.get("reviews") or []
        for c in comments:
            stats["comments_total"] += 1
            rid = str(c.get("comment_id", ""))
            src = stagea.get((pid, aid, rid))
            if not src:
                stats["comments_missing_stagea"] += 1
                continue
            old_pol = str(c.get("polarity", "neu"))
            new_pol = str(src.get("polarity", old_pol) or old_pol)
            if old_pol != new_pol:
                stats["comments_polarity_changed"] += 1
            c["_global_review_polarity_before_repair"] = old_pol
            c["_review_polarity_from_stageA"] = src.get("review_polarity", "")
            c["polarity"] = new_pol
            c["mention_strength"] = src.get("mention_strength", c.get("mention_strength", "weak"))
            c["explicit_fact_hit"] = bool(src.get("explicit_fact_hit", c.get("explicit_fact_hit", False)))
            c["evidence_span"] = src.get("evidence_span", c.get("evidence_span", ""))
            c["review_time"] = src.get("review_time", c.get("review_time", ""))

        aligned = [c for c in comments if int(c.get("y_supportability", 0) or 0) == 1]
        rec["stats"] = {
            "N_total": len(comments),
            "N_aligned": len(aligned),
            "N_aligned_neg": sum(1 for c in aligned if c.get("polarity") == "neg"),
        }
        after_label = compute_label(rec)
        rec["_repair_audit"] = {
            "repair": "stageA_attribute_polarity_v1",
            "y_before": before_label["y"],
            "y_after": after_label["y"],
            "n_neg_aligned_before": before_label["label_audit"].get("n_neg_aligned", 0),
            "n_neg_aligned_after": after_label["label_audit"].get("n_neg_aligned", 0),
        }
        if before_label["y"] != after_label["y"]:
            stats[f"pair_label_flip_{before_label['y']}_to_{after_label['y']}"] += 1
            if len(flip_examples) < 30:
                flip_examples.append({
                    "pair_id": rec.get("pair_id"),
                    "attribute_id": aid,
                    "attribute_name": rec.get("attribute_canonical", aid),
                    "y_before": before_label["y"],
                    "y_after": after_label["y"],
                    "claimful": has_claim(rec),
                    "example_reviews": [
                        {
                            "text": c.get("text", "")[:100],
                            "polarity": c.get("polarity"),
                            "global_before": c.get("_global_review_polarity_before_repair"),
                            "y_supportability": c.get("y_supportability"),
                        }
                        for c in comments[:3]
                    ],
                })
        stats[f"pair_y_before_{before_label['y']}"] += 1
        stats[f"pair_y_after_{after_label['y']}"] += 1
        repaired_rows.append(rec)
        labels.append({
            "pair_id": rec["pair_id"],
            "product_id": pid,
            "attribute_id": aid,
            **after_label,
            "_repair": rec["_repair_audit"],
        })

    report = dict(stats)
    report["pair_label_flip_examples"] = flip_examples
    return repaired_rows, labels, report


def merge_safe_aux(base: dict[str, Any], aux: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    if not aux:
        return out
    for field in SAFE_AUX_FIELDS:
        if field in aux and field not in PROTECTED_AUX_FIELDS:
            out[field] = aux[field]
    if "split" in aux:
        out["_aux_split_ignored"] = aux.get("split")
    return out


def load_aux_by_pair(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    return {pair_id(r): r for r in read_jsonl(path)}


def build_final_dataset(
    base_dataset_path: str | Path,
    labels: list[dict[str, Any]],
    aux_dataset_path: str | Path | None,
    stagea_meta: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    labels_by_key = {key_from_ids(r["product_id"], r["attribute_id"]): r for r in labels}
    aux_by_pair = load_aux_by_pair(aux_dataset_path)
    rows: list[dict[str, Any]] = []
    for rec in read_jsonl(base_dataset_path):
        k = key_from_ids(rec.get("product_id"), rec.get("attribute_id"))
        lab = labels_by_key.get(k)
        if not lab:
            continue
        out = merge_safe_aux(rec, aux_by_pair.get(pair_id(rec)))
        out["_y_before_attrpol_repair"] = int(rec.get("y", 0))
        out["_c_before_attrpol_repair"] = float(rec.get("c", 0.0) or 0.0)
        out["_label_audit_before_attrpol_repair"] = rec.get("label_audit", {})
        out["y"] = int(lab["y"])
        out["c"] = float(lab["c"])
        out["label_audit"] = lab.get("label_audit", {})
        out["_repair"] = lab.get("_repair", {})
        meta = stagea_meta.get(k, {})
        out["_stagea_type"] = meta.get("stagea_type", "")
        out["_stagea_type_counts"] = meta.get("stagea_type_counts", {})
        out["_attribute_scope"] = attribute_scope(out, meta)
        rows.append(out)
    return rows


def split_score(cur: dict[str, Counter], target: dict[str, dict[str, float]]) -> float:
    score = 0.0
    for split in config.SPLIT_RATIO:
        tn = max(1.0, target[split]["n"])
        tp = max(1.0, target[split]["pos"])
        score += ((cur[split]["n"] - target[split]["n"]) / tn) ** 2
        score += 2.0 * ((cur[split]["pos"] - target[split]["pos"]) / tp) ** 2
    return score


def assign_room_splits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("room_id") or r.get("product_id") or pair_id(r))].append(r)
    total_n = len(rows)
    total_pos = sum(int(r.get("y", 0)) for r in rows)
    target = {
        s: {"n": ratio * total_n, "pos": ratio * total_pos}
        for s, ratio in config.SPLIT_RATIO.items()
    }
    cur = {s: Counter({"n": 0, "pos": 0}) for s in config.SPLIT_RATIO}
    assignment: dict[str, str] = {}
    ordered = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1]), -sum(int(r.get("y", 0)) for r in kv[1]), kv[0]),
    )
    for room, vals in ordered:
        add = Counter({"n": len(vals), "pos": sum(int(r.get("y", 0)) for r in vals)})
        best_split = None
        best_score = None
        for split in config.SPLIT_RATIO:
            cur[split].update(add)
            score = split_score(cur, target)
            cur[split].subtract(add)
            if best_score is None or score < best_score:
                best_split = split
                best_score = score
        assignment[room] = str(best_split)
        cur[str(best_split)].update(add)

    out: list[dict[str, Any]] = []
    for r in rows:
        nr = dict(r)
        room = str(nr.get("room_id") or nr.get("product_id") or pair_id(nr))
        nr["split"] = assignment[room]
        out.append(nr)
    return out


def split_leakage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rooms: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        rooms[str(r.get("room_id", ""))].add(str(r.get("split", "")))
    bad = {room: sorted(splits) for room, splits in rooms.items() if len(splits) > 1}
    return {
        "leaky_rooms": len(bad),
        "leaky_rows": sum(1 for r in rows if str(r.get("room_id", "")) in bad),
        "examples": dict(list(bad.items())[:10]),
    }


def summarize_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "labels": dict(Counter(int(r.get("y", 0)) for r in rows)),
        "split": dict(Counter(str(r.get("split", "")) for r in rows)),
        "split_leakage": split_leakage(rows),
        "claimful": sum(1 for r in rows if has_claim(r)),
        "source0": sum(1 for r in rows if source_count(r) == 0),
        "quality_bucket": dict(Counter(quality_bucket(r) for r in rows)),
        "attribute_scope": dict(Counter(str(r.get("_attribute_scope", "")) for r in rows)),
        "category": dict(Counter(str(r.get("category", "")) for r in rows)),
    }


def select_hq_claimful(rows: list[dict[str, Any]], neg_ratio: float) -> list[dict[str, Any]]:
    claimful = [r for r in rows if has_claim(r)]
    pos = [r for r in claimful if int(r.get("y", 0)) == 1]
    neg = [r for r in claimful if int(r.get("y", 0)) == 0]
    neg_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in neg:
        neg_by_bucket[quality_bucket(r)].append(r)
    for vals in neg_by_bucket.values():
        vals.sort(key=lambda r: (-float(r.get("c", 0.0) or 0.0), pair_id(r)))
    target_neg = int(round(len(pos) * neg_ratio))
    selected_neg: list[dict[str, Any]] = []
    for bucket in (
        "neg_core",
        "neg_silver_sourceful",
        "neg_context_sourceful",
        "neg_silver_comment_only",
        "neg_weak",
    ):
        need = target_neg - len(selected_neg)
        if need <= 0:
            break
        selected_neg.extend(neg_by_bucket.get(bucket, [])[:need])
    selected = pos + selected_neg
    selected.sort(key=lambda r: (str(r.get("room_id", "")), pair_id(r)))
    return selected


def objective_dataset(
    repaired_by_pair: dict[str, dict[str, Any]],
    objective_base_path: str | Path,
    adjudication_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not Path(objective_base_path).exists() or not Path(adjudication_path).exists():
        return [], {"missing_inputs": [str(objective_base_path), str(adjudication_path)]}
    base_by_pair = {pair_id(r): r for r in read_jsonl(objective_base_path)}
    selected: list[dict[str, Any]] = []
    decisions = Counter()
    for adj in read_jsonl(adjudication_path):
        pid = pair_id(adj)
        base = base_by_pair.get(pid) or repaired_by_pair.get(pid)
        if not base:
            continue
        cq = str(adj.get("claim_quality", "") or "").strip().lower()
        state = str(adj.get("evidence_state", "") or "").strip().lower()
        risk = str(adj.get("misleading_risk", "") or "").strip().lower()
        if cq in BAD_CLAIM_QUALITY:
            decisions["drop_bad_claim"] += 1
            continue
        y: int | None = None
        decision = ""
        if state == "contradicted":
            y = 1
            decision = "objective_contradicted"
        elif state in SUPPORTED_STATES and risk in LOW_RISK:
            y = 0
            decision = "objective_supported_clean"
        else:
            decisions[f"drop_{state or 'unknown'}_{risk or 'unknown'}"] += 1
            continue
        rec = dict(base)
        repaired_meta = repaired_by_pair.get(pid) or {}
        for field in ("_stagea_type", "_stagea_type_counts", "_attribute_scope"):
            if field in repaired_meta:
                rec[field] = repaired_meta[field]
        rec.setdefault("_attribute_scope", attribute_scope(rec))
        rec["_y_before_objective_relabel"] = int(rec.get("y", 0))
        rec["y"] = y
        rec["c"] = 0.90 if y == 1 else max(0.20, min(0.85, float(rec.get("c", 0.05) or 0.05)))
        rec["_objective_adjudication"] = {
            "claim_quality": adj.get("claim_quality"),
            "evidence_state": adj.get("evidence_state"),
            "misleading_risk": adj.get("misleading_risk"),
            "key_claim": adj.get("key_claim", ""),
            "key_evidence": adj.get("key_evidence", ""),
            "rationale": adj.get("rationale", ""),
            "flags": adj.get("flags", []),
            "model": adj.get("model", ""),
        }
        rec["_objective_decision"] = decision
        decisions[decision] += 1
        selected.append(rec)
    selected.sort(key=lambda r: (str(r.get("room_id", "")), pair_id(r)))
    return selected, {"decisions": dict(decisions)}


def missing_claim_risk_pairs(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rec in pair_rows:
        if has_claim(rec):
            continue
        hits = [
            c for c in rec.get("reviews", [])
            if c.get("polarity") == "neg"
            and c.get("mention_strength") == "strong"
            and c.get("explicit_fact_hit")
        ]
        if hits:
            out.append({
                "pair_id": rec.get("pair_id"),
                "product_id": rec.get("product_id"),
                "attribute_id": rec.get("attribute_id"),
                "attribute_name": rec.get("attribute_canonical", rec.get("attribute_id")),
                "n_hits": len(hits),
                "example": hits[0].get("text", "")[:160],
            })
    return out


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = ["# CLAIMARC Repaired Dataset v1", ""]
    lines.append("## Repair")
    for k, v in report.get("repair", {}).items():
        if k == "pair_label_flip_examples":
            lines.append(f"- `{k}`: {len(v)} examples")
        else:
            lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Datasets")
    for name, stats in report.get("datasets", {}).items():
        lines.append(f"### {name}")
        for k, v in stats.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    lines.append("## Missing Claim Risk")
    m = report.get("missing_claim_risk", {})
    for k, v in m.items():
        if k == "examples":
            lines.append(f"- `{k}`: {len(v)} examples")
            for ex in v[:10]:
                lines.append(f"  - `{ex}`")
        else:
            lines.append(f"- `{k}`: `{v}`")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_records", default=str(config.STAGE_B / "pair_records.jsonl"))
    ap.add_argument("--stagea", default=str(config.STAGE_A / "resolved_aspects.jsonl"))
    ap.add_argument("--base_dataset", default=str(config.FINAL / "dataset.jsonl"))
    ap.add_argument("--aux_dataset", default=str(config.FINAL / "dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl"))
    ap.add_argument("--objective_base", default=str(config.FINAL / "dataset_hq_broad_claimful_enriched_v1.jsonl"))
    ap.add_argument("--adjudication", default=str(config.FINAL / "claim_evidence_adjudication_hq_broad_claimful_enriched_v1.jsonl"))
    ap.add_argument("--out_dir", default=str(config.FINAL / "repaired_v1"))
    ap.add_argument("--neg_ratio", type=float, default=1.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows, labels, repair_report = repair_pair_records(args.pair_records, args.stagea)
    stagea_meta = load_pair_stagea_meta(args.stagea)
    write_jsonl(out_dir / "pair_records_attrpol_v1.jsonl", pair_rows)
    write_jsonl(out_dir / "labels_attrpol_v1.jsonl", labels)

    all_rows = assign_room_splits(build_final_dataset(args.base_dataset, labels, args.aux_dataset, stagea_meta))
    claimful_rows = assign_room_splits([r for r in all_rows if has_claim(r)])
    sourceful_rows = assign_room_splits([r for r in claimful_rows if source_count(r) > 0])
    hq_rows = assign_room_splits(select_hq_claimful(all_rows, args.neg_ratio))
    hq_product_rows = assign_room_splits([
        r for r in hq_rows if str(r.get("_attribute_scope", "")) == "product_attribute"
    ])

    repaired_by_pair = {pair_id(r): r for r in all_rows}
    objective_rows, objective_report = objective_dataset(repaired_by_pair, args.objective_base, args.adjudication)
    objective_rows = assign_room_splits(objective_rows) if objective_rows else []
    objective_product_rows = assign_room_splits([
        r for r in objective_rows if str(r.get("_attribute_scope", "")) == "product_attribute"
    ])

    outputs = {
        "dataset_attrpol_all_v1.jsonl": all_rows,
        "dataset_attrpol_claimful_v1.jsonl": claimful_rows,
        "dataset_attrpol_claimful_sourceful_v1.jsonl": sourceful_rows,
        "dataset_attrpol_hq_claimful_v1.jsonl": hq_rows,
        "dataset_attrpol_hq_product_v1.jsonl": hq_product_rows,
        "dataset_objective_contradicted_v1.jsonl": objective_rows,
        "dataset_objective_contradicted_product_v1.jsonl": objective_product_rows,
    }
    for name, rows in outputs.items():
        write_jsonl(out_dir / name, rows)

    missing_claim = missing_claim_risk_pairs(pair_rows)
    write_jsonl(out_dir / "missing_claim_risk_pairs_v1.jsonl", missing_claim)

    report = {
        "repair": repair_report,
        "datasets": {name: summarize_dataset(rows) for name, rows in outputs.items()},
        "objective": objective_report,
        "missing_claim_risk": {
            "n": len(missing_claim),
            "examples": missing_claim[:30],
        },
        "outputs": {name: str(out_dir / name) for name in outputs},
    }
    write_json(out_dir / "repaired_v1_report.json", report)
    write_markdown(report, out_dir / "REPAIRED_DATASETS_V1.md")
    print(f"[rebuild_repaired_datasets_v1] wrote outputs under {out_dir}")
    for name, stats in report["datasets"].items():
        print(f"  {name}: n={stats['n']} labels={stats['labels']} leakage={stats['split_leakage']['leaky_rooms']}")
    print(f"  missing_claim_risk_pairs={len(missing_claim)}")


if __name__ == "__main__":
    main()
