"""Audit CLAIMARC pair-level label quality and evidence coverage.

This script is intentionally deterministic: it reads existing JSONL datasets
and summarizes where labels are likely learnable, weak, or structurally noisy.
It does not call an LLM and does not change any data.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_count(rec: dict[str, Any]) -> int:
    ev = rec.get("evidence_count") or {}
    if isinstance(ev, dict):
        return sum(int(v or 0) for v in ev.values())
    return int(ev or 0)


def has_claim(rec: dict[str, Any]) -> bool:
    claim = rec.get("claim") or {}
    return bool(claim.get("has_claim_srt") and (claim.get("segments") or claim.get("passage")))


def audit_value(rec: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float((rec.get("label_audit") or {}).get(key, default) or default)


def argument_text(rec: dict[str, Any]) -> str:
    args = rec.get("arguments") or {}
    return "\n".join(str(args.get(k, "") or "") for k in (
        "supporting_argument",
        "refuting_argument",
        "evidence_gap",
    ))


RISK_TERMS = (
    "无法验证", "缺乏", "没有提供", "未提供", "不确定", "冲突", "不一致",
    "无法确认", "缺少", "风险", "夸大", "未标注", "无依据",
)


SUPPORT_TERMS = (
    "直接支持", "相符", "一致", "吻合", "明确显示", "证据显示",
)


def argument_risk_score(rec: dict[str, Any]) -> int:
    text = argument_text(rec)
    return sum(1 for term in RISK_TERMS if term in text)


def argument_support_score(rec: dict[str, Any]) -> int:
    text = argument_text(rec)
    return sum(1 for term in SUPPORT_TERMS if term in text)


def quality_bucket(rec: dict[str, Any]) -> str:
    y = int(rec.get("y", 0))
    c = float(rec.get("c", 0.0) or 0.0)
    n_neg = audit_value(rec, "n_neg_aligned")
    n_pos = audit_value(rec, "n_pos_aligned")
    neg_share = audit_value(rec, "neg_share")
    sourceful = source_count(rec) > 0
    claimful = has_claim(rec)
    suspicious = bool((rec.get("label_audit") or {}).get("suspected_fake"))

    if not claimful:
        return "drop_no_claim"
    if y == 1:
        if c >= 0.15 and n_neg >= 2 and (sourceful or neg_share >= 0.50):
            return "pos_core"
        if c >= 0.10 and n_neg >= 1:
            return "pos_silver"
        return "pos_weak"
    if suspicious:
        return "neg_suspect_fake"
    if sourceful and c >= 0.15 and n_pos >= 2:
        return "neg_core"
    if sourceful and c >= 0.05 and n_pos >= 1:
        return "neg_silver_sourceful"
    if sourceful and c >= 0.05:
        return "neg_context_sourceful"
    if c >= 0.20 and n_pos >= 2:
        return "neg_silver_comment_only"
    return "neg_weak"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["n"] = len(rows)
    out["label"] = dict(Counter(int(r.get("y", 0)) for r in rows))
    out["confidence"] = dict(Counter(str(r.get("confidence", "")) for r in rows))
    out["category"] = dict(Counter(str(r.get("category", "")) for r in rows))
    out["quality_bucket"] = dict(Counter(quality_bucket(r) for r in rows))
    out["source_zero"] = sum(1 for r in rows if source_count(r) == 0)
    out["has_claim"] = sum(1 for r in rows if has_claim(r))
    out["argument_risk_nonzero"] = sum(1 for r in rows if argument_risk_score(r) > 0)
    out["argument_support_nonzero"] = sum(1 for r in rows if argument_support_score(r) > 0)

    by_label = {}
    for y in (0, 1):
        sub = [r for r in rows if int(r.get("y", 0)) == y]
        if not sub:
            continue
        by_label[str(y)] = {
            "n": len(sub),
            "mean_c": round(statistics.mean(float(r.get("c", 0.0) or 0.0) for r in sub), 4),
            "source_zero": sum(1 for r in sub if source_count(r) == 0),
            "has_claim": sum(1 for r in sub if has_claim(r)),
            "confidence": dict(Counter(str(r.get("confidence", "")) for r in sub)),
            "quality_bucket": dict(Counter(quality_bucket(r) for r in sub)),
            "mean_n_aligned": round(statistics.mean(audit_value(r, "n_aligned") for r in sub), 3),
            "mean_n_neg": round(statistics.mean(audit_value(r, "n_neg_aligned") for r in sub), 3),
            "mean_n_pos": round(statistics.mean(audit_value(r, "n_pos_aligned") for r in sub), 3),
        }
    out["by_label"] = by_label

    by_cat = {}
    cats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cats[str(r.get("category", ""))].append(r)
    for cat, sub in sorted(cats.items()):
        by_cat[cat] = {
            "n": len(sub),
            "pos": sum(int(r.get("y", 0)) for r in sub),
            "pos_core": sum(1 for r in sub if quality_bucket(r) == "pos_core"),
            "neg_core": sum(1 for r in sub if quality_bucket(r) == "neg_core"),
            "source_zero": sum(1 for r in sub if source_count(r) == 0),
        }
    out["by_category"] = by_cat
    return out


def write_markdown(report: dict[str, Any], out: Path) -> None:
    lines = ["# CLAIMARC Data Quality Audit\n"]
    for name, rep in report["datasets"].items():
        lines += [
            f"## {name}",
            f"- n={rep['n']} label={rep['label']} source_zero={rep['source_zero']} has_claim={rep['has_claim']}",
            f"- quality_bucket={rep['quality_bucket']}",
            "",
            "| category | n | pos | pos_core | neg_core | source_zero |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for cat, row in rep["by_category"].items():
            lines.append(
                f"| {cat} | {row['n']} | {row['pos']} | {row['pos_core']} | "
                f"{row['neg_core']} | {row['source_zero']} |"
            )
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, help="name=path or path")
    ap.add_argument("--out_json", default="data/final/data_quality_audit_20260612.json")
    ap.add_argument("--out_md", default="docs/DATA_QUALITY_AUDIT_20260612.md")
    args = ap.parse_args()

    report = {"datasets": {}}
    for item in args.dataset:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).stem
        report["datasets"][name] = summarize(read_jsonl(path))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, out_md)
    print(f"[audit_dataset_quality] wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
