#!/usr/bin/env python3
"""Compile fair-rerun artifacts into one reviewer-readable file, table by table."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("acc", "pos_f1", "macro_f1", "auprc", "auroc", "wF1", "ece")
SINGLE_RUN_TAGS = {
    "qwen_flash_zero", "qwen_flash_fs5", "gpt54_zero", "gpt54_fs5",
    "gemini35_zero", "gemini35_fs5", "kimi_zero", "kimi_fs5",
}


def expected_runs(tag: str) -> int:
    if tag in SINGLE_RUN_TAGS or tag.startswith(("BGEfz_", "hp_", "c_")):
        return 1
    return 3


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_rows(root: Path, result_root: Path, evidence_policy: str,
              evidence_policy_overrides=None):
    evidence_policy_overrides = evidence_policy_overrides or {}
    rows = []
    rejected = []
    for path in sorted((result_root / "jobs").glob("*.jsonl")):
        if path.name.startswith("._"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_result_file"] = str(path.relative_to(root))
                reasons = []
                row_tag = str(row.get("tag") or "")
                expected_policy = evidence_policy_overrides.get(
                    row_tag, evidence_policy)
                if row.get("evidence_policy") not in (None, expected_policy):
                    reasons.append(f"non_{expected_policy}")
                for field in ("n_err_val", "n_err_test", "n_err"):
                    if int(row.get(field) or 0) > 0:
                        reasons.append(f"{field}={int(row[field])}")
                if reasons:
                    rejected.append({
                        "result_file": row["_result_file"],
                        "suite_job": row.get("_suite_job"),
                        "tag": row.get("tag"),
                        "reasons": reasons,
                    })
                    continue
                rows.append(row)
    return rows, rejected


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("tag"):
            grouped[row["tag"]].append(row)
    out = {}
    for tag, group in grouped.items():
        item = {"n": len(group), "seeds": sorted({r.get("seed") for r in group
                                                   if r.get("seed") is not None})}
        for metric in METRICS:
            values = [float(r[metric]) for r in group if r.get(metric) is not None]
            if values:
                item[metric] = {
                    "mean": float(np.mean(values)),
                    # Paper tables report dispersion across independent seeds.
                    # Use the sample SD so the canonical 0.8100/0.8072/0.8017
                    # Macro-F1 runs reproduce the archived 80.63 ± 0.42 row.
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "std_ddof": 1,
                }
        item["dataset_sha256"] = sorted({r.get("dataset_sha256", "") for r in group})
        item["evidence_policy"] = sorted({r.get("evidence_policy", "") for r in group})
        out[tag] = item
    return out


def metric_cell(agg, tag, metric, percent=True, expected_n=None):
    value = agg.get(tag, {}).get(metric)
    required = expected_runs(tag) if expected_n is None else expected_n
    if not value or agg.get(tag, {}).get("n", 0) < required:
        return "PENDING"
    scale = 100 if percent else 1
    mean, std = scale * value["mean"], scale * value["std"]
    n = agg[tag]["n"]
    return f"{mean:.2f}±{std:.2f}" if n > 1 else f"{mean:.2f}"


def metric_table(lines, title, entries, agg, metrics=METRICS,
                 expected_n=None):
    lines.extend([f"## {title}", "", "| 方法/设定 | " + " | ".join(metrics) + " | n |",
                  "|---|" + "---:|" * (len(metrics) + 1)])
    for label, tag in entries:
        cells = [metric_cell(agg, tag, metric, expected_n=expected_n) for metric in metrics]
        n = agg.get(tag, {}).get("n", 0)
        required = expected_runs(tag) if expected_n is None else expected_n
        shown_n = n if n >= required else "PENDING"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {shown_n} |")
    lines.append("")


def xdom_table(lines, title, blob):
    lines.extend([f"## {title}", "",
                  "| System | Acc | F1pos | Macro-F1 | AUPRC | AUROC | folds |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    if not blob:
        lines.extend(["| PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |", ""])
        return
    paper_order = (
        "ESIM", "BERT-CLS", "RoBERTa-CLS", "LLM zero-shot (Qwen-Flash)",
        "LLM few-shot (Qwen-Flash)", "CLAIMARC",
    )
    aggregate_rows = blob.get("aggregate", {})
    for name in paper_order:
        if name not in aggregate_rows:
            continue
        values = aggregate_rows[name]
        def cell(key):
            item = values.get(key, {})
            if item.get("mean") is None:
                return "--"
            return f"{item['mean']:.1f}±{item['std']:.1f}"
        lines.append(f"| {name} | {cell('acc')} | {cell('f1pos')} | {cell('macro_f1')} | "
                     f"{cell('auprc')} | {cell('auroc')} | {values.get('n_folds', 0)} |")
    lines.append("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/paper_fair.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy = str(config["fair_comparison"]["evidence_policy"])
    no_fusion_main = bool(config.get("claimarc", {}).get("no_fusion_main", False))
    attribute_blocked_main = bool(
        config.get("claimarc", {}).get("attribute_blocked_contrast", False))
    namespace = str(config.get("paper_suite", {}).get("artifact_namespace", "fair_rerun"))
    result_root = root / "results" / namespace
    evidence_policy_overrides = {
        f"input_{name}": str(view_policy)
        for name, view_policy in config.get("paper_suite", {}).get(
            "evidence_view_ablations", {}).items()
    }
    rows, rejected_rows = load_rows(
        root, result_root, policy, evidence_policy_overrides)
    agg = aggregate(rows)
    agg_seed0 = aggregate([row for row in rows if row.get("seed") == 0])
    audit = load_json(result_root / "reproducibility_report.json")
    if audit is None and namespace == "fair_rerun":
        audit = load_json(root / "results/audit/reproducibility_report.json")
    audit = audit or {}
    data = audit.get("dataset", {})
    if policy == "sources_only":
        title = "# 论文全部实验表：三源证据公平重跑"
        protocol_note = "`sources_only` (PARAM + OCR + VLM)"
    else:
        title = "# 论文全部实验表：Arguments-only 公平重跑"
        protocol_note = "`args_only` (supporting + refuting + evidence-gap arguments)"
    lines = [
        title, "",
        f"> 本文件只汇总 {protocol_note} 公平重跑。"
        "`PENDING` 表示对应 GPU/API 任务尚未成功完成，不会用历史异口径数字填补。", "",
        f"- Dataset SHA-256: `{data.get('sha256', 'PENDING')}`",
        f"- Rows: {data.get('rows', 'PENDING')}",
        f"- Argument records: {data.get('argument_records', 'PENDING')}",
        f"- Accepted fresh RESULT rows: {len(rows)}",
        f"- Rejected incomplete/off-protocol RESULT rows: {len(rejected_rows)}", "",
        "## Table 1. Dataset split and statistics", "",
        "| Split | N | Positive | Positive rate | Rooms |", "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        n = data.get("split_rows", {}).get(split, "PENDING")
        p = data.get("split_positives", {}).get(split, "PENDING")
        rate = data.get("split_positive_rates", {}).get(split)
        rate = f"{100 * rate:.2f}%" if isinstance(rate, (float, int)) else "PENDING"
        rooms = data.get("rooms_by_split", {}).get(split, "PENDING")
        lines.append(f"| {split} | {n} | {p} | {rate} | {rooms} |")
    all_n = sum(data.get("split_rows", {}).values()) if data.get("split_rows") else "PENDING"
    all_p = sum(data.get("split_positives", {}).values()) if data.get("split_positives") else "PENDING"
    all_rate = f"{100 * all_p / all_n:.2f}%" if isinstance(all_n, int) and all_n else "PENDING"
    all_rooms = sum(data.get("rooms_by_split", {}).values()) if data.get("rooms_by_split") else "PENDING"
    lines.append(f"| **All** | **{all_n}** | **{all_p}** | **{all_rate}** | **{all_rooms}** |")
    coverage = data.get("source_coverage_0_1_2_3", {})
    lines.extend([
        "",
        f"Evidence-source availability (0/1/2/3): "
        f"{coverage.get('0', 'PENDING')}/{coverage.get('1', 'PENDING')}/"
        f"{coverage.get('2', 'PENDING')}/{coverage.get('3', 'PENDING')}; "
        f"reliability c mean/median/range: "
        f"{data.get('c_mean', 'PENDING'):.3f}/{data.get('c_median', 'PENDING'):.3f}/"
        f"[{data.get('c_min', 'PENDING'):.3f}, {data.get('c_max', 'PENDING'):.3f}]."
        if isinstance(data.get("c_mean"), (int, float)) else "Dataset summary: PENDING.",
        f"Construction sources: {data.get('construction_sources', {})}; "
        f"aligned-comment pairs: {data.get('pairs_with_aligned_comments', 'PENDING')} "
        f"({100 * data.get('pairs_with_aligned_comments_rate', 0):.2f}%).",
    ])
    lines.extend(["", "## Table 2. Category distribution", "", "| Category | N |", "|---|---:|"])
    categories = data.get("categories", {})
    if categories:
        for name, count in sorted(categories.items()):
            lines.append(f"| {name} | {count} |")
    else:
        lines.append("| PENDING | PENDING |")
    lines.append("")

    metric_table(lines, "Table 3. In-domain main comparison", (
        ("ESIM", "esim"), ("Decomposable Attention", "dam"),
        ("BERT-NLI", "bert_nli"), ("TextCNN", "textcnn"),
        ("BiLSTM", "bilstm"), ("BERT-CLS", "bert_cls"),
        ("RoBERTa-CLS", "roberta_cls"),
        ("BGE frozen + LR", "BGEfz_LR_4tuple"),
        ("BGE frozen + SVM", "BGEfz_SVM_4tuple"),
        ("BGE frozen + MLP", "BGEfz_MLP_4tuple"),
        ("BGE frozen + kNN", "BGEfz_kNN_attr_k15"),
        ("Qwen-Flash zero-shot", "qwen_flash_zero"),
        ("Qwen-Flash five-shot", "qwen_flash_fs5"),
        ("GPT-5.4 zero-shot", "gpt54_zero"),
        ("GPT-5.4 five-shot", "gpt54_fs5"),
        ("Gemini-3.5-Flash zero-shot", "gemini35_zero"),
        ("Gemini-3.5-Flash five-shot", "gemini35_fs5"),
        ("Kimi-K2.6 zero-shot", "kimi_zero"),
        ("Kimi-K2.6 five-shot", "kimi_fs5"),
        ("Qwen2.5-7B QLoRA SFT", "qwen2p5_7b_qlora_sft"),
        ("CLAIMARC", "claimarc_canonical"),
    ), agg)

    category_all = result_root / "table4_xdom_category_all.json"
    rooms_all = result_root / "table4_xdom_rooms_all.json"
    xdom_table(lines, "Table 4a. Leave-one-category transfer",
               load_json(category_all if category_all.exists() else
                         result_root / "table4_xdom_category.json"))
    xdom_table(lines, "Table 4b. Leave-20-streamer transfer",
               load_json(rooms_all if rooms_all.exists() else
                         result_root / "table4_xdom_rooms.json"))

    lines.extend(["## Table 5. Gradient-free target-library injection", "",
                  "| Domain protocol | Condition | AP | AUC | F1 |", "|---|---|---:|---:|---:|"])
    mode = "rooms"
    blob = load_json(result_root / "table5_injection_rooms.json")
    if not blob:
        lines.append(f"| {mode} | PENDING | PENDING | PENDING | PENDING |")
    else:
        for condition in ("forward", "f0.0", "f0.2", "f0.4", "f0.6", "f0.8", "f1.0"):
            a = blob.get("agg", {}).get(f"{condition}_ap", [None, None])
            u = blob.get("agg", {}).get(f"{condition}_auc", [None, None])
            f = blob.get("agg", {}).get(f"{condition}_f1", [None, None])
            cell = lambda x: "PENDING" if x[0] is None else f"{x[0]:.1f}±{x[1]:.1f}"
            lines.append(f"| {mode} | {condition} | {cell(a)} | {cell(u)} | {cell(f)} |")
    lines.append("")

    geom = load_json(result_root / "table6_geometry.json")
    lines.extend(["## Table 6. Representation geometry", "",
                  "| Variant | Silhouette | Hard purity@10 | Alignment | Uniformity |",
                  "|---|---:|---:|---:|---:|"])
    for key, label in (("none", "w/o contrast"), ("supcon", "SupCon"), ("racl", "RACL")):
        row = (geom or {}).get(key, {})
        def gc(name):
            if name not in row:
                return "PENDING"
            std = row.get(name + "_std")
            return f"{row[name]:.3f}±{std:.3f}" if std is not None else f"{row[name]:.3f}"
        lines.append(f"| {label} | {gc('silhouette')} | {gc('hard_knn_purity@10')} | "
                     f"{gc('alignment_pos')} | {gc('uniformity')} |")
    lines.append("")

    metric_table(lines, "Table 7. Core ablations", (
        ("Canonical", "claimarc_canonical"), ("w/o RACL", "no_racl"),
        ("w/o reliability", "no_reliability"),
        ("w/o class balance", "no_class_balance"),
        ("w/o four-tuple", "no_four_tuple"), ("BERT backbone", "bert_backbone"),
    ), agg)
    fusion_ablation = (
        ("With validation-locked fusion", "with_fusion")
        if no_fusion_main else ("w/o fusion", "no_fusion")
    )
    metric_table(lines, "Table 8. Claim/argument interaction ablations", (
        ("Canonical", "claimarc_canonical"), fusion_ablation,
        ("Claim only", "claim_only"), ("Evidence only", "evidence_only"),
        ("Sources only", "input_sources_only"),
        ("Sources + arguments", "input_sources_plus_arguments"),
    ), agg)
    retrieval_ablation = (
        ("Global RACL retrieval", "global_racl_retrieval")
        if attribute_blocked_main
        else ("Same-attribute RACL retrieval", "same_attribute_negative")
    )
    metric_table(lines, "Table 9. RACL mining", (
        ("Canonical", "claimarc_canonical"), ("Hard positive", "hard_positive"),
        retrieval_ablation,
        ("Same-evidence-type negative", "same_evidence_type_negative"),
        ("Kp=1", "kp1"), ("Kp=5", "kp5"), ("Kn=1", "kn1"), ("Kn=10", "kn10"),
    ), agg)
    metric_table(lines, "Table 10. Reliability counterfactuals", (
        ("Canonical c", "claimarc_canonical"), ("Uniform", "no_reliability"),
        ("Inverse", "weight_inverse"), ("Permuted", "weight_permute"),
        ("Binary", "weight_binary"), ("Count only", "weight_count"),
        ("sqrt(c)", "weight_sqrt"),
    ), agg)
    lines.extend(["## Table 11", "", "The current manuscript has no Table 11 (numbering gap).", ""])

    canonical_hp_label = (
        "Canonical LoRA (no fusion, r16, lambda=.10, tau=.10, Kp3/Kn5, BCE)"
        if no_fusion_main
        else "Canonical LoRA (N2, h8, r16, lambda=.5, tau=.07, Kp3/Kn5, BCE)"
    )
    hp_entries = (
        (canonical_hp_label, "hp_lora_canonical"),
        ("Fusion blocks N=1", "hp_fusion1"), ("Fusion blocks N=3", "hp_fusion3"),
        ("Fusion blocks N=4", "hp_fusion4"), ("Attention heads=4", "hp_heads4"),
        ("Attention heads=16", "hp_heads16"), ("LoRA rank=8", "hp_rank8"),
        ("LoRA rank=32", "hp_rank32"), ("lambda_CL=0.05", "hp_lambda0p05"),
        ("lambda_CL=0.20", "hp_lambda0p2"), ("lambda_CL=0.50", "hp_lambda0p5"),
        ("tau=0.05", "hp_tau0p05"), ("tau=0.15", "hp_tau0p15"),
        ("tau=0.20", "hp_tau0p20"), ("Kp/Kn=(1,1)", "hp_k1_1"),
        ("Kp/Kn=(5,10)", "hp_k5_10"), ("ASL", "hp_loss_asl"),
        ("Focal loss", "hp_loss_focal"), ("FFN GeLU", "hp_ffn_gelu"),
        ("cross-attention claim to evidence", "hp_xattn_c2e"),
        ("cross-attention evidence to claim", "hp_xattn_e2c"),
        ("Independent projections", "hp_independent_projection"),
    )
    metric_table(lines, f"Table 12. LoRA-efficient hyperparameter sensitivity ({policy})",
                 hp_entries, agg)
    metric_table(lines, "Table 13. Reliability-formula sensitivity (matched seed 0)", (
        ("Canonical", "claimarc_canonical"), ("k=1.5", "c_k1p5"),
        ("k=6", "c_k6"), ("lambda=0.1", "c_lambda0p1"),
        ("lambda=0.6", "c_lambda0p6"), ("rho=0.2", "c_rho0p2"),
        ("rho=0.6", "c_rho0p6"), ("phi=1.0", "c_phi1p0"),
        ("phi=1.5", "c_phi1p5"),
    ), agg_seed0, expected_n=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_out = args.output.with_suffix(".json")
    json_out.write_text(json.dumps({"aggregates": agg, "fresh_rows": len(rows),
                                    "rejected_rows": rejected_rows},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[written] {args.output}")
    print(f"[written] {json_out}")


if __name__ == "__main__":
    main()
