"""汇总 all_results.jsonl 为可投稿的实验表（§4.4）。
产出：主对比（均值±std + 集成 + 配对显著性）、消融、边界、跨域、几何、检索质量。
用法：python -m models.compile_results > ../docs/experiment_results.md
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import sys as _sys
RES = _sys.argv[1] if len(_sys.argv) > 1 else "../data/final/all_results.jsonl"

NAME = {
    "claimarc": "CLAIMARC (ours)", "bert_cls": "BERT-base+CLS", "roberta_cls": "RoBERTa-wwm+CLS",
    "esim": "ESIM", "bert_nli": "BERT-NLI", "bert_cls_bce": "BERT-base+CLS (BCE)",
    "roberta_cls_bce": "RoBERTa-wwm+CLS (BCE)", "BGE_frozen_LR": "BGE-frozen+LR",
}
MAIN_ORDER = ["claimarc", "roberta_cls", "bert_cls", "bert_nli", "esim", "BGE_frozen_LR",
              "bert_cls_bce", "roberta_cls_bce"]


def load():
    rows = [json.loads(l) for l in open(RES)]
    by_base = defaultdict(list)
    analysis, xdom, ens = {}, [], []
    for r in rows:
        j = r.get("_job", "")
        if "boundary_auprc" in r or r.get("tag") in ("full", "no_cl", "global_neg"):
            if "boundary_auprc" in r:
                analysis[r.get("tag", j)] = r
        elif "holdout" in r:
            xdom.append(r)
        elif "method" in r and "AP_singleseed_mean" in r:
            ens.append(r)
        elif "macro_f1" in r and r.get("macro_f1") is not None:
            base = j.split("__s")[0]
            by_base[base].append(r)
    return by_base, analysis, xdom, ens


def msd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
    return m, sd


def main():
    by_base, analysis, xdom, ens = load()
    print("# CLAIMARC 实验结果（claim-bearing 主任务，正例 17.9%）\n")
    print("> 测试集 653 对（117 正例）；所有方法统一 ASL 损失、可靠性加权、最优单-val 检查点选择。\n")

    # ---- 表1：主对比（均值±std） ----
    print("## 表1 主对比（5 种子均值±std；主指标 Macro-F1 / AP）\n")
    print("| 方法 | Macro-F1 | AP (AUPRC) | AUROC | wF1 | n |")
    print("|---|---|---|---|---|---|")
    clarc = by_base.get("claimarc", [])
    cf = [r["macro_f1"] for r in clarc]
    ca = [r["auprc"] for r in clarc]
    for base in MAIN_ORDER:
        if base not in by_base:
            continue
        rs = by_base[base]
        mf, sf = msd([r.get("macro_f1") for r in rs])
        ma, sa = msd([r.get("auprc") for r in rs])
        mu, su = msd([r.get("auroc") for r in rs])
        mw, sw = msd([r.get("wF1") for r in rs])
        tag = "**" if base == "claimarc" else ""
        print(f"| {tag}{NAME.get(base, base)}{tag} | {mf:.3f}±{sf:.3f} | {ma:.3f}±{sa:.3f} | "
              f"{mu:.3f}±{su:.3f} | {mw:.3f}±{sw:.3f} | {len(rs)} |")

    # 配对显著性（CLAIMARC vs 各基线，按种子配对 t 检验近似）
    print("\n### 配对显著性（CLAIMARC vs 最强基线，per-seed）\n")
    for base in ("roberta_cls", "bert_cls", "bert_cls_bce"):
        if base not in by_base:
            continue
        bf = [r["macro_f1"] for r in by_base[base]]
        ba = [r["auprc"] for r in by_base[base]]
        n = min(len(cf), len(bf))
        if n >= 2:
            df = [cf[i] - bf[i] for i in range(n)]
            da = [ca[i] - ba[i] for i in range(n)]
            mdf, sdf = msd(df); mda, sda = msd(da)
            tf = mdf / (sdf / math.sqrt(n) + 1e-9)
            ta = mda / (sda / math.sqrt(n) + 1e-9)
            print(f"- vs {NAME.get(base, base)}: ΔMacro-F1={mdf:+.3f} (t≈{tf:.2f}), "
                  f"ΔAP={mda:+.3f} (t≈{ta:.2f})")

    # ---- 表2：种子集成 ----
    if ens:
        print("\n## 表2 5 种子集成（降方差后稳定指标）\n")
        print("| 方法 | AP | AUROC | Macro-F1 | wF1 | 单种子AP(±std) |")
        print("|---|---|---|---|---|---|")
        for r in sorted(ens, key=lambda x: -x["AP"]):
            print(f"| {r['method']} | {r['AP']:.3f} | {r['AUROC']:.3f} | {r['Macro_F1']:.3f} | "
                  f"{r['wF1']:.3f} | {r['AP_singleseed_mean']:.3f}±{r['AP_singleseed_std']:.3f} |")

    # ---- 表3：消融 ----
    print("\n## 表3 消融（seed0，相对 canonical）\n")
    print("| 配置 | Macro-F1 | AP | AUROC | wF1 |")
    print("|---|---|---|---|---|")
    base0 = next((r for r in clarc if r.get("seed") == 0), clarc[0] if clarc else None)
    if base0:
        print(f"| canonical | {base0['macro_f1']:.3f} | {base0['auprc']:.3f} | "
              f"{base0['auroc']:.3f} | {base0['wF1']:.3f} |")
    for base in sorted(b for b in by_base if b.startswith("abl_")):
        r = by_base[base][0]
        print(f"| {base[4:]} | {r['macro_f1']:.3f} | {r['auprc']:.3f} | "
              f"{r['auroc']:.3f} | {r['wF1']:.3f} |")

    # ---- 表4：边界 + 几何 ----
    if analysis:
        print("\n## 表4 边界样本（硬混淆）+ 表征几何\n")
        print("| 方法 | overall AP | boundary AP | boundary wF1 | 同属性同标签紧致 | 同属性反标签分离 |")
        print("|---|---|---|---|---|---|")
        for tag in ("full", "no_cl", "global_neg", "bert", "roberta", "esim"):
            r = analysis.get(tag)
            if not r:
                continue
            nm = "CLAIMARC" if tag == "full" else tag
            print(f"| {nm} | {r.get('overall_auprc', 0):.3f} | {r.get('boundary_auprc', 0):.3f} | "
                  f"{r.get('boundary_wF1', 0):.3f} | {r.get('geom_attr_same_label_tight', '-')} | "
                  f"{r.get('geom_attr_opp_label_sep', '-')} |")

    # ---- 表5：跨域 ----
    if xdom:
        print("\n## 表5 跨域留一品类（检索库注入 m 条 support 的少样本适应）\n")
        print("| 留出品类 | forward(无适应) | RKC m=0 | m=1 | m=3 | m=5 | m=10 |")
        print("|---|---|---|---|---|---|---|")
        for r in xdom:
            if r.get("skip"):
                continue
            rk = r.get("rkc", {})
            print(f"| {r['holdout']} | {r.get('forward_macro_f1', 0):.3f} | "
                  f"{rk.get('0', 0):.3f} | {rk.get('1', 0):.3f} | {rk.get('3', 0):.3f} | "
                  f"{rk.get('5', 0):.3f} | {rk.get('10', 0):.3f} |")

    # ---- 检索质量 ----
    if clarc:
        lm, _ = msd([r.get("label_match@10") for r in clarc])
        am, _ = msd([r.get("attr_mAP@10") for r in clarc])
        print(f"\n## 检索表征质量\n- Label-match@10 = {lm:.3f}（先验 0.179）\n- Attr-match mAP@10 = {am:.3f}")


if __name__ == "__main__":
    main()
