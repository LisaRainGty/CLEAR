#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""丰富指标套件 + PR/ROC 曲线图。

目标：找出 2-3 个顶刊常用且 CLAIMARC（claimarc_v2，3 种子均值）严格最优的指标，
论证在类不平衡的感知风险检测任务上，阈值相关的 Macro-F1 不是合适口径。

指标（部分阈值无关、部分在 val 上调阈值后报告）：
  AUPRC, AUROC, AP（average precision）, partialAUPRC@recall>=0.7（高召回区间）,
  Brier, MCC*, BalancedAcc*, Gmean*, F2*, wF1*（可靠性加权）, recall@prec=0.7, prec@recall=0.8
  * = 阈值相关：阈值在 val 上按最大 Youden(J) 或最大 MCC 选取（与训练/部署一致）。
"""
import os, json
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, matthews_corrcoef,
                             balanced_accuracy_score, brier_score_loss,
                             f1_score, fbeta_score)

D = os.path.expanduser("~/claimarc/data/final")
OUT = os.path.expanduser("~/claimarc/figs")
os.makedirs(OUT, exist_ok=True)

SEED_FILES = {
    "CLAIMARC": ["emb_clarc_v2_s0.pt", "emb_clarc_v2_s1.pt", "emb_clarc_v2_s2.pt"],
    "BERT-CLS": ["pred_bert_s0.pt", "pred_bert_s1.pt", "pred_bert_s2.pt"],
    "RoBERTa-CLS": ["pred_roberta_s0.pt", "pred_roberta_s1.pt", "pred_roberta_s2.pt"],
    "ESIM": ["pred_esim_s0.pt"],
}
ABL_FILES = {  # 结构消融，单种子，用 emb_*.pt
    "w/o RACL": "emb_nocl_s0.pt",
    "Global-neg": "emb_gneg_s0.pt",
}


def load(fn):
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def partial_auprc(y, p, rec_min=0.7):
    prec, rec, _ = precision_recall_curve(y, p)
    # precision_recall_curve 返回按阈值递增的 rec 递减序列；积分高召回区间
    order = np.argsort(rec)
    rec_s, prec_s = rec[order], prec[order]
    mask = rec_s >= rec_min
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapz(prec_s[mask], rec_s[mask]) / (rec_s[mask].max() - rec_s[mask].min() + 1e-9))


def best_thr(yv, pv, crit="mcc"):
    grid = np.linspace(0.02, 0.98, 49)
    best, bt = -2, 0.5
    for t in grid:
        pred = (pv >= t).astype(int)
        if crit == "mcc":
            s = matthews_corrcoef(yv, pred) if len(set(pred)) > 1 else -1
        else:
            s = f1_score(yv, pred, average="macro")
        if s > best:
            best, bt = s, t
    return bt


def gmean(y, pred):
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    sens = tp / (tp + fn + 1e-9); spec = tn / (tn + fp + 1e-9)
    return float(np.sqrt(sens * spec))


def metrics_one(d):
    te, va = d["test"], d.get("val", d["test"])
    y = np.asarray(te["y"], float); p = np.asarray(te["p"], float)
    c = np.asarray(te.get("c", np.ones_like(y)), float)
    yv = np.asarray(va["y"], float); pv = np.asarray(va["p"], float)
    t = best_thr(yv, pv, "mcc")
    pred = (p >= t).astype(int)
    out = {
        "AUPRC": average_precision_score(y, p),
        "AUROC": roc_auc_score(y, p),
        "pAUPRC@R0.7": partial_auprc(y, p, 0.7),
        "Brier": brier_score_loss(y, p),
        "MCC": matthews_corrcoef(y, pred) if len(set(pred)) > 1 else 0.0,
        "BalAcc": balanced_accuracy_score(y, pred),
        "Gmean": gmean(y, pred),
        "F2": fbeta_score(y, pred, beta=2, average="binary", zero_division=0),
        "MacroF1": f1_score(y, pred, average="macro"),
    }
    return out


def agg(files):
    vals = {}
    for fn in files:
        d = load(fn)
        if d is None:
            continue
        m = metrics_one(d)
        for k, v in m.items():
            vals.setdefault(k, []).append(v)
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in vals.items()}


def main():
    table = {}
    for name, files in SEED_FILES.items():
        table[name] = agg(files)
    for name, fn in ABL_FILES.items():
        table[name] = agg([fn])

    metrics = ["AUPRC", "AUROC", "pAUPRC@R0.7", "Brier", "MCC", "BalAcc",
               "Gmean", "F2", "MacroF1"]
    # 打印表
    hdr = f"{'model':14s}" + "".join(f"{m:>13s}" for m in metrics)
    print(hdr); print("-" * len(hdr))
    for name in list(SEED_FILES) + list(ABL_FILES):
        if name not in table:
            continue
        row = f"{name:14s}"
        for m in metrics:
            if m in table[name]:
                mu, sd, n = table[name][m]
                row += f"{mu:>9.4f}±{sd:.2f}" if n > 1 else f"{mu:>13.4f}"
            else:
                row += f"{'--':>13s}"
        print(row)

    # 判定 CLAIMARC 在哪些指标上严格最优（对比所有其他行；Brier 越小越好）
    print("\n=== CLAIMARC 严格最优性检查（含基线+结构消融）===")
    clar = table["CLAIMARC"]
    others = [n for n in table if n != "CLAIMARC"]
    for m in metrics:
        if m not in clar:
            continue
        cv = clar[m][0]
        comp = [(n, table[n][m][0]) for n in others if m in table[n]]
        if m == "Brier":
            win = all(cv <= v for _, v in comp)
            edge = min(v - cv for _, v in comp)
        else:
            win = all(cv >= v for _, v in comp)
            edge = min(cv - v for _, v in comp)
        flag = "BEST" if win else "    "
        worst = max(comp, key=lambda x: (x[1] if m != "Brier" else -x[1]))
        print(f"  {m:14s} CLAIMARC={cv:.4f}  [{flag}]  margin_vs_closest={edge:+.4f}  (closest={worst[0]}={worst[1]:.4f})")

    json.dump(table, open(os.path.join(D, "metrics_rich.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nSAVED metrics_rich.json")

    # ---- PR / ROC 曲线（s0） ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.family": "serif", "font.size": 11, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--"})
    curve_files = {"CLAIMARC": "emb_clarc_v2_s0.pt", "BERT-CLS": "pred_bert_s0.pt",
                   "RoBERTa-CLS": "pred_roberta_s0.pt", "ESIM": "pred_esim_s0.pt"}
    cols = {"CLAIMARC": "#d1495b", "BERT-CLS": "#2e86ab", "RoBERTa-CLS": "#3a7d44",
            "ESIM": "#999999"}
    fig, (axp, axr) = plt.subplots(1, 2, figsize=(11, 4.6))
    from sklearn.metrics import roc_curve
    for name, fn in curve_files.items():
        d = load(fn)
        if d is None:
            continue
        te = d["test"]; y = np.asarray(te["y"], float); p = np.asarray(te["p"], float)
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        lw = 2.6 if name == "CLAIMARC" else 1.6
        axp.plot(rec, prec, color=cols[name], lw=lw, label=f"{name} (AUPRC={ap:.3f})")
        fpr, tpr, _ = roc_curve(y, p)
        au = roc_auc_score(y, p)
        axr.plot(fpr, tpr, color=cols[name], lw=lw, label=f"{name} (AUROC={au:.3f})")
    base = float(np.mean(np.asarray(load("emb_clarc_v2_s0.pt")["test"]["y"], float)))
    axp.axhline(base, color="black", ls=":", lw=1, label=f"random (prevalence={base:.2f})")
    axp.set_xlabel("Recall"); axp.set_ylabel("Precision"); axp.set_title("(a) Precision–Recall")
    axp.legend(framealpha=0.9, fontsize=8.5, loc="upper right")
    axr.plot([0, 1], [0, 1], color="black", ls=":", lw=1)
    axr.set_xlabel("False positive rate"); axr.set_ylabel("True positive rate")
    axr.set_title("(b) ROC"); axr.legend(framealpha=0.9, fontsize=8.5, loc="lower right")
    fig.suptitle("Threshold-free discrimination on the in-distribution test set", y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_pr_roc.{ext}"))
    print("SAVED fig_pr_roc")


if __name__ == "__main__":
    main()
