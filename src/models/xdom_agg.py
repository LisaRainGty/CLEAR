"""§4.4.2 跨域聚合：读取 xdom_fold / xdom_llm 存下的各折各模型预测，
统一用 val(Macro-F1) 选阈值，计算 Acc/Precision/Recall/F1pos/Macro-F1/AUPRC/AUROC，
按折宏平均（mean±std），打印对比表并写 JSON。

用法：
  python -m models.xdom_agg --indir /tmp/xdom --mode category --out /tmp/xdom/agg_category.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

from models.train import rkc_attr_predict, best_threshold_macroF1
from models.xdom_common import full_metrics

METRIC_KEYS = ["acc", "prec", "rec", "f1pos", "macro_f1", "auprc", "auroc"]


def clarc_combined(bundle):
    """复现 train.evaluate 的属性分块 RKC：val 调 α，test 上算 forward+RKC 组合概率。"""
    tr, va, te = bundle["train"], bundle["val"], bundle["test"]
    gtr = tr["g"] if torch.is_tensor(tr["g"]) else torch.tensor(np.asarray(tr["g"]))
    gva = va["g"] if torch.is_tensor(va["g"]) else torch.tensor(np.asarray(va["g"]))
    gte = te["g"] if torch.is_tensor(te["g"]) else torch.tensor(np.asarray(te["g"]))
    ty = np.asarray(tr["y"]); tc = np.asarray(tr["c"]); atr = np.asarray(tr["attr"])
    pv = np.asarray(va["p"]); yv = np.asarray(va["y"]); ava = np.asarray(va["attr"])
    pt = np.asarray(te["p"]); yt = np.asarray(te["y"]); ate = np.asarray(te["attr"])
    rkv = rkc_attr_predict(gtr, ty, tc, atr, gva, ava)
    rkt = rkc_attr_predict(gtr, ty, tc, atr, gte, ate)
    best_a, best_s = 1.0, -1.0
    from sklearn.metrics import average_precision_score
    for a in np.linspace(0.0, 1.0, 11):
        if len(set(yv.tolist())) < 2:
            best_a = 0.8; break
        s = average_precision_score(yv, a * pv + (1 - a) * rkv)
        if s > best_s:
            best_s, best_a = s, a
    return (yv, best_a * pv + (1 - best_a) * rkv, yt, best_a * pt + (1 - best_a) * rkt), best_a


def collect(indir, mode):
    """返回 {model_name: {holdout: metrics_dict}}。"""
    out = defaultdict(dict)

    def _fold_key(fname, prefix):
        """category：按品类聚合（去掉 _sN）；rooms/time：每个种子各算一折以得到种子方差。"""
        stem = os.path.basename(fname)[len(prefix):].rsplit(".pt", 1)[0]
        return stem.rsplit("_s", 1)[0] if mode == "category" else stem

    # CLAIMARC（含 forward 与 forward+RKC）
    for f in sorted(glob.glob(os.path.join(indir, f"clarc_{mode}_*.pt"))):
        b = torch.load(f, map_location="cpu", weights_only=False)
        ho = _fold_key(f, f"clarc_{mode}_")
        yv, cv, yt, ct = clarc_combined(b)[0]
        out["CLAIMARC"][ho] = full_metrics(yv, cv, yt, ct)
        out["CLAIMARC (forward-only)"][ho] = full_metrics(
            b["val"]["y"], b["val"]["p"], b["test"]["y"], b["test"]["p"])

    # 微调基线
    name = {"bert_cls": "BERT-CLS", "roberta_cls": "RoBERTa-CLS", "esim": "ESIM"}
    for kind, disp in name.items():
        for f in sorted(glob.glob(os.path.join(indir, f"{kind}_{mode}_*.pt"))):
            b = torch.load(f, map_location="cpu", weights_only=False)
            ho = _fold_key(f, f"{kind}_{mode}_")
            out[disp][ho] = full_metrics(b["val"]["y"], b["val"]["p"],
                                         b["test"]["y"], b["test"]["p"])

    # LLM zero / few
    for f in sorted(glob.glob(os.path.join(indir, f"llm_*_{mode}_*.pt"))):
        b = torch.load(f, map_location="cpu", weights_only=False)
        model = b["model"]; ho = b["holdout"]
        for mk, lbl in (("zero", f"LLM zero-shot ({model})"),
                        ("fewshot", f"LLM few-shot ({model})")):
            if mk in b:
                d = b[mk]
                out[lbl][ho] = full_metrics(d["y_val"], d["p_val"], d["y"], d["p"])
    return out


def aggregate(per_model):
    rows = {}
    for model, byho in per_model.items():
        folds = list(byho.values())
        agg = {"n_folds": len(folds), "folds": sorted(byho.keys())}
        for k in METRIC_KEYS:
            vals = [f[k] for f in folds if f[k] == f[k]]  # drop nan
            agg[k] = {"mean": round(float(np.mean(vals)), 1) if vals else None,
                      "std": round(float(np.std(vals)), 1) if vals else None}
        # 微平均（pos/N 加权可选）：这里另报总样本量
        agg["total_test"] = int(sum(f["n"] for f in folds))
        agg["total_pos"] = int(sum(f["pos"] for f in folds))
        rows[model] = agg
    return rows


ORDER = ["ESIM", "BERT-CLS", "RoBERTa-CLS",
         "LLM zero-shot (Qwen-Flash)", "LLM few-shot (Qwen-Flash)",
         "CLAIMARC (forward-only)", "CLAIMARC"]


def print_table(rows, mode):
    print(f"\n=== Cross-domain ({mode}) — macro-avg over folds (%) ===")
    hdr = f"{'System':30s} " + " ".join(f"{k:>8s}" for k in
          ["Acc", "Prec", "Rec", "F1pos", "MacroF1", "AUPRC", "AUROC"]) + "  folds"
    print(hdr); print("-" * len(hdr))
    keys = list(rows.keys())
    keys.sort(key=lambda m: ORDER.index(m) if m in ORDER else 99)
    for m in keys:
        a = rows[m]
        def cell(k):
            v = a[k]["mean"]
            return f"{v:8.1f}" if v is not None else f"{'--':>8s}"
        line = (f"{m:30s} " + " ".join(cell(k) for k in METRIC_KEYS)
                + f"  {a['n_folds']}")
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--mode", default="category", choices=["category", "rooms", "time"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    per_model = collect(args.indir, args.mode)
    rows = aggregate(per_model)
    print_table(rows, args.mode)
    blob = {"mode": args.mode, "aggregate": rows,
            "per_fold": {m: {h: per_model[m][h] for h in per_model[m]} for m in per_model}}
    if args.out:
        json.dump(blob, open(args.out, "w"), ensure_ascii=False, indent=2)
        print("\n[written]", args.out, flush=True)


if __name__ == "__main__":
    main()
