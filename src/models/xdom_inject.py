"""Frozen-encoder library-injection cross-domain study (RQ3 deployment claim).

For each held-out target domain (leave-one-category / leave-20-streamers), the
encoder is trained ONCE on the source domains and then FROZEN. We never take a
gradient step on the target. Instead we write a growing fraction of labelled
target pairs into the retrieval library and measure how target-domain ranking
improves as the library is filled --- the gradient-free, library-write adaptation
that the framework is designed for.

Conditions, all evaluated on a FIXED held-out target eval set (the half of the
target never eligible for injection):
  forward-only   : parametric classifier p_fwd (no library; flat reference).
  f=0.0          : source-only library RKC (current cross-domain protocol).
  f=0.2..1.0     : source library + fraction f of the target injection pool.

Reads the bundles written by xdom_fold (--save_emb) and reuses CLAIMARC's exact
attribute-blocked RKC vote and val-selected alpha. Pure offline; no GPU.

Usage:
  python -m models.xdom_inject --bundle_dir DIR --mode category --out OUT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from models.train import rkc_attr_predict, best_threshold_macroF1

FRACS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _arr(d, k):
    v = d[k]
    return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _stratified_half(y, seed):
    """Split target indices into eval (fixed) and injection-pool halves, per class."""
    rng = np.random.RandomState(seed)
    evl, pool = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        evl.extend(idx[:h].tolist())
        pool.extend(idx[h:].tolist())
    return np.array(sorted(evl)), np.array(sorted(pool))


def analyze_fold(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    alpha = float(b.get("alpha_rkc", 0.5) or 0.5)
    tr, va, te = b["train"], b["val"], b["test"]
    g_src, y_src, c_src, a_src = (_arr(tr, "g"), _arr(tr, "y"), _arr(tr, "c"), _arr(tr, "attr"))
    g_val, y_val, a_val, p_val = (_arr(va, "g"), _arr(va, "y"), _arr(va, "attr"), _arr(va, "p"))
    g_t, y_t, c_t, a_t, p_t = (_arr(te, "g"), _arr(te, "y"), _arr(te, "c"),
                               _arr(te, "attr"), _arr(te, "p"))
    if len(set(y_t.tolist())) < 2 or y_t.sum() < 3:
        return None

    g_src_t = torch.tensor(g_src, dtype=torch.float32)
    g_t_t = torch.tensor(g_t, dtype=torch.float32)
    evl, pool = _stratified_half(y_t, seed=0)

    # val combination -> alpha-blended threshold (source val), reused for all conditions
    prkc_val = rkc_attr_predict(g_src_t, y_src, c_src, a_src,
                                torch.tensor(g_val, dtype=torch.float32), a_val)
    thr = best_threshold_macroF1(y_val, alpha * p_val + (1 - alpha) * prkc_val) \
        if len(set(y_val.tolist())) > 1 else 0.5

    ye, pe = y_t[evl], p_t[evl]
    out = {"n_eval": int(len(evl)), "pos_eval": int(ye.sum()), "alpha": round(alpha, 2)}
    # forward classifier reference: the FROZEN parametric part, which cannot adapt
    # to the target without a gradient pass.
    out["forward"] = {
        "ap": 100 * average_precision_score(ye, pe),
        "auc": 100 * roc_auc_score(ye, pe),
        "f1": 100 * f1_score(ye, (pe >= thr).astype(int), zero_division=0)}

    g_eval_t = torch.tensor(g_t[evl], dtype=torch.float32)
    rng = np.random.RandomState(0)
    pool_shuf = pool.copy(); rng.shuffle(pool_shuf)
    for f in FRACS:
        k = int(round(f * len(pool_shuf)))
        inj = pool_shuf[:k]
        if len(inj):
            lib_g = torch.cat([g_src_t, g_t_t[inj]], 0)
            lib_y = np.concatenate([y_src, y_t[inj]])
            lib_c = np.concatenate([c_src, c_t[inj]])
            lib_a = np.concatenate([a_src, a_t[inj]])
        else:
            lib_g, lib_y, lib_c, lib_a = g_src_t, y_src, c_src, a_src
        prkc = rkc_attr_predict(lib_g, lib_y, lib_c, lib_a, g_eval_t, a_t[evl])
        thr_r = best_threshold_macroF1(y_val, prkc_val) if len(set(y_val.tolist())) > 1 else 0.5
        out[f"f{f:.1f}"] = {   # pure RKC vote: the gradient-free, library-driven part
            "ap": 100 * average_precision_score(ye, prkc),
            "auc": 100 * roc_auc_score(ye, prkc),
            "f1": 100 * f1_score(ye, (prkc >= thr_r).astype(int), zero_division=0),
            "n_lib": int(len(lib_y))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument("--mode", default="category", choices=["category", "rooms"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.bundle_dir, f"clarc_{args.mode}_*.pt")))
    folds = []
    for p in paths:
        r = analyze_fold(p)
        if r:
            r["fold"] = os.path.basename(p)
            folds.append(r)
            print(f"[{r['fold']}] fwd AP {r['forward']['ap']:.1f} | "
                  f"f0 AP {r['f0.0']['ap']:.1f} -> f1.0 AP {r['f1.0']['ap']:.1f} "
                  f"(+{r['f1.0']['ap']-r['f0.0']['ap']:.1f})", flush=True)

    # aggregate (mean +- s.d. across folds)
    keys = ["forward"] + [f"f{f:.1f}" for f in FRACS]
    agg = {}
    for key in keys:
        for met in ("ap", "auc", "f1"):
            vals = np.array([fd[key][met] for fd in folds])
            agg[f"{key}_{met}"] = [round(float(vals.mean()), 1), round(float(vals.std()), 1)]
    summary = {"mode": args.mode, "n_folds": len(folds), "agg": agg, "folds": folds}
    print("\n=== INJECTION SUMMARY (mean +- s.d. over folds) ===")
    print(f"{'condition':<12} {'AP':>12} {'AUC':>12} {'F1':>12}")
    for key in keys:
        a = agg[f"{key}_ap"]; u = agg[f"{key}_auc"]; ff = agg[f"{key}_f1"]
        print(f"{key:<12} {a[0]:>6.1f}±{a[1]:<4.1f} {u[0]:>6.1f}±{u[1]:<4.1f} {ff[0]:>6.1f}±{ff[1]:<4.1f}")
    if args.out:
        json.dump(summary, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
