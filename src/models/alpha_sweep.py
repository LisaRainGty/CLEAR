"""Offline two-path fusion-ratio sweep on the new canonical (full-FT + B) bundles.

comb(alpha) = alpha * p_CLS + (1 - alpha) * p_RKC
  alpha = 1.0  -> pure forward (CLS) head
  alpha = 0.0  -> pure attribute-blocked retrieval (RKC) vote

Standard protocol: the operating threshold for Acc/F1 is fixed on the validation
split at each alpha (no test leakage); AP/AUC are threshold-free. We report the
full test trajectory over a fine alpha grid (mean +/- s.d. over the three seeds)
for transparency, and separately the alpha that a validation-AP selection rule
picks per seed together with its held-out test performance -- the honest operating
choice.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.train import rkc_attr_predict, best_threshold_macroF1  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score  # noqa: E402

EMB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "data/final/emb_c6")
GRID = np.round(np.linspace(0.0, 1.0, 21), 2)


def one_seed(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    tr, va, te = d["train"], d["val"], d["test"]
    tens = lambda x: x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), dtype=torch.float32)
    gtr = tens(tr["g"]); ytr = np.asarray(tr["y"]); ctr = np.asarray(tr["c"]); atr = list(tr["attr"])
    gva = tens(va["g"]); yva = np.asarray(va["y"]); ava = list(va["attr"]); pva = np.asarray(va["p"])
    gte = tens(te["g"]); yte = np.asarray(te["y"]); ate = list(te["attr"]); pte = np.asarray(te["p"])

    prkc_val = rkc_attr_predict(gtr, ytr, ctr, atr, gva, ava)
    prkc_te = rkc_attr_predict(gtr, ytr, ctr, atr, gte, ate)

    per_alpha = {}
    for a in GRID:
        cva = a * pva + (1 - a) * prkc_val
        cte = a * pte + (1 - a) * prkc_te
        thr = best_threshold_macroF1(yva, cva)
        pred = (cte >= thr).astype(int)
        per_alpha[float(a)] = dict(
            acc=float((pred == yte).mean()),
            f1=float(f1_score(yte, pred, zero_division=0)),
            ap=float(average_precision_score(yte, cte)),
            auc=float(roc_auc_score(yte, cte)),
            val_ap=float(average_precision_score(yva, cva)),
        )
    # validation-AP selection rule
    best_a = max(GRID, key=lambda a: per_alpha[float(a)]["val_ap"])
    return per_alpha, float(best_a)


def main():
    seeds = [os.path.join(EMB, f"c6_canon_s{s}.pt") for s in range(3)]
    seeds = [p for p in seeds if os.path.exists(p)]
    rows = [one_seed(p) for p in seeds]
    per = [r[0] for r in rows]; sel = [r[1] for r in rows]
    n = len(rows)

    def agg(a, key):
        v = np.array([per[i][float(a)][key] for i in range(n)])
        return v.mean(), v.std()

    print(f"# two-path fusion-ratio sweep, {n} seeds (alpha=1 -> CLS, alpha=0 -> RKC)")
    print(f"{'alpha':>6} | {'Acc':>13} {'F1':>13} {'AP':>13} {'AUC':>13}")
    out = {}
    for a in GRID:
        acc = agg(a, "acc"); f1 = agg(a, "f1"); ap = agg(a, "ap"); auc = agg(a, "auc")
        out[float(a)] = dict(acc=list(acc), f1=list(f1), ap=list(ap), auc=list(auc))
        print(f"{a:>6.2f} | {acc[0]*100:>5.1f}\u00b1{acc[1]*100:.1f}   {f1[0]*100:>5.1f}\u00b1{f1[1]*100:.1f}   "
              f"{ap[0]*100:>5.1f}\u00b1{ap[1]*100:.1f}   {auc[0]*100:>5.1f}\u00b1{auc[1]*100:.1f}")

    # val-AP selected alpha and its held-out test performance
    sa = np.array(sel)
    print(f"\n# validation-AP selected alpha per seed: {sel}  (mean {sa.mean():.2f} +/- {sa.std():.2f})")
    selperf = {k: np.array([per[i][sel[i]][k] for i in range(n)]) for k in ("acc", "f1", "ap", "auc")}
    print("# held-out test at val-selected alpha: " +
          "  ".join(f"{k.upper()} {selperf[k].mean()*100:.1f}\u00b1{selperf[k].std()*100:.1f}"
                    for k in ("acc", "f1", "ap", "auc")))

    # test-AP-optimal alpha (descriptive only, for the trajectory shape)
    best_test = max(GRID, key=lambda a: out[float(a)]["ap"][0])
    print(f"# (descriptive) test-AP-optimal alpha on the grid: {best_test:.2f} "
          f"-> AP {out[float(best_test)]['ap'][0]*100:.1f}")

    json.dump({"grid": out, "val_selected_alpha": sel}, open(os.path.join(EMB, "alpha_sweep.json"), "w"), indent=2)
    print("[saved] alpha_sweep.json")


if __name__ == "__main__":
    main()
