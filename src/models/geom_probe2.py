"""Attribute-free, label-centric geometry probe (RQ2) on saved embedding bundles.

Compares the retrieval embedding produced by three controlled variants that share
an identical backbone (BGE full fine-tuning) and classifier and differ ONLY in the
contrastive objective:
    none    -- BCE only (no contrast)
    supcon  -- faithful in-batch supervised contrastive (Khosla et al., 2020)
    racl    -- retrieval-augmented contrast (ours): full-pool pseudo-gold positives
               + hard negatives, reliability- and class-weighted

All metrics are label-conditional (NO attribute conditioning). The train split is
used as the retrieval index; the test split provides queries -- exactly the setup
that RKC voting and gradient-free library-write adaptation rely on.

Metrics
  alignment_pos / alignment_pos_min : E||z_i - z_j||^2 over same-label test pairs
      (all classes / minority "misleading" class); lower = tighter.
  uniformity                        : log E exp(-2||z_i - z_j||^2) over all test pairs.
  silhouette                        : cosine silhouette of the label partition (test).
  knn_purity@k                      : frac of k nearest TRAIN neighbors sharing the
                                      query label (k = 1,5,10,20).
  knn_acc@10 / knn_f1@10            : majority-vote kNN classifier (train index -> test).
  margin_mean / margin_pos_frac     : per-query (max cos to same-label train) minus
                                      (max cos to opposite-label train); separability.
  Hard-region (confusable) split    : margin and purity@10 restricted to the test
                                      points whose nearest opposite-label train
                                      similarity is in the top quartile.

Usage:
  python -m models.geom_probe2 --bundles none=emb_none_s0.pt supcon=emb_supcon_s0.pt racl=emb_racl_s0.pt \
         --out results_artifacts/geom2.json
  # or sweep seeds and aggregate mean/std:
  python -m models.geom_probe2 --emb_dir data/final/emb_geom --seeds 0 1 2 --out ...
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score, f1_score


def _arr(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def probe_bundle(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    gtr = F.normalize(_arr(b["train"]["g"]), dim=-1)
    gte = F.normalize(_arr(b["test"]["g"]), dim=-1)
    ytr = np.asarray(b["train"]["y"]).astype(int)
    yte = np.asarray(b["test"]["y"]).astype(int)
    return geometry_metrics(gtr.numpy(), ytr, gte.numpy(), yte)


def geometry_metrics(gtr, ytr, gte, yte):
    n = len(yte)
    # ---- test-test structure: alignment / uniformity / silhouette ----
    Stt = gte @ gte.T
    iu = np.triu_indices(n, 1)
    s = Stt[iu]
    same = (yte[:, None] == yte[None, :])[iu]
    d2 = 2.0 - 2.0 * s                              # squared euclidean on unit sphere
    align = float(d2[same].mean())
    both1 = ((yte[:, None] == 1) & (yte[None, :] == 1))[iu]
    align_min = float(d2[both1].mean()) if both1.any() else None
    unif = float(np.log(np.exp(-2.0 * d2).mean()))
    try:
        sil = float(silhouette_score(gte, yte, metric="cosine"))
    except Exception:
        sil = None

    # ---- test -> train retrieval (index = train) ----
    Sxt = gte @ gtr.T                               # [n_test, n_train]
    order = np.argsort(-Sxt, axis=1)
    purity = {}
    for k in (1, 5, 10, 20):
        kk = min(k, gtr.shape[0])
        nn = order[:, :kk]
        purity[k] = float(np.mean([(ytr[nn[i]] == yte[i]).mean() for i in range(n)]))
    nn10 = order[:, :min(10, gtr.shape[0])]
    vote = np.array([np.bincount(ytr[nn10[i]], minlength=2).argmax() for i in range(n)])
    knn_acc = float((vote == yte).mean())
    knn_f1 = float(f1_score(yte, vote, pos_label=1, zero_division=0))

    same_tr = (ytr[None, :] == yte[:, None])        # [n_test, n_train]
    sim_same = np.where(same_tr, Sxt, -2.0).max(axis=1)
    sim_opp = np.where(~same_tr, Sxt, -2.0).max(axis=1)
    margin = sim_same - sim_opp
    margin_mean = float(margin.mean())
    margin_pos = float((margin > 0).mean())

    # ---- hard / confusable region: top-quartile nearest opposite-label sim ----
    thr = np.quantile(sim_opp, 0.75)
    hard = sim_opp >= thr
    hard_margin = float(margin[hard].mean()) if hard.any() else None
    hard_purity10 = (float(np.mean([(ytr[nn10[i]] == yte[i]).mean()
                                     for i in np.nonzero(hard)[0]]))
                     if hard.any() else None)

    return {
        "alignment_pos": round(align, 4),
        "alignment_pos_min": round(align_min, 4) if align_min is not None else None,
        "uniformity": round(unif, 4),
        "silhouette": round(sil, 4) if sil is not None else None,
        "knn_purity@1": round(purity[1], 4),
        "knn_purity@5": round(purity[5], 4),
        "knn_purity@10": round(purity[10], 4),
        "knn_purity@20": round(purity[20], 4),
        "knn_acc@10": round(knn_acc, 4),
        "knn_f1@10": round(knn_f1, 4),
        "margin_mean": round(margin_mean, 4),
        "margin_pos_frac": round(margin_pos, 4),
        "hard_margin_mean": round(hard_margin, 4) if hard_margin is not None else None,
        "hard_knn_purity@10": round(hard_purity10, 4) if hard_purity10 is not None else None,
        "n_test": int(n),
    }


def _agg(rows):
    keys = [k for k in rows[0] if k != "n_test" and rows[0][k] is not None]
    out = {"n_test": rows[0]["n_test"], "seeds": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            continue
        out[k] = round(float(np.mean(vals)), 4)
        out[k + "_std"] = round(float(np.std(vals)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="*", default=[],
                    help="name=path entries for single-seed mode")
    ap.add_argument("--emb_dir", default="")
    ap.add_argument("--variants", nargs="*", default=["none", "supcon", "racl"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[])
    ap.add_argument("--prefix", default="emb_geom_")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    result = {}
    if args.seeds and args.emb_dir:
        for v in args.variants:
            rows = []
            for s in args.seeds:
                p = os.path.join(args.emb_dir, f"{args.prefix}{v}_s{s}.pt")
                if os.path.exists(p):
                    rows.append(probe_bundle(p))
                else:
                    print(f"  (missing {p})", flush=True)
            if rows:
                result[v] = _agg(rows) if len(rows) > 1 else rows[0]
    else:
        for ent in args.bundles:
            name, path = ent.split("=", 1)
            if os.path.exists(path):
                result[name] = probe_bundle(path)
            else:
                print(f"  (missing {path})", flush=True)

    cols = ["alignment_pos", "alignment_pos_min", "uniformity", "silhouette",
            "knn_purity@10", "knn_acc@10", "margin_mean", "margin_pos_frac",
            "hard_margin_mean", "hard_knn_purity@10"]
    print(f"{'variant':<10}" + "".join(f"{c:>16}" for c in cols))
    for name, r in result.items():
        print(f"{name:<10}" + "".join(f"{str(r.get(c, '-')):>16}" for c in cols))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[geom_probe2] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
