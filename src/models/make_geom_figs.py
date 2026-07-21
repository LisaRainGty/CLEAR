"""Figures for the redesigned embedding-geometry analysis (§4.6, RQ2).

Reads the controlled 3-variant bundles (none / supcon / racl) and geom2.json
produced by geom_probe2, and writes:
  fig_geometry      grouped bars of label-conditional geometry (3 variants)
  fig_umap_label    3-panel UMAP of the test retrieval embedding, colored by
                    perceived-risk label (none | SupCon-2020 | RACL)
  fig_knn_purity    kNN label purity vs k + retrieval-margin distribution

Run on the GPU host (has umap-learn + matplotlib):
  python -m models.make_geom_figs --emb_dir data/final/emb_geom \
         --geom_json results_artifacts/geom2.json --outdir figs --seed 0
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "savefig.bbox": "tight", "savefig.dpi": 200,
                     "axes.titlesize": 11})

ORDER = ["none", "supcon", "racl"]
LABELS = {"none": "w/o contrast", "supcon": "SupCon (2020)", "racl": "RACL (ours)"}
COLORS = {"none": "#9aa0a6", "supcon": "#4c78a8", "racl": "#d1495b"}


def _g(bundle, split="test"):
    sub = bundle[split]
    g = sub["g"]
    g = g.float() if isinstance(g, torch.Tensor) else torch.tensor(np.asarray(g), dtype=torch.float32)
    return F.normalize(g, dim=-1).numpy(), np.asarray(sub["y"]).astype(int)


def load_bundles(emb_dir, seed, prefix="emb_geom_"):
    out = {}
    for v in ORDER:
        p = os.path.join(emb_dir, f"{prefix}{v}_s{seed}.pt")
        if os.path.exists(p):
            out[v] = torch.load(p, map_location="cpu", weights_only=False)
    return out


# ----------------------------------------------------------- fig_geometry
def fig_geometry(geom, outdir):
    present = [v for v in ORDER if v in geom]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.5))

    # (a) label separability (higher is better)
    sep = [("silhouette", "Silhouette $\\uparrow$"),
           ("hard_knn_purity@10", "Hard-region\npurity@10 $\\uparrow$")]
    x = np.arange(len(sep)); w = 0.26
    for i, v in enumerate(present):
        vals = [geom[v].get(m[0]) for m in sep]
        errs = [geom[v].get(m[0] + "_std", 0) or 0 for m in sep]
        a1.bar(x + (i - (len(present) - 1) / 2) * w, vals, w, yerr=errs, capsize=2,
               label=LABELS[v], color=COLORS[v])
    a1.set_xticks(x); a1.set_xticklabels([m[1] for m in sep])
    a1.set_ylabel("metric value"); a1.set_title("Label separability")
    a1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    # (b) alignment-uniformity plane (lower-left is the retrieval-friendly regime)
    for v in present:
        a2.errorbar(geom[v]["alignment_pos"], geom[v]["uniformity"],
                    xerr=geom[v].get("alignment_pos_std", 0),
                    yerr=geom[v].get("uniformity_std", 0),
                    fmt="o", ms=9, color=COLORS[v], capsize=3, label=LABELS[v])
        a2.annotate(LABELS[v], (geom[v]["alignment_pos"], geom[v]["uniformity"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    a2.set_xlabel("alignment $\\downarrow$ (same-label tightness)")
    a2.set_ylabel("uniformity (more negative = more uniform)")
    a2.set_title("Alignment\u2013uniformity: SupCon collapses, RACL balances")
    fig.savefig(os.path.join(outdir, "fig_geometry.pdf"))
    fig.savefig(os.path.join(outdir, "fig_geometry.png")); plt.close(fig)
    print("saved fig_geometry")


# ----------------------------------------------------------- fig_umap_label
def fig_umap_label(bundles, outdir, seed=0):
    import umap
    present = [v for v in ORDER if v in bundles]
    fig, axes = plt.subplots(1, len(present), figsize=(4.3 * len(present), 4.0))
    if len(present) == 1:
        axes = [axes]
    for ax, v in zip(axes, present):
        g, y = _g(bundles[v])
        emb = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="cosine",
                        random_state=seed).fit_transform(g)
        ax.scatter(emb[y == 0, 0], emb[y == 0, 1], s=7, c="#b0b7bd",
                   label="benign", alpha=0.6, linewidths=0)
        ax.scatter(emb[y == 1, 0], emb[y == 1, 1], s=8, c="#d1495b",
                   label="misleading", alpha=0.85, linewidths=0)
        ax.set_title(LABELS[v]); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(frameon=False, loc="upper left", markerscale=2)
    fig.suptitle("UMAP of the test retrieval embedding $g_{p,a}$ (colored by perceived-risk label)",
                 y=1.02)
    fig.savefig(os.path.join(outdir, "fig_umap_label.pdf"))
    fig.savefig(os.path.join(outdir, "fig_umap_label.png")); plt.close(fig)
    print("saved fig_umap_label")


# ----------------------------------------------------------- fig_knn_purity
def fig_knn_purity(bundles, geom, outdir):
    present = [v for v in ORDER if v in bundles]
    ks = [1, 5, 10, 20]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for v in present:
        ys = [geom[v].get(f"knn_purity@{k}") for k in ks]
        a1.plot(ks, ys, "-o", color=COLORS[v], label=LABELS[v])
    a1.set_xlabel("$k$ (nearest train neighbors)"); a1.set_ylabel("label purity@$k$")
    a1.set_title("Neighborhood label purity"); a1.legend(frameon=False)
    a1.set_xticks(ks)
    for v in present:
        g, y = _g(bundles[v])
        # per-query retrieval margin within test set (proxy for the train-index margin)
        S = g @ g.T; np.fill_diagonal(S, -2)
        same = (y[:, None] == y[None, :])
        m = np.where(same, S, -2).max(1) - np.where(~same, S, -2).max(1)
        a2.hist(m, bins=40, histtype="step", color=COLORS[v], label=LABELS[v], lw=1.6)
    a2.axvline(0, color="k", lw=0.6, ls="--")
    a2.set_xlabel("retrieval margin"); a2.set_ylabel("count")
    a2.set_title("Retrieval-margin distribution"); a2.legend(frameon=False)
    fig.savefig(os.path.join(outdir, "fig_knn_purity.pdf"))
    fig.savefig(os.path.join(outdir, "fig_knn_purity.png")); plt.close(fig)
    print("saved fig_knn_purity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/final/emb_geom")
    ap.add_argument("--geom_json", default="results_artifacts/geom2.json")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    bundles = load_bundles(args.emb_dir, args.seed)
    geom = json.load(open(args.geom_json)) if os.path.exists(args.geom_json) else {}
    if geom:
        fig_geometry(geom, args.outdir)
        if bundles:
            fig_knn_purity(bundles, geom, args.outdir)
    if bundles:
        fig_umap_label(bundles, args.outdir, seed=args.seed)


if __name__ == "__main__":
    main()
