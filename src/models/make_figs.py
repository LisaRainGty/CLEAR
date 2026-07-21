#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 CLAIMARC 论文中由本脚本负责的两张图：

  fig_calibration  可靠性图（reliability diagram）+ 温度缩放（§4.9，图 7）
  fig_hparam       超参敏感性（AUPRC；§4.8，图 5）

其余论文图由专门脚本产出：
  fig_pr_roc            -> metrics_rich.py
  fig_inject            -> make_inject_fig.py
  fig_selective         -> make_selective_fig.py
  fig_umap_label /
  fig_geometry /
  fig_knn_purity (RQ2)  -> make_geom_figs.py

在导出过嵌入 / 结果 JSON 的 GPU 主机上运行（读 ~/claimarc/data/final 下的 *.pt 与
paper_results.jsonl），输出 PDF+PNG 到 ~/claimarc/figs。
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

D = os.path.expanduser("~/claimarc/data/final")
OUT = os.path.expanduser("~/claimarc/figs")
os.makedirs(OUT, exist_ok=True)

C_POS = "#d1495b"   # y=1 误导
C_NEG = "#2e86ab"   # y=0 非误导


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("SAVED", name)


def load_pt(fn):
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def read_jsonl(fn):
    rows = []
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        p = os.path.join(os.path.expanduser("~/claimarc"), fn)
    if not os.path.exists(p):
        return rows
    for ln in open(p):
        ln = ln.strip()
        if ln:
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


# ---------------------------------------------------------------- calibration
def _reliability(y, p, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    xs, ys, ws = [], [], []
    for i in range(nb):
        hi = (p < bins[i + 1]) if i < nb - 1 else (p <= bins[i + 1])
        m = (p >= bins[i]) & hi
        if m.sum() == 0:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ws.append(m.mean())
    return np.array(xs), np.array(ys), np.array(ws)


def _fit_T(logit_v, yv):
    from scipy.optimize import minimize_scalar
    def nll(T):
        z = logit_v / T
        z = np.clip(z, -30, 30)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
    r = minimize_scalar(nll, bounds=(0.05, 10), method="bounded")
    return float(r.x)


def fig_calibration(bundle="emb_geom_racl_s0.pt"):
    """Reliability diagram from a canonical bundle (train.py --save_emb)."""
    d = load_pt(bundle)
    if d is None:
        print(f"SKIP calib: {bundle} not found"); return
    te, va = d["test"], d.get("val", d["test"])
    yt = np.asarray(te["y"], float); pt = np.asarray(te["p"], float)
    yv = np.asarray(va["y"], float); pv = np.asarray(va["p"], float)
    eps = 1e-6
    lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    T = _fit_T(lg(pv), yv)
    pt_cal = 1 / (1 + np.exp(-lg(pt) / T))

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle=":", label="perfect calibration")
    for p, lab, col in [(pt, "CLAIMARC (raw)", C_NEG),
                        (pt_cal, f"CLAIMARC (T-scaled, T={T:.2f})", C_POS)]:
        xs, ys, ws = _reliability(yt, p)
        ax.plot(xs, ys, marker="o", color=col, linewidth=2, label=lab)
    ax.set_xlabel("mean predicted P(misleading)")
    ax.set_ylabel("empirical positive rate")
    ax.set_title("Reliability diagram (test set)")
    ax.legend(framealpha=0.9, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig_calibration")


# ---------------------------------------------------------------- hyperparam
def fig_hparam():
    rows = read_jsonl("../paper_results.jsonl") or read_jsonl("paper_results.jsonl")
    by = {r["tag"]: r for r in rows if "tag" in r}

    def g(tag, key="auprc"):
        return by.get(tag, {}).get(key, np.nan)

    lam_x = [0.1, 0.3, 0.5, 1.0]
    lam_y = [g("abl_lambda_0.1"), g("claimarc"), g("abl_lambda_0.5"), g("abl_lambda_1.0")]
    tau_x = [0.05, 0.07, 0.10]
    tau_y = [g("abl_tau_0.05"), g("claimarc"), g("abl_tau_0.10")]
    n_x = [1, 2, 4]
    n_y = [g("abl_nfusion_1"), g("claimarc"), g("abl_nfusion_4")]
    lora_x = [8, 16, 32]
    lora_y = [g("abl_lora_8"), g("claimarc"), g("abl_lora_32")]
    K_x = [1, 3, 5]  # (1,1),(3,5)默认,(5,10)
    K_y = [g("abl_K_1_1"), g("claimarc"), g("abl_K_5_10")]

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    series = [
        (axes[0], lam_x, lam_y, "$\\lambda_{CL}$", "contrastive weight", 2),
        (axes[1], tau_x, tau_y, "$\\tau$", "temperature", 1),
        (axes[2], n_x, n_y, "$N$", "fusion layers", 1),
        (axes[3], lora_x, lora_y, "LoRA rank", "LoRA rank", 1),
        (axes[4], K_x, K_y, "$K_p$ ($(K_p,K_n)$)", "retrieved samples", 1),
    ]
    for ax, xs, ys, xl, ttl, ci in series:
        ax.plot(xs, ys, marker="o", color=C_NEG, linewidth=2)
        ax.scatter([xs[ci]], [ys[ci]], s=90, facecolors="none", edgecolors=C_POS,
                   linewidths=2, zorder=5)
        ax.set_xlabel(xl); ax.set_title(ttl)
        ax.set_xticks(xs)
    axes[0].set_ylabel("AUPRC")
    fig.suptitle("Hyperparameter sensitivity (AUPRC; circle = canonical setting)", y=1.05)
    save(fig, "fig_hparam")


if __name__ == "__main__":
    for f in (fig_calibration, fig_hparam):
        try:
            f()
        except Exception as e:
            import traceback
            print("ERR", f.__name__, e)
            traceback.print_exc()
    print("ALLDONE", sorted(os.listdir(OUT)))
