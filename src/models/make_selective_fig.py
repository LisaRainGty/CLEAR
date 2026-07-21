"""Regenerate fig_selective.pdf from the new-canonical risk--coverage recompute."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(_ROOT, "docs/paper/figs")

_cands = [os.path.join(_ROOT, "data/final/selective_canon.json"),
          os.path.join(_ROOT, "data/final/emb_c6/selective_canon.json")]
SEL = next(p for p in _cands if os.path.exists(p))
d = json.load(open(SEL))
keys = sorted(d.keys(), key=lambda k: float(k), reverse=True)
x = [float(k) * 100 for k in keys]
acc = [d[k]["acc"][0] for k in keys]
ap = [d[k]["ap"][0] for k in keys]
racc = [d[k]["racc"][0] for k in keys]
rap = [d[k]["rap"][0] for k in keys]

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].plot(x, acc, "o-", color="#1f77b4", label="disagreement gate")
ax[0].plot(x, racc, "s--", color="#999999", label="random abstention")
ax[0].set_xlabel("Coverage (%)"); ax[0].set_ylabel("Accuracy"); ax[0].set_title("Accuracy")
ax[0].invert_xaxis(); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
ax[1].plot(x, ap, "o-", color="#d62728", label="disagreement gate")
ax[1].plot(x, rap, "s--", color="#999999", label="random abstention")
ax[1].set_xlabel("Coverage (%)"); ax[1].set_ylabel("AP"); ax[1].set_title("Average Precision")
ax[1].invert_xaxis(); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(FIGS, f"fig_selective.{ext}")
    fig.savefig(out, bbox_inches="tight"); print(f"[saved] {out}")
