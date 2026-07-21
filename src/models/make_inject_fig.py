"""Plot the gradient-free library-adaptation trajectory (fig_inject.pdf)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                 "font.size": 11, "axes.titlesize": 12, "figure.dpi": 150,
                 "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
                 "grid.linestyle": "--"})
C_POS, C_NEG, C_G = "#d1495b", "#2e86ab", "#3a7d44"
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(_ROOT, "data/final")
OUT = os.path.join(_ROOT, "docs/paper/figs"); os.makedirs(OUT, exist_ok=True)

s = json.load(open(os.path.join(D, "inject_rooms.json")))
agg = s["agg"]
fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
x = [100 * f for f in fracs]
ap = [agg[f"f{f:.1f}_ap"][0] for f in fracs]; ap_e = [agg[f"f{f:.1f}_ap"][1] for f in fracs]
auc = [agg[f"f{f:.1f}_auc"][0] for f in fracs]; auc_e = [agg[f"f{f:.1f}_auc"][1] for f in fracs]
f1 = [agg[f"f{f:.1f}_f1"][0] for f in fracs]; f1_e = [agg[f"f{f:.1f}_f1"][1] for f in fracs]
fwd_ap, fwd_auc, fwd_f1 = agg["forward_ap"][0], agg["forward_auc"][0], agg["forward_f1"][0]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
a1.errorbar(x, ap, yerr=ap_e, marker="o", color=C_NEG, linewidth=2.2, capsize=3,
            label="retrieval vote (AP)")
a1.errorbar(x, auc, yerr=auc_e, marker="s", color=C_G, linewidth=2.2, capsize=3,
            label="retrieval vote (AUC)")
a1.axhline(fwd_ap, color=C_NEG, linestyle=":", linewidth=1.6, label="frozen forward (AP)")
a1.axhline(fwd_auc, color=C_G, linestyle=":", linewidth=1.6, label="frozen forward (AUC)")
a1.set_xlabel("% of held-out-streamer pairs written to library")
a1.set_ylabel("score (%)"); a1.set_title("(a) Ranking metrics")
a1.legend(fontsize=8, framealpha=0.9, loc="center right"); a1.set_xticks(x)

a2.errorbar(x, f1, yerr=f1_e, marker="D", color=C_POS, linewidth=2.2, capsize=3,
            label="retrieval vote (F1)")
a2.axhline(fwd_f1, color="black", linestyle=":", linewidth=1.6,
           label="frozen forward (F1)")
a2.set_xlabel("% of held-out-streamer pairs written to library")
a2.set_ylabel("positive-class F1 (%)"); a2.set_title("(b) Operating-point F1")
a2.legend(fontsize=9, framealpha=0.9, loc="lower right"); a2.set_xticks(x)

fig.suptitle("Gradient-free library adaptation on unseen streamers", y=1.01)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_inject.{ext}"))
print("SAVED fig_inject", "AP", ap, "AUC", auc, "F1", f1)
