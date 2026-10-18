"""RELIC-IDS figures. Reads ../results/relic.json, writes ../figures/fig_zero_day.png and fig_adversarial.png."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "results", "relic.json")))
OUT = os.path.join(HERE, "..", "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "MLP (DNN)"]
SHORT = ["LR", "DT", "RF", "XGB", "MLP"]

# ---- zero-day: detection of known vs unseen attack types
z = pd.DataFrame(R["zero_day"]["models"]).groupby("model")[["known_dr", "novel_dr"]].agg(["mean", "std"]).loc[MODELS]
fig, ax = plt.subplots(figsize=(7.2, 2.8)); x = np.arange(len(MODELS)); w = 0.36
ax.bar(x - w/2 - 0.01, z[("known_dr", "mean")], w, yerr=z[("known_dr", "std")], color="#2a78d6",
       label=f"Attack types seen in training (n = {R['zero_day']['n_known']:,})", error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
ax.bar(x + w/2 + 0.01, z[("novel_dr", "mean")], w, yerr=z[("novel_dr", "std")], color="#eb6834",
       label=f"Unseen (zero-day) attack types (n = {R['zero_day']['n_novel']:,})", error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
for i in range(len(MODELS)):
    ax.text(i - w/2, z[("known_dr", "mean")].iloc[i] + z[("known_dr", "std")].iloc[i] + 0.025, f"{z[('known_dr','mean')].iloc[i]:.2f}", ha="center", fontsize=7)
    ax.text(i + w/2, z[("novel_dr", "mean")].iloc[i] + z[("novel_dr", "std")].iloc[i] + 0.025, f"{z[('novel_dr','mean')].iloc[i]:.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(SHORT); ax.set_ylim(0, 1); ax.set_ylabel("Detection rate (recall)")
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.savefig(os.path.join(OUT, "fig_zero_day.png"), dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- adversarial mimicry: detection rate vs alpha for k = 1, 2, 3 perturbed features
a = pd.DataFrame(R["adversarial"])
shades = {1: 0.45, 2: 0.7, 3: 1.0}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, ds in zip(axes, ["NSL-KDD", "UNSW-NB15"]):
    for k in (1, 2, 3):
        d = a[(a.dataset == ds) & (a.k == k)].sort_values("alpha")
        feats = d.features.iloc[0]
        ax.plot(d.alpha, d.detection_rate, marker="o", ms=4, lw=2, color="#2a78d6", alpha=shades[k],
                label=f"k = {k}: " + ", ".join(feats))
    ax.set_title(ds, fontsize=9)
    ax.set_ylim(0, 1.05); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=6.3, frameon=False, loc="lower left")
axes[0].set_ylabel("Attacks still detected")
fig.supxlabel("Perturbation strength \u03b1 (0 = original flow, 1 = feature set to benign median)", fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_adversarial.png"), dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("RELIC figures written")
