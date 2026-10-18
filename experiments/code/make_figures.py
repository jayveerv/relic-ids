"""Figures for the paper. Reads ../results/results.json, writes ../figures/*.png (300 dpi)."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, "..", "results", "results.json")))
OUT = os.path.join(HERE, "..", "figures"); os.makedirs(OUT, exist_ok=True)

MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "MLP (DNN)"]
COL = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

def grid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)

# ---------------- Fig: binary F1 / FPR per model, both datasets (mean of 3 seeds) ----------------
b = pd.DataFrame(RES["binary"])
agg = b.groupby(["dataset", "model"]).agg(f1=("f1", "mean"), f1s=("f1", "std"), fpr=("fpr", "mean"),
                                          acc=("accuracy", "mean")).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, ds in zip(axes, ["NSL-KDD", "UNSW-NB15"]):
    d = agg[agg.dataset == ds].set_index("model").loc[MODELS]
    x = np.arange(len(MODELS))
    bars = ax.bar(x, d.f1, color=[COL[m] for m in MODELS], width=0.7, edgecolor="white", lw=1.5,
                  yerr=d.f1s, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    for xi, v in zip(x, d.f1):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["LR", "DT", "RF", "XGB", "MLP"])
    ax.set_title(ds, fontsize=9, color=INK); ax.set_ylim(0.5, 1.0); grid(ax)
axes[0].set_ylabel("F1-score (official test set)")
save(fig, "fig_binary_f1.png")

# ---------------- Fig: random split vs official split ----------------
g = pd.DataFrame(RES["split_gap"])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, ds in zip(axes, ["NSL-KDD", "UNSW-NB15"]):
    d = g[g.dataset == ds]; x = np.arange(len(d)); w = 0.36
    ax.bar(x - w/2 - 0.01, d.random_split_acc, w, color="#2a78d6", label="Random 80/20 split of training file")
    ax.bar(x + w/2 + 0.01, d.official_acc, w, color="#eb6834", label="Official held-out test set")
    for xi, (r, o) in enumerate(zip(d.random_split_acc, d.official_acc)):
        ax.text(xi - w/2, r + 0.01, f"{r:.3f}", ha="center", fontsize=6.5, color=INK)
        ax.text(xi + w/2, o + 0.01, f"{o:.3f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["DT", "RF", "XGB"]); ax.set_title(ds, fontsize=9, color=INK)
    ax.set_ylim(0.5, 1.07); grid(ax)
axes[0].set_ylabel("Accuracy")
axes[0].legend(loc="lower left", fontsize=7, frameon=False, bbox_to_anchor=(0, -0.42), ncol=2)
save(fig, "fig_split_gap.png")

# ---------------- Fig: feature selection (F1 and training time vs k) ----------------
f = pd.DataFrame(RES["feature_selection"])
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2))
for j, ds in enumerate(["NSL-KDD", "UNSW-NB15"]):
    for m in ["Random Forest", "XGBoost"]:
        d = f[(f.dataset == ds) & (f.model == m)].sort_values("k")
        lab = [str(k) if k != d.k.max() else f"all ({k})" for k in d.k]
        axes[0, j].plot(lab, d.f1, marker="o", ms=4, lw=2, color=COL[m], label=m)
        axes[1, j].plot(lab, d.train_s, marker="o", ms=4, lw=2, color=COL[m], label=m)
    axes[0, j].set_title(ds, fontsize=9, color=INK); grid(axes[0, j]); grid(axes[1, j])
    axes[1, j].set_xlabel("Number of features kept (mutual information ranking)")
axes[0, 0].set_ylabel("F1-score"); axes[1, 0].set_ylabel("Training time (s)")
axes[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout(); save(fig, "fig_feature_selection.png")

# ---------------- Fig: SHAP top-10 (XGBoost) ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, ds in zip(axes, ["NSL-KDD", "UNSW-NB15"]):
    items = RES["shap_top10"][ds][::-1]
    ax.barh([n for n, _ in items], [v for _, v in items], color="#2a78d6", height=0.65)
    ax.set_title(ds, fontsize=9, color=INK); ax.set_xlabel("mean |SHAP value|")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", labelsize=7)
fig.tight_layout(); save(fig, "fig_shap.png")

# ---------------- Diagram helpers ----------------
def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#2a78d6", fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))

# ---------------- Fig: taxonomy ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.4)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
box(ax, 0.33, 0.84, 0.34, 0.11, "AI-based IDS for cloud environments", fc="#dbe7f8", bold=True, fs=8.5)
cats = [("Classical ML", "LR, DT, SVM,\nRF, XGBoost"),
        ("Deep learning", "DNN/MLP, CNN,\nLSTM/GRU,\nautoencoder,\ntransformer"),
        ("Hybrid &\noptimisation", "Metaheuristic\ntuning (ABC,\ngrowth optimiser),\nensembles"),
        ("Feature selection", "Filter (MI, chi2),\nwrapper,\nmetaheuristic"),
        ("Explainable &\ntrustworthy AI", "SHAP, LIME,\nadversarial\nrobustness,\nfederated learning")]
for i, (t, d) in enumerate(cats):
    x = 0.01 + i * 0.198
    box(ax, x, 0.55, 0.18, 0.14, t, fc="#fdf0ea", ec="#eb6834", bold=True)
    box(ax, x, 0.14, 0.18, 0.32, d, fc="white", ec=GRID, fs=7)
    arrow(ax, 0.5, 0.84, x + 0.09, 0.69); arrow(ax, x + 0.09, 0.55, x + 0.09, 0.46)
ax.text(0.5, 0.05, "Deployment layers: network (NIDS), host/VM (HIDS), hypervisor, and hybrid",
        ha="center", fontsize=7.5, color=INK2)
save(fig, "fig_taxonomy.png")

# ---------------- Fig: cloud AI-IDS architecture ----------------
fig, ax = plt.subplots(figsize=(7.2, 2.6)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
steps = ["Cloud traffic\n& logs (VPC\nflows, VM,\nhypervisor)", "Pre-\nprocessing\n(encoding,\nscaling)",
         "Feature\nselection\n(mutual\ninformation)", "AI detection\nengine (RF /\nXGBoost /\nDNN)",
         "Explanation\n(SHAP per\nalert)", "Response\n(SOC alert,\nblock or\nisolate VM)"]
for i, s in enumerate(steps):
    x = 0.005 + i * 0.167
    box(ax, x, 0.30, 0.15, 0.52, s, fc="#eef3fb" if i not in (3,) else "#fdf0ea",
        ec="#2a78d6" if i != 3 else "#eb6834", fs=6.8)
    if i: arrow(ax, x - 0.017, 0.56, x, 0.56)
ax.annotate("", xy=(0.585, 0.28), xytext=(0.92, 0.28),
            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1, connectionstyle="arc3,rad=-0.35"))
ax.text(0.75, 0.04, "feedback: analyst labels used to retrain (concept drift)", ha="center", fontsize=7, color=INK2)
save(fig, "fig_architecture.png")

# ---------------- Fig: literature selection flow ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
box(ax, 0.02, 0.70, 0.40, 0.2, "Seed set: citations from\npreliminary survey (n = 24)", bold=True)
box(ax, 0.56, 0.70, 0.42, 0.2, "Targeted open-access searches\n(IEEE Xplore, arXiv, MDPI, SpringerOpen,\nFrontiers, CEUR; 2009-2025)", bold=True)
box(ax, 0.02, 0.38, 0.40, 0.2, "Excluded (n = 18)\n10 bibliographic records not verifiable\n8 no free full text / low-credibility venue",
    fc="#fdecec", ec="#e34948")
box(ax, 0.56, 0.38, 0.42, 0.2, "Added after full-text check (n = 24)\nopen access, peer-reviewed or\nwidely cited preprint/dataset paper")
box(ax, 0.02, 0.06, 0.40, 0.2, "Retained from seed set (n = 6)")
arrow(ax, 0.22, 0.70, 0.22, 0.58); arrow(ax, 0.77, 0.70, 0.77, 0.58); arrow(ax, 0.22, 0.38, 0.22, 0.26)
box(ax, 0.56, 0.06, 0.42, 0.2, "Studies included in this review (n = 30)", fc="#e6f6ee", ec="#1baf7a", bold=True)
arrow(ax, 0.42, 0.16, 0.56, 0.16); arrow(ax, 0.77, 0.38, 0.77, 0.26)
save(fig, "fig_selection_flow.png")
print("figures written to", OUT)
