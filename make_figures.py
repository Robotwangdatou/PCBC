# -*- coding: utf-8 -*-
"""Generate Figure 1 (method schematic), Figure 2 (audit-cost frontier),
and Figure 3 (residual-decay) for paper_revised_MSH3.tex.

Data sources (all authoritative, in the repository root):
  - Figure 2: pcbc_cost_frontier_full9_summary.csv  (nine-dataset mean cost)
  - Figure 3: pcbc_residual_decay_verify_detail.csv (45 folds -> dataset mean -> 9-set mean)
Figure 1 is an illustrative schematic; the vertex residuals and barycentric
weights shown in panel (b) are taken from the real German audit case of
Table XII (sample #1, fold 0) and are labelled as illustrative.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import Delaunay

import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
})

# ---------------------------------------------------------------- Figure 1
rng = np.random.default_rng(42)
n_good, n_bad = 45, 30
good = np.vstack([
    rng.normal([0.30, 0.32], 0.14, (n_good, 2)),
])
good = np.vstack([good, rng.normal([0.15, 0.72], 0.10, (10, 2))])
bad = np.vstack([
    rng.normal([0.68, 0.70], 0.11, (n_bad, 2)),
    rng.normal([0.88, 0.30], 0.09, (10, 2)),
])
X = np.vstack([good, bad])
y = np.array([0] * len(good) + [1] * len(bad))
q = np.array([[0.585, 0.545]])          # query point

tri = Delaunay(X)
sid = tri.find_simplex(q)[0]
simp = tri.simplices[sid]
print("Figure 1: query falls in simplex", sid, "vertices", simp)

# vertex residuals / barycentric weights from the real German audit case
res_vals = [0.6053, -0.7132, -0.1539]
w_vals = [0.4467, 0.0074, 0.5459]
corr = sum(w * r for w, r in zip(w_vals, res_vals))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
ax = axes[0]
# Delaunay edges
seen = set()
for s in tri.simplices:
    for i in range(3):
        e = tuple(sorted((s[i], s[(i + 1) % 3])))
        if e not in seen:
            seen.add(e)
            p1, p2 = X[e[0]], X[e[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="0.75", lw=0.7, zorder=1)
# highlight enclosing simplex
poly = X[simp]
ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor="#1a7f37",
                        lw=2.2, zorder=3))
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=16, c="#2b6cb0", marker="o",
           edgecolor="white", lw=0.4, zorder=2, label="Good credit (train)")
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, c="#c0392b", marker="x",
           lw=1.2, zorder=2, label="Default (train)")
ax.scatter(*q.T, s=110, c="black", marker="*", zorder=5, label="Query $x_{new}$")
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("Feature $x_j$"); ax.set_ylabel("Feature $x_k$")
ax.set_title("(a) Delaunay triangulation of the 2-D subspace")
ax.legend(loc="lower right", framealpha=0.9, fontsize=7.5)

ax = axes[1]
# local view: enclosing simplex scaled around query
cx, cy = q[0]
verts = poly
ax.add_patch(MplPolygon(verts, closed=True, fill=True, facecolor="#eaf5ee",
                        edgecolor="#1a7f37", lw=2.0, zorder=1))
for i in range(3):
    v = verts[i]
    ax.scatter(*v, s=46, c="#1a7f37", marker="o", edgecolor="white", lw=0.6, zorder=3)
    ax.annotate(r"$v_{%d}$" % (i + 1), (v[0], v[1]),
                textcoords="offset points", xytext=(7, 6), fontsize=9)
ax.scatter(*q.T, s=130, c="black", marker="*", zorder=5)
ax.annotate(r"$x_{new}$", (q[0, 0] + 0.012, q[0, 1] + 0.012),
            textcoords="offset points", xytext=(6, 6), fontsize=9)
# vertex weights and residuals in a clean box at the lower-right blank area
info = ("$v_1$: $w=0.447$, $r=+0.605$\n"
        "$v_2$: $w=0.007$, $r=-0.713$\n"
        "$v_3$: $w=0.546$, $r=-0.154$")
ax.text(0.97, 0.06, info, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fbfbfb", ec="#aaaaaa", lw=0.7))
ax.set_xlim(q[0, 0] - 0.22, q[0, 0] + 0.22)
ax.set_ylim(q[0, 1] - 0.20, q[0, 1] + 0.24)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("Feature $x_j$ (zoom)"); ax.set_ylabel("Feature $x_k$ (zoom)")
ax.set_title("(b) Barycentric interpolation of the residual")
ax.text(0.5, -0.14,
        r"correction $= \sum_i w_i\, r_i = %.4f$" % corr +
        "  (weights and residuals from Table XII case; geometry illustrative)",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
        color="#333333")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_method.pdf"), bbox_inches="tight")
plt.close(fig)
print("Figure 1 written.")

# ---------------------------------------------------------------- Figure 2
# Nine-dataset means (per-dataset mean over the five folds, then averaged over
# the nine datasets) computed from pcbc_cost_frontier_full9_summary_allmodels.csv;
# values match Table 6 (tab:cost) of the paper (incl. the EBM lie-flat row).
ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
costs = {
    "LR":          [0.0896, 0.1425, 0.2482, 0.4597, 0.8826],
    "EBM":         [0.1690, 0.1734, 0.1823, 0.2000, 0.2354],
    "PCBC-1":      [0.0763, 0.1057, 0.1644, 0.2819, 0.5170],
    "PCBC":        [0.0767, 0.0987, 0.1426, 0.2304, 0.4061],
}
colors = {"LR": "#8c8c8c", "EBM": "#2f855a", "PCBC-1": "#2b6cb0",
          "PCBC": "#c0392b"}
markers = {"LR": "o", "EBM": "D", "PCBC-1": "s", "PCBC": "^"}
linestyles = {"LR": "-", "EBM": "--", "PCBC-1": "-", "PCBC": "-"}

fig, ax = plt.subplots(figsize=(4.6, 3.2))
for col in costs:
    ax.plot(ratios, costs[col], marker=markers[col], ms=5.5, lw=1.8,
            color=colors[col], ls=linestyles[col], label=col, zorder=3)
ax.set_xscale("log", base=2)
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax.set_xlabel("Cost ratio $r$ (false-positive vs false-negative cost)")
ax.set_ylabel("Mean normalized audit cost")
ax.set_title("Audit-cost frontier (nine-dataset mean)")
ax.grid(True, ls=":", alpha=0.5)
ax.legend(framealpha=0.9)
# annotate the reduction at r=4 (consistent with Table 6)
v_lr, v_ms = costs["LR"][-1], costs["PCBC"][-1]
red = (v_lr - v_ms) / v_lr * 100
ax.annotate("$-$%.1f\\%% vs LR" % red, xy=(4.0, v_ms),
            xytext=(1.6, v_ms + 0.14), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_cost_frontier.pdf"), bbox_inches="tight")
plt.close(fig)
print("Figure 2 written. reduction at r=4: %.1f%%" % red)

# ---------------------------------------------------------------- Figure 3
d = pd.read_csv(os.path.join(_REPO, "pcbc_residual_decay_verify_detail.csv"))
per = d.groupby("dataset")[[
    "serial_norm_train_0", "serial_norm_train_1", "serial_norm_train_2", "serial_norm_train_3",
    "serial_norm_test_0", "serial_norm_test_1", "serial_norm_test_2", "serial_norm_test_3"]].mean()
train = per[[c for c in per.columns if "train" in c]].mean().values
test = per[[c for c in per.columns if "test" in c]].mean().values
layers = [0, 1, 2, 3]

fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.plot(layers, train, marker="o", ms=5.5, lw=1.8, color="#8c8c8c", label="Train (serial)")
ax.plot(layers, test, marker="s", ms=5.5, lw=1.8, color="#2b6cb0", label="Test (serial)")
ax.set_xticks(layers)
ax.set_xticklabels(["Base", "Layer 1", "Layer 2", "Layer 3"])
ax.set_xlabel("Serial correction layer")
ax.set_ylabel(r"Mean $\ell_2$ residual norm")
ax.set_title("Residual decay along the serial chain (nine-dataset mean)")
ax.grid(True, ls=":", alpha=0.5)
ax.legend(framealpha=0.9)
ax.annotate("training plateau\n(%.4f $\\rightarrow$ %.4f)" % (train[0], train[-1]),
            xy=(3.0, train[-1]), xytext=(1.35, train[-1] - 3.2), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
ax.annotate("test monotone\n(%.4f $\\rightarrow$ %.4f)" % (test[0], test[-1]),
            xy=(3.0, test[-1]), xytext=(1.9, test[-1] + 2.2), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_residual_decay.pdf"), bbox_inches="tight")
plt.close(fig)
print("Figure 3 written. train %.2f->%.2f  test %.2f->%.2f" % (train[0], train[-1], test[0], test[-1]))
