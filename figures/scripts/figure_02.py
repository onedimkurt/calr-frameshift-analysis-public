#!/usr/bin/env python3
"""
figure_02.py — Charge Architecture and IDP Regime Classification
================================================================
Blood Cancer Journal — Article submission
CALR MPN Manuscript

Figure 2 (6 panels, cost = 2 budget slots):
  A: NCPR across 3 groups
  B: FCR across 3 groups
  C: SCD across 3 groups
  D: kappa across 3 groups
  E: Mean Kyte-Doolittle hydropathy across 3 groups
  F: Das-Pappu diagram (|NCPR| vs FCR) with IDP regime boundaries

Data sources:
  - RECOMPUTED_FEATURES_76_VARIANTS.tsv
  - cluster_assignments.tsv
  - sequence_three_group_pairwise.tsv
"""

import csv, os, sys, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np

import os as _os
DATA_ROOT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data", "derived")
FEAT_FILE = os.path.join(DATA_ROOT, "RECOMPUTED_FEATURES_76_VARIANTS.tsv")
CLUST_FILE = os.path.join(DATA_ROOT, "unsupervised_clustering", "statistics", "cluster_assignments.tsv")
STATS_FILE = os.path.join(DATA_ROOT, "three_group_validation", "statistics", "sequence_three_group_pairwise.tsv")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

for label, path in [("Features", FEAT_FILE), ("Clusters", CLUST_FILE), ("Stats", STATS_FILE)]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} not found: {path}"); sys.exit(1)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 300, "savefig.dpi": 300,
})

COLOR_1A = "#0072B2"; COLOR_1B = "#E69F00"; COLOR_T2 = "#009E73"
COLORS = [COLOR_1A, COLOR_1B, COLOR_T2]
HATCHES = ["///", "\\\\\\\\", "xxx"]
GROUP_ORDER = ["Type 1-A", "Type 1-B", "Type 2"]
GROUP_LABELS = ["Type 1-A\n(n=25)", "Type 1-B\n(n=19)", "Type 2\n(n=32)"]
COMPARISON_MAP = {
    "Type2_vs_Type1A": ("Type 2", "Type 1-A"),
    "Type2_vs_Type1B": ("Type 2", "Type 1-B"),
    "Type1A_vs_Type1B": ("Type 1-A", "Type 1-B"),
}

def read_clusters(path):
    m = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m[row["sequence_id"].strip()] = row["seq_subgroup"].strip()
    return m

def read_features(path, clusters, features):
    data = {feat: {g: [] for g in GROUP_ORDER} for feat in features}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["sequence_id"].strip()
            pc = row["primary_class"].strip()
            group = clusters.get(sid, "Type 2") if pc == "Type 1-like" else "Type 2"
            if group not in GROUP_ORDER: continue
            for feat in features:
                val = row.get(feat, "").strip()
                if val: data[feat][group].append(float(val))
    return data

def read_variant_pairs(path, clusters):
    variants = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["sequence_id"].strip()
            pc = row["primary_class"].strip()
            group = clusters.get(sid, "Type 2") if pc == "Type 1-like" else "Type 2"
            if group not in GROUP_ORDER: continue
            ncpr = row.get("ncpr", "").strip()
            fcr = row.get("fcr", "").strip()
            if ncpr and fcr:
                variants.append({"group": group, "ncpr": float(ncpr), "fcr": float(fcr)})
    return variants

def read_stats(path, features):
    stats = {f: {} for f in features}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            feat = row["feature"].strip()
            if feat not in features: continue
            comp = row["comparison"].strip()
            if comp in COMPARISON_MAP:
                g1, g2 = COMPARISON_MAP[comp]
                stats[feat][f"{g1} vs {g2}"] = {"p_BH": float(row["p_BH"]), "sig": row["sig"].strip()}
    return stats

def sig_label(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "ns"

def draw_bracket(ax, x1, x2, y, h, label, fontsize=5, lw=0.6):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c="0.2", clip_on=False)
    ax.text((x1+x2)/2, y+h, label, ha="center", va="bottom", fontsize=fontsize, color="0.2")

def draw_strip_box(ax, data, feat, ylabel, stats, panel_label, force_zero=True):
    positions = [0, 1, 2]
    random.seed(42 + hash(feat) % 100)
    all_vals = [data[feat][g] for g in GROUP_ORDER]

    bp = ax.boxplot(all_vals, positions=positions, widths=0.5, patch_artist=True,
        showfliers=False, medianprops=dict(color="black", linewidth=0.8),
        whiskerprops=dict(linewidth=0.6, color="0.3"),
        capprops=dict(linewidth=0.6, color="0.3"), boxprops=dict(linewidth=0.5))
    for patch, color, hatch in zip(bp["boxes"], COLORS, HATCHES):
        patch.set_facecolor(color); patch.set_alpha(0.30)
        patch.set_hatch(hatch); patch.set_edgecolor(color)

    for gi, (g, color) in enumerate(zip(GROUP_ORDER, COLORS)):
        vals = data[feat][g]
        jitter = [positions[gi] + random.uniform(-0.15, 0.15) for _ in vals]
        ax.scatter(jitter, vals, s=8, color=color, alpha=0.7,
                   edgecolors="white", linewidths=0.3, zorder=3)

    ax.set_ylabel(ylabel, fontsize=7, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_LABELS, fontsize=5)

    ymin_data = min(min(v) for v in all_vals)
    ymax_data = max(max(v) for v in all_vals)
    if force_zero and ymin_data >= 0:
        ax.set_ylim(0, ymax_data * 1.45)
    else:
        margin = (ymax_data - ymin_data) * 0.45
        ax.set_ylim(ymin_data - margin * 0.15, ymax_data + margin)
        if ymin_data < 0 < ymax_data:
            ax.axhline(0, color="0.5", linewidth=0.4, linestyle="--", zorder=1)

    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.20, 1.10, panel_label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="right")

    feat_stats = stats.get(feat, {})
    ylim_lo, ylim_hi = ax.get_ylim()
    y_range = ylim_hi - ylim_lo
    bracket_h = y_range * 0.022
    tier_base = ymax_data + y_range * 0.04
    tier_gap = y_range * 0.07

    for comp_key, xi, xj, y_pos in [
        ("Type 1-A vs Type 1-B", 0, 1, tier_base),
        ("Type 1-B vs Type 2", 1, 2, tier_base + tier_gap),
        ("Type 1-A vs Type 2", 0, 2, tier_base + tier_gap * 2),
    ]:
        s = feat_stats.get(comp_key)
        if s is None:
            parts = comp_key.split(" vs ")
            s = feat_stats.get(f"{parts[1]} vs {parts[0]}")
        if s is not None:
            draw_bracket(ax, positions[xi], positions[xj], y_pos, bracket_h, sig_label(s["p_BH"]))

def draw_das_pappu(ax, variants, panel_label):
    for g, color in zip(GROUP_ORDER, COLORS):
        pts = [v for v in variants if v["group"] == g]
        ncpr_abs = [abs(v["ncpr"]) for v in pts]
        fcr_vals = [v["fcr"] for v in pts]
        ax.scatter(ncpr_abs, fcr_vals, s=14, color=color, alpha=0.8,
                   edgecolors="white", linewidths=0.3, zorder=3)

    # Regime boundary: FCR = |NCPR| diagonal
    x_diag = np.linspace(0, 0.6, 100)
    ax.plot(x_diag, x_diag, color="0.5", linewidth=0.6, linestyle="--", zorder=1)
    # Weak/strong threshold
    ax.axhline(0.35, color="0.7", linewidth=0.4, linestyle=":", zorder=1)

    # localCIDER phasePlotRegion: all three groups = region 3 (strong polyampholyte);
    # Type-1 lies at the high-|NCPR| edge approaching the cationic-polyelectrolyte boundary.
    ax.text(0.06, 0.53, "Strong\npolyampholyte (R3)", fontsize=5, color="0.45",
            fontstyle="italic", ha="center", va="center")
    ax.text(0.34, 0.45, "Cationic\npolyelectrolyte\nboundary", fontsize=4.5, color="0.55",
            fontstyle="italic", ha="center", va="center")
    ax.text(0.12, 0.18, "Weak polyampholyte\n& polyelectrolyte", fontsize=4.5,
            color="0.6", fontstyle="italic", ha="center", va="center")

    ax.set_xlabel("|NCPR|", fontsize=7, fontweight="bold")
    ax.set_ylabel("FCR", fontsize=7, fontweight="bold")
    ax.set_xlim(0, 0.45); ax.set_ylim(0, 0.65)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    leg = [Line2D([0],[0], marker="o", color="w", markerfacecolor=c,
            markersize=4.5, markeredgecolor="white", markeredgewidth=0.3,
            label=g) for g, c in zip(GROUP_ORDER, COLORS)]
    ax.legend(handles=leg, fontsize=5, loc="upper right", frameon=True,
              framealpha=0.9, edgecolor="0.8", handletextpad=0.3, borderpad=0.4)
    ax.text(-0.20, 1.10, panel_label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="right")

def make_figure():
    features = ["ncpr", "fcr", "SCD", "kappa", "kd_novel"]
    clusters = read_clusters(CLUST_FILE)
    data = read_features(FEAT_FILE, clusters, features)
    stats = read_stats(STATS_FILE, features)
    variants = read_variant_pairs(FEAT_FILE, clusters)

    for feat in features:
        for g in GROUP_ORDER:
            n = len(data[feat][g])
            expected = {"Type 1-A": 25, "Type 1-B": 19, "Type 2": 32}[g]
            if n != expected:
                print(f"WARNING: {feat} {g} n={n}, expected {expected}")

    fig_w = 175 / 25.4; fig_h = 160 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.38,
                           left=0.10, right=0.97, bottom=0.05, top=0.97)

    draw_strip_box(fig.add_subplot(gs[0,0]), data, "ncpr", "NCPR", stats, "A")
    draw_strip_box(fig.add_subplot(gs[0,1]), data, "fcr", "FCR", stats, "B")
    draw_strip_box(fig.add_subplot(gs[1,0]), data, "SCD", "SCD", stats, "C", force_zero=False)
    draw_strip_box(fig.add_subplot(gs[1,1]), data, "kappa",
                   r"$\kappa$ (charge segregation)", stats, "D")
    draw_strip_box(fig.add_subplot(gs[2,0]), data, "kd_novel",
                   "Mean hydropathy\n(Kyte-Doolittle)", stats, "E", force_zero=False)
    draw_das_pappu(fig.add_subplot(gs[2,1]), variants, "F")

    for ext, fmt, dpi in [("pdf", "pdf", None), ("png", "png", 300)]:
        path = os.path.join(OUT_DIR, f"figure_02.{ext}")
        kw = {"format": fmt, "bbox_inches": "tight", "pad_inches": 0.03}
        if dpi: kw["dpi"] = dpi
        fig.savefig(path, **kw)
        print(f"✓ Saved: {path}")
    plt.close(fig)
    print(f"  Width: 175 mm | Panels: 6 (A-F) | Budget cost: 2")

if __name__ == "__main__":
    make_figure()
