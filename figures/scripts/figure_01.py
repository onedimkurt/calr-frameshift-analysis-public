#!/usr/bin/env python3
"""
figure_01.py — CALR Variant Architecture and Sequence Composition
=================================================================
Blood Cancer Journal — Article submission
CALR MPN Manuscript

Figure 1 (3 panels, cost = 1 budget slot):
  A: Generalized linear protein schematic (WT + 3 variant groups)
  B: Novel tail length across 3 groups
  C: WT C-domain fragment length across 3 groups

Data sources:
  - RECOMPUTED_FEATURES_76_VARIANTS.tsv
  - cluster_assignments.tsv
  - sequence_three_group_pairwise.tsv

Output:
  - FIGURES_FINAL/outputs/figure_01.pdf  (vector)
  - FIGURES_FINAL/outputs/figure_01.png  (300 dpi)

Journal specs: Arial 5–7 pt, RGB, colorblind-safe, 175 mm wide.
"""

import csv
import os
import sys
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# ── Paths ─────────────────────────────────────────────────────────
import os as _os
DATA_ROOT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data", "derived")
FEAT_FILE = os.path.join(DATA_ROOT, "RECOMPUTED_FEATURES_76_VARIANTS.tsv")
CLUST_FILE = os.path.join(
    DATA_ROOT, "unsupervised_clustering", "statistics", "cluster_assignments.tsv"
)
STATS_FILE = os.path.join(
    DATA_ROOT, "three_group_validation", "statistics",
    "sequence_three_group_pairwise.tsv"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

for label, path in [("Features", FEAT_FILE), ("Clusters", CLUST_FILE),
                     ("Stats", STATS_FILE)]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} file not found: {path}")
        sys.exit(1)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

COLOR_1A = "#0072B2"
COLOR_1B = "#E69F00"
COLOR_T2 = "#009E73"
COLORS = [COLOR_1A, COLOR_1B, COLOR_T2]
HATCHES = ["///", "\\\\\\\\", "xxx"]
GROUP_ORDER = ["Type 1-A", "Type 1-B", "Type 2"]
GROUP_LABELS = ["Type 1-A\n(n=25)", "Type 1-B\n(n=19)", "Type 2\n(n=32)"]

COL_SIGNAL = "#B0B0B0"
COL_N_DOMAIN = "#7BAFD4"
COL_P_DOMAIN = "#A8D5BA"
COL_C_DOMAIN = "#F4A582"
COL_APPROACH = "#E8D5B7"
COL_ANCHOR = "#C44E52"
COL_POST_ANCHOR = "#F0E0F0"
COL_KKRK = "#8856A7"
COL_ACIDIC = "#E7298A"

WT_LEN = 417
SIGNAL = (1, 17)
N_DOMAIN = (18, 197)
P_DOMAIN = (198, 308)
C_DOMAIN = (309, 417)
KDEL = (414, 417)
ANCHOR_LEN = 22

GROUPS_ARCH = {
    "Type 1-A": {
        "color": COLOR_1A, "fs_range": (363, 370), "fs_median": 366,
        "anchor_median": 391, "tail_len_mean": 67.0, "total_median": 433,
        "n": 25, "has_kkrk": False, "has_acidic_clusters": False,
        "label": "Type 1-A (n = 25)",
    },
    "Type 1-B": {
        "color": COLOR_1B, "fs_range": (361, 385), "fs_median": 365,
        "anchor_median": 383, "tail_len_mean": 60.8, "total_median": 427,
        "n": 19, "has_kkrk": False, "has_acidic_clusters": False,
        "label": "Type 1-B (n = 19)",
    },
    "Type 2": {
        "color": COLOR_T2, "fs_range": (381, 386), "fs_median": 384,
        "anchor_median": 417, "tail_len_mean": 78.2, "total_median": 462,
        "n": 32, "has_kkrk": True, "has_acidic_clusters": True,
        "label": "Type 2 (n = 32)",
    },
}

COMPARISON_MAP = {
    "Type2_vs_Type1A": ("Type 2", "Type 1-A"),
    "Type2_vs_Type1B": ("Type 2", "Type 1-B"),
    "Type1A_vs_Type1B": ("Type 1-A", "Type 1-B"),
}


def read_clusters(path):
    mapping = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            mapping[row["sequence_id"].strip()] = row["seq_subgroup"].strip()
    return mapping


def read_features(path, clusters, features):
    data = {feat: {g: [] for g in GROUP_ORDER} for feat in features}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["sequence_id"].strip()
            pc = row["primary_class"].strip()
            group = clusters.get(sid, "Type 2") if pc == "Type 1-like" else "Type 2"
            if group not in GROUP_ORDER:
                continue
            for feat in features:
                val = row.get(feat, "").strip()
                if val:
                    data[feat][group].append(float(val))
    return data


def read_stats(path, features):
    stats = {f: {} for f in features}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            feat = row["feature"].strip()
            if feat not in features:
                continue
            comp = row["comparison"].strip()
            if comp in COMPARISON_MAP:
                g1, g2 = COMPARISON_MAP[comp]
                key = f"{g1} vs {g2}"
                stats[feat][key] = {
                    "p_BH": float(row["p_BH"]),
                    "sig": row["sig"].strip(),
                }
    return stats


def sig_label(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"


def draw_bracket(ax, x1, x2, y, h, label, fontsize=5, lw=0.6):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=lw, c="0.2", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom",
            fontsize=fontsize, color="0.2")


def draw_simple_bar(ax, x_start, x_end, y, height, color,
                     edgecolor="0.3", lw=0.5, alpha=1.0):
    rect = plt.Rectangle(
        (x_start, y - height / 2), x_end - x_start, height,
        facecolor=color, edgecolor=edgecolor, linewidth=lw,
        alpha=alpha, zorder=2
    )
    ax.add_patch(rect)


def draw_fs_range(ax, x_lo, x_hi, y, height, color):
    rect = plt.Rectangle(
        (x_lo, y - height / 2), x_hi - x_lo, height,
        facecolor=color, edgecolor=color, linewidth=0.5,
        alpha=0.25, hatch="///", zorder=1
    )
    ax.add_patch(rect)


def draw_schematic(ax):
    bar_height = 3.0
    y_gap = 6.5
    x_max = 480
    y_wt = 24

    # WT CALR
    draw_simple_bar(ax, SIGNAL[0], SIGNAL[1], y_wt, bar_height, COL_SIGNAL, edgecolor="0.5")
    ax.text((SIGNAL[0] + SIGNAL[1]) / 2, y_wt, "SP", ha="center", va="center",
            fontsize=4, color="0.3", zorder=3)
    draw_simple_bar(ax, N_DOMAIN[0], N_DOMAIN[1], y_wt, bar_height, COL_N_DOMAIN)
    ax.text((N_DOMAIN[0] + N_DOMAIN[1]) / 2, y_wt, "N-domain", ha="center", va="center",
            fontsize=5, fontweight="bold", color="0.15", zorder=3)
    draw_simple_bar(ax, P_DOMAIN[0], P_DOMAIN[1], y_wt, bar_height, COL_P_DOMAIN)
    ax.text((P_DOMAIN[0] + P_DOMAIN[1]) / 2, y_wt, "P-domain", ha="center", va="center",
            fontsize=5, fontweight="bold", color="0.15", zorder=3)
    draw_simple_bar(ax, C_DOMAIN[0], C_DOMAIN[1], y_wt, bar_height, COL_C_DOMAIN)
    ax.text((C_DOMAIN[0] + C_DOMAIN[1]) / 2, y_wt, "C-domain", ha="center", va="center",
            fontsize=5, fontweight="bold", color="0.15", zorder=3)
    ax.annotate("KDEL", xy=(KDEL[1], y_wt + bar_height / 2),
                xytext=(KDEL[1] + 8, y_wt + 2.2),
                fontsize=4.5, fontweight="bold", color="#D62728",
                arrowprops=dict(arrowstyle="-|>", color="#D62728", lw=0.5),
                ha="left", va="center")
    ax.text(-10, y_wt, "WT CALR\n(417 aa)", ha="right", va="center",
            fontsize=5.5, fontweight="bold", color="0.2")
    for pos, label in [(1, "1"), (197, "197"), (308, "308"), (309, "309"), (417, "417")]:
        ax.text(pos, y_wt - bar_height / 2 - 0.6, str(label),
                ha="center", va="top", fontsize=3.5, color="0.5")

    y_sep = y_wt - 4.5
    ax.plot([0, x_max - 30], [y_sep, y_sep], lw=0.3, color="0.7", linestyle=":")

    y_current = y_sep - 3.0
    for group_name, gdata in GROUPS_ARCH.items():
        y = y_current
        color = gdata["color"]
        fs_lo, fs_hi = gdata["fs_range"]
        fs_med = gdata["fs_median"]
        anch_med = gdata["anchor_median"]
        total_med = gdata["total_median"]

        draw_simple_bar(ax, SIGNAL[0], SIGNAL[1], y, bar_height, COL_SIGNAL, edgecolor="0.5")
        draw_simple_bar(ax, N_DOMAIN[0], N_DOMAIN[1], y, bar_height, COL_N_DOMAIN)
        ax.text((N_DOMAIN[0] + N_DOMAIN[1]) / 2, y, "N", ha="center", va="center",
                fontsize=4.5, color="0.15", fontweight="bold", zorder=3)
        draw_simple_bar(ax, P_DOMAIN[0], P_DOMAIN[1], y, bar_height, COL_P_DOMAIN)
        ax.text((P_DOMAIN[0] + P_DOMAIN[1]) / 2, y, "P", ha="center", va="center",
                fontsize=4.5, color="0.15", fontweight="bold", zorder=3)
        draw_simple_bar(ax, C_DOMAIN[0], fs_med, y, bar_height, COL_C_DOMAIN, edgecolor="0.4")
        draw_fs_range(ax, fs_lo, fs_hi, y, bar_height + 0.4, color)
        ax.annotate("", xy=(fs_med, y + bar_height / 2 + 0.2),
                    xytext=(fs_med, y + bar_height / 2 + 1.8),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.7))
        ax.text(fs_med, y + bar_height / 2 + 2.0,
                f"fs {fs_lo}\u2013{fs_hi}",
                ha="center", va="bottom", fontsize=4, color=color, fontweight="bold")

        draw_simple_bar(ax, fs_med, anch_med, y, bar_height, COL_APPROACH, edgecolor="0.4")
        az_len = anch_med - fs_med
        if az_len > 12:
            ax.text((fs_med + anch_med) / 2, y, "Approach\nzone",
                    ha="center", va="center", fontsize=3.5, color="0.4", zorder=3)

        anch_end = anch_med + ANCHOR_LEN
        draw_simple_bar(ax, anch_med, anch_end, y, bar_height, COL_ANCHOR, edgecolor="#8B0000")
        ax.text((anch_med + anch_end) / 2, y, "Anchor",
                ha="center", va="center", fontsize=4, color="white",
                fontweight="bold", zorder=3)

        post_end = fs_med + int(gdata["tail_len_mean"])
        if post_end > anch_end:
            draw_simple_bar(ax, anch_end, post_end, y, bar_height,
                             COL_POST_ANCHOR, edgecolor="0.4")

        cys32_pos = anch_med + ANCHOR_LEN + 10
        cys36_pos = anch_med + ANCHOR_LEN + 14
        if cys32_pos < post_end and cys36_pos < post_end:
            for cpos, clabel in [(cys32_pos, "C+32"), (cys36_pos, "C+36")]:
                ax.plot(cpos, y - bar_height / 2 - 0.2, marker="v",
                        markersize=2.5, color="#DAA520", zorder=4)
                ax.text(cpos, y - bar_height / 2 - 1.0, clabel,
                        ha="center", va="top", fontsize=3, color="#DAA520",
                        fontweight="bold")

        if gdata["has_kkrk"]:
            kkrk_pos = anch_end + 2
            kkrk_end = kkrk_pos + 4
            draw_simple_bar(ax, kkrk_pos, kkrk_end, y, bar_height * 0.65,
                             COL_KKRK, edgecolor=COL_KKRK)
            ax.text((kkrk_pos + kkrk_end) / 2, y + bar_height / 2 + 0.8,
                    "KKRK", ha="center", va="bottom", fontsize=3.5,
                    color=COL_KKRK, fontweight="bold")

        if gdata["has_acidic_clusters"]:
            acid_start = anch_end + 8
            for k in range(3):
                a_s = acid_start + k * 5
                a_e = a_s + 3
                if a_e < post_end:
                    draw_simple_bar(ax, a_s, a_e, y, bar_height * 0.45,
                                     COL_ACIDIC, edgecolor=COL_ACIDIC, alpha=0.6)
            ax.text(acid_start + 7, y - bar_height / 2 - 1.0,
                    "DE clusters", ha="center", va="top", fontsize=3,
                    color=COL_ACIDIC, fontweight="bold")

        ax.text(-10, y, gdata["label"], ha="right", va="center",
                fontsize=5.5, fontweight="bold", color=color)
        ax.text(post_end + 3, y, f"~{total_med} aa",
                ha="left", va="center", fontsize=4, color="0.4")
        y_current -= y_gap

    # Legend
    leg_y = y_current - 1
    leg_items = [
        (COL_N_DOMAIN, "N-domain"), (COL_P_DOMAIN, "P-domain"),
        (COL_C_DOMAIN, "C-domain (WT)"), (COL_APPROACH, "Approach zone"),
        (COL_ANCHOR, "Anchor (22 aa)"), (COL_POST_ANCHOR, "Post-anchor"),
        (COL_KKRK, "KKRK motif"), (COL_ACIDIC, "Acidic clusters"),
    ]
    for i, (col, lab) in enumerate(leg_items):
        x = 30 + (i % 4) * 55
        ly = leg_y - (i // 4) * 2.2
        draw_simple_bar(ax, x, x + 7, ly, 1.5, col, edgecolor="0.4")
        ax.text(x + 9, ly, lab, ha="left", va="center", fontsize=4.5, color="0.3")

    ax.set_xlim(-75, x_max + 5)
    ax.set_ylim(leg_y - 4, y_wt + 5)
    ax.axis("off")


def draw_strip_box(ax, data, feat, ylabel, stats, panel_label):
    positions = [0, 1, 2]
    jitter_width = 0.15
    random.seed(42 + hash(feat) % 100)

    all_vals = [data[feat][g] for g in GROUP_ORDER]

    bp = ax.boxplot(
        all_vals, positions=positions, widths=0.5, patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=0.8),
        whiskerprops=dict(linewidth=0.6, color="0.3"),
        capprops=dict(linewidth=0.6, color="0.3"),
        boxprops=dict(linewidth=0.5),
    )
    for patch, color, hatch in zip(bp["boxes"], COLORS, HATCHES):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_hatch(hatch)
        patch.set_edgecolor(color)

    for gi, (g, color) in enumerate(zip(GROUP_ORDER, COLORS)):
        vals = data[feat][g]
        jitter = [positions[gi] + random.uniform(-jitter_width, jitter_width)
                  for _ in vals]
        ax.scatter(jitter, vals, s=8, color=color, alpha=0.7,
                   edgecolors="white", linewidths=0.3, zorder=3)

    ax.set_ylabel(ylabel, fontsize=7, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_LABELS, fontsize=5.5)

    ymax_data = max(max(v) for v in all_vals)
    ax.set_ylim(0, ymax_data * 1.40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(-0.18, 1.08, panel_label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="right")

    feat_stats = stats.get(feat, {})
    ylim_lo, ylim_hi = ax.get_ylim()
    y_range = ylim_hi - ylim_lo
    bracket_h = y_range * 0.025
    tier_base = ymax_data + y_range * 0.04
    tier_gap = y_range * 0.07

    bracket_defs = [
        ("Type 1-A vs Type 1-B", 0, 1, tier_base),
        ("Type 1-B vs Type 2", 1, 2, tier_base + tier_gap),
        ("Type 1-A vs Type 2", 0, 2, tier_base + tier_gap * 2),
    ]
    for comp_key, xi, xj, y_pos in bracket_defs:
        s = feat_stats.get(comp_key)
        if s is None:
            parts = comp_key.split(" vs ")
            s = feat_stats.get(f"{parts[1]} vs {parts[0]}")
        if s is not None:
            draw_bracket(ax, positions[xi], positions[xj],
                         y_pos, bracket_h, sig_label(s["p_BH"]))


def make_figure():
    features = ["novel_cterminus_residues", "wt_fragment_length"]
    clusters = read_clusters(CLUST_FILE)
    data = read_features(FEAT_FILE, clusters, features)
    stats = read_stats(STATS_FILE, features)

    for feat in features:
        for g in GROUP_ORDER:
            n = len(data[feat][g])
            expected = {"Type 1-A": 25, "Type 1-B": 19, "Type 2": 32}[g]
            if n != expected:
                print(f"WARNING: {feat} {g} has n={n}, expected {expected}")

    fig_w = 175 / 25.4
    fig_h = 145 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[1.2, 1],
                           hspace=0.30, wspace=0.35,
                           left=0.08, right=0.98,
                           bottom=0.06, top=0.97)

    ax_a = fig.add_subplot(gs[0, :])
    draw_schematic(ax_a)
    ax_a.text(0.0, 1.02, "A", transform=ax_a.transAxes,
              fontsize=9, fontweight="bold", va="bottom", ha="right")

    ax_b = fig.add_subplot(gs[1, 0])
    draw_strip_box(ax_b, data, "novel_cterminus_residues",
                   "Novel tail length\n(residues)", stats, "B")

    ax_c = fig.add_subplot(gs[1, 1])
    draw_strip_box(ax_c, data, "wt_fragment_length",
                   "WT C-domain fragment\nlength (residues)", stats, "C")

    pdf_path = os.path.join(OUT_DIR, "figure_01.pdf")
    png_path = os.path.join(OUT_DIR, "figure_01.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, format="png", bbox_inches="tight", pad_inches=0.03,
                dpi=300)
    plt.close(fig)

    print(f"✓ PDF saved: {pdf_path}")
    print(f"✓ PNG saved: {png_path}")
    print(f"  Width: 175 mm | Panels: 3 (A schematic + B,C strip+box)")
    print(f"  Budget cost: 1 (≤3 panels)")


if __name__ == "__main__":
    make_figure()
