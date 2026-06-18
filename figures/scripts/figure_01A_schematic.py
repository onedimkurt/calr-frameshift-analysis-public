#!/usr/bin/env python3
"""
figure_01A_schematic.py — CALR Protein Architecture Schematic (Proof of Concept)
=================================================================================
Blood Cancer Journal — Article submission
CALR MPN Manuscript

Figure 1, Panel A: Generalized linear protein diagram showing:
  - WT CALR domain structure (N-domain, P-domain, C-domain, KDEL)
  - Three variant group architectures (Type 1-A, Type 1-B, Type 2)
  - Frameshift position ranges per group
  - Conserved WT content upstream of frameshift
  - Novel C-terminal tail with: approach zone, anchor sequence, post-anchor
  - Cys+32 and Cys+36 positions
  - Type 2-specific motifs: KKRK, acidic clusters

Data-driven: all positions derived from RECOMPUTED_FEATURES_76_VARIANTS.tsv

Output:
  - FIGURES_FINAL/outputs/figure_01A_schematic.pdf
  - FIGURES_FINAL/outputs/figure_01A_schematic.png (300 dpi)

Journal specs: Arial 5-7 pt, RGB, colorblind-safe, 175 mm wide.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ── Colors ────────────────────────────────────────────────────────
COLOR_1A = "#0072B2"
COLOR_1B = "#E69F00"
COLOR_T2 = "#009E73"

# Domain colors (muted, professional)
COL_SIGNAL = "#B0B0B0"
COL_N_DOMAIN = "#7BAFD4"
COL_P_DOMAIN = "#A8D5BA"
COL_C_DOMAIN = "#F4A582"
COL_NOVEL = "#D4D4D4"       # Base novel tail
COL_APPROACH = "#E8D5B7"
COL_ANCHOR = "#C44E52"
COL_POST_ANCHOR = "#F0E0F0"
COL_KKRK = "#8856A7"
COL_ACIDIC = "#E7298A"

# ── WT CALR domain boundaries (UniProt P27797, 417 aa) ───────────
WT_LEN = 417
SIGNAL = (1, 17)
N_DOMAIN = (18, 197)
P_DOMAIN = (198, 308)
C_DOMAIN = (309, 417)
KDEL = (414, 417)

# ── Variant group data (from audit) ──────────────────────────────
# Frameshift ranges, anchor ranges, tail lengths, total protein lengths
GROUPS = {
    "Type 1-A": {
        "color": COLOR_1A,
        "fs_range": (363, 370),
        "fs_median": 366,
        "anchor_range": (386, 398),
        "anchor_median": 391,
        "tail_len_mean": 67.0,
        "tail_len_range": (65, 79),
        "total_len_range": (429, 441),
        "total_median": 433,
        "wt_frag_mean": 58.2,
        "n": 25,
        "has_kkrk": False,
        "has_acidic_clusters": False,
        "label": "Type 1-A (n = 25)",
    },
    "Type 1-B": {
        "color": COLOR_1B,
        "fs_range": (361, 385),
        "fs_median": 365,
        "anchor_range": (377, 403),
        "anchor_median": 383,
        "tail_len_mean": 60.8,
        "tail_len_range": (58, 63),
        "total_len_range": (420, 446),
        "total_median": 427,
        "wt_frag_mean": 57.9,
        "n": 19,
        "has_kkrk": False,
        "has_acidic_clusters": False,
        "label": "Type 1-B (n = 19)",
    },
    "Type 2": {
        "color": COLOR_T2,
        "fs_range": (381, 386),
        "fs_median": 384,
        "anchor_range": (413, 421),
        "anchor_median": 417,
        "tail_len_mean": 78.2,
        "tail_len_range": (76, 80),
        "total_len_range": (456, 464),
        "total_median": 462,
        "wt_frag_mean": 74.8,
        "n": 32,
        "has_kkrk": True,
        "has_acidic_clusters": True,
        "label": "Type 2 (n = 32)",
    },
}

ANCHOR_LEN = 22  # RRMMRTKMRMRRMRRTRRKMRR


def draw_bar(ax, x_start, x_end, y, height, color, label=None,
             edgecolor="0.3", lw=0.5, alpha=1.0, hatch=None):
    """Draw a horizontal bar representing a protein region."""
    width = x_end - x_start
    rect = FancyBboxPatch(
        (x_start, y - height / 2), width, height,
        boxstyle="round,pad=0.3",
        facecolor=color, edgecolor=edgecolor, linewidth=lw,
        alpha=alpha, zorder=2
    )
    ax.add_patch(rect)
    if label and width > 15:
        ax.text(x_start + width / 2, y, label,
                ha="center", va="center", fontsize=5, color="0.15",
                fontweight="bold", zorder=3)


def draw_simple_bar(ax, x_start, x_end, y, height, color,
                     edgecolor="0.3", lw=0.5, alpha=1.0):
    """Draw a simple rectangular bar (no rounded corners)."""
    width = x_end - x_start
    rect = plt.Rectangle(
        (x_start, y - height / 2), width, height,
        facecolor=color, edgecolor=edgecolor, linewidth=lw,
        alpha=alpha, zorder=2
    )
    ax.add_patch(rect)


def draw_region_label(ax, x_start, x_end, y, y_offset, text,
                       fontsize=5, color="0.3"):
    """Draw a label above/below a region with thin lines."""
    xmid = (x_start + x_end) / 2
    ax.text(xmid, y + y_offset, text, ha="center", va="center",
            fontsize=fontsize, color=color, fontstyle="italic")


def draw_bracket_below(ax, x_start, x_end, y, depth, text,
                        fontsize=5, color="0.4"):
    """Draw a bracket below a bar with label."""
    xmid = (x_start + x_end) / 2
    ax.plot([x_start, x_start, x_end, x_end],
            [y, y - depth, y - depth, y],
            lw=0.5, color=color, clip_on=False)
    ax.text(xmid, y - depth - 0.8, text, ha="center", va="top",
            fontsize=fontsize, color=color)


def draw_fs_range(ax, x_lo, x_hi, y, height, color):
    """Draw a hatched region showing frameshift position range."""
    width = x_hi - x_lo
    rect = plt.Rectangle(
        (x_lo, y - height / 2), width, height,
        facecolor=color, edgecolor=color, linewidth=0.5,
        alpha=0.25, hatch="///", zorder=1
    )
    ax.add_patch(rect)


def make_schematic():
    # Scale: 1 residue = 1 unit on x-axis
    # Max protein length ~ 464 (Type 2), add margins
    x_max = 480
    bar_height = 3.0
    y_gap = 7.0  # vertical gap between protein bars

    fig_w = 175 / 25.4
    fig_h = 110 / 25.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)

    y_positions = {}
    y_current = 28  # Start from top

    # ══════════════════════════════════════════════════════════════
    # ROW 0: Wild-type CALR (417 aa)
    # ══════════════════════════════════════════════════════════════
    y_wt = y_current
    y_positions["WT"] = y_wt

    # Signal peptide
    draw_simple_bar(ax, SIGNAL[0], SIGNAL[1], y_wt, bar_height,
                     COL_SIGNAL, edgecolor="0.5")
    ax.text((SIGNAL[0] + SIGNAL[1]) / 2, y_wt, "SP",
            ha="center", va="center", fontsize=4.5, color="0.3", zorder=3)

    # N-domain
    draw_simple_bar(ax, N_DOMAIN[0], N_DOMAIN[1], y_wt, bar_height,
                     COL_N_DOMAIN)
    ax.text((N_DOMAIN[0] + N_DOMAIN[1]) / 2, y_wt, "N-domain",
            ha="center", va="center", fontsize=5.5, fontweight="bold",
            color="0.15", zorder=3)

    # P-domain
    draw_simple_bar(ax, P_DOMAIN[0], P_DOMAIN[1], y_wt, bar_height,
                     COL_P_DOMAIN)
    ax.text((P_DOMAIN[0] + P_DOMAIN[1]) / 2, y_wt, "P-domain",
            ha="center", va="center", fontsize=5.5, fontweight="bold",
            color="0.15", zorder=3)

    # C-domain
    draw_simple_bar(ax, C_DOMAIN[0], C_DOMAIN[1], y_wt, bar_height,
                     COL_C_DOMAIN)
    ax.text((C_DOMAIN[0] + C_DOMAIN[1]) / 2, y_wt, "C-domain",
            ha="center", va="center", fontsize=5.5, fontweight="bold",
            color="0.15", zorder=3)

    # KDEL marker
    ax.annotate("KDEL", xy=(KDEL[1], y_wt + bar_height / 2),
                xytext=(KDEL[1] + 8, y_wt + 2.5),
                fontsize=5, fontweight="bold", color="#D62728",
                arrowprops=dict(arrowstyle="-|>", color="#D62728", lw=0.6),
                ha="left", va="center")

    # WT label
    ax.text(-12, y_wt, "WT CALR\n(417 aa)", ha="right", va="center",
            fontsize=6, fontweight="bold", color="0.2")

    # Residue numbers below WT bar
    for pos, label in [(1, "1"), (17, "17"), (197, "197"),
                        (308, "308"), (309, "309"), (417, "417")]:
        ax.text(pos, y_wt - bar_height / 2 - 0.8, str(label),
                ha="center", va="top", fontsize=4, color="0.5")

    # Separator line
    y_current -= 5.5
    ax.plot([0, x_max - 20], [y_current, y_current],
            lw=0.3, color="0.7", linestyle=":")

    # ══════════════════════════════════════════════════════════════
    # ROWS 1–3: Variant groups
    # ══════════════════════════════════════════════════════════════
    for gi, (group_name, gdata) in enumerate(GROUPS.items()):
        y_current -= y_gap
        y = y_current
        y_positions[group_name] = y
        color = gdata["color"]

        fs_lo, fs_hi = gdata["fs_range"]
        fs_med = gdata["fs_median"]
        anch_lo, anch_hi = gdata["anchor_range"]
        anch_med = gdata["anchor_median"]
        total_med = gdata["total_median"]

        # ── Conserved WT regions (same as WT up to frameshift) ────
        # Signal peptide
        draw_simple_bar(ax, SIGNAL[0], SIGNAL[1], y, bar_height,
                         COL_SIGNAL, edgecolor="0.5")

        # N-domain
        draw_simple_bar(ax, N_DOMAIN[0], N_DOMAIN[1], y, bar_height,
                         COL_N_DOMAIN)
        ax.text((N_DOMAIN[0] + N_DOMAIN[1]) / 2, y, "N",
                ha="center", va="center", fontsize=5, color="0.15",
                fontweight="bold", zorder=3)

        # P-domain
        draw_simple_bar(ax, P_DOMAIN[0], P_DOMAIN[1], y, bar_height,
                         COL_P_DOMAIN)
        ax.text((P_DOMAIN[0] + P_DOMAIN[1]) / 2, y, "P",
                ha="center", va="center", fontsize=5, color="0.15",
                fontweight="bold", zorder=3)

        # Retained WT C-domain (309 to fs_median)
        draw_simple_bar(ax, C_DOMAIN[0], fs_med, y, bar_height,
                         COL_C_DOMAIN, edgecolor="0.4")

        # Frameshift range indicator (hatched zone)
        draw_fs_range(ax, fs_lo, fs_hi, y, bar_height + 0.6, color)

        # Frameshift position arrow
        ax.annotate("", xy=(fs_med, y + bar_height / 2 + 0.3),
                    xytext=(fs_med, y + bar_height / 2 + 2.2),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8))
        ax.text(fs_med, y + bar_height / 2 + 2.5,
                f"fs {fs_lo}–{fs_hi}",
                ha="center", va="bottom", fontsize=4.5, color=color,
                fontweight="bold")

        # ── Novel C-terminal tail ─────────────────────────────────
        # Approach zone (fs_med to anch_med)
        draw_simple_bar(ax, fs_med, anch_med, y, bar_height,
                         COL_APPROACH, edgecolor="0.4")
        az_len = anch_med - fs_med
        if az_len > 12:
            ax.text((fs_med + anch_med) / 2, y, "Approach\nzone",
                    ha="center", va="center", fontsize=4, color="0.35",
                    zorder=3)

        # Anchor sequence (22 aa)
        anch_end = anch_med + ANCHOR_LEN
        draw_simple_bar(ax, anch_med, anch_end, y, bar_height,
                         COL_ANCHOR, edgecolor="#8B0000")
        ax.text((anch_med + anch_end) / 2, y, "Anchor",
                ha="center", va="center", fontsize=4.5, color="white",
                fontweight="bold", zorder=3)

        # Post-anchor region
        post_anch_end = fs_med + int(gdata["tail_len_mean"])
        if post_anch_end > anch_end:
            draw_simple_bar(ax, anch_end, post_anch_end, y, bar_height,
                             COL_POST_ANCHOR, edgecolor="0.4")

        # Cys+32 and Cys+36 markers
        cys32_pos = anch_med + ANCHOR_LEN + 10  # approximate +32 from anchor start
        cys36_pos = anch_med + ANCHOR_LEN + 14
        if cys32_pos < post_anch_end and cys36_pos < post_anch_end:
            for cpos, clabel in [(cys32_pos, "C+32"), (cys36_pos, "C+36")]:
                ax.plot(cpos, y - bar_height / 2 - 0.3, marker="v",
                        markersize=3, color="#DAA520", zorder=4)
                ax.text(cpos, y - bar_height / 2 - 1.2, clabel,
                        ha="center", va="top", fontsize=3.5,
                        color="#DAA520", fontweight="bold")

        # Type 2-specific motifs
        if gdata["has_kkrk"]:
            kkrk_pos = anch_end + 2
            kkrk_end = kkrk_pos + 4
            draw_simple_bar(ax, kkrk_pos, kkrk_end, y, bar_height * 0.7,
                             COL_KKRK, edgecolor=COL_KKRK)
            ax.text((kkrk_pos + kkrk_end) / 2, y + bar_height / 2 + 1.0,
                    "KKRK", ha="center", va="bottom", fontsize=4,
                    color=COL_KKRK, fontweight="bold")

        if gdata["has_acidic_clusters"]:
            # Three acidic clusters downstream of KKRK
            acid_start = anch_end + 8
            for k in range(3):
                a_s = acid_start + k * 5
                a_e = a_s + 3
                if a_e < post_anch_end:
                    draw_simple_bar(ax, a_s, a_e, y, bar_height * 0.5,
                                     COL_ACIDIC, edgecolor=COL_ACIDIC,
                                     alpha=0.6)
            ax.text(acid_start + 7, y - bar_height / 2 - 1.2,
                    "DE clusters", ha="center", va="top", fontsize=3.5,
                    color=COL_ACIDIC, fontweight="bold")

        # ── Group label ───────────────────────────────────────────
        ax.text(-12, y, gdata["label"], ha="right", va="center",
                fontsize=6, fontweight="bold", color=color)

        # Total length annotation
        ax.text(post_anch_end + 3, y, f"~{total_med} aa",
                ha="left", va="center", fontsize=4.5, color="0.4")

    # ══════════════════════════════════════════════════════════════
    # Legend for regions
    # ══════════════════════════════════════════════════════════════
    leg_y = -1
    leg_items = [
        (COL_N_DOMAIN, "N-domain"),
        (COL_P_DOMAIN, "P-domain"),
        (COL_C_DOMAIN, "C-domain (WT)"),
        (COL_APPROACH, "Approach zone"),
        (COL_ANCHOR, "Anchor (22 aa)"),
        (COL_POST_ANCHOR, "Post-anchor"),
        (COL_KKRK, "KKRK motif"),
        (COL_ACIDIC, "Acidic clusters"),
    ]
    leg_x_start = 30
    leg_spacing = 55

    for i, (col, lab) in enumerate(leg_items):
        x = leg_x_start + (i % 4) * leg_spacing
        row = i // 4
        ly = leg_y - row * 2.5
        draw_simple_bar(ax, x, x + 8, ly, 1.8, col, edgecolor="0.4")
        ax.text(x + 10, ly, lab, ha="left", va="center",
                fontsize=5, color="0.3")

    # ── Axes cleanup ──────────────────────────────────────────────
    ax.set_xlim(-80, x_max + 10)
    ax.set_ylim(leg_y - 5, y_positions["WT"] + 5)
    ax.axis("off")

    # ── Save ──────────────────────────────────────────────────────
    pdf_path = os.path.join(OUT_DIR, "figure_01A_schematic.pdf")
    png_path = os.path.join(OUT_DIR, "figure_01A_schematic.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, format="png", bbox_inches="tight", pad_inches=0.05,
                dpi=300)
    plt.close(fig)

    print(f"✓ PDF saved: {pdf_path}")
    print(f"✓ PNG saved: {png_path}")
    print(f"  Width: 175 mm | Schematic panel")


if __name__ == "__main__":
    make_schematic()
