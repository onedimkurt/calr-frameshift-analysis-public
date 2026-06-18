#!/usr/bin/env python3
"""
figure_05.py — Additional Sequence and Structural Features
===========================================================
Blood Cancer Journal — Article submission
CALR MPN Manuscript

Figure 5 (6 panels, cost = 2 budget slots):
  A: Feature importance (top 15 Type 1-A vs 1-B discriminators by |d|)
  B: Polyanion run length across 3 groups
  C: WT-novel charge contrast across 3 groups
  D: Inter-chain PAE across 3 groups
  E: RRR motif count across 3 groups
  F: Coacervation score across 3 groups

Data sources:
  - RECOMPUTED_FEATURES_76_VARIANTS.tsv
  - AF2_DEFINITIVE.tsv
  - cluster_assignments.tsv
  - feature_importance_posthoc.tsv
  - sequence_three_group_pairwise.tsv
  - all_comparisons.tsv
"""

import csv, os, sys, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import os as _os
DATA_ROOT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data", "derived")
FEAT_FILE = os.path.join(DATA_ROOT, "RECOMPUTED_FEATURES_76_VARIANTS.tsv")
AF2_FILE = os.path.join(DATA_ROOT, "af2_definitive", "statistics", "AF2_DEFINITIVE.tsv")
CLUST_FILE = os.path.join(DATA_ROOT, "unsupervised_clustering", "statistics", "cluster_assignments.tsv")
IMPORTANCE_FILE = os.path.join(DATA_ROOT, "unsupervised_clustering", "statistics", "feature_importance_posthoc.tsv")
SEQ_STATS = os.path.join(DATA_ROOT, "three_group_validation", "statistics", "sequence_three_group_pairwise.tsv")
AF2_STATS = os.path.join(DATA_ROOT, "af2_definitive", "statistics", "all_comparisons.tsv")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

for label, path in [("Features", FEAT_FILE), ("AF2", AF2_FILE), ("Clusters", CLUST_FILE),
                     ("Importance", IMPORTANCE_FILE), ("SeqStats", SEQ_STATS), ("AF2Stats", AF2_STATS)]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} not found: {path}"); sys.exit(1)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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

SEQ_COMP_MAP = {
    "Type2_vs_Type1A": ("Type 2", "Type 1-A"),
    "Type2_vs_Type1B": ("Type 2", "Type 1-B"),
    "Type1A_vs_Type1B": ("Type 1-A", "Type 1-B"),
}
AF2_COMP_MAP = {
    "Type1A_vs_Type1B": "Type 1-A vs Type 1-B",
    "Type2_vs_Type1A": "Type 2 vs Type 1-A",
    "Type2_vs_Type1B": "Type 2 vs Type 1-B",
}

FEAT_DISPLAY = {
    "approach_len": "Approach zone length", "novel_cterminus_residues": "Novel tail length",
    "az_charge_-10": "Charge at az pos \u221210", "az_charge_-6": "Charge at az pos \u22126",
    "az_charge_-5": "Charge at az pos \u22125", "az_charge_-7": "Charge at az pos \u22127",
    "az_charge_-12": "Charge at az pos \u221212", "az_charge_-14": "Charge at az pos \u221214",
    "az_charge_-16": "Charge at az pos \u221216", "az_charge_-8": "Charge at az pos \u22128",
    "az_charge_-4": "Charge at az pos \u22124", "az_charge_-11": "Charge at az pos \u221211",
    "az_charge_-15": "Charge at az pos \u221215",
    "SCD": "SCD", "ncpr": "NCPR", "fcr": "FCR", "kappa": "\u03BA (kappa)",
    "kd_novel": "Mean hydropathy", "coac_novel": "Coacervation score",
    "tail_RK_fraction": "Arg+Lys fraction", "grp_Positive": "Positive residues",
    "grp_Special": "Special residues", "f_P": "Pro fraction", "f_M": "Met fraction",
    "f_R": "Arg fraction", "f_G": "Gly fraction", "f_C": "Cys fraction",
    "f_S": "Ser fraction", "uversky_llps": "Uversky LLPS metric",
    "RRR_n": "RRR motif count", "entropy_novel": "Sequence entropy",
    "beta_novel": "Beta-sheet propensity", "polycation_len": "Longest polycation run",
    "n_rtrr": "RTRR motif count",
}

def sig_label(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "ns"

def draw_bracket(ax, x1, x2, y, h, label, fontsize=5, lw=0.6):
    ax.plot([x1,x1,x2,x2], [y,y+h,y+h,y], lw=lw, c="0.2", clip_on=False)
    ax.text((x1+x2)/2, y+h, label, ha="center", va="bottom", fontsize=fontsize, color="0.2")

def read_clusters(path):
    m = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m[row["sequence_id"].strip()] = row["seq_subgroup"].strip()
    return m

def read_seq_features(path, clusters, features):
    data = {f: {g: [] for g in GROUP_ORDER} for f in features}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["sequence_id"].strip()
            pc = row["primary_class"].strip()
            g = clusters.get(sid, "Type 2") if pc == "Type 1-like" else "Type 2"
            if g not in GROUP_ORDER: continue
            for feat in features:
                val = row.get(feat, "").strip()
                if val: data[feat][g].append(float(val))
    return data

def read_af2_features(path, features):
    data = {f: {g: [] for g in GROUP_ORDER} for f in features}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sg = row["subgroup"].strip(); pc = row["primary_class"].strip()
            if sg in ("Type 1-A","Type 1-B"): g = sg
            elif pc == "Type 2-like": g = "Type 2"
            else: continue
            for feat in features:
                val = row.get(feat, "").strip()
                if val: data[feat][g].append(float(val))
    return data

def read_seq_stats(path, features):
    stats = {f: {} for f in features}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            feat = row["feature"].strip()
            if feat not in features: continue
            comp = row["comparison"].strip()
            if comp in SEQ_COMP_MAP:
                g1, g2 = SEQ_COMP_MAP[comp]
                stats[feat][f"{g1} vs {g2}"] = {"p_BH": float(row["p_BH"]), "sig": row["sig"].strip()}
    return stats

def read_af2_stats(path, features):
    stats = {f: {} for f in features}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            feat = row["feature"].strip()
            if feat not in features: continue
            comp = row.get("comparison","").strip()
            p_bh = row.get("p_BH","").strip()
            if not p_bh: continue
            label = AF2_COMP_MAP.get(comp, comp)
            stats[feat][label] = {"p_BH": float(p_bh), "sig": row.get("sig","").strip()}
    return stats

def compute_missing_af2_pairwise(data, stats, features):
    try:
        from scipy.stats import mannwhitneyu
    except ImportError: return
    for feat in features:
        if "Type 2 vs Type 1-B" not in stats.get(feat, {}):
            v_t2, v_1b = data[feat]["Type 2"], data[feat]["Type 1-B"]
            if v_t2 and v_1b:
                U, p = mannwhitneyu(v_t2, v_1b, alternative="two-sided")
                p_bh = min(p*3, 1.0)
                stats.setdefault(feat,{})["Type 2 vs Type 1-B"] = {"p_BH": p_bh, "sig": sig_label(p_bh)}

def read_importance(path, top_n=15):
    feats = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            feats.append({"feature": row["feature"].strip(), "d": float(row["d"]),
                "abs_d": float(row["abs_d"]), "p_BH": float(row["p_BH"]), "sig": row["sig"].strip()})
    feats.sort(key=lambda x: x["abs_d"], reverse=True)
    return feats[:top_n]

def draw_strip_box(ax, data, feat, ylabel, stats, panel_label):
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

    ax.set_ylabel(ylabel, fontsize=6.5, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_LABELS, fontsize=5)

    ymin_data = min(min(v) for v in all_vals if v)
    ymax_data = max(max(v) for v in all_vals if v)
    if ymin_data >= 0:
        ax.set_ylim(0, ymax_data * 1.45)
    else:
        margin = (ymax_data - ymin_data) * 0.45
        ax.set_ylim(ymin_data - margin*0.15, ymax_data + margin)

    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.22, 1.10, panel_label, transform=ax.transAxes,
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

def draw_feature_importance(ax, top_feats, panel_label):
    n = len(top_feats)
    top_rev = list(reversed(top_feats))
    y_pos = list(range(n))
    d_vals = [f["d"] for f in top_rev]
    colors_bar = [COLOR_1A if d > 0 else COLOR_1B for d in d_vals]

    ax.barh(y_pos, [abs(d) for d in d_vals], height=0.7,
            color=colors_bar, edgecolor="white", linewidth=0.3, alpha=0.75)

    labels = [FEAT_DISPLAY.get(f["feature"], f["feature"]) for f in top_rev]
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("|Cohen's d|", fontsize=7, fontweight="bold")

    for i, f in enumerate(top_rev):
        ax.text(abs(f["d"])+0.05, i, sig_label(f["p_BH"]),
                va="center", ha="left", fontsize=4, color="0.3")

    ax.text(0.97, 0.02, "Blue: higher in Type 1-A\nOrange: higher in Type 1-B",
            transform=ax.transAxes, fontsize=4.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9))

    ax.set_xlim(0, max(abs(d) for d in d_vals)*1.15)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.28, 1.04, panel_label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="right")

def make_figure():
    clusters = read_clusters(CLUST_FILE)

    seq_feats = ["polyanion_len", "wt_novel_charge_contrast", "RRR_n", "coac_novel"]
    seq_data = read_seq_features(FEAT_FILE, clusters, seq_feats)
    seq_stats = read_seq_stats(SEQ_STATS, seq_feats)

    af2_feats = ["inter_pae"]
    af2_data = read_af2_features(AF2_FILE, af2_feats)
    af2_stats = read_af2_stats(AF2_STATS, af2_feats)
    compute_missing_af2_pairwise(af2_data, af2_stats, af2_feats)

    top_feats = read_importance(IMPORTANCE_FILE, top_n=15)

    # Verify
    for feat in seq_feats:
        for g in GROUP_ORDER:
            n = len(seq_data[feat][g])
            expected = {"Type 1-A": 25, "Type 1-B": 19, "Type 2": 32}[g]
            if n != expected: print(f"WARNING: {feat} {g} n={n}, expected {expected}")

    fig_w = 175/25.4; fig_h = 195/25.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1],
                           hspace=0.40, wspace=0.38,
                           left=0.10, right=0.97, bottom=0.04, top=0.97)

    # Row 1: Feature importance (A) + Polyanion run (B)
    draw_feature_importance(fig.add_subplot(gs[0, 0]), top_feats, "A")
    draw_strip_box(fig.add_subplot(gs[0, 1]), seq_data, "polyanion_len",
                   "Longest polyanion\nrun (residues)", seq_stats, "B")

    # Row 2: Charge contrast (C) + Inter-chain PAE (D)
    draw_strip_box(fig.add_subplot(gs[1, 0]), seq_data, "wt_novel_charge_contrast",
                   "WT\u2013novel charge\ncontrast", seq_stats, "C")
    draw_strip_box(fig.add_subplot(gs[1, 1]), af2_data, "inter_pae",
                   "Inter-chain PAE", af2_stats, "D")

    # Row 3: RRR motif (E) + Coacervation (F)
    draw_strip_box(fig.add_subplot(gs[2, 0]), seq_data, "RRR_n",
                   "RRR motif count", seq_stats, "E")
    draw_strip_box(fig.add_subplot(gs[2, 1]), seq_data, "coac_novel",
                   "Coacervation score", seq_stats, "F")

    for ext, fmt, dpi in [("pdf","pdf",None), ("png","png",300)]:
        path = os.path.join(OUT_DIR, f"figure_05.{ext}")
        kw = {"format": fmt, "bbox_inches": "tight", "pad_inches": 0.03}
        if dpi: kw["dpi"] = dpi
        fig.savefig(path, **kw); print(f"\u2713 Saved: {path}")
    plt.close(fig)
    print(f"  Width: 175 mm | Panels: 6 (A-F) | Budget cost: 2")

if __name__ == "__main__":
    make_figure()
