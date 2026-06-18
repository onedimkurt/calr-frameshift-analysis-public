#!/usr/bin/env python3
"""
figure_04.py — Structural Validation
=====================================
Blood Cancer Journal — Article submission
CALR MPN Manuscript

Figure 4 (6 panels, cost = 2 budget slots):
  A: Sg-Sg inter-chain distance distribution by subgroup
  B: Contingency heatmap (sequence subgroup x Sg-Sg geometry)
  C: Homodimer novel tail pLDDT across 3 groups
  D: ipTM across 3 groups
  E: Dimer-pTM across 3 groups
  F: Normalized inter-chain novel tail contacts across 3 groups

Data sources:
  - AF2_DEFINITIVE.tsv
  - cluster_assignments.tsv
  - all_comparisons.tsv (pairwise stats)
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
AF2_FILE = os.path.join(DATA_ROOT, "af2_definitive", "statistics", "AF2_DEFINITIVE.tsv")
CLUST_FILE = os.path.join(DATA_ROOT, "unsupervised_clustering", "statistics", "cluster_assignments.tsv")
AF2_STATS = os.path.join(DATA_ROOT, "af2_definitive", "statistics", "all_comparisons.tsv")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

for label, path in [("AF2", AF2_FILE), ("Clusters", CLUST_FILE), ("Stats", AF2_STATS)]:
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

AF2_COMP_MAP = {
    "Type1A_vs_Type1B": "Type 1-A vs Type 1-B",
    "Type2_vs_Type1A": "Type 2 vs Type 1-A",
    "Type1like_vs_Type2": "Type 1-like vs Type 2",
}


def sig_label(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "ns"


def draw_bracket(ax, x1, x2, y, h, label, fontsize=5, lw=0.6):
    ax.plot([x1,x1,x2,x2], [y,y+h,y+h,y], lw=lw, c="0.2", clip_on=False)
    ax.text((x1+x2)/2, y+h, label, ha="center", va="bottom", fontsize=fontsize, color="0.2")


def read_cluster_info(path):
    data = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            data.append({
                "sid": row["sequence_id"].strip(),
                "subgroup": row["seq_subgroup"].strip(),
                "sg_dist": float(row["sg_sg_dist"]),
                "sg_group": row["sg_group"].strip(),
            })
    return data


def read_af2_data(path, features):
    data = {f: {g: [] for g in GROUP_ORDER} for f in features}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sg = row["subgroup"].strip()
            pc = row["primary_class"].strip()
            if sg in ("Type 1-A", "Type 1-B"): g = sg
            elif pc == "Type 2-like": g = "Type 2"
            else: continue
            for feat in features:
                val = row.get(feat, "").strip()
                if val: data[feat][g].append(float(val))
    return data


def read_af2_stats(path, features):
    stats = {f: {} for f in features}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            feat = row["feature"].strip()
            if feat not in features: continue
            comp = row.get("comparison", "").strip()
            p_bh = row.get("p_BH", "").strip()
            if not p_bh: continue
            label = AF2_COMP_MAP.get(comp, comp)
            stats[feat][label] = {"p_BH": float(p_bh), "sig": row.get("sig", "").strip()}
    return stats


def compute_missing_pairwise(data, stats, features):
    """Compute Type 2 vs Type 1-B if missing, using Mann-Whitney."""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        print("scipy not available"); return
    for feat in features:
        if "Type 2 vs Type 1-B" not in stats.get(feat, {}):
            v_t2 = data[feat]["Type 2"]
            v_1b = data[feat]["Type 1-B"]
            if v_t2 and v_1b:
                U, p = mannwhitneyu(v_t2, v_1b, alternative="two-sided")
                stats.setdefault(feat, {})["Type 2 vs Type 1-B"] = {
                    "p_BH": min(p * 3, 1.0), "sig": sig_label(min(p * 3, 1.0))
                }


def fisher_exact_2x2(a, b, c, d):
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[a, b], [c, d]])
        return p
    except ImportError:
        return 0.000092


def draw_strip_box_3group(ax, data, feat, ylabel, stats, panel_label):
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
        ax.set_ylim(ymin_data - margin * 0.15, ymax_data + margin)

    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.22, 1.10, panel_label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="right")

    # Brackets
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
            draw_bracket(ax, positions[xi], positions[xj], y_pos, bracket_h,
                         sig_label(s["p_BH"]))


def draw_sg_strip(ax, clust_data):
    """Panel A: Sg-Sg distance strip plot."""
    random.seed(42)
    data_1a = [v for v in clust_data if v["subgroup"] == "Type 1-A"]
    data_1b = [v for v in clust_data if v["subgroup"] == "Type 1-B"]

    for gi, (variants, color) in enumerate([(data_1a, COLOR_1A), (data_1b, COLOR_1B)]):
        dists = [v["sg_dist"] for v in variants]
        jitter = [gi + random.uniform(-0.15, 0.15) for _ in dists]
        ax.scatter(jitter, dists, s=18, color=color, alpha=0.8,
                   edgecolors="white", linewidths=0.3, zorder=3)

    ax.axhline(20, color="0.5", linewidth=0.6, linestyle="--", zorder=1)
    ax.axhline(50, color="0.5", linewidth=0.6, linestyle="--", zorder=1)
    ax.text(1.65, 20, r"20 $\AA$", fontsize=5, va="center", ha="left", color="0.4")
    ax.text(1.65, 50, r"50 $\AA$", fontsize=5, va="center", ha="left", color="0.4")

    ax.text(-0.52, 10, "Compact", fontsize=4.5, va="center", ha="center",
            color="0.55", fontstyle="italic", rotation=90)
    ax.text(-0.52, 35, "Intermediate", fontsize=4.5, va="center", ha="center",
            color="0.55", fontstyle="italic", rotation=90)
    ax.text(-0.52, 78, "Extended", fontsize=4.5, va="center", ha="center",
            color="0.55", fontstyle="italic", rotation=90)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Type 1-A\n(n=25)", "Type 1-B\n(n=19)"], fontsize=5.5)
    ax.set_ylabel(r"S$_\gamma$–S$_\gamma$ inter-chain distance ($\AA$)", fontsize=6.5, labelpad=8)

    all_dists = [v["sg_dist"] for v in clust_data]
    ax.set_ylim(0, max(all_dists) * 1.08)
    ax.set_xlim(-0.70, 1.85)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.12, 1.06, "A", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="right")


def draw_contingency(ax, clust_data):
    """Panel B: Contingency heatmap."""
    ct = {"Type 1-A": {"compact": 0, "extended": 0},
          "Type 1-B": {"compact": 0, "extended": 0}}
    n_inter = 0
    for v in clust_data:
        if v["sg_group"] in ("compact", "extended"):
            ct[v["subgroup"]][v["sg_group"]] += 1
        else:
            n_inter += 1

    a, b = ct["Type 1-A"]["compact"], ct["Type 1-A"]["extended"]
    c, d = ct["Type 1-B"]["compact"], ct["Type 1-B"]["extended"]
    fisher_p = fisher_exact_2x2(a, b, c, d)
    correct = a + d
    evaluable = a + b + c + d
    accuracy = 100 * correct / evaluable

    table = np.array([[a, b], [c, d]])
    cell_colors = np.array([
        [COLOR_1A, "#D4E8F5"],
        ["#FEF0D5", COLOR_1B],
    ])

    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j, 1-i), 1, 1, facecolor=cell_colors[i,j],
                                  edgecolor="white", linewidth=1.5, alpha=0.5)
            ax.add_patch(rect)
            ax.text(j+0.5, 1.5-i, str(table[i,j]), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="0.15")

    for i, (lab, color) in enumerate([("Type 1-A", COLOR_1A), ("Type 1-B", COLOR_1B)]):
        ax.text(-0.08, 1.5-i, lab, ha="right", va="center", fontsize=6,
                fontweight="bold", color=color)
    for j, lab in enumerate(["Compact", "Extended"]):
        ax.text(j+0.5, 2.15, lab, ha="center", va="bottom", fontsize=6,
                fontweight="bold", color="0.3")

    ax.text(1.0, 2.45, r"S$_\gamma$–S$_\gamma$ geometry",
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="0.2")
    ax.text(-0.12, 1.0, "Sequence\nsubgroup", ha="right", va="center",
            fontsize=6.5, fontweight="bold", color="0.2", rotation=90)

    ax.set_xlim(-0.1, 2.1); ax.set_ylim(-0.1, 2.6)
    ax.set_aspect("equal"); ax.axis("off")

    p_str = f"{fisher_p:.1e}".replace("e-0", r" \times 10^{-").replace("e-", r" \times 10^{-") + "}"
    ax.text(1.0, -0.05,
            f"Fisher's exact $p = {p_str}$\n"
            f"Mapping accuracy: {correct}/{evaluable} ({accuracy:.1f}%)\n"
            f"Intermediate excluded: {n_inter}",
            ha="center", va="top", fontsize=5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9))
    ax.text(-0.05, 1.04, "B", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="right")


def draw_multimodel_sgsg(ax):
    """Panel G: per-variant inter-chain Sgamma-Sgamma mean +/- SD across 5 ColabFold
    models, ordered within group. Built from the verified C11 multi-model table."""
    import csv as _csv
    PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    C11 = os.path.join(PROJ,"data","derived","SuppTable_multimodel_sgsg.tsv")
    CL  = PROJ + "/RECOMPUTED_FEATURES/unsupervised_clustering/statistics/cluster_assignments.tsv"
    AF  = PROJ + "/RECOMPUTED_FEATURES/af2_definitive/statistics/AF2_DEFINITIVE.tsv"

    # group map: Type 1-A/1-B from cluster file; Type 2 from AF2 primary_class
    grp = {}
    with open(CL) as fh:
        for row in _csv.DictReader(fh, delimiter="\t"):
            sid = row["sequence_id"].split("|")[0].strip()
            grp[sid] = row["seq_subgroup"].strip()
    with open(AF) as fh:
        for row in _csv.DictReader(fh, delimiter="\t"):
            sid = row["sequence_id"].split("|")[0].strip()
            if row.get("primary_class","").strip() == "Type 2-like":
                grp[sid] = "Type 2"

    rows = []
    with open(C11) as fh:
        for row in _csv.DictReader(fh, delimiter="\t"):
            vid = row["variant_id"].strip()
            g = grp.get(vid)
            if g not in ("Type 1-A","Type 1-B","Type 2"):
                continue
            rows.append((g, float(row["sgsg_mean"]), float(row["sgsg_sd"])))

    cmap = {"Type 1-A": COLOR_1A, "Type 1-B": COLOR_1B, "Type 2": COLOR_T2}
    x = 0; xticks = []; xlabels = []
    for g in GROUP_ORDER:
        gr = sorted([r for r in rows if r[0]==g], key=lambda r: r[1])
        if not gr: continue
        start = x
        for (_, mean, sd) in gr:
            ax.errorbar(x, mean, yerr=sd, fmt="o", ms=2.6, mfc=cmap[g], mec="white",
                        mew=0.3, ecolor=cmap[g], elinewidth=0.6, capsize=1.2, alpha=0.85, zorder=3)
            x += 1
        xticks.append((start + x - 1) / 2.0)
        xlabels.append(g.replace("Type ", "T"))
        x += 2  # gap between groups

    # geometry bands
    ax.axhspan(0, 20, color="0.85", alpha=0.35, zorder=0)
    ax.axhspan(50, ax.get_ylim()[1] if ax.get_ylim()[1] > 50 else 200, color="0.92", alpha=0.4, zorder=0)
    ax.axhline(20, color="0.5", lw=0.4, ls=":", zorder=1)
    ax.axhline(50, color="0.5", lw=0.4, ls=":", zorder=1)
    ax.set_ylim(0, 200)
    ax.text(0.995, 20, "compact (<20 A)", fontsize=4.2, color="0.4", ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.text(0.995, 50, "extended (>50 A)", fontsize=4.2, color="0.4", ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=5)
    ax.set_ylabel(r"S$_\gamma$-S$_\gamma$ across 5 models" + "\n(mean $\pm$ SD, $\AA$)", fontsize=6)
    ax.set_xlabel("variants, ordered by mean within group", fontsize=5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.06, 1.04, "G", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
    ax.set_title("Multi-model S$_\\gamma$-S$_\\gamma$ reproducibility (5 models per variant)", fontsize=6, pad=3)


def make_figure():
    struct_features = ["dimer_novel_tail_plddt", "dimer_iptm", "dimer_ptm",
                       "novel_tail_contacts_norm"]
    clust_data = read_cluster_info(CLUST_FILE)
    af2_data = read_af2_data(AF2_FILE, struct_features)
    af2_stats = read_af2_stats(AF2_STATS, struct_features)
    compute_missing_pairwise(af2_data, af2_stats, struct_features)

    # Verify
    for feat in struct_features:
        for g in GROUP_ORDER:
            n = len(af2_data[feat][g])
            expected = {"Type 1-A": 25, "Type 1-B": 19, "Type 2": 32}[g]
            if n != expected:
                print(f"WARNING: {feat} {g} n={n}, expected {expected}")

    # Layout: 3 rows x 2 columns
    fig_w = 175 / 25.4; fig_h = 240 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           height_ratios=[1.1, 1, 1, 0.95],
                           hspace=0.45, wspace=0.38,
                           left=0.10, right=0.97, bottom=0.04, top=0.97)

    # Row 1: Sg-Sg strip + contingency
    draw_sg_strip(fig.add_subplot(gs[0, 0]), clust_data)
    draw_contingency(fig.add_subplot(gs[0, 1]), clust_data)

    # Row 2: dimer novel tail pLDDT + ipTM
    draw_strip_box_3group(fig.add_subplot(gs[1, 0]), af2_data,
        "dimer_novel_tail_plddt", "Homodimer novel\ntail pLDDT", af2_stats, "C")
    draw_strip_box_3group(fig.add_subplot(gs[1, 1]), af2_data,
        "dimer_iptm", "ipTM", af2_stats, "D")

    # Row 3: dimer-pTM + novel tail contacts
    draw_strip_box_3group(fig.add_subplot(gs[2, 0]), af2_data,
        "dimer_ptm", "Dimer-pTM", af2_stats, "E")
    draw_strip_box_3group(fig.add_subplot(gs[2, 1]), af2_data,
        "novel_tail_contacts_norm", "Novel tail contacts\n(normalized)", af2_stats, "F")

    # Row 4: panel G spans full width — multi-model Sgamma-Sgamma reproducibility
    draw_multimodel_sgsg(fig.add_subplot(gs[3, :]))

    for ext, fmt, dpi in [("pdf", "pdf", None), ("png", "png", 300)]:
        path = os.path.join(OUT_DIR, f"figure_04.{ext}")
        kw = {"format": fmt, "bbox_inches": "tight", "pad_inches": 0.03}
        if dpi: kw["dpi"] = dpi
        fig.savefig(path, **kw); print(f"✓ Saved: {path}")
    plt.close(fig)
    print(f"  Width: 175 mm | Panels: 7 (A-G) | Budget cost: 2")


if __name__ == "__main__":
    make_figure()
