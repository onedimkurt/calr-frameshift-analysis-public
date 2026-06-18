#!/usr/bin/env python3
"""
figure_03.py — Clustering, Approach Zone, and Feature Importance
================================================================
Blood Cancer Journal — Article submission
CALR MPN Manuscript

Figure 3 (4 panels, cost = 2 budget slots):
  A: PCA colored by sequence subgroup (Type 1-A, Type 1-B)
  B: PCA colored by Sg-Sg geometry (compact, extended, intermediate)
  C: Approach zone charge polarity (lollipop, positions -1 to -14)
  D: Feature importance — top 15 discriminators by |d| (horizontal bar)

Data sources:
  - RECOMPUTED_FEATURES_76_VARIANTS.tsv (87 features, Type 1-like)
  - cluster_assignments.tsv
  - feature_importance_posthoc.tsv
"""

import csv, os, sys, random, math
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
IMPORTANCE_FILE = os.path.join(DATA_ROOT, "unsupervised_clustering", "statistics", "feature_importance_posthoc.tsv")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

for label, path in [("Features", FEAT_FILE), ("Clusters", CLUST_FILE), ("Importance", IMPORTANCE_FILE)]:
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

COLOR_1A = "#0072B2"; COLOR_1B = "#E69F00"
META = {"sequence_id","variant_id","primary_class","frameshift_position","has_anchor","anchor_start"}
POST_KKRK = {"post_kkrk_length","post_kkrk_charge","post_kkrk_rk_frac","post_kkrk_de_frac","post_kkrk_seq"}
CLASS_SEP = {"KKRK_present"}
ZERO_VAR = {"f_F","f_H","f_I","f_Y","has_cys32","has_cys36","KRKEE_n","CK2_sites","PKA_sites"}
EXCLUDE = META | POST_KKRK | CLASS_SEP | ZERO_VAR

FEAT_DISPLAY = {
    "approach_len": "Approach zone length", "novel_cterminus_residues": "Novel tail length",
    "az_charge_-10": "Charge at az pos \u221210", "az_charge_-6": "Charge at az pos \u22126",
    "az_charge_-5": "Charge at az pos \u22125", "az_charge_-7": "Charge at az pos \u22127",
    "az_charge_-12": "Charge at az pos \u221212", "az_charge_-14": "Charge at az pos \u221214",
    "az_charge_-16": "Charge at az pos \u221216", "az_charge_-8": "Charge at az pos \u22128",
    "SCD": "SCD", "ncpr": "NCPR", "fcr": "FCR", "kappa": "\u03BA (kappa)",
    "kd_novel": "Mean hydropathy", "coac_novel": "Coacervation score",
    "tail_RK_fraction": "Arg+Lys fraction", "grp_Positive": "Positive residues",
    "grp_Special": "Special residues", "f_P": "Pro fraction", "f_M": "Met fraction",
    "f_R": "Arg fraction", "f_G": "Gly fraction", "f_C": "Cys fraction",
    "f_S": "Ser fraction", "uversky_llps": "Uversky LLPS metric",
    "RRR_n": "RRR motif count", "entropy_novel": "Sequence entropy",
    "beta_novel": "Beta-sheet propensity", "polycation_len": "Longest polycation run",
    "n_rtrr": "RTRR motif count", "az_charge_-4": "Charge at az pos \u22124",
    "az_charge_-11": "Charge at az pos \u221211", "az_charge_-15": "Charge at az pos \u221215",
}

def read_cluster_info(path):
    subgroups, sg_groups = {}, {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["sequence_id"].strip()
            subgroups[sid] = row["seq_subgroup"].strip()
            sg_groups[sid] = row["sg_group"].strip()
    return subgroups, sg_groups

def read_type1_features(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        feat_cols = [c for c in reader.fieldnames if c not in EXCLUDE]
        sids, rows = [], []
        for row in reader:
            if row["primary_class"].strip() != "Type 1-like": continue
            sids.append(row["sequence_id"].strip())
            vals = [float(row[c].strip()) if row[c].strip() else float("nan") for c in feat_cols]
            rows.append(vals)
    return sids, feat_cols, rows

def read_importance(path, top_n=15):
    feats = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            feats.append({"feature": row["feature"].strip(), "d": float(row["d"]),
                "abs_d": float(row["abs_d"]), "p_BH": float(row["p_BH"]), "sig": row["sig"].strip()})
    feats.sort(key=lambda x: x["abs_d"], reverse=True)
    return feats[:top_n]

def read_az_stats(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            feat = row["feature"].strip()
            if feat.startswith("az_charge_-"):
                pos = int(feat.replace("az_charge_-", ""))
                data[pos] = {"mean_1A": float(row["strong_mean"]), "mean_1B": float(row["weak_mean"]),
                    "d": float(row["d"]), "p_BH": float(row["p_BH"]), "sig": row["sig"].strip()}
    return data

def sig_label(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "ns"

def compute_pca(sids, feat_cols, rows):
    X = np.array(rows, dtype=float)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:,j])
        if mask.any(): X[mask,j] = np.nanmedian(X[:,j])
    means = X.mean(axis=0); stds = X.std(axis=0, ddof=0); stds[stds==0] = 1.0
    Xs = (X - means) / stds
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    explained = (S**2) / (S**2).sum() * 100
    return Xs @ Vt[:2].T, explained[0], explained[1]

def draw_pca_subgroup(ax, scores, sids, subgroups, pc1v, pc2v):
    for i in range(len(sids)):
        c = COLOR_1A if subgroups.get(sids[i]) == "Type 1-A" else COLOR_1B
        ax.scatter(scores[i,0], scores[i,1], c=c, s=18, edgecolors="white", linewidths=0.4, zorder=3, alpha=0.85)
    ax.set_xlabel(f"PC1 ({pc1v:.1f}%)", fontsize=7); ax.set_ylabel(f"PC2 ({pc2v:.1f}%)", fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.axhline(0, color="0.85", lw=0.3, zorder=1); ax.axvline(0, color="0.85", lw=0.3, zorder=1)
    xm = (scores[:,0].max()-scores[:,0].min())*0.08; ym = (scores[:,1].max()-scores[:,1].min())*0.08
    ax.set_xlim(scores[:,0].min()-xm, scores[:,0].max()+xm)
    ax.set_ylim(scores[:,1].min()-ym, scores[:,1].max()+ym)
    leg = [Line2D([0],[0], marker="o", color="w", markerfacecolor=COLOR_1A, markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.3, label="Type 1-A (n=25)"),
           Line2D([0],[0], marker="o", color="w", markerfacecolor=COLOR_1B, markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.3, label="Type 1-B (n=19)")]
    ax.legend(handles=leg, fontsize=5, loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.text(-0.15, 1.06, "A", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")

def draw_pca_geometry(ax, scores, sids, subgroups, sg_groups, pc1v, pc2v):
    for i in range(len(sids)):
        c = COLOR_1A if subgroups.get(sids[i]) == "Type 1-A" else COLOR_1B
        sg = sg_groups.get(sids[i], "unknown")
        if sg == "compact":
            ax.scatter(scores[i,0], scores[i,1], c=c, s=18, marker="o", edgecolors="white", linewidths=0.4, zorder=3, alpha=0.85)
        elif sg == "extended":
            ax.scatter(scores[i,0], scores[i,1], c="none", s=24, marker="^", edgecolors=c, linewidths=1.0, zorder=3, alpha=0.9)
        else:
            ax.scatter(scores[i,0], scores[i,1], c="none", s=24, marker="D", edgecolors=c, linewidths=1.0, zorder=3, alpha=0.9)
    ax.set_xlabel(f"PC1 ({pc1v:.1f}%)", fontsize=7); ax.set_ylabel(f"PC2 ({pc2v:.1f}%)", fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.axhline(0, color="0.85", lw=0.3, zorder=1); ax.axvline(0, color="0.85", lw=0.3, zorder=1)
    xm = (scores[:,0].max()-scores[:,0].min())*0.08; ym = (scores[:,1].max()-scores[:,1].min())*0.08
    ax.set_xlim(scores[:,0].min()-xm, scores[:,0].max()+xm)
    ax.set_ylim(scores[:,1].min()-ym, scores[:,1].max()+ym)
    leg = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="0.5", markersize=4.5, markeredgecolor="white", markeredgewidth=0.3,
               label=r"Compact (S$_\gamma$–S$_\gamma$ < 20 $\AA$)"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="none", markersize=4.5, markeredgecolor="0.5", markeredgewidth=1.0,
               label=r"Extended (S$_\gamma$–S$_\gamma$ > 50 $\AA$)"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="none", markersize=4, markeredgecolor="0.5", markeredgewidth=1.0, label="Intermediate"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=COLOR_1A, markersize=4.5, markeredgecolor="white", markeredgewidth=0.3, label="Type 1-A"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=COLOR_1B, markersize=4.5, markeredgecolor="white", markeredgewidth=0.3, label="Type 1-B")]
    ax.legend(handles=leg, fontsize=4.5, loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.text(0.03, 0.03, "Rank-1 models:\nFisher's exact $p = 9.2 \\times 10^{-5}$\nMapping accuracy 81.4% (rank-1)\n5-model concordance 36.4%",
            transform=ax.transAxes, fontsize=5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9))
    ax.text(-0.15, 1.06, "B", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")

def draw_approach_zone(ax, az_data):
    positions_present = [p for p in range(1,15) if p in az_data]
    y_offset = 0.18
    for i, pos in enumerate(positions_present):
        d = az_data[pos]; y_center = len(positions_present) - i
        for mean_val, y_off, color in [(d["mean_1A"], y_offset, COLOR_1A), (d["mean_1B"], -y_offset, COLOR_1B)]:
            yy = y_center + y_off
            ax.plot([0, mean_val], [yy, yy], color=color, linewidth=1.0, solid_capstyle="round", zorder=2)
            ax.scatter(mean_val, yy, s=12, color=color, edgecolors="white", linewidths=0.3, zorder=3)
        sig = sig_label(d["p_BH"]); sig_color = "0.2" if sig != "ns" else "0.6"
        ax.text(1.12, y_center, sig, fontsize=4, va="center", ha="left", color=sig_color, fontweight="bold" if sig != "ns" else "normal")
        if d["mean_1A"] * d["mean_1B"] < 0 and sig != "ns":
            ax.axhspan(y_center-0.42, y_center+0.42, facecolor="#FFF3E0", alpha=0.5, zorder=0)
    ax.axvline(0, color="0.4", linewidth=0.6, zorder=1)
    ax.set_xlabel("Mean formal charge", fontsize=6); ax.set_ylabel("Approach zone position\n(relative to anchor)", fontsize=6)
    ax.set_yticks([len(positions_present)-i for i in range(len(positions_present))])
    ax.set_yticklabels([f"\u2212{p}" for p in positions_present], fontsize=5)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(0.2, len(positions_present)+0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.6, -0.12, "Anionic", fontsize=4.5, ha="center", va="top", color="0.4", fontstyle="italic", transform=ax.get_xaxis_transform())
    ax.text(+0.6, -0.12, "Cationic", fontsize=4.5, ha="center", va="top", color="0.4", fontstyle="italic", transform=ax.get_xaxis_transform())
    leg = [Line2D([0],[0], marker="o", color=COLOR_1A, markerfacecolor=COLOR_1A, markersize=3.5, markeredgecolor="white", markeredgewidth=0.3, linewidth=1.0, label="Type 1-A"),
           Line2D([0],[0], marker="o", color=COLOR_1B, markerfacecolor=COLOR_1B, markersize=3.5, markeredgecolor="white", markeredgewidth=0.3, linewidth=1.0, label="Type 1-B")]
    ax.legend(handles=leg, fontsize=4.5, loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.text(1.12, len(positions_present)+0.6, "Sig.", fontsize=4.5, va="center", ha="left", fontweight="bold", color="0.3")
    ax.text(-0.15, 1.04, "C", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")

def draw_feature_importance(ax, top_feats):
    n = len(top_feats)
    top_rev = list(reversed(top_feats))
    y_pos = list(range(n))
    d_vals = [f["d"] for f in top_rev]
    colors_bar = [COLOR_1A if d > 0 else COLOR_1B for d in d_vals]
    ax.barh(y_pos, [abs(d) for d in d_vals], height=0.7, color=colors_bar, edgecolor="white", linewidth=0.3, alpha=0.75)
    labels = [FEAT_DISPLAY.get(f["feature"], f["feature"]) for f in top_rev]
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("|Cohen's d|", fontsize=7, fontweight="bold")
    for i, f in enumerate(top_rev):
        ax.text(abs(f["d"])+0.05, i, sig_label(f["p_BH"]), va="center", ha="left", fontsize=4, color="0.3")
    ax.text(0.97, 0.02, "Blue: higher in Type 1-A\nOrange: higher in Type 1-B",
            transform=ax.transAxes, fontsize=4.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9))
    ax.set_xlim(0, max(abs(d) for d in d_vals)*1.15)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.28, 1.04, "D", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")

def make_figure():
    subgroups, sg_groups = read_cluster_info(CLUST_FILE)
    sids, feat_cols, rows = read_type1_features(FEAT_FILE)
    scores, pc1v, pc2v = compute_pca(sids, feat_cols, rows)
    az_data = read_az_stats(IMPORTANCE_FILE)
    top_feats = read_importance(IMPORTANCE_FILE, top_n=15)
    print(f"PCA: PC1={pc1v:.1f}%, PC2={pc2v:.1f}%, n={len(sids)}, p={len(feat_cols)}")

    fig_w = 175/25.4; fig_h = 175/25.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.3],
                           hspace=0.32, wspace=0.35, left=0.10, right=0.97, bottom=0.05, top=0.97)

    draw_pca_subgroup(fig.add_subplot(gs[0,0]), scores, sids, subgroups, pc1v, pc2v)
    draw_pca_geometry(fig.add_subplot(gs[0,1]), scores, sids, subgroups, sg_groups, pc1v, pc2v)
    draw_approach_zone(fig.add_subplot(gs[1,0]), az_data)
    draw_feature_importance(fig.add_subplot(gs[1,1]), top_feats)

    for ext, fmt, dpi in [("pdf","pdf",None), ("png","png",300)]:
        path = os.path.join(OUT_DIR, f"figure_03.{ext}")
        kw = {"format": fmt, "bbox_inches": "tight", "pad_inches": 0.03}
        if dpi: kw["dpi"] = dpi
        fig.savefig(path, **kw); print(f"✓ Saved: {path}")
    plt.close(fig)
    print(f"  Width: 175 mm | Panels: 4 (A-D) | Budget cost: 2")

if __name__ == "__main__":
    make_figure()
