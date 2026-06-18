#!/usr/bin/env python3
"""
============================================================
TRUE UNSUPERVISED CLUSTERING — TYPE 2 (Type 2) VARIANTS
============================================================
Mirrors the Type 1 clustering analysis exactly, applied to the
32 Type 2 (Type 2 / net +2 frameshift) variants.

Key differences from Type 1 analysis:
  - Clusters Type 2 variants (not Type 1)
  - No SG-SG validation (no structural geometry groups for Type 2)
  - post_kkrk_* features ARE defined for Type 2 → included
  - KKRK_present = 1 for ALL Type 2 → excluded (zero variance)
  - No seq_subgroup label assigned (purely exploratory)

Output goes to a NEW directory:
  RECOMPUTED_FEATURES/type2_unsupervised_clustering/
  → does NOT overwrite anything in unsupervised_clustering/

Run:
  conda run -n calr_env python ~/Downloads/type2_unsupervised_clustering.py
============================================================
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import pdist
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ── PATHS ─────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
FEAT_FILE = DATA / "derived" / "RECOMPUTED_FEATURES_76_VARIANTS.tsv"

# New output directory — does NOT overwrite Type 1 results
OUTDIR = DATA / "derived" / "type2_unsupervised_clustering"
(OUTDIR / "figures").mkdir(parents=True, exist_ok=True)
(OUTDIR / "statistics").mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TYPE 2 UNSUPERVISED CLUSTERING")
print("=" * 70)
print(f"Output directory: {OUTDIR}")
print()

# ── LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv(FEAT_FILE, sep="\t")

# Separate Type 2 (Type 2) variants
type2 = df[df["primary_class"].str.contains("2", na=False)].copy()
type1 = df[df["primary_class"].str.contains("1", na=False)].copy()
print(f"Total variants: {len(df)}  |  Type 2: {len(type2)}  |  Type 1 (Type 1-like): {len(type1)}")
assert len(type2) == 32, f"Expected 32 Type 2 variants, got {len(type2)}"

# ── FEATURE SELECTION ─────────────────────────────────────
META_COLS = {"sequence_id", "variant_id", "primary_class",
             "frameshift_position", "has_anchor", "anchor_start", "post_kkrk_seq"}

EXCLUDE = META_COLS | {
    "sg_sg_dist",     # structural validation — not available for Type 2
    "KKRK_present",   # zero variance in Type 2 (all = 1)
}

all_feats = [c for c in df.columns
             if c not in EXCLUDE
             and df[c].dtype in ["float64", "int64", "bool"]
             and c not in META_COLS]

# Keep features that have variance within Type 2 and sufficient non-missing values
usable = []
dropped = []
for f in all_feats:
    vals = type2[f].dropna()
    if len(vals) >= 10 and vals.std() > 1e-10:
        usable.append(f)
    else:
        dropped.append(f)

print(f"\nFeature selection:")
print(f"  All numeric features considered: {len(all_feats)}")
print(f"  Retained (variance ≥1e-10, n≥10): {len(usable)}")
print(f"  Dropped (zero variance or sparse in Type 2): {len(dropped)}")
if dropped:
    for f in dropped:
        vals = type2[f].dropna()
        reason = "zero variance" if len(vals) >= 10 else f"only {len(vals)} non-missing"
        print(f"    {f}  [{reason}]")

# ── PREPARE Type 2 DATA ────────────────────────────────────────
X_raw = type2[usable].copy()

n_imputed = 0
imputed_cols = []
for col in usable:
    n_na = X_raw[col].isna().sum()
    if n_na > 0:
        median_val = X_raw[col].median()
        X_raw[col] = X_raw[col].fillna(median_val)
        n_imputed += n_na
        imputed_cols.append((col, n_na, median_val))

print(f"\nImputation:")
print(f"  Total imputed values: {n_imputed} ({n_imputed / (len(type2) * len(usable)) * 100:.1f}% of matrix)")
if imputed_cols:
    for col, n_na, med in imputed_cols:
        print(f"    {col}: {n_na} NaN → median={med:.3f}")
else:
    print("  No NaN values — no imputation needed")

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), columns=usable, index=X_raw.index)
print(f"\nData matrix: {X.shape[0]} Type 2 variants × {X.shape[1]} features")

# ============================================================
# STEP 1: SILHOUETTE + GAP STATISTIC
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: CLUSTER STRUCTURE ASSESSMENT")
print("=" * 70)

sil = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = km.fit_predict(X)
    sil[k] = silhouette_score(X, labels)
    print(f"  k={k}: silhouette={sil[k]:.3f}")

best_k_sil = max(sil, key=sil.get)
print(f"\n  Best k by silhouette: {best_k_sil} (sil={sil[best_k_sil]:.3f})")
print(f"  k=2 silhouette: {sil[2]:.3f}")

# Gap statistic
print(f"\n  Gap statistic (k=1 to 4, 50 reference datasets):")
np.random.seed(42)
n_ref = 50
gap_scores = {}
for k in [1, 2, 3, 4]:
    if k == 1:
        obs = np.sum(np.var(X.values, axis=0)) * len(X)
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init=50)
        km.fit(X)
        obs = km.inertia_
    refs = []
    for _ in range(n_ref):
        Xr = np.random.uniform(X.min().values, X.max().values, size=X.shape)
        if k == 1:
            ri = np.sum(np.var(Xr, axis=0)) * len(Xr)
        else:
            kmr = KMeans(n_clusters=k, random_state=None, n_init=10)
            kmr.fit(Xr)
            ri = kmr.inertia_
        refs.append(np.log(ri))
    gap = np.mean(refs) - np.log(obs)
    se = np.std(refs) * np.sqrt(1 + 1 / n_ref)
    gap_scores[k] = {"gap": gap, "se": se}
    print(f"    k={k}: gap={gap:.3f} ± {se:.3f}")

if gap_scores[1]["gap"] >= gap_scores[2]["gap"] - gap_scores[2]["se"]:
    gap_verdict = "k=1 sufficient (no subgroups detected)"
else:
    gap_verdict = "k≥2 supported"
print(f"\n  Gap verdict: {gap_verdict}")

# ============================================================
# STEP 2: K-MEANS k=2 AND HIERARCHICAL (for reference)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: CLUSTERING (k=2, for comparison)")
print("=" * 70)

km2 = KMeans(n_clusters=2, random_state=42, n_init=50)
type2["cluster_k2"] = km2.fit_predict(X) + 1

Z = linkage(pdist(X.values, "euclidean"), method="ward")
cc, _ = cophenet(Z, pdist(X.values, "euclidean"))
type2["hclust_k2"] = fcluster(Z, t=2, criterion="maxclust")

c1 = type2[type2["cluster_k2"] == 1]
c2 = type2[type2["cluster_k2"] == 2]
print(f"  K-means k=2: Cluster 1={len(c1)}, Cluster 2={len(c2)}")
print(f"  Cophenetic correlation (Ward): {cc:.3f}")
print(f"  K-means / Hclust agreement: {(type2['cluster_k2'] == type2['hclust_k2']).mean():.0%}")

# k-means for all k=2..7 (save cluster sizes)
kmeans_results = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = km.fit_predict(X) + 1
    sizes = sorted([int((labels == c).sum()) for c in range(1, k + 1)], reverse=True)
    kmeans_results[k] = sizes
    print(f"  k={k}: cluster sizes = {sizes}  silhouette={sil[k]:.3f}")

# ============================================================
# STEP 3: PCA
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: PCA")
print("=" * 70)

pca = PCA(n_components=min(10, X.shape[1]))
X_pca = pca.fit_transform(X.values)
ve = pca.explained_variance_ratio_
print(f"  PC1: {ve[0]:.1%}  PC2: {ve[1]:.1%}  PC1+PC2: {sum(ve[:2]):.1%}")
print(f"  Components for 80% variance: {(np.cumsum(ve) >= 0.80).argmax() + 1}")

# ============================================================
# STEP 4: POST-HOC FEATURE IMPORTANCE (k=2, for reference)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: POST-HOC FEATURE IMPORTANCE (k=2 clusters)")
print("=" * 70)

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    v1, v2 = g1.var(), g2.var()
    ps = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / ps if ps > 1e-10 else 0.0

def bh(pvals):
    n = len(pvals)
    if n == 0:
        return []
    si = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    adj[si[-1]] = min(1.0, pvals[si[-1]])
    for i in range(n - 2, -1, -1):
        adj[si[i]] = min(pvals[si[i]] * n / (i + 1), adj[si[i + 1]], 1.0)
    return adj

feat_results = []
raw_p = []
for f in usable:
    s = type2.loc[type2["cluster_k2"] == 1, f].dropna()
    w = type2.loc[type2["cluster_k2"] == 2, f].dropna()
    if len(s) < 3 or len(w) < 3:
        continue
    try:
        _, p = stats.mannwhitneyu(s, w, alternative="two-sided")
    except Exception:
        continue
    d = cohens_d(s, w)
    feat_results.append({"feature": f, "cluster1_mean": s.mean(), "cluster2_mean": w.mean(),
                          "d": d, "abs_d": abs(d), "p_raw": p})
    raw_p.append(p)

adj = bh(raw_p)
for i, r in enumerate(feat_results):
    r["p_BH"] = adj[i]
    r["sig"] = "***" if adj[i] < 0.001 else "**" if adj[i] < 0.01 else "*" if adj[i] < 0.05 else "ns"

df_feat = pd.DataFrame(feat_results).sort_values("abs_d", ascending=False)
n_sig = (df_feat["sig"] != "ns").sum()
print(f"  BH-significant features discriminating k=2 clusters: {n_sig}/{len(feat_results)}")
print(f"\n  Top 20 by |d|:")
print(f"  {'Feature':<30} {'Cl1 mean':>9} {'Cl2 mean':>9} {'d':>7} {'p_BH':>10} {'sig':>4}")
print("  " + "-" * 70)
for _, r in df_feat.head(20).iterrows():
    print(f"  {r['feature']:<30} {r['cluster1_mean']:>9.3f} {r['cluster2_mean']:>9.3f} "
          f"{r['d']:>+7.2f} {r['p_BH']:>10.4f} {r['sig']:>4}")

# ============================================================
# STEP 5: COMPREHENSIVE FIGURE
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: FIGURE")
print("=" * 70)

# Color scheme matching manuscript
C1 = "#4393c3"    # blue-ish for Type 2 cluster 1
C2 = "#d6604d"    # red-ish for Type 2 cluster 2

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 300, "savefig.dpi": 300,
})

fig = plt.figure(figsize=(22, 16))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel A: Silhouette across k ──────────────────────────
ax = fig.add_subplot(gs[0, 0])
ks = sorted(sil.keys())
ss = [sil[k] for k in ks]
bar_colors = ["#e74c3c" if k == best_k_sil else "#bdc3c7" for k in ks]
bars = ax.bar(ks, ss, color=bar_colors, edgecolor="black", linewidth=0.7)
# Annotate bars
for bar, s_val in zip(bars, ss):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{s_val:.3f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xlabel("k (number of clusters)", fontsize=7)
ax.set_ylabel("Silhouette coefficient", fontsize=7)
ax.set_title(f"A. Silhouette vs k  |  {len(usable)} features\n"
             f"Best k={best_k_sil} (sil={sil[best_k_sil]:.3f})   k=2: {sil[2]:.3f}",
             fontsize=9, fontweight="bold")
ax.set_xticks(ks)
ax.set_ylim(0, max(ss) * 1.25)
ax.axhline(sil[2], ls="--", lw=0.8, color="#e74c3c", alpha=0.5, label=f"k=2 ({sil[2]:.3f})")
ax.legend(fontsize=7)

# Annotate gap verdict
verdict_color = "#c0392b" if "no subgroups" in gap_verdict else "#27ae60"
ax.text(0.98, 0.98, f"Gap: {gap_verdict}", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.5, color=verdict_color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=verdict_color, alpha=0.8))

# ── Panel B: Cluster sizes at each k ─────────────────────
ax = fig.add_subplot(gs[0, 1])
# Stacked bar showing cluster size composition at each k
all_sizes_by_k = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels_k = km.fit_predict(X) + 1
    sizes_k = [int((labels_k == c).sum()) for c in range(1, k + 1)]
    # Sort descending for consistent stacking
    sizes_k_sorted = sorted(sizes_k, reverse=True)
    all_sizes_by_k[k] = sizes_k_sorted

cluster_colors = ["#4393c3", "#d6604d", "#74c476", "#fdae6b", "#9e9ac8", "#969696", "#e7ba52"]

x_pos = list(range(2, 8))
bottoms = np.zeros(len(x_pos))
# Find max number of clusters to plot
max_k = 7
for cluster_idx in range(max_k):
    heights = []
    for k in x_pos:
        sizes = all_sizes_by_k[k]
        heights.append(sizes[cluster_idx] if cluster_idx < len(sizes) else 0)
    color = cluster_colors[cluster_idx] if cluster_idx < len(cluster_colors) else "#cccccc"
    label = f"Cluster {cluster_idx + 1}" if cluster_idx < 3 else None
    ax.bar(x_pos, heights, bottom=bottoms,
           color=color, edgecolor="black", linewidth=0.5,
           label=label, alpha=0.85)
    # Label non-zero segments
    for xi, (h, b, k) in enumerate(zip(heights, bottoms, x_pos)):
        if h > 0:
            ax.text(xi + 2, b + h / 2, str(h), ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="black")
    bottoms += np.array(heights, dtype=float)

ax.set_xlabel("k (number of clusters)", fontsize=7)
ax.set_ylabel("Number of variants", fontsize=7)
ax.set_title(f"B. Cluster size composition per k\n(Type 2, n={len(type2)})",
             fontsize=9, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_ylim(0, len(type2) * 1.15)
ax.axhline(len(type2), ls=":", lw=0.7, color="gray", alpha=0.5)
ax.legend(fontsize=7, loc="upper right")

# ── Panel C: PCA colored by k=2 cluster ──────────────────
ax = fig.add_subplot(gs[0, 2])
for cl, color, marker, label in [(1, C1, "o", f"Cluster 1 (n={len(c1)})"),
                                   (2, C2, "s", f"Cluster 2 (n={len(c2)})")]:
    mask = type2["cluster_k2"] == cl
    idx_list = [list(type2.index).index(i) for i in type2[mask].index if i in type2.index]
    ax.scatter(X_pca[idx_list, 0], X_pca[idx_list, 1],
               c=color, s=60, marker=marker,
               edgecolor="black", linewidth=0.5, alpha=0.85, label=label)
ax.set_xlabel(f"PC1 ({ve[0]:.1%})", fontsize=7)
ax.set_ylabel(f"PC2 ({ve[1]:.1%})", fontsize=7)
ax.set_title(f"C. PCA — k=2 cluster membership\n(PC1+PC2 = {sum(ve[:2]):.1%} variance)",
             fontsize=9, fontweight="bold")
ax.legend(fontsize=8)
ax.text(0.02, 0.02,
        f"Agreement k-means/Ward: {(type2['cluster_k2'] == type2['hclust_k2']).mean():.0%}\n"
        f"Cophenetic r = {cc:.3f}",
        transform=ax.transAxes, fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#dee2e6"))

# ── Panel D: Feature importance (top 20, k=2 post-hoc) ───
ax = fig.add_subplot(gs[1, 0])
top20 = df_feat.head(20).sort_values("abs_d", ascending=True)
bar_c = ["#c0392b" if s != "ns" else "#bdc3c7" for s in top20["sig"]]
ax.barh(range(len(top20)), top20["d"],
        color=bar_c, edgecolor="black", linewidth=0.4, alpha=0.85)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["feature"], fontsize=6)
ax.axvline(0, color="black", lw=0.7)
ax.axvline(-0.8, ls=":", color="gray", alpha=0.4)
ax.axvline(0.8, ls=":", color="gray", alpha=0.4)
ax.set_xlabel("Cohen's d (Cluster 1 − Cluster 2)", fontsize=7)
ax.set_title(f"D. Post-hoc feature importance (k=2)\n"
             f"{n_sig} BH-significant / {len(feat_results)} tested",
             fontsize=9, fontweight="bold")
ax.text(0.98, 0.02, "Red = BH-sig\nGray = ns",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#dee2e6"))

# ── Panel E: Dendrogram ───────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
from scipy.cluster.hierarchy import set_link_color_palette
set_link_color_palette(["#333333"])
dendrogram(Z, ax=ax, leaf_rotation=90, no_labels=True,
           color_threshold=0, above_threshold_color="#333333")
ax.set_title(f"E. Ward hierarchical clustering\nCophenetic r = {cc:.3f}",
             fontsize=9, fontweight="bold")
ax.set_ylabel("Distance", fontsize=7)

# ── Panel F: Summary text ─────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")

# Gap table
gap_lines = "\n".join([f"    k={k}: gap={v['gap']:.3f} ± {v['se']:.3f}"
                        for k, v in gap_scores.items()])
# Cluster size table
size_lines = "\n".join([f"    k={k}: {' | '.join(str(s) for s in all_sizes_by_k[k])}"
                         for k in range(2, 8)])

summary = f"""TYPE 2 (Type 2) UNSUPERVISED CLUSTERING
{'=' * 42}

Variants: {len(type2)} Type 2 (Type 2 / net +2 frameshift)
Features: {len(usable)} (all recomputed, no cherry-picking)
Excluded: KKRK_present (zero variance in Type 2)
          sg_sg_dist (not available for Type 2)
NaN handling: median imputation
  Imputed values: {n_imputed} ({n_imputed / (len(type2) * len(usable)) * 100:.1f}% of matrix)

Cluster structure (silhouette):
  k=2: {sil[2]:.3f}   k=3: {sil[3]:.3f}   k=4: {sil[4]:.3f}
  k=5: {sil[5]:.3f}   k=6: {sil[6]:.3f}   k=7: {sil[7]:.3f}
  Best k = {best_k_sil} (sil = {sil[best_k_sil]:.3f})

Gap statistic (50 references):
{gap_lines}
  Verdict: {gap_verdict}

Cluster sizes by k:
{size_lines}

K-means/Ward agreement at k=2:
  {(type2['cluster_k2'] == type2['hclust_k2']).mean():.0%}  (Cophenetic r = {cc:.3f})

Post-hoc (k=2): {n_sig}/{len(feat_results)} BH-significant features

PCA: PC1={ve[0]:.1%}  PC2={ve[1]:.1%}  total={sum(ve[:2]):.1%}

NOTE: No SG-SG validation available for Type 2.
This analysis is purely exploratory."""

ax.text(0.03, 0.97, summary, transform=ax.transAxes, fontsize=7.5,
        va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa",
                  edgecolor="#dee2e6", alpha=0.95))

title_color = "#c0392b" if "no subgroups" in gap_verdict else "#27ae60"
verdict_display = "NO MEANINGFUL SUBSTRUCTURE DETECTED" if "no subgroups" in gap_verdict else "SUBSTRUCTURE DETECTED — INTERPRType 2 WITH CAUTION"
fig.suptitle(f"Unsupervised Clustering of Type 2 (Type 2) Variants\n"
             f"ALL Recomputed Sequence Features — No Pre-Selection\n"
             f"[{verdict_display}]",
             fontsize=13, fontweight="bold", y=0.998, color=title_color)

fig.savefig(OUTDIR / "figures" / "type2_unsupervised_clustering.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(OUTDIR / "figures" / "type2_unsupervised_clustering.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: type2_unsupervised_clustering.pdf/png")

# ── SAVE STATISTICS ────────────────────────────────────────
# Silhouette table
sil_df = pd.DataFrame([{"k": k, "silhouette": sil[k]} for k in sorted(sil)])
sil_df.to_csv(OUTDIR / "statistics" / "silhouette_by_k.tsv", sep="\t", index=False)

# Gap statistic table
gap_df = pd.DataFrame([{"k": k, "gap": v["gap"], "se": v["se"]}
                        for k, v in gap_scores.items()])
gap_df.to_csv(OUTDIR / "statistics" / "gap_statistic.tsv", sep="\t", index=False)

# Feature importance (k=2 post-hoc)
df_feat.to_csv(OUTDIR / "statistics" / "feature_importance_posthoc_k2.tsv",
               sep="\t", index=False)

# Cluster assignments
assign = type2[["sequence_id", "primary_class", "cluster_k2", "hclust_k2"]].copy()
assign["kmeans_hclust_agree"] = (type2["cluster_k2"] == type2["hclust_k2"])
assign.to_csv(OUTDIR / "statistics" / "cluster_assignments_et.tsv",
              sep="\t", index=False)

# Summary stats
summary_stats = {
    "n_variants": len(type2),
    "n_features": len(usable),
    "n_imputed": n_imputed,
    "sil_k2": sil[2],
    "best_k": best_k_sil,
    "sil_best_k": sil[best_k_sil],
    "gap_verdict": gap_verdict,
    "cophenetic_r": cc,
    "kmeans_hclust_agreement": float((type2["cluster_k2"] == type2["hclust_k2"]).mean()),
    "n_BH_sig_features_k2": n_sig,
    "n_features_tested_k2": len(feat_results),
}
pd.DataFrame([summary_stats]).to_csv(OUTDIR / "statistics" / "summary.tsv",
                                      sep="\t", index=False)

print(f"\n{'=' * 70}")
print("ALL OUTPUTS SAVED")
print(f"{'=' * 70}")
print(f"\nDirectory: {OUTDIR}")
print(f"  figures/type2_unsupervised_clustering.pdf")
print(f"  figures/type2_unsupervised_clustering.png")
print(f"  statistics/silhouette_by_k.tsv")
print(f"  statistics/gap_statistic.tsv")
print(f"  statistics/feature_importance_posthoc_k2.tsv")
print(f"  statistics/cluster_assignments_et.tsv")
print(f"  statistics/summary.tsv")
print()
print("NOTHING overwritten in unsupervised_clustering/ (Type 1 results intact)")
print()
print(f"CONCLUSION: {gap_verdict}")
print(f"  k=2 silhouette = {sil[2]:.3f}")
print(f"  Best k = {best_k_sil} (sil = {sil[best_k_sil]:.3f})")
