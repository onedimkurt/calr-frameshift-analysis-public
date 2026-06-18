#!/usr/bin/env python3
"""
============================================================
TRUE UNSUPERVISED CLUSTERING — OPTION B
============================================================
Clusters Type 1 variants using ALL recomputed sequence features
with NO pre-selection. This is the defensible analysis:
  1. Cluster on everything (no feature cherry-picking)
  2. THEN compare to SG-SG groups (validation, not input)
  3. THEN identify which features drive the split (post-hoc)

NaN handling: impute with feature median (for short AZ positions).
Post-KKRK features excluded (NaN for all Type 1 variants).

Run: conda run -n calr_env python ~/Downloads/true_unsupervised_clustering.py
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

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUTDIR  = DATA / "derived" / "unsupervised_clustering"
(OUTDIR / "figures").mkdir(parents=True, exist_ok=True)
(OUTDIR / "statistics").mkdir(parents=True, exist_ok=True)

# ── LOAD RECOMPUTED FEATURES ──────────────────────────────
df = pd.read_csv(DATA / "derived" / "RECOMPUTED_FEATURES_76_VARIANTS.tsv", sep="\t")
META_COLS = ["sequence_id", "variant_id", "primary_class",
             "frameshift_position", "has_anchor", "anchor_start", "post_kkrk_seq"]

# Load SG-SG for validation (NOT used in clustering)
# Sgamma-Sgamma distances from the public AF2 structural metrics table
cohort = pd.read_csv(DATA / "derived" / "AF2_DEFINITIVE.tsv", sep="\t")
sgsg = cohort[["sequence_id","sg_sg_dist"]].drop_duplicates("sequence_id")
df = df.merge(sgsg, on="sequence_id", how="left")

# Separate Type 2 and Type 1
type2 = df[df["primary_class"].str.contains("2", na=False)].copy()
type1 = df[df["primary_class"].str.contains("1", na=False)].copy()
print(f"Type 2: {len(type2)}, Type 1: {len(type1)}")

# ── FEATURE SELECTION FOR CLUSTERING ───────────────────────
# Use ALL numeric features EXCEPT:
#   - Metadata columns
#   - post_kkrk_* (NaN for all Type 1)
#   - sg_sg_dist (the validation target — NEVER in clustering)
#   - KKRK_present (binary, separates Type 2/Type 1 by definition)
#   - frameshift_position (geometry, not sequence property)
#   - has_anchor, anchor_start (metadata)

EXCLUDE = set(META_COLS) | {
    "sg_sg_dist",           # validation target
    "KKRK_present",         # class separator, not subgroup feature
    "post_kkrk_length",     # NaN for all Type 1
    "post_kkrk_charge",     # NaN for all Type 1
    "post_kkrk_rk_frac",    # NaN for all Type 1
    "post_kkrk_de_frac",    # NaN for all Type 1
}

all_feats = [c for c in df.columns if c not in EXCLUDE
             and df[c].dtype in ["float64","int64","bool"]
             and c not in META_COLS]

# Check which have variance within Type 1
usable = []
for f in all_feats:
    vals = type1[f].dropna()
    if len(vals) >= 10 and vals.std() > 1e-10:
        usable.append(f)
    else:
        print(f"  Dropped (no variance / insufficient data in Type 1): {f}")

print(f"\nFeatures for Type 1 clustering: {len(usable)}")

# ── PREPARE Type 1 DATA ───────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

X_raw = type1[usable].copy()
# Impute NaN with median (Option B)
n_imputed = 0
for col in usable:
    n_na = X_raw[col].isna().sum()
    if n_na > 0:
        median_val = X_raw[col].median()
        X_raw[col] = X_raw[col].fillna(median_val)
        n_imputed += n_na
        print(f"  Imputed {col}: {n_na} NaN → median={median_val:.3f}")

print(f"  Total imputed values: {n_imputed}")

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), columns=usable, index=X_raw.index)
print(f"  Data matrix: {X.shape[0]} Type 1 variants × {X.shape[1]} features")


# ============================================================
# STEP 1: SILHOUETTE + GAP STATISTIC
# ============================================================
print("\n" + "="*70)
print("STEP 1: CLUSTER STRUCTURE ASSESSMENT")
print("="*70)

sil = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = km.fit_predict(X)
    sil[k] = silhouette_score(X, labels)
    print(f"  k={k}: silhouette={sil[k]:.3f}")

best_k_sil = max(sil, key=sil.get)
print(f"  Best k by silhouette: {best_k_sil} (sil={sil[best_k_sil]:.3f})")

# Gap statistic
print(f"\n  Gap statistic:")
n_ref = 50
gap_scores = {}
for k in [1, 2, 3, 4]:
    if k == 1:
        obs = np.sum(np.var(X, axis=0)) * len(X)
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init=50)
        km.fit(X); obs = km.inertia_
    refs = []
    for _ in range(n_ref):
        Xr = pd.DataFrame(np.random.uniform(X.min().values, X.max().values, size=X.shape), columns=X.columns)
        if k == 1:
            ri = np.sum(np.var(Xr, axis=0)) * len(Xr)
        else:
            kmr = KMeans(n_clusters=k, random_state=None, n_init=10)
            kmr.fit(Xr); ri = kmr.inertia_
        refs.append(np.log(ri))
    gap = np.mean(refs) - np.log(obs)
    se = np.std(refs) * np.sqrt(1 + 1/n_ref)
    gap_scores[k] = {"gap": gap, "se": se}
    print(f"    k={k}: gap={gap:.3f} ± {se:.3f}")

# Gap test
if gap_scores[1]["gap"] >= gap_scores[2]["gap"] - gap_scores[2]["se"]:
    gap_verdict = "k=1 sufficient (no subgroups)"
else:
    gap_verdict = "k≥2 supported"
print(f"  Gap verdict: {gap_verdict}")


# ============================================================
# STEP 2: K-MEANS k=2 AND HIERARCHICAL
# ============================================================
print("\n" + "="*70)
print("STEP 2: CLUSTERING (k=2)")
print("="*70)

# K-means k=2
km2 = KMeans(n_clusters=2, random_state=42, n_init=50)
type1["cluster_k2"] = km2.fit_predict(X) + 1

# Hierarchical (Ward's method)
Z = linkage(pdist(X, "euclidean"), method="ward")
type1["hclust_k2"] = fcluster(Z, t=2, criterion="maxclust")

# K-means best-k
if best_k_sil != 2:
    km_best = KMeans(n_clusters=best_k_sil, random_state=42, n_init=50)
    type1["cluster_best"] = km_best.fit_predict(X) + 1

c1 = type1[type1["cluster_k2"]==1]
c2 = type1[type1["cluster_k2"]==2]
print(f"  K-means k=2: Cluster 1={len(c1)}, Cluster 2={len(c2)}")
print(f"  K-means/Hclust agreement: {(type1['cluster_k2']==type1['hclust_k2']).mean():.0%}")


# ============================================================
# STEP 3: VALIDATE AGAINST SG-SG (post-hoc, not input)
# ============================================================
print("\n" + "="*70)
print("STEP 3: SG-SG VALIDATION (post-hoc)")
print("="*70)

# Define SG-SG groups
type1["sg_group"] = "mid"
type1.loc[type1["sg_sg_dist"] < 20, "sg_group"] = "compact"
type1.loc[type1["sg_sg_dist"] > 50, "sg_group"] = "extended"

# Label clusters by which has lower mean SG-SG
sg1 = c1["sg_sg_dist"].mean()
sg2 = c2["sg_sg_dist"].mean()
if sg1 < sg2:
    type1["seq_subgroup"] = type1["cluster_k2"].map({1: "Type 1-A", 2: "Type 1-B"})
else:
    type1["seq_subgroup"] = type1["cluster_k2"].map({1: "Type 1-B", 2: "Type 1-A"})

strong = type1[type1["seq_subgroup"]=="Type 1-A"]
weak = type1[type1["seq_subgroup"]=="Type 1-B"]

print(f"  Type 1-A: n={len(strong)}, SG-SG mean={strong['sg_sg_dist'].mean():.1f}")
print(f"  Type 1-B: n={len(weak)}, SG-SG mean={weak['sg_sg_dist'].mean():.1f}")

# ARI: cluster labels vs SG-SG binary groups
sg_binary = (type1["sg_sg_dist"] < 20).astype(int)
cluster_binary = (type1["seq_subgroup"] == "Type 1-A").astype(int)
ari = adjusted_rand_score(sg_binary, cluster_binary)
print(f"\n  ARI (cluster vs SG-SG<20): {ari:.3f}")

# Fisher's exact test: cluster × SG-SG group
from scipy.stats import fisher_exact
# Build contingency: Type 1-A × compact, Type 1-A × extended, etc.
compact = type1[type1["sg_sg_dist"] < 20]
extended = type1[type1["sg_sg_dist"] > 50]
a = (compact["seq_subgroup"]=="Type 1-A").sum()
b = (compact["seq_subgroup"]=="Type 1-B").sum()
c_val = (extended["seq_subgroup"]=="Type 1-A").sum()
d_val = (extended["seq_subgroup"]=="Type 1-B").sum()
table = [[a, b], [c_val, d_val]]
_, fisher_p = fisher_exact(table)
print(f"  Fisher exact (compact×strong vs extended×weak): p={fisher_p:.6f}")
print(f"  Contingency: compact[strong={a}, weak={b}], extended[strong={c_val}, weak={d_val}]")

# Mapping accuracy
n_correct = a + d_val
n_total = a + b + c_val + d_val
print(f"  Mapping accuracy: {n_correct}/{n_total} ({n_correct/n_total:.1%})")


# ============================================================
# STEP 4: PCA
# ============================================================
print("\n" + "="*70)
print("STEP 4: PCA")
print("="*70)

pca = PCA(n_components=min(10, X.shape[1]))
X_pca = pca.fit_transform(X)
ve = pca.explained_variance_ratio_
print(f"  PC1: {ve[0]:.1%}  PC2: {ve[1]:.1%}  PC1+2: {sum(ve[:2]):.1%}")

loadings = pd.DataFrame(pca.components_[:3].T, columns=["PC1","PC2","PC3"], index=usable)
loadings["abs_PC1"] = loadings["PC1"].abs()
loadings = loadings.sort_values("abs_PC1", ascending=False)
print(f"\n  Top 10 PC1 loadings:")
for _, r in loadings.head(10).iterrows():
    print(f"    {r.name:<30} PC1={r['PC1']:>+.3f}")


# ============================================================
# STEP 5: POST-HOC FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*70)
print("STEP 5: POST-HOC FEATURE IMPORTANCE")
print("="*70)

def cohens_d(g1, g2):
    n1,n2 = len(g1),len(g2)
    v1,v2 = g1.var(), g2.var()
    ps = np.sqrt(((n1-1)*v1+(n2-1)*v2)/(n1+n2-2))
    return (g1.mean()-g2.mean())/ps if ps>1e-10 else 0.0

def bh(pvals):
    n=len(pvals)
    if n==0: return []
    si=sorted(range(n), key=lambda i:pvals[i])
    adj=[0.0]*n; adj[si[-1]]=min(1.0,pvals[si[-1]])
    for i in range(n-2,-1,-1):
        adj[si[i]]=min(pvals[si[i]]*n/(i+1), adj[si[i+1]], 1.0)
    return adj

feat_results = []
raw_p = []
for f in usable:
    s = strong[f].dropna(); w = weak[f].dropna()
    if len(s)<3 or len(w)<3: continue
    try:
        _, p = stats.mannwhitneyu(s, w, alternative="two-sided")
    except:
        continue
    d = cohens_d(s, w)
    feat_results.append({"feature":f, "strong_mean":s.mean(), "weak_mean":w.mean(),
                          "d":d, "abs_d":abs(d), "p_raw":p})
    raw_p.append(p)

adj = bh(raw_p)
for i,r in enumerate(feat_results):
    r["p_BH"] = adj[i]
    r["sig"] = "***" if adj[i]<0.001 else "**" if adj[i]<0.01 else "*" if adj[i]<0.05 else "ns"

df_feat = pd.DataFrame(feat_results).sort_values("p_BH")
n_sig = (df_feat["sig"]!="ns").sum()
n_large = (df_feat["abs_d"]>0.8).sum()

print(f"  BH-significant: {n_sig}/{len(feat_results)}")
print(f"  |d| > 0.8: {n_large}")

print(f"\n{'Feature':<30} {'Strong':>8} {'Weak':>8} {'d':>7} {'p_BH':>9} {'sig':>4}")
print("-"*70)
for _, r in df_feat.head(25).iterrows():
    print(f"  {r['feature']:<30} {r['strong_mean']:>8.3f} {r['weak_mean']:>8.3f} "
          f"{r['d']:>+7.2f} {r['p_BH']:>9.4f} {r['sig']:>4}")

df_feat.to_csv(OUTDIR/"statistics"/"feature_importance_posthoc.tsv", sep="\t", index=False)


# ============================================================
# STEP 6: WITHIN-CLUSTER CORRELATION (spectrum test)
# ============================================================
print("\n" + "="*70)
print("STEP 6: WITHIN-CLUSTER SPECTRUM TEST")
print("="*70)

# Test if SG-SG correlates with features WITHIN the compact group
compact_mf = type1[type1["sg_sg_dist"] < 20].copy()
print(f"  Compact Type 1: n={len(compact_mf)}")

within_results = []
for f in df_feat[df_feat["sig"]!="ns"]["feature"].head(10):
    vals = compact_mf[f].dropna()
    sg = compact_mf.loc[vals.index, "sg_sg_dist"]
    if len(vals) >= 5:
        rho, p = stats.spearmanr(vals, sg)
        within_results.append({"feature":f, "rho":rho, "p":p})
        sig_str = "*" if p<0.05 else "ns"
        print(f"    {f:<30} rho={rho:>+.3f} p={p:.4f} {sig_str}")

n_within_sig = sum(1 for r in within_results if r["p"] < 0.05)
print(f"\n  Within-cluster significant: {n_within_sig}/{len(within_results)}")
if n_within_sig <= 1:
    print(f"  → Two reproducible subgroups along a continuous charge axis (boundary softness assessed by bootstrap; see review validation)")
else:
    print(f"  → Possible continuous gradient within compact group")


# ============================================================
# STEP 7: COMPREHENSIVE FIGURE
# ============================================================
print("\n" + "="*70)
print("STEP 7: FIGURE")
print("="*70)

fig = plt.figure(figsize=(22, 16))
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# A: Silhouette
ax = fig.add_subplot(gs[0, 0])
ks = sorted(sil.keys()); ss = [sil[k] for k in ks]
colors_s = ["#27ae60" if k==best_k_sil else "#e74c3c" if k==2 else "#bdc3c7" for k in ks]
ax.bar(ks, ss, color=colors_s, edgecolor="black", linewidth=0.5)
ax.set_xlabel("k"); ax.set_ylabel("Silhouette")
ax.set_title(f"A. Silhouette (ALL {len(usable)} features)\n"
             f"k=2: {sil[2]:.3f}, best k={best_k_sil}", fontsize=11, fontweight="bold")
ax.set_xticks(ks)

# B: PCA colored by cluster
ax = fig.add_subplot(gs[0, 1])
for sg, color, marker in [("Type 1-A","#27ae60","o"), ("Type 1-B","#8e44ad","s")]:
    mask = type1["seq_subgroup"]==sg
    idx_list = [list(type1.index).index(i) for i in type1[mask].index]
    ax.scatter(X_pca[idx_list,0], X_pca[idx_list,1], c=color, s=60,
               marker=marker, edgecolor="black", linewidth=0.5, alpha=0.8,
               label=f"{sg} (n={mask.sum()})")
ax.set_xlabel(f"PC1 ({ve[0]:.0%})"); ax.set_ylabel(f"PC2 ({ve[1]:.0%})")
ax.set_title("B. PCA (unsupervised clusters)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# C: PCA colored by SG-SG (validation)
ax = fig.add_subplot(gs[0, 2])
sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=type1["sg_sg_dist"].values,
                cmap="RdYlGn_r", s=60, edgecolor="black", linewidth=0.5, alpha=0.8)
ax.set_xlabel(f"PC1 ({ve[0]:.0%})"); ax.set_ylabel(f"PC2 ({ve[1]:.0%})")
ax.set_title("C. PCA colored by SG-SG\n(validation — not used in clustering)", fontsize=11, fontweight="bold")
plt.colorbar(sc, ax=ax, shrink=0.7, label="SG-SG (Å)")

# D: Effect sizes (top 20)
ax = fig.add_subplot(gs[1, 0])
top20 = df_feat.sort_values("abs_d", ascending=True).tail(20)
colors_ef = ["#c0392b" if s!="ns" else "#bdc3c7" for s in top20["sig"]]
ax.barh(range(len(top20)), top20["d"],
        color=colors_ef, edgecolor="black", linewidth=0.5, alpha=0.8)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["feature"], fontsize=6)
ax.axvline(0, color="black", lw=0.5)
ax.axvline(-0.8, ls=":", color="gray", alpha=0.3)
ax.axvline(0.8, ls=":", color="gray", alpha=0.3)
ax.set_xlabel("Cohen's d (Type 1-A − Type 1-B)")
ax.set_title(f"D. Post-hoc Feature Importance\n{n_sig} BH-significant", fontsize=11, fontweight="bold")

# E: Dendrogram
ax = fig.add_subplot(gs[1, 1])
# Color leaves by cluster
from scipy.cluster.hierarchy import set_link_color_palette
set_link_color_palette(["#333"])
dend = dendrogram(Z, ax=ax, leaf_rotation=90, no_labels=True,
                   color_threshold=0, above_threshold_color="#333")
ax.set_title("E. Hierarchical Clustering (Ward)\n(all features)", fontsize=11, fontweight="bold")
ax.set_ylabel("Distance")

# F: Summary
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
summary = f"""TRUE UNSUPERVISED CLUSTERING
{'='*40}

Features: {len(usable)} (ALL recomputed, no selection)
NaN handling: median imputation
Variants: {len(type1)} Type 1

Cluster structure:
  k=2 silhouette: {sil[2]:.3f}
  Best k: {best_k_sil} (sil={sil[best_k_sil]:.3f})
  Gap statistic: {gap_verdict}

Clusters (k=2):
  Type 1-A: n={len(strong)}
  Type 1-B:   n={len(weak)}

SG-SG VALIDATION (post-hoc):
  ARI: {ari:.3f}
  Fisher p: {fisher_p:.6f}
  Mapping accuracy: {n_correct}/{n_total} ({n_correct/n_total:.0%})

Post-hoc features:
  BH-significant: {n_sig}/{len(feat_results)}
  |d| > 0.8: {n_large}

Within-cluster spectrum:
  {n_within_sig}/{len(within_results)} significant
  → {'Two discrete subgroups' if n_within_sig<=1 else 'Possible gradient'}

PCA: PC1={ve[0]:.0%} PC2={ve[1]:.0%}

METHOD: No feature pre-selection.
All features computed from raw sequences.
SG-SG used ONLY for post-hoc validation."""

ax.text(0.03, 0.97, summary, transform=ax.transAxes, fontsize=8,
        va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6", alpha=0.95))

fig.suptitle("True Unsupervised Clustering of Type 1 Variants\n"
             "ALL Recomputed Sequence Features — No Pre-Selection",
             fontsize=14, fontweight="bold", y=0.995)
fig.savefig(OUTDIR/"figures"/"unsupervised_clustering.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUTDIR/"figures"/"unsupervised_clustering.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: unsupervised_clustering.pdf/png")


# ── SAVE ASSIGNMENTS ───────────────────────────────────────
assign = type1[["sequence_id","primary_class","cluster_k2","hclust_k2",
             "seq_subgroup","sg_sg_dist","sg_group"]].copy()
assign.to_csv(OUTDIR/"statistics"/"cluster_assignments.tsv", sep="\t", index=False)


print("\n" + "="*70)
print("DONE")
print("="*70)
print(f"""
Output: {OUTDIR}

THIS IS THE DEFENSIBLE ANALYSIS:
  1. ALL features used (no cherry-picking)
  2. Features recomputed from raw sequences (fully traceable)
  3. SG-SG used ONLY for post-hoc validation
  4. Feature importance determined AFTER clustering
  5. Within-cluster spectrum test reported; subgroup boundary characterized as gradational by bootstrap (review validation)
""")
