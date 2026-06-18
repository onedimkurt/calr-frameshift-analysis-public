import sys, hashlib
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

P = str(Path(__file__).resolve().parents[1])
FEAT = f"{P}/data/derived/RECOMPUTED_FEATURES_76_VARIANTS.tsv"
COH  = f"{P}/data/derived/af2_definitive/statistics/AF2_DEFINITIVE.tsv"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

print("INPUT SHA256:")
print("  FEAT:", sha(FEAT))
print("  COH :", sha(COH))

# ---- replicate original load + merge ----
df = pd.read_csv(FEAT, sep="\t")
cohort = pd.read_csv(COH, sep="\t")
sgsg = cohort[["sequence_id","sg_sg_dist"]].drop_duplicates("sequence_id")
df = df.merge(sgsg, on="sequence_id", how="left")

META_COLS = ["sequence_id","variant_id","primary_class",
             "frameshift_position","has_anchor","anchor_start","post_kkrk_seq"]
type1 = df[df["primary_class"].str.contains("1", na=False)].copy()
print(f"\nType 1 subset (Type 1-like): n={len(type1)}  (anchor: 44)")

EXCLUDE = set(META_COLS) | {"sg_sg_dist","KKRK_present",
    "post_kkrk_length","post_kkrk_charge","post_kkrk_rk_frac","post_kkrk_de_frac"}
all_feats = [c for c in df.columns if c not in EXCLUDE
             and df[c].dtype in ["float64","int64","bool"]
             and c not in META_COLS]
usable=[]
for f in all_feats:
    vals = type1[f].dropna()
    if len(vals) >= 10 and vals.std() > 1e-10:
        usable.append(f)
    else:
        print(f"  Dropped (no var/insufficient in Type 1): {f}")
print(f"\nFeatures for Type 1 clustering: {len(usable)}  (anchor: ~87)")

X_raw = type1[usable].copy().astype(float)
n_imputed=0
for col in usable:
    n_na = int(X_raw[col].isna().sum())
    if n_na>0:
        mv = X_raw[col].median()
        X_raw[col] = X_raw[col].fillna(mv)
        n_imputed += n_na
        print(f"  Imputed {col}: {n_na} NaN -> median={mv:.3f}")
print(f"  Total imputed values: {n_imputed}")

X = pd.DataFrame(StandardScaler().fit_transform(X_raw), columns=usable, index=X_raw.index)
print(f"  Data matrix: {X.shape[0]} x {X.shape[1]}")

# ---- silhouette k=2..7 ----
print("\nSILHOUETTE k=2..7 (anchor k=2=0.213, lowest across range):")
sil={}
for k in range(2,8):
    lab = KMeans(n_clusters=k, random_state=42, n_init=50).fit_predict(X)
    sil[k]=silhouette_score(X,lab)
    print(f"  k={k}: silhouette={sil[k]:.3f}")
flag = "PASS" if abs(sil[2]-0.213)<0.005 else "FLAG"
print(f"  -> k=2 = {sil[2]:.3f}  vs anchor 0.213  [{flag}]")

# ---- kmeans vs ward agreement at k=2 ----
km2 = KMeans(n_clusters=2, random_state=42, n_init=50).fit_predict(X)
Z = linkage(pdist(X.values,"euclidean"), method="ward")
hc2 = fcluster(Z, t=2, criterion="maxclust")
agree = max((km2==hc2).mean(), (km2==(3-hc2)).mean())  # label-invariant
print(f"\nK-means/Ward agreement @k=2: {agree:.0%}  (anchor 100%)  [{'PASS' if agree>=0.999 else 'FLAG'}]")
ari_kw = adjusted_rand_score(km2,hc2)
print(f"ARI(kmeans,ward) @k=2: {ari_kw:.3f}  (anchor 1.000)")

# ---- subgroup counts ----
type1=type1.copy(); type1["cluster_k2"]=km2
vc = pd.Series(km2).value_counts().sort_index().tolist()
print(f"k=2 cluster sizes: {sorted(vc, reverse=True)}  (anchor 25/19)")

print("\nNOTE: sensitivity ARIs (1.000/1.000/0.662) require the original's "
      "3 perturbations; will reproduce in 3b once recipe is confirmed.")
sys.stdout.flush()
