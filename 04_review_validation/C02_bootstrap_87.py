import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
P=Path(__file__).resolve().parents[1]; DATA=P/"data"
FEAT=DATA/"derived"/"RECOMPUTED_FEATURES_76_VARIANTS.tsv"
CLUS=DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
df=pd.read_csv(FEAT,sep="\t"); df["sid"]=df["sequence_id"].astype(str).str.strip()
ref=pd.read_csv(CLUS,sep="\t"); ref["sid"]=ref["sequence_id"].astype(str).str.strip()
t1=df[df["sid"].isin(ref["sid"])].copy()
# EXACT exclusion: metadata + structural target + cys, AND the two positional metadata columns
META_DROP={"frameshift_position","anchor_start"}
drop=[c for c in t1.columns if c.lower() in ("sequence_id","sid","primary_class","type","seq_subgroup")
      or "sg_sg" in c.lower() or "sgamma" in c.lower() or c.lower().startswith("cys_")
      or c in META_DROP]
num=t1.select_dtypes(include=[np.number]).columns.tolist()
keep=[c for c in num if c not in drop and t1[c].notna().sum()>=10 and t1[c].std(skipna=True)>0]
print(f"feature count: {len(keep)} (expect 87)")
assert len(keep)==87, f"GOT {len(keep)} NOT 87 — STOP, do not trust bootstrap"
X_raw=t1[keep].copy(); X_raw=X_raw.fillna(X_raw.median())
X=StandardScaler().fit_transform(X_raw)
ref_labels=KMeans(2,n_init=50,random_state=42).fit(X).labels_
print(f"partition sizes: {np.bincount(ref_labels)} (expect 25/19)")

B=1000; rng=np.random.default_rng(42); n=X.shape[0]; jac={0:[],1:[]}
for b in range(B):
    idx=rng.choice(n,n,replace=True); Xb=X[idx]
    labb=KMeans(2,n_init=10,random_state=int(rng.integers(1e9))).fit(Xb).labels_
    for c in (0,1):
        orig=set(np.where(ref_labels==c)[0]); best=0.0
        for cb in (0,1):
            boot=set(idx[labb==cb]); u=len(orig|boot)
            if u>0: best=max(best,len(orig&boot)/u)
        jac[c].append(best)
print("\n===== C02 BOOTSTRAP (87 features, B=1000, per-cluster Jaccard) =====")
for c in (0,1):
    arr=np.array(jac[c])
    print(f"  cluster {c} (n={(ref_labels==c).sum()}): mean Jaccard={arr.mean():.3f}, "
          f"median={np.median(arr):.3f}, frac>=0.75={np.mean(arr>=0.75):.2f}, frac>=0.6={np.mean(arr>=0.6):.2f}")
# also overall ARI stability: mean ARI of bootstrap partition vs reference
aris=[]
for b in range(200):
    idx=rng.choice(n,n,replace=True)
    lb=KMeans(2,n_init=10,random_state=int(rng.integers(1e9))).fit(X[idx]).labels_
    # map back: ARI on the sampled points
    aris.append(adjusted_rand_score(ref_labels[idx],lb))
print(f"  mean bootstrap ARI (partition reproducibility) = {np.mean(aris):.3f} ± {np.std(aris):.3f}")
sys.stdout.flush()
