import sys, hashlib, datetime, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.stats import mannwhitneyu
P=Path(__file__).resolve().parents[1]; DATA=P/"data"; RF=DATA/"derived"
FEAT=DATA/"derived"/"RECOMPUTED_FEATURES_76_VARIANTS.tsv"
CLUS=DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()[:12]
print("INPUT feature matrix sha:",sha(FEAT))
print("INPUT cluster assignments sha:",sha(CLUS))

df=pd.read_csv(FEAT,sep="\t"); df["sid"]=df["sequence_id"].astype(str).str.strip()
ref=pd.read_csv(CLUS,sep="\t"); ref["sid"]=ref["sequence_id"].astype(str).str.strip()
sub=dict(zip(ref["sid"],ref["seq_subgroup"].astype(str).str.strip()))
t1=df[df["sid"].isin(ref["sid"])].copy()
print(f"\nType-1-like variants: {len(t1)} (expect 44)")
drop=[c for c in t1.columns if c.lower() in ("sequence_id","sid","primary_class","type","seq_subgroup")
      or "sg_sg" in c.lower() or "sgamma" in c.lower() or c.lower().startswith("cys_")]
num=t1.select_dtypes(include=[np.number]).columns.tolist()
feats=[c for c in num if c not in drop]
keep=[c for c in feats if t1[c].notna().sum()>=10 and t1[c].std(skipna=True)>0]
X_raw=t1[keep].copy(); X_raw=X_raw.fillna(X_raw.median())
X=StandardScaler().fit_transform(X_raw)
print(f"feature matrix for clustering: {X.shape} (expect ~44 x 87; got {X.shape[1]} features)")
km0=KMeans(2,n_init=50,random_state=42).fit(X); ref_labels=km0.labels_
print(f"reference partition sizes: {np.bincount(ref_labels)} (expect 25/19)")

B=1000; rng=np.random.default_rng(42); n=X.shape[0]
jac={0:[],1:[]}
for b in range(B):
    idx=rng.choice(n,n,replace=True); Xb=X[idx]
    labb=KMeans(2,n_init=10,random_state=int(rng.integers(1e9))).fit(Xb).labels_
    for c in (0,1):
        orig=set(np.where(ref_labels==c)[0]); best=0.0
        for cb in (0,1):
            boot=set(idx[labb==cb]); inter=len(orig&boot); union=len(orig|boot)
            if union>0: best=max(best,inter/union)
        jac[c].append(best)
print("\n===== C02 BOOTSTRAP STABILITY (B=1000, per-cluster Jaccard) =====")
for c in (0,1):
    arr=np.array(jac[c]); size=(ref_labels==c).sum()
    print(f"  cluster {c} (n={size}): mean Jaccard={arr.mean():.3f}, median={np.median(arr):.3f}, "
          f"frac>=0.75={np.mean(arr>=0.75):.2f}, frac>=0.6={np.mean(arr>=0.6):.2f}")
print("  interpretation: >=0.75 stable; 0.6-0.75 patterns/doubtful; <0.6 dissolved")

print("\n===== C03 PC-SPACE CLUSTERING (address p>>n) =====")
pca=PCA().fit(X); cum=np.cumsum(pca.explained_variance_ratio_)
print(f"  PC1={pca.explained_variance_ratio_[0]:.3f} PC2={pca.explained_variance_ratio_[1]:.3f} cum2={cum[1]:.3f}; PCs for 80%={np.argmax(cum>=0.8)+1}")
for kpc in (2,3,5,10):
    Xp=PCA(n_components=kpc,random_state=42).fit_transform(X)
    lp=KMeans(2,n_init=50,random_state=42).fit(Xp).labels_
    print(f"  cluster on first {kpc} PCs: ARI vs full-feature partition = {adjusted_rand_score(ref_labels,lp):.3f}")

print("\n===== FCR p re-verification (corrected 0.504 / 0.487) =====")
if "fcr" in df.columns:
    g={}
    for s,lab in zip(t1["sid"],ref_labels):
        g.setdefault(lab,[]).append(t1.loc[t1["sid"]==s,"fcr"].values[0])
    a=np.array(g[0]); b=np.array(g[1]); hi,lo=(a,b) if a.mean()>b.mean() else (b,a)
    u,p=mannwhitneyu(hi,lo,alternative="two-sided")
    print(f"  1-A FCR={hi.mean():.3f}±{hi.std(ddof=1):.3f}, 1-B FCR={lo.mean():.3f}±{lo.std(ddof=1):.3f}")
    print(f"  Mann-Whitney U={u:.0f}, p={p:.3e}  (manuscript claims p<0.001)")
else:
    print("  'fcr' col not found:", [c for c in df.columns if 'fcr' in c.lower()])

with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C02/C03/FCR-recheck run\n")
sys.stdout.flush()
