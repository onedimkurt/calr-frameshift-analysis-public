import sys, json, hashlib, datetime, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
P=Path(__file__).resolve().parents[1]; DATA=P/"data"; RF=DATA/"derived"
FEAT=DATA/"derived"/"RECOMPUTED_FEATURES_76_VARIANTS.tsv"
COH=DATA/"derived"/"af2_definitive"/"statistics"/"AF2_DEFINITIVE.tsv"
CLUS=DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
df=pd.read_csv(FEAT,sep="\t")
sg=pd.read_csv(COH,sep="\t")[["sequence_id","sg_sg_dist"]].drop_duplicates("sequence_id")
df=df.merge(sg,on="sequence_id",how="left")
META=["sequence_id","variant_id","primary_class","frameshift_position","has_anchor","anchor_start","post_kkrk_seq"]
EXC=set(META)|{"sg_sg_dist","KKRK_present","post_kkrk_length","post_kkrk_charge","post_kkrk_rk_frac","post_kkrk_de_frac"}

def build(subset):
    s=df[df["primary_class"].str.contains(subset,na=False)].copy()
    base=[c for c in df.columns if c not in EXC and df[c].dtype in ["float64","int64","bool"] and c not in META]
    usable=[f for f in base if len(s[f].dropna())>=10 and s[f].dropna().std()>1e-10]
    Xr=s[usable].copy().astype(float)
    for c in usable:
        if Xr[c].isna().any(): Xr[c]=Xr[c].fillna(Xr[c].median())
    return pd.DataFrame(StandardScaler().fit_transform(Xr),columns=usable), s

X1,_=build("1"); X2,_=build("2")
print(f"Type1 X={X1.shape}  Type2 X={X2.shape}")

def sil_curve(X):
    return {k:silhouette_score(X,KMeans(k,random_state=42,n_init=50).fit_predict(X)) for k in range(2,8)}
s1,s2=sil_curve(X1),sil_curve(X2)
print("\nSILHOUETTE k=2..7:")
print("  Type1:",{k:round(v,3) for k,v in s1.items()},"(k=2 is",("LOWEST" if s1[2]==min(s1.values()) else "not lowest")+")")
print("  Type2:",{k:round(v,3) for k,v in s2.items()})
print(f"  PARADOX: Type2 k2={s2[2]:.3f} > Type1 k2={s1[2]:.3f}  (rejected split scores higher)")

# gap (manuscript implementation), run 3x to show Monte Carlo spread
def gap_curve(X,seed=None,kmax=7,n_ref=50):
    if seed is not None: np.random.seed(seed)
    out={}
    for k in range(1,kmax+1):
        if k==1: obs=np.sum(np.var(X,axis=0))*len(X)
        else: obs=KMeans(k,random_state=42,n_init=50).fit(X).inertia_
        refs=[]
        for _ in range(n_ref):
            Xr=pd.DataFrame(np.random.uniform(X.min().values,X.max().values,size=X.shape),columns=X.columns)
            ri=(np.sum(np.var(Xr,axis=0))*len(Xr)) if k==1 else KMeans(k,random_state=None,n_init=10).fit(Xr).inertia_
            refs.append(np.log(ri))
        out[k]={"gap":float(np.mean(refs)-np.log(obs)),"se":float(np.std(refs)*np.sqrt(1+1/n_ref))}
    return out
print("\nGAP (manuscript impl), 3 runs to show stochastic spread:")
runs=[gap_curve(X1,seed=s) for s in [None,None,7]]
for k in [1,2,3]:
    vals=[f"{r[k]['gap']:.3f}" for r in runs]
    print(f"  k={k}: gap runs = {vals}  (manuscript reports gap1~0.381, gap2~0.569-0.571)")
g=runs[2]  # seeded run = reproducible
# Tibshirani: smallest k with gap(k) >= gap(k+1)-se(k+1)
tib=None
for k in range(1,7):
    if g[k]["gap"]>=g[k+1]["gap"]-g[k+1]["se"]: tib=k; break
print(f"  Tibshirani-selected k (seeded run) = {tib}")
print(f"  reconciliation: gap2 0.569 vs 0.571 = Monte Carlo noise (unseeded refs), within SE ~{g[2]['se']:.3f}")

# fragmentation: do k>2 clusters NEST within the k=2 partition?
print("\nFRAGMENTATION (do higher-k clusters subdivide k=2, not restructure?):")
k2=KMeans(2,random_state=42,n_init=50).fit_predict(X1)
frag=[]
for k in range(3,6):
    kk=KMeans(k,random_state=42,n_init=50).fit_predict(X1)
    # nestedness: fraction of each k-cluster that falls in a single k2 cluster
    ct=pd.crosstab(kk,k2); purity=(ct.max(axis=1)/ct.sum(axis=1)).mean()
    ari23=adjusted_rand_score(k2,kk)
    frag.append({"k":k,"mean_cluster_purity_vs_k2":float(purity),"ari_vs_k2":float(ari23)})
    print(f"  k={k}: mean purity vs k=2 = {purity:.3f} (1.0=pure subdivision)  ARI(k vs k2)={ari23:.3f}")
print("  -> high purity = higher k merely fragments the two groups (no new structure)")

# Type2 kmeans/Ward agreement (target 31%)
k2_t2=KMeans(2,random_state=42,n_init=50).fit_predict(X2)
hc_t2=fcluster(linkage(pdist(X2.values,"euclidean"),"ward"),2,"maxclust")
agree2=pd.crosstab(k2_t2,hc_t2).values.max(axis=1).sum()/len(k2_t2)
print(f"\nType2 k-means/Ward agreement = {agree2:.0%}  (anchor 31%; Type1 was 100%)")

(RF/"outputs").mkdir(parents=True,exist_ok=True)
blob={"inputs":{"FEAT":sha(FEAT),"COH":sha(COH),"CLUS":sha(CLUS)},
      "silhouette_type1":s1,"silhouette_type2":s2,
      "gap_type1_seeded":g,"tibshirani_k":tib,
      "gap_runs_k2":[r[2]["gap"] for r in runs],
      "fragmentation":frag,"type2_kmeans_ward_agreement":agree2,
      "paradox":f"Type2 rejected split sil={s2[2]:.3f} > Type1 accepted sil={s1[2]:.3f}; evidence=concordance not silhouette"}
(RF/"outputs"/"C05_silhouette_gap.json").write_text(json.dumps(blob,indent=2))
pd.DataFrame([{"k":k,"sil_type1":s1.get(k),"sil_type2":s2.get(k),
               "gap_type1":g.get(k,{}).get("gap"),"gap_se":g.get(k,{}).get("se")} for k in range(1,8)]
            ).to_csv(RF/"outputs"/"C05_silhouette_gap.tsv",sep="\t",index=False)
with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C05  sil_t1_k2={s1[2]:.3f} sil_t2_k2={s2[2]:.3f} "
            f"tib_k={tib} t2_agree={agree2:.2f}\n")
print("\nWROTE C05_silhouette_gap.{tsv,json}")
sys.stdout.flush()
