import json, hashlib, datetime, sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_auc_score
from scipy import stats

P=Path(__file__).resolve().parents[1]; DATA=P/"data"; RF=DATA/"derived"
FEAT=DATA/"derived"/"RECOMPUTED_FEATURES_76_VARIANTS.tsv"
COH=DATA/"derived"/"af2_definitive"/"statistics"/"AF2_DEFINITIVE.tsv"
CLUS=DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
SEED=42
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
df=pd.read_csv(FEAT,sep="\t")
sg=pd.read_csv(COH,sep="\t")[["sequence_id","sg_sg_dist"]].drop_duplicates("sequence_id")
df=df.merge(sg,on="sequence_id",how="left")
ref=pd.read_csv(CLUS,sep="\t"); rmap=dict(zip(ref["sequence_id"].str.strip(),ref["seq_subgroup"].str.strip()))
META=["sequence_id","variant_id","primary_class","frameshift_position","has_anchor","anchor_start","post_kkrk_seq"]
EXC=set(META)|{"sg_sg_dist","KKRK_present","post_kkrk_length","post_kkrk_charge","post_kkrk_rk_frac","post_kkrk_de_frac"}
type1=df[df["primary_class"].str.contains("1",na=False)].copy().reset_index(drop=True)
feats=[c for c in df.columns if c not in EXC and df[c].dtype in ["float64","int64","bool"] and c not in META]
usable=[f for f in feats if len(type1[f].dropna())>=10 and type1[f].dropna().std()>1e-10]
Xr=type1[usable].copy().astype(float)
for c in usable:
    if Xr[c].isna().any(): Xr[c]=Xr[c].fillna(Xr[c].median())
X=StandardScaler().fit_transform(Xr)
type1["sid"]=type1["sequence_id"].str.strip(); type1["ref"]=type1["sid"].map(rmap)
ref_lab=np.array([{"Type 1-A":0,"Type 1-B":1}[g] for g in type1["ref"]])

# anchor self-check
km=KMeans(2,random_state=SEED,n_init=50).fit(X); sil2=silhouette_score(X,km.labels_)
print(f"ANCHOR sil@k2={sil2:.3f} (exp 0.213)  sizes={sorted(pd.Series(km.labels_).value_counts().tolist(),reverse=True)}")
if abs(sil2-0.213)>0.005: print("!! drift, abort"); sys.exit(1)

# ---- C06: PCA ----
pca=PCA(random_state=SEED).fit(X)
ve=pca.explained_variance_ratio_
cum=np.cumsum(ve)
n80=int(np.argmax(cum>=0.80))+1
print("\nC06 PCA explained variance:")
print(f"  PC1={ve[0]:.1%} (exp 29.6%)  PC2={ve[1]:.1%} (exp 15.6%)  cum2={cum[1]:.1%} (exp 45.2%)")
print(f"  #PCs for >=80% = {n80} (exp 9)")
Xp=pca.transform(X)
# which PC separates clusters? per-PC t-stat + AUC of ref labels
print("\nC06 per-PC cluster separation (1-A vs 1-B):")
sep=[]
for i in range(min(5,Xp.shape[1])):
    a=Xp[ref_lab==0,i]; b=Xp[ref_lab==1,i]
    t,p=stats.ttest_ind(a,b,equal_var=False)
    auc=roc_auc_score(ref_lab, Xp[:,i]); auc=max(auc,1-auc)
    sep.append({"PC":i+1,"explained_var":float(ve[i]),"t":float(t),"p":float(p),"auc":float(auc)})
    print(f"  PC{i+1}: EV={ve[i]:.1%}  t={t:+.2f} p={p:.2e}  AUC={auc:.3f}")
top=max(sep,key=lambda r:r['auc'])
print(f"  -> clusters separate best on PC{top['PC']} (AUC={top['auc']:.3f})  [anchor: PC2]")

# ---- C03: clustering in reduced PC space (p>>n remedy) ----
print("\nC03 clustering in reduced PC space vs full-feature reference:")
res=[]
for npc in [2,3,5,9,n80]:
    Xpc=Xp[:,:npc]
    lab=KMeans(2,random_state=SEED,n_init=50).fit_predict(Xpc)
    ari=adjusted_rand_score(ref_lab,lab)
    sil=silhouette_score(Xpc,lab)
    res.append({"n_pcs":int(npc),"ari_vs_ref":float(ari),"silhouette":float(sil)})
    print(f"  {npc:2d} PCs: ARI vs ref={ari:.3f}  silhouette={sil:.3f}")

blob={"fix":"C03_C06","seed":SEED,"inputs":{"FEAT":sha(FEAT),"COH":sha(COH),"CLUS":sha(CLUS)},
      "n_features":len(usable),"silhouette_k2":sil2,
      "pca":{"PC1":float(ve[0]),"PC2":float(ve[1]),"cum2":float(cum[1]),"n_pcs_80":n80,
             "explained_variance_first10":[float(x) for x in ve[:10]]},
      "per_pc_separation":sep,"best_separating_pc":top["PC"],
      "pc_space_clustering":res}
(RF/"outputs").mkdir(parents=True,exist_ok=True)
pd.DataFrame(res).to_csv(RF/"outputs"/"C03_pc_space_clustering.tsv",sep="\t",index=False)
pd.DataFrame(sep).to_csv(RF/"outputs"/"C06_pca_separation.tsv",sep="\t",index=False)
(RF/"outputs"/"C03_C06_pca.json").write_text(json.dumps(blob,indent=2))
with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C03_C06  PC1={ve[0]:.3f} PC2={ve[1]:.3f} "
            f"cum2={cum[1]:.3f} n80={n80} bestPC={top['PC']}\n")
print("\nWROTE C03_pc_space_clustering.tsv, C06_pca_separation.tsv, C03_C06_pca.json")
sys.stdout.flush()
