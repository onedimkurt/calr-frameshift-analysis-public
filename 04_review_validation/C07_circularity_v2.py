import sys, json, hashlib, datetime, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
P=Path(__file__).resolve().parents[1]; DATA=P/"data"; RF=DATA/"derived"
S1=DATA/"sequences"/"Table_S1_Sequences_76variants.tsv"
CLUS=DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
ANCHOR="RRMMRTKMRMRRMRRTRRKMRR"; TAIL_START=360; SEED=42
AAS="ACDEFGHIKLMNPQRSTVWY"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
d=pd.read_csv(S1,sep="\t")
ref=pd.read_csv(CLUS,sep="\t"); rmap=dict(zip(ref["sequence_id"].astype(str).str.strip(),
                                              ref["seq_subgroup"].astype(str).str.strip()))
d["sid"]=d["variant_id"].astype(str)+"|Type_1-like"
t1=d[d["type"].astype(str).str.contains("1",na=False)].copy().reset_index(drop=True)
def az(seq,fs):
    nf=seq[max(TAIL_START,int(fs)):]
    return nf[:nf.index(ANCHOR)]
t1["az"]=t1.apply(lambda r:az(str(r["full_sequence"]),r["frameshift_position"]),axis=1)
t1["sub"]=t1["sid"].map(rmap)
ref_lab=np.array([{"Type 1-A":0,"Type 1-B":1}[g] for g in t1["sub"]])

# featurizations (all UPSTREAM of charge/grantham)
comp=np.array([[s.count(a)/len(s) for a in AAS] for s in t1["az"]])      # 20 aa-fractions
length=np.array([[len(s)] for s in t1["az"]]).astype(float)             # az length
comp_len=np.hstack([comp,length])                                       # composition + length

def evaluate(name,M):
    X=StandardScaler().fit_transform(M)
    km=KMeans(2,random_state=SEED,n_init=50).fit_predict(X)
    wl=AgglomerativeClustering(2,linkage="ward").fit_predict(X)
    ari_km=adjusted_rand_score(ref_lab,km); ari_w=adjusted_rand_score(ref_lab,wl)
    sz=sorted(pd.Series(km).value_counts().tolist(),reverse=True)
    # supervised separability of reference labels (does the signal EXIST in this space)
    auc=cross_val_score(LogisticRegression(max_iter=5000),X,ref_lab,cv=5,scoring="roc_auc").mean() if M.shape[1]>1 else np.nan
    return {"featurization":name,"n_dim":M.shape[1],"ari_kmeans":float(ari_km),
            "ari_ward":float(ari_w),"kmeans_sizes":sz,"cv_auc_ref":float(auc) if auc==auc else None}

print("C07 (corrected): recover 25/19 split from UPSTREAM az features?")
print("reference: az_charge-based clustering -> Type 1-A=25 / 1-B=19\n")
res=[]
for name,M in [("az_raw_composition(20aa)",comp),
               ("az_length_only",length),
               ("az_composition+length",comp_len)]:
    r=evaluate(name,M); res.append(r)
    verdict=("RECOVERS (not circular on charge)" if r["ari_kmeans"]>=0.8 or r["ari_ward"]>=0.8
             else "partial" if max(r["ari_kmeans"],r["ari_ward"])>=0.5 else "does NOT recover")
    auc=f"{r['cv_auc_ref']:.3f}" if r["cv_auc_ref"] else "n/a"
    print(f"  {name:28s} dim={r['n_dim']:2d}  ARI_km={r['ari_kmeans']:+.3f}  ARI_ward={r['ari_ward']:+.3f}  "
          f"sizes={r['kmeans_sizes']}  CV-AUC={auc}  -> {verdict}")

(RF/"outputs").mkdir(parents=True,exist_ok=True)
pd.DataFrame(res).to_csv(RF/"outputs"/"C07_circularity.tsv",sep="\t",index=False)
(RF/"outputs"/"C07_circularity.json").write_text(json.dumps(
   {"inputs":{"S1":sha(S1),"CLUS":sha(CLUS)},
    "approach":"recover charge-defined 25/19 split from upstream az features (raw composition, length); grantham EXCLUDED (charge-entangled, hardcoded)",
    "results":res},indent=2))
with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C07v2  "
            +"  ".join(f"{r['featurization'][:12]}:ARIkm={r['ari_kmeans']:.2f}" for r in res)+"\n")
print("\nWROTE C07_circularity.tsv, C07_circularity.json")
sys.stdout.flush()
