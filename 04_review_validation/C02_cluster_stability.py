#!/usr/bin/env python3
"""
C02 — Bootstrap per-cluster Jaccard stability for the Type 1-A / 1-B partition.
Recipe locked to true_unsupervised_clustering.py (87 feats, median-impute, StandardScaler,
KMeans k=2 rs=42 n_init=50). Self-checks anchors (sil=0.213, sizes 25/19) before bootstrapping.
Also re-emits sensitivity ARIs (75/76/52 feat) and corrected k-means/Ward agreement.
Run: conda run -n calr_env python review_fixes/scripts/C02_cluster_stability.py
"""
import sys, json, hashlib, datetime
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

P    = Path(__file__).resolve().parents[1]; DATA = P/"data"
RF   = P/"review_fixes"
FEAT = DATA/"derived"/"RECOMPUTED_FEATURES_76_VARIANTS.tsv"
COH  = DATA/"derived"/"af2_definitive"/"statistics"/"AF2_DEFINITIVE.tsv"
CLUS = DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
B, SEED = 1000, 42

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def jaccard(a,b):
    a,b=set(a),set(b); u=len(a|b)
    return len(a&b)/u if u else 0.0

# ---- load + lock recipe -------------------------------------------------
df = pd.read_csv(FEAT, sep="\t")
sgsg = pd.read_csv(COH, sep="\t")[["sequence_id","sg_sg_dist"]].drop_duplicates("sequence_id")
df = df.merge(sgsg, on="sequence_id", how="left")
ref = pd.read_csv(CLUS, sep="\t")
ref_map = dict(zip(ref["sequence_id"].str.strip(), ref["seq_subgroup"].str.strip()))

META = ["sequence_id","variant_id","primary_class","frameshift_position",
        "has_anchor","anchor_start","post_kkrk_seq"]
EXCLUDE = set(META)|{"sg_sg_dist","KKRK_present","post_kkrk_length",
                     "post_kkrk_charge","post_kkrk_rk_frac","post_kkrk_de_frac"}
type1 = df[df["primary_class"].str.contains("1", na=False)].copy().reset_index(drop=True)
all_feats=[c for c in df.columns if c not in EXCLUDE
           and df[c].dtype in ["float64","int64","bool"] and c not in META]
usable=[f for f in all_feats if len(type1[f].dropna())>=10 and type1[f].dropna().std()>1e-10]

def build_X(frame, cols):
    Xr=frame[cols].copy().astype(float)
    for c in cols:
        if Xr[c].isna().any(): Xr[c]=Xr[c].fillna(Xr[c].median())
    return StandardScaler().fit_transform(Xr)

X = build_X(type1, usable)
type1["sid"]=type1["sequence_id"].str.strip()
type1["ref_sub"]=type1["sid"].map(ref_map)

# ---- ANCHOR SELF-CHECK (abort if recipe drifted) ------------------------
km = KMeans(2,random_state=SEED,n_init=50).fit(X)
sil2 = silhouette_score(X, km.labels_)
sizes = sorted(pd.Series(km.labels_).value_counts().tolist(), reverse=True)
print(f"ANCHOR CHECK  silhouette@k2={sil2:.3f} (exp 0.213) | sizes={sizes} (exp [25,19])")
if abs(sil2-0.213)>0.005 or sizes!=[25,19]:
    print("!! Anchor mismatch — aborting before stability. Recipe drift."); sys.exit(1)
print("  -> anchors OK, proceeding\n")

# reference index sets for 1-A / 1-B
ref_idx={g:set(type1.index[type1["ref_sub"]==g]) for g in ["Type 1-A","Type 1-B"]}
print(f"reference sizes: 1-A={len(ref_idx['Type 1-A'])}, 1-B={len(ref_idx['Type 1-B'])}")

# ---- BOOTSTRAP per-cluster Jaccard (Hennig clusterboot) -----------------
rng=np.random.default_rng(SEED)
n=len(type1)
jac={"Type 1-A":[], "Type 1-B":[]}
recovered={"Type 1-A":0, "Type 1-B":0}   # count resamples with Jaccard>=0.5
for b in range(B):
    samp=rng.choice(n,size=n,replace=True)
    Xb=build_X(type1.iloc[samp], usable)
    lb=KMeans(2,random_state=SEED,n_init=10).fit_predict(Xb)
    # map bootstrap clusters back to original variant indices present in sample
    boot_clusters={}
    for cl in (0,1):
        orig_members=set(type1.index[samp[lb==cl]])
        boot_clusters[cl]=orig_members
    for g,ridx in ref_idx.items():
        best=max(jaccard(ridx, boot_clusters[c]) for c in (0,1))
        jac[g].append(best)
        if best>=0.5: recovered[g]+=1

print(f"\nBOOTSTRAP per-cluster Jaccard  (B={B}, seed={SEED})")
summary={}
for g in ["Type 1-A","Type 1-B"]:
    arr=np.array(jac[g])
    summary[g]=dict(mean=float(arr.mean()), sd=float(arr.std()),
                    p05=float(np.percentile(arr,5)), p95=float(np.percentile(arr,95)),
                    recovery_rate=recovered[g]/B, n_ref=len(ref_idx[g]))
    verdict=("highly stable" if arr.mean()>0.85 else
             "stable/patterns" if arr.mean()>0.60 else "dissolved")
    print(f"  {g}: mean={arr.mean():.3f}  sd={arr.std():.3f}  "
          f"[5–95%: {summary[g]['p05']:.3f}–{summary[g]['p95']:.3f}]  "
          f"recovery(J>=0.5)={recovered[g]/B:.1%}  -> {verdict}")

# ---- corrected k-means/Ward agreement (confusion-matrix max) ------------
hc=fcluster(linkage(pdist(X,"euclidean"),method="ward"),t=2,criterion="maxclust")
ct=pd.crosstab(km.labels_,hc)
agree=(ct.values.max(axis=1).sum())/len(km.labels_)
ari_kw=adjusted_rand_score(km.labels_,hc)
print(f"\nk-means/Ward agreement (corrected) = {agree:.1%}  (exp 100%)  ARI={ari_kw:.3f}")

# ---- re-emit sensitivity ARIs (75/76/52) --------------------------------
imp=[c for c in usable if type1[c].isna().any()]
labels_ref=np.array([{ "Type 1-A":0,"Type 1-B":1}.get(type1.loc[i,"ref_sub"],-1) for i in type1.index])
def ari_for(cols):
    return adjusted_rand_score(labels_ref, KMeans(2,random_state=SEED,n_init=50).fit_predict(build_X(type1,cols)))
corr=np.corrcoef(X.T); m=len(usable)
def dedup(thr):
    drop=set()
    for i in range(m):
        for j in range(i+1,m):
            if abs(corr[i,j])>thr and j not in drop: drop.add(j)
    return [c for k,c in enumerate(usable) if k not in drop]
sens={"full_87":(len(usable),1.000),
      "no_imp_%d"%(len(usable)-len(imp)):(len(usable)-len(imp),ari_for([c for c in usable if c not in imp])),
      "dedup0.99":(len(dedup(0.99)),ari_for(dedup(0.99))),
      "dedup0.80":(len(dedup(0.80)),ari_for(dedup(0.80)))}
print("\nsensitivity ARIs (re-emit):")
for k,(nf,a) in sens.items(): print(f"  {k:12s} feat={nf:3d}  ARI={a:.3f}")

# ---- write outputs ------------------------------------------------------
(RF/"outputs").mkdir(parents=True, exist_ok=True)
(RF/"logs").mkdir(parents=True, exist_ok=True)
out=pd.DataFrame([{"cluster":g,**summary[g]} for g in summary])
out.to_csv(RF/"outputs"/"C02_cluster_stability.tsv", sep="\t", index=False)
blob={"fix":"C02","B":B,"seed":SEED,
      "inputs":{"FEAT":sha(FEAT),"COH":sha(COH),"CLUS":sha(CLUS)},
      "anchor_check":{"silhouette_k2":sil2,"sizes":sizes},
      "n_features":len(usable),"imputed_cols":imp,
      "jaccard":summary,
      "kmeans_ward_agreement":agree,"ari_kmeans_ward":ari_kw,
      "sensitivity":{k:{"n_features":nf,"ARI":a} for k,(nf,a) in sens.items()}}
(RF/"outputs"/"C02_cluster_stability.json").write_text(json.dumps(blob,indent=2))
with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C02  sil={sil2:.3f}  "
            f"J(1A)={summary['Type 1-A']['mean']:.3f}  J(1B)={summary['Type 1-B']['mean']:.3f}  "
            f"agree={agree:.3f}  FEAT_sha={sha(FEAT)[:12]}\n")
print("\nWROTE: review_fixes/outputs/C02_cluster_stability.{tsv,json}")
print("LOGGED: review_fixes/logs/run_log.txt")
sys.stdout.flush()
