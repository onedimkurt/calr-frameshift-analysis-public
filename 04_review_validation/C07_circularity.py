import sys, json, hashlib, datetime, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
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
type1["sid"]=type1["sequence_id"].str.strip(); type1["ref"]=type1["sid"].map(rmap)
ref_lab=np.array([{"Type 1-A":0,"Type 1-B":1}[g] for g in type1["ref"]])
base=[c for c in df.columns if c not in EXC and df[c].dtype in ["float64","int64","bool"] and c not in META]
usable=[f for f in base if len(type1[f].dropna())>=10 and type1[f].dropna().std()>1e-10]

def cluster(cols):
    Xr=type1[cols].copy().astype(float)
    for c in cols:
        if Xr[c].isna().any(): Xr[c]=Xr[c].fillna(Xr[c].median())
    X=StandardScaler().fit_transform(Xr)
    lab=KMeans(2,random_state=SEED,n_init=50).fit_predict(X)
    return adjusted_rand_score(ref_lab,lab), silhouette_score(X,lab), sorted(pd.Series(lab).value_counts().tolist(),reverse=True)

# baseline
ari0,sil0,sz0=cluster(usable)
print(f"BASELINE 87 feats: ARI={ari0:.3f} (exp 1.000) sil={sil0:.3f} sizes={sz0}")

az_charge=[c for c in usable if c.startswith("az_charge")]
az_all   =[c for c in usable if c.startswith("az_")]
global_charge=[c for c in ["ncpr","fcr","SCD","kappa","tail_RK_fraction","tail_DE_fraction",
               "polycation_len","polyanion_len","wt_novel_charge_contrast","uversky_llps"] if c in usable]

tiers={
 "T1_drop_az_charge": [c for c in usable if c not in az_charge],
 "T2_drop_all_az":    [c for c in usable if c not in az_all],
 "T3_drop_az_and_global_charge": [c for c in usable if c not in set(az_all)|set(global_charge)],
}
print(f"\naz_charge cols={len(az_charge)}  az_all cols={len(az_all)}  global_charge cols={len(global_charge)}")
res=[{"tier":"baseline","n_features":len(usable),"ari_vs_ref":ari0,"silhouette":sil0,"sizes":sz0}]
print("\nCIRCULARITY TIERS (re-cluster without charge features):")
for name,cols in tiers.items():
    ari,sil,sz=cluster(cols)
    res.append({"tier":name,"n_features":len(cols),"ari_vs_ref":ari,"silhouette":sil,"sizes":sz})
    verdict="partition SURVIVES" if ari>=0.8 else "partition degrades" if ari>=0.5 else "partition LOST"
    print(f"  {name:30s} feats={len(cols):2d}  ARI={ari:.3f}  sil={sil:.3f}  sizes={sz}  -> {verdict}")

(RF/"outputs").mkdir(parents=True,exist_ok=True)
pd.DataFrame(res).to_csv(RF/"outputs"/"C07_circularity.tsv",sep="\t",index=False)
(RF/"outputs"/"C07_circularity.json").write_text(json.dumps(
   {"inputs":{"FEAT":sha(FEAT),"COH":sha(COH),"CLUS":sha(CLUS)},
    "az_charge_n":len(az_charge),"az_all_n":len(az_all),"global_charge_n":len(global_charge),
    "results":res},indent=2))
with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C07  "
            +"  ".join(f"{r['tier']}:ARI={r['ari_vs_ref']:.3f}" for r in res)+"\n")
print("\nWROTE C07_circularity.tsv, C07_circularity.json")
sys.stdout.flush()
