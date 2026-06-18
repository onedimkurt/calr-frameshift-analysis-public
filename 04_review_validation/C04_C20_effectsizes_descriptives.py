import sys, json, hashlib, datetime, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
P=Path(__file__).resolve().parents[1]; DATA=P/"data"; RF=DATA/"derived"
FEAT=DATA/"derived"/"RECOMPUTED_FEATURES_76_VARIANTS.tsv"
CLUS=DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
df=pd.read_csv(FEAT,sep="\t")
ref=pd.read_csv(CLUS,sep="\t"); rmap=dict(zip(ref["sequence_id"].str.strip(),ref["seq_subgroup"].str.strip()))
df["sid"]=df["sequence_id"].str.strip()
df["group3"]=df.apply(lambda r:"Type 2" if "2" in str(r["primary_class"]) else rmap.get(r["sid"],"UNMAPPED"),axis=1)
assert (df["group3"].value_counts().reindex(["Type 2","Type 1-A","Type 1-B"]).tolist()==[32,25,19])

METRICS={"ncpr":"NCPR","fcr":"FCR","SCD":"SCD","kappa":"kappa",
         "novel_cterminus_residues":"novel_tail_length",
         "wt_fragment_length":"wt_cdomain_length",
         "uversky_llps":"CH_disorder_score","kd_novel":"kd_hydropathy",
         "tail_RK_fraction":"tail_RK_frac","tail_DE_fraction":"tail_DE_frac",
         "polycation_len":"polycation_run","polyanion_len":"polyanion_run",
         "wt_novel_charge_contrast":"charge_contrast"}

def hedges_g(a,b):
    na,nb=len(a),len(b); sa,sb=a.std(ddof=1),b.std(ddof=1)
    sp=np.sqrt(((na-1)*sa**2+(nb-1)*sb**2)/(na+nb-2))
    d=(a.mean()-b.mean())/sp if sp>0 else np.nan
    J=1-3/(4*(na+nb)-9)         # bias correction
    return d*J
def glass_delta(a,b,ctrl):     # ctrl = which group's SD to use
    s=ctrl.std(ddof=1)
    return (a.mean()-b.mean())/s if s>0 else np.nan

groups={g:df[df["group3"]==g] for g in ["Type 2","Type 1-A","Type 1-B"]}
desc=[]; eff=[]
for col,name in METRICS.items():
    for g in ["Type 2","Type 1-A","Type 1-B"]:
        v=groups[g][col].dropna()
        desc.append({"metric":name,"col":col,"group":g,"n":len(v),
                     "mean":float(v.mean()),"sd":float(v.std(ddof=1))})
    for a,b in [("Type 1-A","Type 1-B"),("Type 1-A","Type 2"),("Type 1-B","Type 2")]:
        va,vb=groups[a][col].dropna(),groups[b][col].dropna()
        u,p=stats.mannwhitneyu(va,vb,alternative="two-sided")
        eff.append({"metric":name,"contrast":f"{a} vs {b}",
                    "mean_a":float(va.mean()),"mean_b":float(vb.mean()),
                    "sd_a":float(va.std(ddof=1)),"sd_b":float(vb.std(ddof=1)),
                    "hedges_g":float(hedges_g(va,vb)),
                    "glass_delta_ctrlB":float(glass_delta(va,vb,vb)),
                    "mannwhitney_p":float(p)})
D=pd.DataFrame(desc); E=pd.DataFrame(eff)

print("=== C20 per-group mean±SD (key anchors) ===")
piv=D.pivot(index="metric",columns="group",values="mean")
for m in ["NCPR","FCR","SCD","kappa","novel_tail_length","wt_cdomain_length"]:
    r=piv.loc[m]; print(f"  {m:20s} T2={r['Type 2']:.3f}  1-A={r['Type 1-A']:.3f}  1-B={r['Type 1-B']:.3f}")

print("\n=== C04 effect sizes (Hedges g; pooled-d replaced) ===")
for m in ["NCPR","SCD","novel_tail_length","wt_cdomain_length"]:
    sub=E[E["metric"]==m]
    for _,r in sub.iterrows():
        print(f"  {m:18s} {r['contrast']:20s} g={r['hedges_g']:+.2f}  Δ(Glass)={r['glass_delta_ctrlB']:+.2f}  p={r['mannwhitney_p']:.1e}")

# d≈14 artifact: tail-length T2 vs 1-B
tl_t2=groups["Type 2"]["novel_cterminus_residues"]; tl_1b=groups["Type 1-B"]["novel_cterminus_residues"]
sp=np.sqrt(((len(tl_t2)-1)*tl_t2.std(ddof=1)**2+(len(tl_1b)-1)*tl_1b.std(ddof=1)**2)/(len(tl_t2)+len(tl_1b)-2))
cohen_d=(tl_t2.mean()-tl_1b.mean())/sp
print(f"\n=== C04 d≈14 artifact check (tail length T2 vs 1-B) ===")
print(f"  means {tl_t2.mean():.2f} vs {tl_1b.mean():.2f}; SDs {tl_t2.std(ddof=1):.2f}/{tl_1b.std(ddof=1):.2f}")
print(f"  pooled Cohen d = {cohen_d:.2f}  (huge because within-group SD tiny) -> VARIANCE ARTIFACT")
print(f"  Hedges g = {hedges_g(tl_t2,tl_1b):+.2f}; report with CI, cap interpretation")

# mislabel reconciliation
t1=df[df["primary_class"].str.contains('1',na=False)]["wt_fragment_length"]
print(f"\n=== C20 mislabel reconciliation (WT C-domain length) ===")
print(f"  pooled Type-1-like (n=44): {t1.mean():.2f} ± {t1.std(ddof=1):.2f}  [MS '58.05±3.64']")
print(f"  Type 1-A only (n=25):      {groups['Type 1-A']['wt_fragment_length'].mean():.2f} ± {groups['Type 1-A']['wt_fragment_length'].std(ddof=1):.2f}  [MS '58.16±2.08']")
print(f"  -> 58.05=pooled, 58.16=1-A subgroup; manuscript conflated the two.")

(RF/"outputs").mkdir(parents=True,exist_ok=True)
D.to_csv(RF/"outputs"/"C20_group_descriptives.tsv",sep="\t",index=False)
E.to_csv(RF/"outputs"/"C04_effect_sizes.tsv",sep="\t",index=False)
(RF/"outputs"/"C04_C20.json").write_text(json.dumps(
   {"inputs":{"FEAT":sha(FEAT),"CLUS":sha(CLUS)},
    "descriptives":desc,"effect_sizes":eff,
    "d14_artifact":{"cohen_d":float(cohen_d),"hedges_g":float(hedges_g(tl_t2,tl_1b))}},indent=2))
with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C04_C20  groups=32/25/19  "
            f"d14_tail_T2vs1B={cohen_d:.1f} g={hedges_g(tl_t2,tl_1b):.2f}\n")
print("\nWROTE C20_group_descriptives.tsv, C04_effect_sizes.tsv, C04_C20.json")
sys.stdout.flush()
