import os, sys, json, hashlib, datetime, time
for v in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ[v]="4"
import numpy as np, pandas as pd
from pathlib import Path
import torch; torch.set_num_threads(4)
from sparrow import Protein
P=Path(__file__).resolve().parents[1]; DATA=P/"data"; RF=DATA/"derived"
S1=DATA/"sequences"/"Table_S1_Sequences_76variants.tsv"
CLUS=DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
COH=DATA/"derived"/"af2_definitive"/"statistics"/"AF2_DEFINITIVE.tsv"
ANCHOR="RRMMRTKMRMRRMRRTRRKMRR"; TAIL_START=360
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
d=pd.read_csv(S1,sep="\t")
ref=pd.read_csv(CLUS,sep="\t"); rmap=dict(zip(ref["sequence_id"].astype(str).str.strip(),ref["seq_subgroup"].astype(str).str.strip()))
sg=pd.read_csv(COH,sep="\t")[["sequence_id","sg_sg_dist"]].drop_duplicates("sequence_id")
sgmap=dict(zip(sg["sequence_id"].astype(str).str.strip(),sg["sg_sg_dist"]))
d["sid"]=d["variant_id"].astype(str)+"|"+d["type"].astype(str).str.replace(" ","_").str.replace("-like","-like")
def group3(r):
    if "2" in str(r["type"]): return "Type 2"
    return rmap.get(f"{r['variant_id']}|Type_1-like","UNMAPPED")
d["group3"]=d.apply(group3,axis=1)
def novel_full(seq,fs): return seq[max(TAIL_START,int(fs)):]

t0=time.time(); rows=[]
for _,r in d.iterrows():
    nf=novel_full(str(r["full_sequence"]),r["frameshift_position"])
    if len(nf)<5: continue
    pr=Protein(nf).predictor
    rg=float(pr.radius_of_gyration()); re=float(pr.end_to_end_distance())
    nu=float(pr.scaling_exponent()); asph=float(pr.asphericity())
    sid1=f"{r['variant_id']}|Type_1-like"
    rows.append({"variant_id":r["variant_id"],"group3":r["group3"],"len":len(nf),
                 "Rg":rg,"Re":re,"nu":nu,"asph":asph,
                 "sg_sg_dist":sgmap.get(sid1,np.nan)})
R=pd.DataFrame(rows)
print(f"computed {len(R)} variants in {time.time()-t0:.1f}s")

# guard: var63 reproduces probe values
g=R[R["variant_id"]=="var63"].iloc[0]
print(f"GUARD var63: Rg={g['Rg']:.2f}(exp 24.86) Re={g['Re']:.2f}(exp 57.60) nu={g['nu']:.3f}(exp 0.547)")

print("\nC10 per-group ensemble dimensions (mean±SD):")
summ={}
for grp in ["Type 2","Type 1-A","Type 1-B"]:
    s=R[R["group3"]==grp]
    summ[grp]={k:[float(s[k].mean()),float(s[k].std(ddof=1))] for k in ["Rg","Re","nu","asph","len"]}
    print(f"  {grp:9s}: Rg={s['Rg'].mean():.2f}±{s['Rg'].std(ddof=1):.2f}  "
          f"Re={s['Re'].mean():.2f}±{s['Re'].std(ddof=1):.2f}  "
          f"nu={s['nu'].mean():.3f}  asph={s['asph'].mean():.3f}  n={len(s)}")

# cross-check: does ALBATROSS Rg track the structural Sg-Sg compact/extended?
sub=R.dropna(subset=["sg_sg_dist"])
if len(sub)>5:
    rho=sub[["Rg","sg_sg_dist"]].corr(method="spearman").iloc[0,1]
    print(f"\nALBATROSS Rg vs structural Sγ-Sγ distance: Spearman rho={rho:.3f} (n={len(sub)})")
    print("  (positive rho => sequences ALBATROSS calls expanded also have larger Sγ-Sγ -> concordant 'compact vs extended')")

(RF/"outputs").mkdir(parents=True,exist_ok=True)
R.to_csv(RF/"outputs"/"C10_albatross_per_variant.tsv",sep="\t",index=False)
(RF/"outputs"/"C10_albatross.json").write_text(json.dumps(
   {"inputs":{"S1":sha(S1),"CLUS":sha(CLUS)},"span":"novel_full = full_sequence[max(360,fs):]",
    "backend":"CPU (4 threads)","sparrow_torch":torch.__version__,
    "guard_var63":{"Rg":g["Rg"],"Re":g["Re"],"nu":g["nu"]},
    "group_summary":summ,
    "spearman_Rg_vs_sgsg":float(rho) if len(sub)>5 else None},indent=2))
with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C10  Rg T2={summ['Type 2']['Rg'][0]:.1f} "
            f"1A={summ['Type 1-A']['Rg'][0]:.1f} 1B={summ['Type 1-B']['Rg'][0]:.1f} "
            f"rho_sgsg={rho if len(sub)>5 else 'na'}\n")
print("\nWROTE C10_albatross_per_variant.tsv, C10_albatross.json")
sys.stdout.flush()
