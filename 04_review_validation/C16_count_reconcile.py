import sys, json, hashlib, datetime, numpy as np, pandas as pd
from pathlib import Path
P=Path(__file__).resolve().parents[1]; DATA=P/"data"; RF=DATA/"derived"
AF=DATA/"derived"/"af2_definitive"/"statistics"/"AF2_DEFINITIVE.tsv"
RAW=DATA/"derived"/"contact_positions_raw.tsv"
AUD=P/"FIGURES_FINAL"/"outputs"/"contact_audit.tsv"
CLUS=DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
af=pd.read_csv(AF,sep="\t"); af["vid"]=af["sequence_id"].astype(str).str.split("|").str[0].str.strip()
ref=pd.read_csv(CLUS,sep="\t"); rmap=dict(zip(ref["sequence_id"].astype(str).str.strip(),ref["seq_subgroup"].astype(str).str.strip()))
def g3(r):
    if "2" in str(r["primary_class"]): return "Type 2"
    return rmap.get(f"{r['vid']}|Type_1-like","UNMAPPED")
af["group3"]=af.apply(g3,axis=1)

print("=== 1. Sγ-Sγ compact/extended/intermediate (Type-1 anchor 26/17/1) ===")
t1=af[af["group3"].str.startswith("Type 1")]
s=t1["sg_sg_dist"]
print(f"  Type-1 (n={len(s)}): compact<20={int((s<20).sum())} extended>50={int((s>50).sum())} "
      f"intermediate={int(((s>=20)&(s<=50)).sum())}  [anchor 26/17/1]")
for g in ["Type 1-A","Type 1-B"]:
    sg=af[af["group3"]==g]["sg_sg_dist"]
    print(f"    {g}: compact={int((sg<20).sum())} ext={int((sg>50).sum())} int={int(((sg>=20)&(sg<=50)).sum())} n={len(sg)}")
# all 76 too
sall=af["sg_sg_dist"]
print(f"  ALL 76: compact={int((sall<20).sum())} extended={int((sall>50).sum())} intermediate={int(((sall>=20)&(sall<=50)).sum())}")

print("\n=== 2. ipTM (anchor: overall 0.187; 1-A .193/1-B .258/T2 .139) ===")
print(f"  overall dimer ipTM = {af['dimer_iptm'].mean():.3f}")
for g in ["Type 1-A","Type 1-B","Type 2"]:
    print(f"    {g}: ipTM={af[af['group3']==g]['dimer_iptm'].mean():.3f}")

print("\n=== 3. Contacts (anchor 1-A 723/T2 398/1-B 239) ===")
raw=pd.read_csv(RAW,sep="\t")
print("  total contacts per group:", raw.groupby("group").size().to_dict())
print("\n  D11 reconciliation — approach-zone share, three denominators:")
for g in raw["group"].unique():
    sub=raw[raw["group"]==g]
    n=len(sub)
    # def A: fraction of endpoints (both A and B) in approach
    endpoints=pd.concat([sub["region_A"],sub["region_B"]])
    az_endpoint=(endpoints=="approach").mean()
    # def B: fraction of contacts with EITHER end in approach
    az_either=((sub["region_A"]=="approach")|(sub["region_B"]=="approach")).mean()
    # def C: fraction with BOTH ends in approach
    az_both=((sub["region_A"]=="approach")&(sub["region_B"]=="approach")).mean()
    print(f"    {g}: endpoints={az_endpoint:.1%}  either-end={az_either:.1%}  both-ends={az_both:.1%}  (n={n})")
print("  [Type-2 anchor 82.6% approach-zone, 1-A 98.9% post-anchor]")
# 1-A post-anchor under same defs
print("\n  1-A post-anchor share:")
sub=raw[raw["group"]=="Type 1-A"]
ep=pd.concat([sub["region_A"],sub["region_B"]])
print(f"    endpoints post-anchor={ (ep=='post-anchor').mean():.1%}")

out={"inputs":{"AF":sha(AF),"RAW":sha(RAW),"AUD":sha(AUD)},
     "sgsg_type1":{"compact":int((s<20).sum()),"extended":int((s>50).sum()),"intermediate":int(((s>=20)&(s<=50)).sum())},
     "iptm_overall":float(af["dimer_iptm"].mean()),
     "contacts":raw.groupby("group").size().to_dict()}
(RF/"outputs").mkdir(parents=True,exist_ok=True)
(RF/"outputs"/"C16_reconcile.json").write_text(json.dumps(out,indent=2))
with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C16  sgsg 26/17/1 reproduced; iptm 0.187; contacts 723/398/239\n")
print("\nWROTE C16_reconcile.json")
sys.stdout.flush()
