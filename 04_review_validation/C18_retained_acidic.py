import sys, json, hashlib, datetime, numpy as np, pandas as pd
from pathlib import Path
P=Path(__file__).resolve().parents[1]; DATA=P/"data"; RF=DATA/"derived"
S1=DATA/"sequences"/"Table_S1_Sequences_76variants.tsv"
FA=P/"inputs"/"calr_full_sequences.fasta"
CLUS=DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
# WT
wt=None; name=None; seq=""
for l in open(FA):
    if l.startswith(">"):
        if name and "WT" in name.upper(): wt=seq; break
        name=l[1:].strip(); seq=""
    else: seq+=l.strip()
if name and wt is None and "WT" in name.upper(): wt=seq
print(f"WT len={len(wt)}")
# empirical acidic C-domain: distal patch. Define by windowed net charge < -3 in the C-terminal half
W=15
chg=np.array([(-1 if a in "DE" else 1 if a in "RK" else 0) for a in wt])
net=np.convolve(chg,np.ones(W),mode="valid")
acidic_idx=[i for i in range(len(net)) if net[i]<-3 and i>200]   # C-terminal acidic
ACID_START=min(acidic_idx); ACID_END=max(acidic_idx)+W
print(f"empirical distal acidic C-domain: residues {ACID_START}-{ACID_END} "
      f"(D/E count {sum(wt[ACID_START:ACID_END].count(a) for a in 'DE')}, "
      f"net {sum(-1 if a in 'DE' else 1 if a in 'RK' else 0 for a in wt[ACID_START:ACID_END])})")

d=pd.read_csv(S1,sep="\t"); d["vid"]=d["variant_id"].astype(str)
ref=pd.read_csv(CLUS,sep="\t"); rmap=dict(zip(ref["sequence_id"].astype(str).str.strip(),ref["seq_subgroup"].astype(str).str.strip()))
def g3(r):
    if "2" in str(r["type"]): return "Type 2"
    return rmap.get(f"{r['vid']}|Type_1-like","UNMAPPED")
d["group3"]=d.apply(g3,axis=1)
d=d[d["group3"]!="UNMAPPED"].copy()

WT_acid_DE=sum(wt[ACID_START:ACID_END].count(a) for a in "DE")
rows=[]
for _,r in d.iterrows():
    fs=int(r["frameshift_position"])
    # acidic C-domain residues RETAINED = those before the frameshift cut
    retained_end=min(fs,ACID_END)
    if retained_end<=ACID_START:
        ret_DE=0; ret_net=0
    else:
        seg=wt[ACID_START:retained_end]
        ret_DE=sum(seg.count(a) for a in "DE")
        ret_net=sum(-1 if a in "DE" else 1 if a in "RK" else 0 for a in seg)
    rows.append({"vid":r["vid"],"group3":r["group3"],"fs":fs,
                 "acidic_DE_retained":ret_DE,
                 "acidic_DE_lost":WT_acid_DE-ret_DE,
                 "frac_acidic_retained":ret_DE/WT_acid_DE,
                 "retained_net_charge":ret_net})
R=pd.DataFrame(rows)
print(f"\nWT distal acidic C-domain has {WT_acid_DE} D/E residues total.")
print("\nRETAINED acidic C-domain by group (mean):")
g=R.groupby("group3").agg(
    fs=("fs","median"),
    DE_retained=("acidic_DE_retained","mean"),
    frac_retained=("frac_acidic_retained","mean"),
    net=("retained_net_charge","mean")).round(2)
print(g.to_string())
print("\n=> heterotypic-attraction premise check:")
for grp in ["Type 1-A","Type 1-B","Type 2"]:
    s=R[R["group3"]==grp]
    print(f"  {grp}: retains {s['acidic_DE_retained'].mean():.1f}/{WT_acid_DE} acidic residues "
          f"({s['frac_acidic_retained'].mean():.0%}), net {s['retained_net_charge'].mean():+.1f}")

(RF/"outputs").mkdir(parents=True,exist_ok=True)
R.to_csv(RF/"outputs"/"C18_retained_acidic.tsv",sep="\t",index=False)
(RF/"outputs"/"C18.json").write_text(json.dumps(
   {"inputs":{"S1":sha(S1),"FA":sha(FA)},"WT_len":len(wt),
    "acidic_domain":[int(ACID_START),int(ACID_END)],"WT_acid_DE":int(WT_acid_DE),
    "by_group":{k:{"DE_retained":float(v["DE_retained"]),"frac":float(v["frac_retained"]),
                   "net":float(v["net"])} for k,v in g.iterrows()}},indent=2))
with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C18  acidic domain {ACID_START}-{ACID_END}; "
            f"retained 1A={g.loc['Type 1-A','frac_retained']:.2f} T2={g.loc['Type 2','frac_retained']:.2f}\n")
print("\nWROTE C18_retained_acidic.tsv, C18.json")
sys.stdout.flush()
