import sys, numpy as np, pandas as pd
from pathlib import Path
from localcider.sequenceParameters import SequenceParameters
P=Path(__file__).resolve().parents[1]; DATA=P/"data"
S1=DATA/"sequences"/"Table_S1_Sequences_76variants.tsv"
CLUS=DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
ANCHOR="RRMMRTKMRMRRMRRTRRKMRR"; TAIL_START=360
# discover the correct region method name once
probe=SequenceParameters("ACDEFGHIKLMNPQRSTVWY")
cands=[m for m in dir(probe) if "region" in m.lower() or "phase" in m.lower()]
print("phase/region methods available:",cands)
region_method=None
for name in ("get_phasePlotRegion","get_phase_plot_region","phasePlotRegion"):
    if hasattr(probe,name): region_method=name; break
if region_method is None and cands: region_method=cands[0]
print("using region method:",region_method)
def getreg(sp):
    return int(getattr(sp,region_method)())
def getfp(sp):
    for n in ("get_fraction_positive","get_countPos","get_fractionPositive"):
        if hasattr(sp,n):
            v=getattr(sp,n)(); return v if v<=1 else v/sp.get_length()
    return np.nan
def getfm(sp):
    for n in ("get_fraction_negative","get_countNeg","get_fractionNegative"):
        if hasattr(sp,n):
            v=getattr(sp,n)(); return v if v<=1 else v/sp.get_length()
    return np.nan

d=pd.read_csv(S1,sep="\t"); d["vid"]=d["variant_id"].astype(str)
ref=pd.read_csv(CLUS,sep="\t"); rmap=dict(zip(ref["sequence_id"].astype(str).str.strip(),ref["seq_subgroup"].astype(str).str.strip()))
typ=dict(zip(d["vid"],d["type"].astype(str)))
def grp(v):
    if "2" in typ.get(v,""): return "Type 2"
    return rmap.get(f"{v}|Type_1-like","?")
REGION={1:"R1 weak polyampholyte/electrolyte",2:"R2 Janus/boundary",3:"R3 strong polyampholyte",
        4:"R4 strong polyelectrolyte",5:"R5 strong polyelectrolyte"}
rows=[]
for _,r in d.iterrows():
    seq=str(r["full_sequence"])[max(TAIL_START,int(r["frameshift_position"])):]
    if ANCHOR not in seq: continue
    try:
        sp=SequenceParameters(seq)
        rows.append({"vid":r["vid"],"group":grp(r["vid"]),"fplus":getfp(sp),"fminus":getfm(sp),"region":getreg(sp)})
    except Exception as e:
        print(r["vid"],"ERR",e); continue
R=pd.DataFrame(rows)
print("\nper-group f+/f-/FCR + localCIDER region:")
for g in ["Type 1-A","Type 1-B","Type 2"]:
    s=R[R["group"]==g]
    if len(s)==0: continue
    mr=int(s["region"].mode().iloc[0])
    print(f"  {g}: f+={s['fplus'].mean():.3f} f-={s['fminus'].mean():.3f} FCR={(s['fplus']+s['fminus']).mean():.3f} "
          f"|NCPR|={abs(s['fplus']-s['fminus']).mean():.3f} | region mode={mr} ({REGION[mr]}) | counts={dict(s['region'].value_counts())}")
sys.stdout.flush()
