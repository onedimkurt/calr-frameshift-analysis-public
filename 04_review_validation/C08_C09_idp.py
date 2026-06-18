import sys, json, hashlib, datetime, numpy as np, pandas as pd
from pathlib import Path
from localcider.sequenceParameters import SequenceParameters
P=Path(__file__).resolve().parents[1]; DATA=P/"data"; RF=DATA/"derived"
S1=DATA/"sequences"/"Table_S1_Sequences_76variants.tsv"
CLUS=DATA/"derived"/"unsupervised_clustering"/"statistics"/"cluster_assignments.tsv"
ANCHOR="RRMMRTKMRMRRMRRTRRKMRR"; TAIL_START=360
POS=set("RK"); NEG=set("DE")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

d=pd.read_csv(S1,sep="\t")
ref=pd.read_csv(CLUS,sep="\t"); rmap=dict(zip(ref["sequence_id"].astype(str).str.strip(),
                                              ref["seq_subgroup"].astype(str).str.strip()))
def group3(r):
    if "2" in str(r["type"]): return "Type 2"
    sid=f"{r['variant_id']}|Type_1-like"
    return rmap.get(sid,"UNMAPPED")
d["group3"]=d.apply(group3,axis=1)
print("groups:",d["group3"].value_counts().to_dict())

def novel_full(seq,fs):
    return seq[max(TAIL_START,int(fs)):]

rows=[]
for _,r in d.iterrows():
    nf=novel_full(str(r["full_sequence"]), r["frameshift_position"])
    if len(nf)<5: 
        rows.append({"variant_id":r["variant_id"],"group3":r["group3"],"skipped":True}); continue
    SP=SequenceParameters(nf)
    n=len(nf)
    fpos=sum(1 for a in nf if a in POS)/n
    fneg=sum(1 for a in nf if a in NEG)/n
    rows.append({"variant_id":r["variant_id"],"group3":r["group3"],"len":n,
                 "fpos":fpos,"fneg":fneg,
                 "ncpr":SP.get_NCPR(),"fcr":SP.get_FCR(),
                 "kappa":SP.get_kappa(),"scd":SP.get_SCD(),
                 "hydro":SP.get_mean_hydropathy()})
R=pd.DataFrame(rows)

# ---- GUARD: var63 must reproduce stored CIDER values ----
g=R[R["variant_id"]=="var63"].iloc[0]
print(f"\nGUARD var63: ncpr={g['ncpr']:.4f}(exp .2712) fcr={g['fcr']:.4f}(exp .5085) scd={g['scd']:.4f}(exp 5.0385)")
ok=abs(g['ncpr']-0.2712)<0.002 and abs(g['fcr']-0.5085)<0.002 and abs(g['scd']-5.0385)<0.01
print("  span reproduces CIDER:", "YES" if ok else "NO -- ABORT")
if not ok: sys.exit(1)

# ---- identity checks ----
R["id_ncpr"]=(R["fpos"]-R["fneg"]-R["ncpr"]).abs()
R["id_fcr"]=(R["fpos"]+R["fneg"]-R["fcr"]).abs()
print(f"\nidentity check max|f+ - f- - NCPR|={R['id_ncpr'].max():.4f}  max|f+ + f- - FCR|={R['id_fcr'].max():.4f}")

# ---- C08: f+/f- per group + Das-Pappu region ----
def das_pappu(fpos,fneg):
    fcr=fpos+fneg; ncpr=abs(fpos-fneg)
    if fcr<0.25: return "R1 (weak polyampholyte/polyelectrolyte)"
    if fcr>=0.35 and ncpr<=0.35 and fpos>0.15 and fneg>0.15: return "R3 (strong polyampholyte)"
    if ncpr>0.35: return "R4 (polyelectrolyte)"
    return "R2 (boundary/Janus)"
print("\nC08 f+/f- per group + Das-Pappu (f+/f- space):")
summ={}
for grp in ["Type 2","Type 1-A","Type 1-B"]:
    s=R[R["group3"]==grp]
    fp,fn=s["fpos"].mean(),s["fneg"].mean()
    reg=das_pappu(fp,fn)
    summ[grp]={"fpos":float(fp),"fneg":float(fn),"fcr":float(fp+fn),"ncpr":float(fp-fn),
               "region":reg,"n":len(s)}
    print(f"  {grp:9s}: f+={fp:.3f} f-={fn:.3f}  (FCR={fp+fn:.3f} NCPR={fp-fn:.3f})  -> {reg}")

# ---- C09: SCD prefactor 1/N vs 2/N ----
def scd_manual(seq,pref):
    chg=[1 if a in POS else -1 if a in NEG else 0 for a in seq]
    N=len(seq); s=0.0
    for i in range(N):
        for j in range(i+1,N):
            s+=chg[i]*chg[j]*np.sqrt(j-i)
    return pref*s
ex=novel_full(str(d[d["variant_id"]=="var63"].iloc[0]["full_sequence"]),
              d[d["variant_id"]=="var63"].iloc[0]["frameshift_position"])
scd_lc=SequenceParameters(ex).get_SCD()
scd_1N=scd_manual(ex,1/len(ex)); scd_2N=scd_manual(ex,2/len(ex))
print(f"\nC09 SCD prefactor check (var63): localCIDER={scd_lc:.4f}")
print(f"   manual 1/N={scd_1N:.4f}  | 2/N={scd_2N:.4f}  -> localCIDER uses {'1/N' if abs(scd_lc-scd_1N)<abs(scd_lc-scd_2N) else '2/N'}")
print("   (Sawle & Ghosh 2015 canonical SCD uses 1/N)")

# ---- C09: kappa caveat for low f- (Type 1-A) ----
low_fneg=R[(R["group3"]=="Type 1-A")&(R["fneg"]<0.10)]
print(f"\nC09 kappa caveat: Type 1-A variants with f- < 0.10: {len(low_fneg)}/{summ['Type 1-A']['n']}")
print("   -> kappa is ill-defined/unreliable when one charge species is near-absent; flag for 1-A.")

(RF/"outputs").mkdir(parents=True,exist_ok=True)
R.to_csv(RF/"outputs"/"C08_fpos_fneg_per_variant.tsv",sep="\t",index=False)
(RF/"outputs"/"C08_C09_idp.json").write_text(json.dumps(
   {"inputs":{"S1":sha(S1),"CLUS":sha(CLUS)},"span":"novel_full = full_sequence[max(360,fs):]",
    "guard_var63_ok":bool(ok),"group_summary":summ,
    "scd":{"localcider":scd_lc,"manual_1N":scd_1N,"manual_2N":scd_2N},
    "kappa_caveat_1A_low_fneg":int(len(low_fneg))},indent=2))
with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C08_C09  guard_ok={ok}  "
            f"regions T2={summ['Type 2']['region'][:2]} 1A={summ['Type 1-A']['region'][:2]} 1B={summ['Type 1-B']['region'][:2]}\n")
print("\nWROTE C08_fpos_fneg_per_variant.tsv, C08_C09_idp.json")
sys.stdout.flush()
