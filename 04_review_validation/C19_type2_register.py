import sys, glob, json, hashlib, datetime, numpy as np, pandas as pd
import os
from pathlib import Path
P=Path(__file__).resolve().parents[1]; DATA=P/"data"; RF=DATA/"derived"
S1=DATA/"sequences"/"Table_S1_Sequences_76variants.tsv"
ANCHOR="RRMMRTKMRMRRMRRTRRKMRR"; TAIL_START=360
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def load_ca(pdb):
    ca={}
    for l in open(pdb):
        if l.startswith("ATOM") and l[12:16].strip()=="CA":
            ca.setdefault(l[21],[]).append((int(l[22:26]),
                np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])))
    for c in ca: ca[c].sort()
    return ca

def register_analysis(pdb, az_len, anchor_len=22):
    ca=load_ca(pdb); chs=sorted(ca)
    A=np.array([x for _,x in ca[chs[0]]]); B=np.array([x for _,x in ca[chs[1]]])
    nA,nB=len(A),len(B)
    # all inter-chain CA-CA contacts < 8 A
    contacts=[]
    for i in range(nA):
        dd=np.linalg.norm(B-A[i],axis=1)
        for j in np.where(dd<8.0)[0]:
            contacts.append((i,j,dd[j]))
    if not contacts: return None
    iA=np.array([c[0] for c in contacts]); jB=np.array([c[1] for c in contacts])
    # parallel vs antiparallel: correlation of contacting residue indices
    r=np.corrcoef(iA,jB)[0,1] if len(iA)>2 else np.nan
    # region labels: 0..az_len = approach, az..az+anchor = anchor, rest = post-anchor
    def region(i):
        if i<az_len: return "approach"
        if i<az_len+anchor_len: return "anchor"
        return "post-anchor"
    regA=[region(i) for i in iA]; regB=[region(j) for j in jB]
    pairs=pd.Series([f"{a}|{b}" for a,b in zip(regA,regB)]).value_counts(normalize=True)
    # symmetry: is A-region-X<->B-region-Y matched by A-region-Y<->B-region-X? (self-complementary)
    return {"n_contacts":len(contacts),"index_corr":float(r),
            "orientation":"parallel" if r>0.3 else "antiparallel" if r<-0.3 else "mixed/orthogonal",
            "region_pairs":pairs.head(6).to_dict(),
            "mean_contact_dist":float(np.mean([c[2] for c in contacts]))}

# az lengths
d=pd.read_csv(S1,sep="\t"); d["vid"]=d["variant_id"].astype(str)
def azlen(seq,fs):
    nf=seq[max(TAIL_START,int(fs)):]
    return nf.index(ANCHOR) if ANCHOR in nf else 0
azmap=dict(zip(d["vid"],d.apply(lambda r:azlen(str(r["full_sequence"]),r["frameshift_position"]),axis=1)))

# primary example: var114 (highest ipTM Type-2); plus 2 more Type-2 for robustness
t2=set(d[d["type"].astype(str).str.contains("2",na=False)]["vid"])
print("=== C19: Type-2 inter-chain register ===")
results={}
for v in ["var114","var124","var48","var104"]:
    pdb=os.environ.get("CALR_HOMODIMER_DIR", str(P/"data"/"structures"/"homodimer_pdbs"))+f"/{v}_homodimer_unrelaxed_rank1.pdb"
    if not Path(pdb).exists(): print(f"  {v}: no pdb"); continue
    res=register_analysis(pdb, azmap.get(v,0))
    if res is None: print(f"  {v}: no inter-chain contacts"); continue
    results[v]=res
    print(f"\n  {v} (az_len={azmap.get(v)}): {res['n_contacts']} CA contacts, "
          f"index_corr={res['index_corr']:+.2f} -> {res['orientation']}")
    print(f"     mean contact dist {res['mean_contact_dist']:.1f} Å")
    print(f"     top region pairings (A|B): {res['region_pairs']}")

(RF/"outputs").mkdir(parents=True,exist_ok=True)
(RF/"outputs"/"C19_register.json").write_text(json.dumps(results,indent=2))
with open(RF/"logs"/"run_log.txt","a") as f:
    f.write(f"{datetime.datetime.now().isoformat()}  C19  type2 register: "
            +", ".join(f"{v}={r['orientation']}" for v,r in results.items())+"\n")
print("\nWROTE C19_register.json")
sys.stdout.flush()
