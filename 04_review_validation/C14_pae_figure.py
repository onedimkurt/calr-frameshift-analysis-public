import sys, json, glob, numpy as np
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
P=Path(__file__).resolve().parents[1]; DATA=P/"data"; RF=DATA/"derived"
src=sorted(glob.glob(os.environ.get("CALR_PAE_GLOB", str(Path.home()/"Downloads"/"C14_dimer_pae*.json"))))[-1]
data=json.load(open(src))
print("loaded:", src)

order=["Type 1-A","Type 1-B","Type 2"]
meta={"Type 1-A":("compact","ipTM 0.19"),"Type 1-B":("extended","ipTM 0.26"),"Type 2":("extended","ipTM 0.13")}
fig=plt.figure(figsize=(13,4.6))
gs=GridSpec(1,4,width_ratios=[1,1,1,0.06],wspace=0.25)
ims=[]
for k,grp in enumerate(order):
    d=data[grp]; a=np.array(d["pae"]); n=a.shape[0]; half=n//2
    ax=fig.add_subplot(gs[0,k])
    im=ax.imshow(a,cmap="Greens_r",vmin=0,vmax=31.75,origin="upper")
    ims.append(im)
    # chain boundary lines (A|B)
    ax.axhline(half-0.5,color="k",lw=0.8,ls="--"); ax.axvline(half-0.5,color="k",lw=0.8,ls="--")
    ax.set_title(f"{grp} — {d['vid']}\n{meta[grp][0]}, {meta[grp][1]}",fontsize=10)
    ax.set_xlabel("residue (chain A | chain B)",fontsize=8)
    if k==0: ax.set_ylabel("residue (chain A | chain B)",fontsize=8)
    ax.set_xticks([half/2,half+half/2]); ax.set_xticklabels(["A","B"],fontsize=8)
    ax.set_yticks([half/2,half+half/2]); ax.set_yticklabels(["A","B"],fontsize=8)
    # annotate the inter-chain (off-diagonal) blocks
    ax.text(half/2, half+half/2, "inter-\nchain", ha="center",va="center",fontsize=7,color="#444",alpha=0.7)
cax=fig.add_subplot(gs[0,3])
cb=fig.colorbar(ims[0],cax=cax); cb.set_label("Predicted Aligned Error (Å)",fontsize=8)
fig.suptitle("Representative homodimer PAE matrices (rank-1, AF2-multimer); dashed lines = chain boundary",
             fontsize=11,y=1.02)
out=RF/"outputs"/"C14_representative_PAE.png"
fig.savefig(out,dpi=200,bbox_inches="tight")
fig.savefig(RF/"outputs"/"C14_representative_PAE.pdf",bbox_inches="tight")
print("WROTE",out)
# quick quantitative summary of inter-chain confidence (off-diagonal mean) for the caption
for grp in order:
    a=np.array(data[grp]["pae"]); h=a.shape[0]//2
    inter=np.concatenate([a[:h,h:].ravel(),a[h:,:h].ravel()])
    intra=np.concatenate([a[:h,:h].ravel(),a[h:,h:].ravel()])
    print(f"  {grp} {data[grp]['vid']}: inter-chain mean PAE={inter.mean():.1f} Å, intra-chain={intra.mean():.1f} Å")
sys.stdout.flush()
