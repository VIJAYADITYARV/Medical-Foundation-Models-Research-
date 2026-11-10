import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.linewidth":0.6})
BLUE="#cfe2f3"; AMBER="#fde3a7"; GRAY="#e6e6e6"; EDGE="#33475b"

# ---------------- Fig 1: pipeline ----------------
fig, ax = plt.subplots(figsize=(7.2,3.35)); ax.set_xlim(0,100); ax.set_ylim(0,43); ax.axis("off")
W,H,G=17.2,8.2,3.5
def box(col,row,text,kind):
    x=col*(W+G); y={0:33,1:19,2:5}[row]
    fc={"base":BLUE,"new":AMBER,"off":GRAY}[kind]
    ls="--" if kind=="off" else "-"
    p=FancyBboxPatch((x,y),W,H,boxstyle="round,pad=0.25,rounding_size=1.2",fc=fc,ec=EDGE,lw=0.8,ls=ls)
    ax.add_patch(p); ax.text(x+W/2,y+H/2,text,ha="center",va="center",fontsize=6.2,linespacing=1.15)
    return (x,y)
def arr(x1,y1,x2,y2,style="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=7,lw=0.8,color=EDGE))
r0=["25 gold-standard\nlabels (GS)","Train UNet++\n(Adam, 200 ep.)","Coarse masks on\nunlabeled pool","Easy-first\nordering","Complexity gate\n(Random Forest)"]
k0=["base","base","base","new","new"]
r1=["Rotation-consistency\ncheck (6 angles)","MedSAM refinement\n(box prompt)","Morphological\ngate V2","Quality predictor\n(GBR)","Weak labels\nsaved"]
k1=["new","base","new","new","base"]
r2=["Augmented set\n(GS + weak)","Retrain UNet++\n(distance-weighted)","Evaluate\n(optional flip-TTA)","SimCLR pretraining\n(not loaded)","Centroid point\n(not passed)"]
k2=["base","new","new","off","off"]
for i,(t,k) in enumerate(zip(r0,k0)): box(i,0,t,k)
for i,(t,k) in enumerate(zip(r1,k1)): box(i,1,t,k)
for i,(t,k) in enumerate(zip(r2,k2)): box(i,2,t,k)
for row,y in [(0,33),(1,19),(2,5)]:
    n=5 if row<2 else 3
    for i in range(n-1):
        x=i*(W+G)+W; arr(x+0.3,y+H/2,x+G-0.3,y+H/2)
# row connectors (elbow)
for ytop,ybot in [(33,19),(19,5)]:
    xe=4*(W+G)+W/2
    ax.plot([xe,xe],[ytop-0.2,ytop-3.2],color=EDGE,lw=0.8)
    ax.plot([xe,W/2],[ytop-3.2,ytop-3.2],color=EDGE,lw=0.8)
    arr(W/2,ytop-3.2,W/2,ybot+H+0.3)
ax.text(50,1.6,"Rejected images leave the cascade at each gate; only images passing every gate are saved as weak labels.",ha="center",fontsize=6.2,style="italic")
fig.savefig("figs/fig1_pipeline.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 2: results ----------------
fig,axs=plt.subplots(1,2,figsize=(7.2,2.6),gridspec_kw={"width_ratios":[1.55,1]})
ax=axs[0]
cfg=["25 GS\nonly","25 GS + 100 WL\n(box)","25 GS + 100 WL\n(points)","20 GS + 33 WL\n(cascade, DW loss)"]
pub=[0.3059,0.4661,0.5302,np.nan]; our=[0.4209,0.4631,0.6111,0.3905]
x=np.arange(4); w=0.36
b1=ax.bar(x-w/2,pub,w,color="#bdbdbd",edgecolor="#555",hatch="//",label="Published [1] (reference)")
b2=ax.bar(x+w/2,our,w,color="#4c78a8",edgecolor="#1f3a5a",label="Ours (eval.py, single seed)")
for xi,v in zip(x-w/2,pub):
    if not np.isnan(v): ax.text(xi,v+0.01,f"{v:.4f}",ha="center",fontsize=5.4)
for xi,v in zip(x+w/2,our): ax.text(xi,v+0.01,f"{v:.4f}",ha="center",fontsize=5.4)
ax.set_xticks(x); ax.set_xticklabels(cfg,fontsize=6.2); ax.set_ylabel("Dice"); ax.set_ylim(0,0.72)
ax.set_title("BUSI test set (published: 130 images, ours: 129)",fontsize=7); ax.legend(fontsize=6,loc="upper left",frameon=False)
ax.spines[["top","right"]].set_visible(False)
ax=axs[1]
x=np.arange(2); w=0.36
iou=[0.6585,0.5868]; dice=[0.6779,0.6149]
ax.bar(x-w/2,iou,w,color="#72b7b2",edgecolor="#2c5a57",label="IoU"); ax.bar(x+w/2,dice,w,color="#4c78a8",edgecolor="#1f3a5a",label="Dice")
for xi,v in zip(x-w/2,iou): ax.text(xi,v+0.01,f"{v:.4f}",ha="center",fontsize=5.4)
for xi,v in zip(x+w/2,dice): ax.text(xi,v+0.01,f"{v:.4f}",ha="center",fontsize=5.4)
ax.set_xticks(x); ax.set_xticklabels(["25 GS only","20 GS + 6 WL\n(cascade, DW loss)"],fontsize=6.2)
ax.set_ylim(0,0.85); ax.set_title("ISIC 2018 test set (518 images)",fontsize=7); ax.legend(fontsize=6,frameon=False,loc="upper right")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figs/fig2_results.png",dpi=300); plt.close(fig)

# ---------------- Fig 3: cascade ----------------
fig,ax=plt.subplots(figsize=(3.45,2.25))
labels=["BUSI\n(493 candidates)","ISIC\n(200 candidates)"]
data={"Complexity gate":[328,160],"Rotation inconsistency":[82,20],"Morphology":[11,3],"Quality predictor":[39,11],"Saved as weak label":[33,6]}
cols=["#9e9e9e","#e45756","#f58518","#b279a2","#54a24b"]
tot=np.array([493,200]); left=np.zeros(2)
for (k,v),c in zip(data.items(),cols):
    frac=np.array(v)/tot*100
    ax.barh(labels,frac,left=left,color=c,edgecolor="white",label=k)
    for i,(f,n) in enumerate(zip(frac,v)):
        if f>5: ax.text(left[i]+f/2,i,str(n),ha="center",va="center",fontsize=6,color="white")
    left+=frac
ax.set_xlim(0,100); ax.set_xlabel("% of candidate images",fontsize=6.5)
ax.legend(fontsize=5.4,ncol=2,loc="upper center",bbox_to_anchor=(0.5,1.42),frameon=False)
ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=6.2)
fig.savefig("figs/fig3_cascade.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 4: contrastive ----------------
fig,axs=plt.subplots(1,2,figsize=(3.45,1.9))
busi=[0.7060,0.6390,0.6077,0.5981,0.5730,0.5512,0.5257,0.5249,0.5178,0.5463,0.5137,0.5179,0.5020,0.5087,0.5229]
isic=[4.1071,4.0381,3.8357,3.6519,3.6013]
axs[0].plot(range(1,16),busi,marker="o",ms=2.2,lw=0.9,color="#4c78a8"); axs[0].set_title("BUSI (15 epochs)",fontsize=6.5)
axs[1].plot(range(1,6),isic,marker="o",ms=2.2,lw=0.9,color="#e45756"); axs[1].set_title("ISIC (5 epochs)",fontsize=6.5)
for a in axs:
    a.set_xlabel("epoch",fontsize=6.2); a.tick_params(labelsize=5.8); a.spines[["top","right"]].set_visible(False)
axs[0].set_ylabel("NT-Xent loss",fontsize=6.2)
fig.tight_layout(); fig.savefig("figs/fig4_contrastive.png",dpi=300); plt.close(fig)
print("ok")
