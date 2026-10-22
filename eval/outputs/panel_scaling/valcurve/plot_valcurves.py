#!/usr/bin/env python3
"""Distillation-time curves for arm A (paper's 25) vs arm B (all 996): synthetic val loss, child val NLL,
SP2013 accuracy and MAE at each saved checkpoint (update 0 = base Qwen3-4B)."""
import json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
TR='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication'
V='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/valcurve'
def hist(arm):
    d=f'{TR}/qwen3_4b_{arm}_v10_e1_valcurve_lora'
    for p in [f'{d}/history.json',f'{d}/final/history.json',f'{d}/best/history.json']:
        try:
            h=pd.DataFrame(json.load(open(p))); h=h[h.val_loss.notna()]
            return h[['updates_done','val_loss']].drop_duplicates('updates_done').sort_values('updates_done')
        except Exception: pass
    import re
    rows=[]
    for f in glob.glob(f'{V}/logs/distill_{arm}_*.out'):
        for line in open(f):
            m=re.search(r'\[eval full e\d+_u(\d+)\].*? val=([0-9.]+)',line)
            if m: rows.append((int(m.group(1)),float(m.group(2))))
    return pd.DataFrame(rows,columns=['updates_done','val_loss']).drop_duplicates('updates_done').sort_values('updates_done')
sns.set_style('ticks'); sns.set_context('paper'); plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':8})
c=plt.cm.viridis([0.6,0.0]); style={'A':dict(color=c[0],lw=1.1,label="Paper's 25 learners (arm A)"),'B':dict(color=c[1],lw=1.5,label='All 996 learners (arm B)')}
fig,ax=plt.subplots(2,2,figsize=(3.5,3.3),dpi=300,sharex=True); ax=ax.ravel()
out={}
for arm in 'AB':
    h=hist(arm); s=pd.read_csv(f'{V}/ckpt_scores_{arm}.csv').sort_values('update'); out[arm]=(h,s)
    ax[0].plot(h.updates_done,h.val_loss,marker='o',ms=2,**style[arm])
    ax[1].plot(s['update'],s.child_val_nll,marker='o',ms=2,**style[arm])
    ax[2].plot(s['update'],s.sp2013_acc,marker='o',ms=2,**style[arm])
    ax[3].plot(s['update'],s.sp2013_mae,marker='o',ms=2,**style[arm])
ax[0].set_ylabel('Val loss, held-out\nsynthetic traces',fontsize=7)
ax[1].set_ylabel('Val NLL, held-out\nchildren',fontsize=7); ax[2].set_ylabel('SP2013 accuracy (%)',fontsize=7); ax[3].set_ylabel('MAE vs. children (pp)',fontsize=7)
ax[2].axhline(51.6,color='0.45',ls='--',lw=0.7,label='Children (51.6%)')
for a in ax[2:]: a.set_xlabel('Distillation updates',fontsize=7)
for a in ax: a.tick_params(labelsize=6)
h1,l1=ax[0].get_legend_handles_labels(); h2,l2=ax[2].get_legend_handles_labels(); h1+=[h2[-1]]; l1+=[l2[-1]]
sns.despine(); plt.tight_layout(rect=(0,0.09,1,1)); fig.legend(h1,l1,loc='lower center',ncol=2,fontsize=5.8,frameon=False,bbox_to_anchor=(0.5,0.0))
for ext in ('png','pdf'): fig.savefig(f'{V}/distill_valcurves_A_vs_B.{ext}',bbox_inches='tight')
for arm,(h,s) in out.items(): print(arm,'synthetic val points',len(h),'| checkpoints scored',len(s)); print(s.to_string(index=False))
