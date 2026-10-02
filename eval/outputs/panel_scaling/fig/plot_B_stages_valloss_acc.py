#!/usr/bin/env python3
"""Arm B (all 996 learners) vs arm A (paper's 25): MAE and NLL at the distill and distill+human-FT stages.
Distill stage = one checkpoint (no seeds); human-FT stage = mean ± SD over seeds 42-45."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0,'/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp')
from rescore_frac_dir import extract_final_answer, to_frac
PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'; JOB='/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc'; FC=f'{JOB}/tmp/frac100'
h=pd.read_csv(f'{JOB}/emnlp-paper/tables/sp2013_fraction_cell_accuracy_profiles.csv')
h['cell']=(h.cell.str.lower().str[:3]+' '+h.cell.str.split().str[-1].str.upper()); href=h.set_index('cell')['Human']
def mae_acc(p):
    d=pd.read_csv(p); c=d.correct_answer.apply(to_frac)
    d['ok']=[int(x is not None and x in extract_final_answer(r)) for r,x in zip(d.model_response,c)]
    d['cell']=(d.operation.str.lower().str[:3]+' '+d.denom_type.str.upper()); ca=d.groupby('cell').ok.mean()*100
    return float((ca-href.loc[ca.index]).abs().mean()), float(d.ok.mean()*100)
seeds=pd.read_csv(f'{PS}/results/A_vs_B_per_seed.csv'); dn=pd.read_csv(f'{PS}/fig/distill_nll.csv').set_index('checkpoint')
S={'A':dict(label="Paper's 25 learners (arm A)", dacc=mae_acc(f'{FC}/arm_v10_p25regen_distill.csv')[1], dnll=dn.loc['A paper25 distill','nll_human_val']),
   'B':dict(label='All 996 learners (arm B)', dacc=mae_acc(f'{PS}/results/B_all996_sub_distill.csv')[1], dnll=dn.loc['B all996 distill','nll_human_val'])}
for k in S:
    t=seeds[seeds.arm==k]; S[k].update(hacc=t.acc.mean(), hacc_sd=t.acc.std(ddof=1), hnll=t.val_NLL.mean(), hnll_sd=t.val_NLL.std(ddof=1), n=len(t))
sns.set_style('ticks'); sns.set_context('paper'); plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':8})
cols=plt.cm.viridis([0.0,0.6]); col={'B':cols[0],'A':cols[1]}
fig,axes=plt.subplots(1,2,figsize=(3.5,2.8),dpi=300); x=[0,1]; xt=['Distill','Distill +\nhuman FT']
for ax,(key,ylab) in zip(axes,[('nll','Val loss (NLL), held-out children'),('acc','Accuracy on SP2013 problems (%)')]):
    for arm,alpha,lw,z in [('A',0.75,1.0,1),('B',1.0,1.6,2)]:
        s=S[arm]; y=[s['d'+key],s['h'+key]]; e=[0,s['h'+key+'_sd']]
        ax.errorbar(x,y,yerr=e,color=col[arm],alpha=alpha,lw=lw,marker='o',ms=3.5,capsize=2,zorder=z,label=s['label'])
        dy={'acc':0,'nll':(5 if arm=='A' else -5)}[key]
        ax.annotate(f"{y[1]:.1f}" if key=='acc' else f"{y[1]:.2f}",(1,y[1]),xytext=(6,dy),textcoords='offset points',va='center',fontsize=7,color=col[arm])
    ax.set_xticks(x); ax.set_xticklabels(xt,fontsize=7.5); ax.set_xlim(-0.3,1.45); ax.set_ylabel(ylab,fontsize=8); ax.tick_params(labelsize=7)
axes[0].set_ylim(bottom=0); axes[1].set_ylim(0,70)
axes[1].axhline(51.6,color='0.45',ls='--',lw=0.8,label='Children (51.6%)')
h,l=axes[0].get_legend_handles_labels(); h2,l2=axes[1].get_legend_handles_labels(); h+= [x for x,y in zip(h2,l2) if y.startswith('Children')]; l+=['Children (51.6%)']
sns.despine(); plt.tight_layout(rect=(0,0.1,1,1))
fig.legend(h,l,loc='lower center',ncol=2,fontsize=6,frameon=False,handlelength=1.5,bbox_to_anchor=(0.5,0.0))
for ext in ('png','pdf'): fig.savefig(f'{PS}/fig/armB_vs_A_valloss_acc.{ext}',bbox_inches='tight')
print({k:{kk:(round(v,3) if isinstance(v,float) else v) for kk,v in s.items()} for k,s in S.items()})
