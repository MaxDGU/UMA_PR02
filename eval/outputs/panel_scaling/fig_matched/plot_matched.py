#!/usr/bin/env python3
"""25 vs 1000 learners with every learner on all 20,720 problems (tag 50%). Stages: distill -> standard human FT (solid)
and distill -> KL-anchored human FT (dashed). Human-FT points: mean +- SD over seeds 42-45. Spaced (training-format) eval."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.lines import Line2D
sys.path.insert(0,'/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp')
from rescore_frac_dir import extract_final_answer, to_frac
JOB='/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc'; PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'; C=f'{PS}/c1000'; K=f'{PS}/kl_primary'
h=pd.read_csv(f'{JOB}/emnlp-paper/tables/sp2013_fraction_cell_accuracy_profiles.csv'); h['cell']=(h.cell.str.lower().str[:3]+' '+h.cell.str.split().str[-1].str.upper()); href=h.set_index('cell')['Human']
def sc(p):
    d=pd.read_csv(p); c=d.correct_answer.apply(to_frac); d['ok']=[int(x is not None and x in extract_final_answer(r)) for r,x in zip(d.model_response,c)]
    d['cell']=(d.operation.str.lower().str[:3]+' '+d.denom_type.str.upper()); ca=d.groupby('cell').ok.mean()*100
    return float((ca-href.loc[ca.index]).abs().mean()), float(d.ok.mean()*100)
def nll(r): return json.load(open(r+'/history.json'))[-1]['val_nll']
ARMS={'Adrop':("25 learners",'Acontrol_tag50'),'C1000drop':("1,000 learners",'C1000_tag50')}
S={}
for a,(lab,kname) in ARMS.items():
    m,acc=sc(f'{C}/results_spaced/{a}_distill.csv'); dn=json.load(open(f'{C}/results/{a}_distill_child_nll.json'))['nll_human_val']
    std=[(*sc(f'{C}/results_spaced/{a}_seed{s}_dh.csv'),nll(f'{C}/hft_runs/{a}_seed{s}')) for s in (42,43,44,45)]
    kl=[(*sc(f'{C}/results_spaced/{a}_kl1.0_seed{s}_dh.csv'),nll(f'{C}/hft_runs/{a}_kl1.0_seed{s}')) for s in (42,43)]+\
       [(*sc(f'{K}/results/{kname}_kl1.0_seed{s}_dh.csv'),nll(f'{K}/runs/{kname}_kl1.0_seed{s}')) for s in (44,45)]
    S[a]=dict(label=lab,d=dict(mae=m,acc=acc,nll=dn),
              std={k:(np.mean([x[i] for x in std]),np.std([x[i] for x in std],ddof=1)) for i,k in enumerate(['mae','acc','nll'])},
              kl={k:(np.mean([x[i] for x in kl]),np.std([x[i] for x in kl],ddof=1)) for i,k in enumerate(['mae','acc','nll'])})
sns.set_style('ticks'); sns.set_context('paper'); plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':8})
col={'Adrop':plt.cm.viridis(0.6),'C1000drop':plt.cm.viridis(0.0)}
def fig(panels,fname,ref=None):
    f,axes=plt.subplots(1,2,figsize=(3.5,2.8),dpi=300)
    for ax,(key,ylab) in zip(axes,panels):
        for a,s in S.items():
            for stage,ls,dx in [('std','-',-0.04),('kl','--',0.04)]:
                y0=s['d'][key]; y1,e1=s[stage][key]
                ax.plot([0,1+dx],[y0,y1],ls=ls,color=col[a],lw=1.4,marker='o',ms=3.2)
                ax.errorbar([1+dx],[y1],yerr=[e1],color=col[a],capsize=2,lw=1.0)
                ax.annotate(f"{y1:.1f}" if key!='nll' else f"{y1:.2f}",(1+dx,y1),xytext=(7,(0 if key=='mae' else (-4 if stage=='std' else 4))),textcoords='offset points',va='center',fontsize=6,color=col[a])
        if key=='acc' and ref: ax.axhline(ref,color='0.45',ls=':',lw=0.8)
        ax.set_xticks([0,1]); ax.set_xticklabels(['Distill','Distill +\nhuman FT'],fontsize=7.5); ax.set_xlim(-0.3,1.5); ax.set_ylim(bottom=0); ax.set_ylabel(ylab,fontsize=8); ax.tick_params(labelsize=7)
    hd=[Line2D([],[],color=col[a],lw=1.4,marker='o',ms=3,label=s['label']) for a,s in S.items()]+[Line2D([],[],color='0.3',ls='-',lw=1.2,label='standard human FT'),Line2D([],[],color='0.3',ls='--',lw=1.2,label='KL-anchored human FT')]
    if ref: hd.append(Line2D([],[],color='0.45',ls=':',lw=0.8,label=f'Children ({ref}%)'))
    sns.despine(); plt.tight_layout(rect=(0,0.15,1,1)); f.legend(handles=hd,loc='lower center',ncol=2,fontsize=5.8,frameon=False,bbox_to_anchor=(0.5,0.0))
    for ext in ('png','pdf'): f.savefig(f'{PS}/fig_matched/{fname}.{ext}',bbox_inches='tight')
fig([('mae','MAE vs. children (pp)'),('nll','Val NLL, held-out children')],'matched_25v1000_mae_nll')
fig([('nll','Val loss (NLL), held-out children'),('acc','Accuracy on SP2013 problems (%)')],'matched_25v1000_valloss_acc',ref=51.6)
for a,s in S.items(): print(s['label'],'distill',{k:round(v,2) for k,v in s['d'].items()},'| std',{k:(round(v[0],2),round(v[1],2)) for k,v in s['std'].items()},'| kl',{k:(round(v[0],2),round(v[1],2)) for k,v in s['kl'].items()})
