#!/usr/bin/env python3
"""Arm C (all 1000 learners x all 20,720 problems, half-tagged prompts) vs arm A control (25 learners, same prompt setting).
Distill stage: MAE/acc on SP2013 (256 samples x 16) + NLL on held-out children. Human-FT: 4 standard seeds, 2 KL-anchored seeds."""
import pandas as pd, numpy as np, sys, json, os
from scipy import stats
sys.path.insert(0,'/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp')
from rescore_frac_dir import extract_final_answer, to_frac
JOB='/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc'; C='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000'
h=pd.read_csv(f'{JOB}/emnlp-paper/tables/sp2013_fraction_cell_accuracy_profiles.csv'); h['cell']=(h.cell.str.lower().str[:3]+' '+h.cell.str.split().str[-1].str.upper()); href=h.set_index('cell')['Human']
def sc(p):
    d=pd.read_csv(p); c=d.correct_answer.apply(to_frac); d['ok']=[int(x is not None and x in extract_final_answer(r)) for r,x in zip(d.model_response,c)]
    d['cell']=(d.operation.str.lower().str[:3]+' '+d.denom_type.str.upper()); ca=d.groupby('cell').ok.mean()*100
    return float((ca-href.loc[ca.index]).abs().mean()), float(d.ok.mean()*100), ca
def nll(r):
    try: return json.load(open(r+'/history.json'))[-1]['val_nll']
    except Exception: return np.nan
rows=[]; cells={}; per={}
for arm,lab in [('Adrop','A control: 25 learners'),('C1000drop','C: 1000 learners, all problems')]:
    m,a,ca=sc(f'{C}/results_spaced/{arm}_distill.csv'); dn=json.load(open(f'{C}/results/{arm}_distill_child_nll.json'))['nll_human_val']; cells[(lab,'distill')]=ca
    rows.append(dict(arm=lab,stage='distill',MAE=f'{m:.1f}',acc=f'{a:.1f}',child_NLL=f'{dn:.3f}',n=1))
    for st,tags in [('human-FT',[f'seed{s}' for s in (42,43,44,45)]),('KL human-FT',[f'kl1.0_seed{s}' for s in (42,43)])]:
        R=[(sc(f'{C}/results_spaced/{arm}_{t}_dh.csv'),nll(f'{C}/hft_runs/{arm}_{t}')) for t in tags if os.path.exists(f'{C}/results_spaced/{arm}_{t}_dh.csv')]
        if not R: continue
        M=[x[0][0] for x in R]; A=[x[0][1] for x in R]; N=[x[1] for x in R]; per[(lab,st)]=M; cells[(lab,st)]=pd.concat([x[0][2] for x in R],axis=1).mean(axis=1)
        rows.append(dict(arm=lab,stage=st,MAE=f'{np.mean(M):.1f} ± {np.std(M,ddof=1):.1f}' if len(M)>1 else f'{M[0]:.1f}',acc=f'{np.mean(A):.1f}',child_NLL=f'{np.nanmean(N):.3f}',n=len(R)))
t=pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv(f'{C}/results/summary_C_vs_Acontrol_SPACED.csv',index=False)
for st in ('human-FT','KL human-FT'):
    a=per.get(('A control: 25 learners',st)); b=per.get(('C: 1000 learners, all problems',st))
    if a and b and len(a)>1 and len(b)>1: print(f'{st}: Welch MAE C vs A control t={stats.ttest_ind(b,a,equal_var=False).statistic:.2f} p={stats.ttest_ind(b,a,equal_var=False).pvalue:.3f}')
pc=pd.DataFrame({'Human':href, **{f'{k[0][:9]}|{k[1]}':v for k,v in cells.items()}}).round(1); print(pc.to_string()); pc.to_csv(f'{C}/results/per_cell_C_vs_Acontrol_SPACED.csv')
print('\nchildren 51.6% accuracy. Reference (tag on every prompt, 518K rows): A 7.6 +- 2.1, B1000 20.0 +- 2.0, paper ckpt 7.2')
p=f'{C}/results_spaced/B1000_check_seed42_dh.csv'
if os.path.exists(p): m,a,_=sc(p); print(f'B1000 seed 42 human-FT: spaced MAE {m:.1f} acc {a:.1f}  (unspaced original 18.7 / 32.9)')
