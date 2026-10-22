#!/usr/bin/env python3
"""Primary metric: KL-anchored human-FT (beta 1.0), seeds 42-45, all arms, spaced (training-format) evaluation."""
import pandas as pd, numpy as np, sys, json, os
from scipy import stats
sys.path.insert(0,'/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp')
from rescore_frac_dir import extract_final_answer, to_frac
JOB='/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc'; PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'; K=f'{PS}/kl_primary'; C=f'{PS}/c1000'
h=pd.read_csv(f'{JOB}/emnlp-paper/tables/sp2013_fraction_cell_accuracy_profiles.csv'); h['cell']=(h.cell.str.lower().str[:3]+' '+h.cell.str.split().str[-1].str.upper()); href=h.set_index('cell')['Human']
def sc(p):
    d=pd.read_csv(p); c=d.correct_answer.apply(to_frac); d['ok']=[int(x is not None and x in extract_final_answer(r)) for r,x in zip(d.model_response,c)]
    d['cell']=(d.operation.str.lower().str[:3]+' '+d.denom_type.str.upper()); ca=d.groupby('cell').ok.mean()*100
    return float((ca-href.loc[ca.index]).abs().mean()), float(d.ok.mean()*100), ca
def nll(r):
    try: return json.load(open(r+'/history.json'))[-1]['val_nll']
    except Exception: return np.nan
def src(arm,s):
    if arm=='Acontrol_tag50' and s in (42,43): return f'{C}/results_spaced/Adrop_kl1.0_seed{s}_dh.csv', f'{C}/hft_runs/Adrop_kl1.0_seed{s}'
    if arm=='C1000_tag50' and s in (42,43): return f'{C}/results_spaced/C1000drop_kl1.0_seed{s}_dh.csv', f'{C}/hft_runs/C1000drop_kl1.0_seed{s}'
    return f'{K}/results/{arm}_kl1.0_seed{s}_dh.csv', f'{K}/runs/{arm}_kl1.0_seed{s}'
LAB={'A_tag100':'A: 25 learners, tag 100%','Acontrol_tag50':'A control: 25 learners, tag 50%','B1000_tag100':'B1000: 1000 learners x 520, tag 100%','C1000_tag50':'C: 1000 learners x 20,720, tag 50%'}
rows=[]; per={}; cells={}
for arm,lab in LAB.items():
    R=[]
    for s in (42,43,44,45):
        p,r=src(arm,s)
        if os.path.exists(p): m,a,ca=sc(p); R.append((s,m,a,nll(r),ca))
    if not R: continue
    M=[x[1] for x in R]; per[arm]=M; cells[lab]=pd.concat([x[4] for x in R],axis=1).mean(axis=1)
    rows.append(dict(arm=lab,seeds=len(R),KL_MAE=f'{np.mean(M):.1f} ± {np.std(M,ddof=1):.1f}' if len(M)>1 else f'{M[0]:.1f}',acc=f'{np.mean([x[2] for x in R]):.1f}',
                     child_NLL=f'{np.nanmean([x[3] for x in R]):.3f}',per_seed=' '.join(f'{x:.1f}' for x in M)))
t=pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv(f'{K}/summary_KL_primary.csv',index=False)
ks=[k for k in per if len(per[k])>1]
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        r=stats.ttest_ind(per[ks[i]],per[ks[j]],equal_var=False); print(f'  {ks[i]} vs {ks[j]}: Welch p = {r.pvalue:.3f}')
pd.DataFrame({'Human':href,**cells}).round(1).to_csv(f'{K}/per_cell_KL_primary.csv'); print(pd.DataFrame({'Human':href,**{k.split(':')[0]:v for k,v in cells.items()}}).round(1).to_string())
