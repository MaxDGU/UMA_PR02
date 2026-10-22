#!/usr/bin/env python3
import pandas as pd, numpy as np, sys, json, os
sys.path.insert(0,'/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp')
from rescore_frac_dir import extract_final_answer, to_frac
JOB='/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc'; L='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/longhft'
h=pd.read_csv(f'{JOB}/emnlp-paper/tables/sp2013_fraction_cell_accuracy_profiles.csv'); h['cell']=(h.cell.str.lower().str[:3]+' '+h.cell.str.split().str[-1].str.upper()); href=h.set_index('cell')['Human']
def sc(p):
    d=pd.read_csv(p); c=d.correct_answer.apply(to_frac); d['ok']=[int(x is not None and x in extract_final_answer(r)) for r,x in zip(d.model_response,c)]
    d['cell']=(d.operation.str.lower().str[:3]+' '+d.denom_type.str.upper()); ca=d.groupby('cell').ok.mean()*100
    return float((ca-href.loc[ca.index]).abs().mean()), float(d.ok.mean()*100)
rows=[]
for arm,lab in [('C1000_tag50','1,000 learners'),('Acontrol_tag50','25 learners')]:
    for s in (42,43):
        tag=f'{arm}_kl1.0_ep10_seed{s}'; run=f'{L}/runs/{tag}'
        try: hist={e['epoch']:e['val_nll'] for e in json.load(open(f'{run}/history.json')) if e.get('val_nll') is not None}
        except Exception: hist={}
        try: be=json.load(open(f'{run}/summary_metrics.json')).get('best_epoch')
        except Exception: be=None
        for ck,ep in [('epoch_1',1),('epoch_3',3),('epoch_5',5),('epoch_10',10),('best',be)]:
            p=f'{L}/results/{tag}_{ck}.csv'
            if not os.path.exists(p): continue
            m,a=sc(p); rows.append(dict(arm=lab,seed=s,checkpoint=ck if ck!='best' else f'best (epoch {be})',epoch=ep,MAE=m,acc=a,child_NLL=hist.get(ep,np.nan)))
t=pd.DataFrame(rows); t.to_csv(f'{L}/long_per_seed.csv',index=False)
g=t.groupby(['arm','checkpoint'],sort=False).agg(MAE=('MAE','mean'),MAE_sd=('MAE','std'),acc=('acc','mean'),child_NLL=('child_NLL','mean'),n=('MAE','size')).round(2)
print(g.to_string()); print('\nper-epoch val NLL (from training history):')
for arm,lab in [('C1000_tag50','1,000'),('Acontrol_tag50','25')]:
    for s in (42,43):
        try: print(f'  {lab} seed {s}:',[round(e['val_nll'],3) for e in json.load(open(f'{L}/runs/{arm}_kl1.0_ep10_seed{s}/history.json')) if e.get('val_nll') is not None])
        except Exception: pass
print('\n1-epoch reference (4 seeds, spaced): 25 learners MAE 6.1 NLL 1.55 | 1,000 learners MAE 7.3 NLL 6.56')
