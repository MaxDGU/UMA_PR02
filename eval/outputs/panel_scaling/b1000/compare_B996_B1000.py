#!/usr/bin/env python3
"""Arm B (996 learners) vs arm B1000 (same rows + 4 regenerated learners), distill and distill+human-FT stages."""
import pandas as pd, numpy as np, sys, json, os
from scipy import stats
sys.path.insert(0,'/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp')
from rescore_frac_dir import extract_final_answer, to_frac
JOB='/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc'; PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'; K=f'{PS}/b1000'
TR='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication'
h=pd.read_csv(f'{JOB}/emnlp-paper/tables/sp2013_fraction_cell_accuracy_profiles.csv'); h['cell']=(h.cell.str.lower().str[:3]+' '+h.cell.str.split().str[-1].str.upper()); href=h.set_index('cell')['Human']
def sc(p):
    d=pd.read_csv(p); c=d.correct_answer.apply(to_frac); d['ok']=[int(x is not None and x in extract_final_answer(r)) for r,x in zip(d.model_response,c)]
    d['cell']=(d.operation.str.lower().str[:3]+' '+d.denom_type.str.upper()); ca=d.groupby('cell').ok.mean()*100
    return float((ca-href.loc[ca.index]).abs().mean()), float(d.ok.mean()*100), ca
def nll(r):
    try: return json.load(open(f'{r}/history.json'))[-1]['val_nll']
    except Exception: return np.nan
arms={'B996':dict(distill=f'{PS}/results/B_all996_sub_distill.csv', dnll=pd.read_csv(f'{PS}/fig/distill_nll.csv').set_index('checkpoint').loc['B all996 distill','nll_human_val'],
                  dh=lambda s:f'{PS}/results/B_all996_sub_dh_seed{s}.csv', run=lambda s:f'{PS}/hft_runs/B_all996_sub_seed{s}', summ=f'{TR}/qwen3_4b_B_all996_sub_v10_e1_v10regen_lora/summary_metrics.json'),
      'B1000':dict(distill=f'{K}/results/B1000_sub_distill.csv', dnll=pd.read_csv(f'{K}/distill_nll.csv').set_index('checkpoint').loc['B1000 distill','nll_human_val'],
                  dh=lambda s:f'{K}/results/B1000_sub_dh_seed{s}.csv', run=lambda s:f'{K}/hft_runs/B1000_sub_seed{s}', summ=f'{TR}/qwen3_4b_B1000_sub_v10_e1_v10regen_lora/summary_metrics.json')}
seed_rows=[]; stage_rows=[]; cells={}
for a,c in arms.items():
    s=json.load(open(c['summ'])); m,ac,ca=sc(c['distill']); cells[(a,'distill')]=ca
    stage_rows.append(dict(arm=a,stage='distill',train_rows=s['train_rows'],updates=s.get('best_updates_done'),synthetic_val_loss=round(s['best_val_loss'],4),MAE=round(m,1),acc=round(ac,1),child_val_NLL=round(c['dnll'],3)))
    cc=[]
    for sd in (42,43,44,45):
        m,ac,ca=sc(c['dh'](sd)); cc.append(ca); seed_rows.append(dict(arm=a,seed=sd,MAE=round(m,2),acc=round(ac,2),val_NLL=round(nll(c['run'](sd)),4)))
    cells[(a,'dh')]=pd.concat(cc,axis=1).mean(axis=1)
sr=pd.DataFrame(seed_rows)
for a in arms:
    t=sr[sr.arm==a]; stage_rows.append(dict(arm=a,stage='distill+humanFT (4 seeds)',MAE=f'{t.MAE.mean():.1f} ± {t.MAE.std(ddof=1):.1f}',acc=f'{t.acc.mean():.1f} ± {t.acc.std(ddof=1):.1f}',child_val_NLL=f'{t.val_NLL.mean():.3f} ± {t.val_NLL.std(ddof=1):.3f}'))
st=pd.DataFrame(stage_rows); print(st.to_string(index=False)); print('\nper seed:'); print(sr.pivot(index='seed',columns='arm',values=['MAE','acc','val_NLL']).to_string())
for col in ['MAE','acc','val_NLL']:
    x=sr[sr.arm=='B996'].sort_values('seed')[col].values; y=sr[sr.arm=='B1000'].sort_values('seed')[col].values
    print(f'{col}: mean diff (B1000 - B996) {np.mean(y-x):+.2f}; paired t p={stats.ttest_rel(y,x).pvalue:.2f}')
pc=pd.DataFrame({'Human':href, **{f'{a}_{s}':v for (a,s),v in cells.items()}}).round(1); print('\nper cell:'); print(pc.to_string())
st.to_csv(f'{K}/compare_stages.csv',index=False); sr.to_csv(f'{K}/compare_per_seed.csv',index=False); pc.to_csv(f'{K}/compare_per_cell.csv')
