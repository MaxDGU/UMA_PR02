#!/usr/bin/env python3
"""Score every arm of the panel-scaling study with the paper's fixed parser: 8-cell MAE vs SP2013 children + accuracy, plus human-FT val NLL."""
import pandas as pd, sys, os, glob, json, numpy as np
sys.path.insert(0,'/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp')
from rescore_frac_dir import extract_final_answer, to_frac
JOB='/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc'; FC=f'{JOB}/tmp/frac100'; PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'
h=pd.read_csv(f'{JOB}/emnlp-paper/tables/sp2013_fraction_cell_accuracy_profiles.csv')
h['cell']=(h.cell.str.lower().str[:3]+' '+h.cell.str.split().str[-1].str.upper()); href=h.set_index('cell')['Human']
def score(p):
    d=pd.read_csv(p); corr=d.correct_answer.apply(to_frac)
    d['ok']=[int(c is not None and c in extract_final_answer(r)) for r,c in zip(d.model_response,corr)]
    d['cell']=(d.operation.str.lower().str[:3]+' '+d.denom_type.str.upper()); ca=d.groupby('cell')['ok'].mean()*100
    return float((ca-href.loc[ca.index]).abs().mean()), float(d.ok.mean()*100)
def nll(run):
    try: return json.load(open(run+'/history.json'))[-1]['val_nll']
    except Exception: return np.nan
TR='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication'
arms=[  # label, distill csv, list of (dh csv, hft run dir)
 ('published 25 (paper ckpt, this harness)', None, [(f'{FC}/control_published_dh.csv', f'{TR}/qwen_humanft_from_distill_4b_ep1_20260503_124020')]),
 ('A  paper25 regen (518K)', f'{FC}/arm_v10_p25regen_distill.csv', [(f'{FC}/arm_v10_p25regen_dh.csv', f'{TR}/qwen_humanft_panel100bal_base_v10regen_p25')]+[(f'{FC}/arm_v10_p25regen_dh_seed{s}.csv', f'{FC}/hft_seeds/runs/p25regen_seed{s}') for s in (43,44,45)]),
 ('   balanced98 sub (518K)', f'{FC}/arm_v10_p98sub_distill.csv', [(f'{FC}/arm_v10_p98sub_dh.csv', f'{TR}/qwen_humanft_panel100bal_base_v10regen_p98sub')]+[(f'{FC}/arm_v10_p98sub_dh_seed{s}.csv', f'{FC}/hft_seeds/runs/p98sub_seed{s}') for s in (43,44,45)]),
 ('   balanced98 full (1.83M)', f'{FC}/arm_v10_p98_distill.csv', [(f'{FC}/arm_v10_p98_dh.csv', f'{TR}/qwen_humanft_panel100bal_base_v10regen_p98')]),
]
for arm in ['B_all996_sub','D_rand25_s42','D_rand25_s43','C_all996_full']:
    arms.append((f'{arm}', f'{PS}/results/{arm}_distill.csv', [(f'{PS}/results/{arm}_dh_seed{s}.csv', f'{PS}/hft_runs/{arm}_seed{s}') for s in (42,43,44,45)]))
rows=[]
for label,dcsv,dhs in arms:
    dm,da=score(dcsv) if dcsv and os.path.exists(dcsv) else (np.nan,np.nan)
    m=[score(c) for c,_ in dhs if os.path.exists(c)]; n=[nll(r) for c,r in dhs if os.path.exists(c)]
    rows.append(dict(arm=label, distill_MAE=round(dm,1), distill_acc=round(da,1), dh_MAE_mean=round(np.mean([x[0] for x in m]),1) if m else np.nan,
        dh_MAE_sd=round(np.std([x[0] for x in m]),1) if len(m)>1 else np.nan, dh_acc_mean=round(np.mean([x[1] for x in m]),1) if m else np.nan,
        hft_val_NLL=round(np.nanmean(n),3) if n else np.nan, n_seeds=len(m)))
out=pd.DataFrame(rows); out.to_csv(f'{PS}/results/summary.csv',index=False); print(out.to_string(index=False)); print('\nchildren: 51.6% accuracy; paper reports MAE 5.2 / 46.9% / NLL 1.361')
