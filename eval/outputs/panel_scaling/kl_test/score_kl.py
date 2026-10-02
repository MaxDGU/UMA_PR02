#!/usr/bin/env python3
"""Test 2: does keeping human-FT close to the distilled model (KL penalty) close arm B's gap?"""
import pandas as pd, numpy as np, sys, json, os, re
from fractions import Fraction
sys.path.insert(0,'/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp')
from rescore_frac_dir import extract_final_answer, to_frac
JOB='/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc'; FC=f'{JOB}/tmp/frac100'; PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'; K=f'{PS}/kl_test'
h=pd.read_csv(f'{JOB}/emnlp-paper/tables/sp2013_fraction_cell_accuracy_profiles.csv')
h['cell']=(h.cell.str.lower().str[:3]+' '+h.cell.str.split().str[-1].str.upper()); href=h.set_index('cell')['Human']
def recog(prob,ans,op):
    (a,b),(c,d)=[tuple(map(int,x)) for x in re.findall(r'(\d+)/(\d+)',prob)]; f=to_frac(str(ans)); s=1 if op=='Add' else -1
    if f is None: return False
    c_=[Fraction(a,b)+s*Fraction(c,d),Fraction(a+s*c,b),Fraction(a+s*c,d),Fraction(a+s*c,b*d)]+([Fraction(a+s*c,b+s*d)] if b+s*d else [])
    return f in c_
def score(p):
    x=pd.read_csv(p); c=x.correct_answer.apply(to_frac)
    x['ok']=[int(v is not None and v in extract_final_answer(r)) for r,v in zip(x.model_response,c)]
    x['cell']=(x.operation.str.lower().str[:3]+' '+x.denom_type.str.upper()); ca=x.groupby('cell').ok.mean()*100
    u=x[x.operation.isin(['Add','Sub'])&(x.denom_type=='UD')]
    incoh=np.mean([not recog(p_,a,o) for p_,a,o in zip(u.problem,u.parsed_answer,u.operation)])*100
    return dict(MAE=float((ca-href.loc[ca.index]).abs().mean()),acc=x.ok.mean()*100,UD_addsub=float(ca[['add UD','sub UD']].mean()),
                incoherent_UD=incoh,words=float(x.model_response.astype(str).str.split().str.len().median()))
def nll(r):
    try: return json.load(open(r+'/history.json'))[-1]['val_nll']
    except Exception: return np.nan
runs=[]
for s in (42,43):
    runs.append(('A',0.0,s,f'{FC}/arm_v10_p25regen_dh.csv' if s==42 else f'{FC}/arm_v10_p25regen_dh_seed{s}.csv',
                 '/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication/qwen_humanft_panel100bal_base_v10regen_p25' if s==42 else f'{FC}/hft_seeds/runs/p25regen_seed{s}'))
    runs.append(('B',0.0,s,f'{PS}/results/B_all996_sub_dh_seed{s}.csv',f'{PS}/hft_runs/B_all996_sub_seed{s}'))
    for arm in 'AB':
        for b in ('0.1','1.0'): runs.append((arm,float(b),s,f'{K}/{arm}_kl{b}_seed{s}.csv',f'{K}/runs/{arm}_kl{b}_seed{s}'))
rows=[]
for arm,b,s,f,r in runs:
    if not os.path.exists(f): rows.append(dict(arm=arm,kl_beta=b,seed=s,status='missing')); continue
    d=score(f); d.update(arm=arm,kl_beta=b,seed=s,val_NLL=nll(r),status='ok'); rows.append(d)
t=pd.DataFrame(rows); t.to_csv(f'{K}/kl_per_run.csv',index=False)
ok=t[t.status=='ok']
agg=ok.groupby(['arm','kl_beta'])[['MAE','acc','UD_addsub','incoherent_UD','words','val_NLL']].mean().round(2); agg['n']=ok.groupby(['arm','kl_beta']).size()
print(agg.to_string()); print('\nmissing:',t[t.status!='ok'][['arm','kl_beta','seed']].values.tolist())
for b in sorted(ok.kl_beta.unique()):
    try: print(f'B-A MAE gap at kl_beta={b}: {agg.loc[("B",b),"MAE"]-agg.loc[("A",b),"MAE"]:.1f}')
    except KeyError: pass
print('children: acc 51.6, UD add/sub', round(float(href[['add UD','sub UD']].mean()),1))
