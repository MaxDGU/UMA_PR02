#!/usr/bin/env python3
"""Convert saved grade-6 UMA learners (xlsx: subj/proc/ans sheets) into the
model_subjid_N.pkl.gz payload that run_saved_models_on_problem_set.py loads.
Payload = {proc_mem, ans_mem, params, rule_names}; rule order follows all_rules
(the ordering validated by gen_from_saved.py, 47.0% vs 46.5% published)."""
import argparse, gzip, json, os, pickle, sys, glob, re
import pandas as pd
REPO="/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft"
UMA_DIR=os.path.join(REPO,"1. Model")
MODELS="/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractionGPT_repo/uma_trained_students"
ap=argparse.ArgumentParser()
ap.add_argument("--out-dir",required=True)
ap.add_argument("--subjids",default="all",help="'all' or comma list")
a=ap.parse_args()
os.chdir(UMA_DIR); sys.path.insert(0,UMA_DIR)
from models import all_rules
os.makedirs(a.out_dir,exist_ok=True)
files=sorted(glob.glob(os.path.join(MODELS,"sim subjid_* grade_6 model.xlsx")))
want=None if a.subjids=="all" else {int(x) for x in a.subjids.split(",")}
n=0
for f in files:
    sid=int(re.search(r"subjid_(\d+) ",os.path.basename(f)).group(1))
    if want is not None and sid not in want: continue
    out=os.path.join(a.out_dir,f"model_subjid_{sid}.pkl.gz")
    if os.path.exists(out): n+=1; continue
    S=pd.read_excel(f,sheet_name='subj',index_col=0)
    P=pd.read_excel(f,sheet_name='proc',index_col=0)
    A=pd.read_excel(f,sheet_name='ans',index_col=0)
    raw=S['params'].iloc[0]; params=json.loads(raw) if isinstance(raw,str) else raw
    rule_names=[r.name for r in all_rules if r.name in P.columns]
    assert set(rule_names)==set(P.columns), (sid, set(P.columns)-set(rule_names))
    payload={"subjid":sid,"proc_mem":P,"ans_mem":A,"params":params,"rule_names":rule_names}
    with gzip.open(out,"wb") as fh: pickle.dump(payload,fh)
    n+=1
    if n%100==0: print(f"  {n} converted",flush=True)
print(f"DONE {n} models -> {a.out_dir}")
