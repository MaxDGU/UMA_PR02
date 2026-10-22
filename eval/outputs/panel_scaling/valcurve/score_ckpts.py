#!/usr/bin/env python3
"""Score every saved distillation checkpoint of one arm: NLL on the 96 held-out child responses (same tokenization
as human-FT) and SP2013 accuracy/MAE from 64 samples per problem (T=1.0, top_p=0.95). Base model included as update 0."""
import sys, glob, os, re, json, torch, pandas as pd, numpy as np, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0,'/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp')
from rescore_frac_dir import extract_final_answer, to_frac
R='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft'; TR=f'{R}/results/transformer_replication'
PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'; JOB='/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc'
sys.path.insert(0,TR); from train_transformer_hf import render_problem_for_prompt
ARM=sys.argv[1]; OUT=f'{TR}/qwen3_4b_{ARM}_v10_e1_valcurve_lora'; BASE='Qwen/Qwen3-4B-Base'; NS=64
tok=AutoTokenizer.from_pretrained(BASE); tok.pad_token=tok.pad_token or tok.eos_token
hv=pd.read_csv(f'{R}/data/human_ft/data_val_nlp.csv'); pairs=[(f"Solve this fraction problem: {render_problem_for_prompt(str(p))}=?",str(y)) for p,y in zip(hv.prob,hv.response_nl)]
probs=pd.read_csv(f'{PS}/results/B_all996_sub_distill.csv').drop_duplicates('problem')[['problem','operation','denom_type','correct_answer']]
h=pd.read_csv(f'{JOB}/emnlp-paper/tables/sp2013_fraction_cell_accuracy_profiles.csv'); h['cell']=(h.cell.str.lower().str[:3]+' '+h.cell.str.split().str[-1].str.upper()); href=h.set_index('cell')['Human']
@torch.no_grad()
def child_nll(m):
    tot=0.;n=0
    for p,y in pairs:
        pi=tok.encode(p.strip(),add_special_tokens=False); ri=tok.encode(y.strip(),add_special_tokens=False)+[tok.eos_token_id]
        ids=torch.tensor([(pi+ri)[:256]]).cuda(); L=ids.shape[1]-len(pi)
        lo=m(ids).logits[0,len(pi)-1:-1].float(); tot+=F.cross_entropy(lo,ids[0,len(pi):],reduction='sum').item(); n+=L
    return tot/n
@torch.no_grad()
def sp2013(m):
    tok.padding_side='left'; rows=[]
    for r in probs.itertuples():
        enc=tok([f"Solve this fraction problem: {r.problem}=?"]*NS,return_tensors='pt',padding=True).to('cuda')
        out=m.generate(**enc,do_sample=True,temperature=1.0,top_p=0.95,max_new_tokens=192,pad_token_id=tok.pad_token_id)
        c=to_frac(r.correct_answer)
        for o in out:
            t=tok.decode(o[enc.input_ids.shape[1]:],skip_special_tokens=True); rows.append((r.operation.lower()[:3]+' '+r.denom_type.upper(), int(c is not None and c in extract_final_answer(t))))
    d=pd.DataFrame(rows,columns=['cell','ok']); ca=d.groupby('cell').ok.mean()*100
    return float((ca-href.loc[ca.index]).abs().mean()), float(d.ok.mean()*100)
ck=sorted(glob.glob(f'{OUT}/eval_e01_u*'),key=lambda p:int(re.search(r'_u(\d+)$',p).group(1)))
print(ARM,'checkpoints:',[os.path.basename(c) for c in ck],flush=True)
res=[]; outcsv=f'{PS}/valcurve/ckpt_scores_{ARM}.csv'
for c in [None]+ck:
    torch.manual_seed(0)
    base=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16)
    m=(PeftModel.from_pretrained(base,c).merge_and_unload() if c else base).cuda().eval()
    u=0 if c is None else int(re.search(r'_u(\d+)$',c).group(1))
    nll=child_nll(m); mae,acc=sp2013(m); res.append(dict(arm=ARM,update=u,child_val_nll=round(nll,4),sp2013_mae=round(mae,2),sp2013_acc=round(acc,2)))
    print(res[-1],flush=True); pd.DataFrame(res).to_csv(outcsv,index=False)
    del m,base; torch.cuda.empty_cache()
print('DONE')
