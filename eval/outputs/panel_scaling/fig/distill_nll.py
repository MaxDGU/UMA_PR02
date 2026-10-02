#!/usr/bin/env python3
"""Step-0 diagnostics for distill checkpoints: per-token NLL (response tokens only, eos appended,
same tokenization as finetune_humandata.py) on human train/val and on samples from three corpora,
plus mean next-token entropy over the human response positions."""
import sys, os, re, json, math, pandas as pd, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
R='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft'; T='/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/a50e2446/tmp'
TR=f'{R}/results/transformer_replication'
sys.path.insert(0,TR)
from train_transformer_hf import render_problem_for_prompt
CKPTS={'A paper25 distill':f'{TR}/qwen3_4b_p25regen_v10_e1_v10regen_lora/best',
 'B all996 distill':f'{TR}/qwen3_4b_B_all996_sub_v10_e1_v10regen_lora/best'}
BASE='Qwen/Qwen3-4B-Base'; MAXLEN=256
tok=AutoTokenizer.from_pretrained(BASE); tok.pad_token=tok.pad_token or tok.eos_token
def prompt(p): return f"Solve this fraction problem: {render_problem_for_prompt(str(p))}=?"
sets={}
for lab,f in [('human_train',f'{R}/data/human_ft/data_train_nlp.csv'),('human_val',f'{R}/data/human_ft/data_val_nlp.csv')]:
    d=pd.read_csv(f); sets[lab]=list(zip(d.prob.map(prompt),d.response_nl.astype(str)))
for lab in ['pub25','all996','v8p25']:
    d=pd.read_csv(f'{T}/corpus_sample_{lab}.csv'); sets['corpus_'+lab]=list(zip(d.prob.map(prompt),d.response_nl.astype(str)))
@torch.no_grad()
def score(model,pairs,bs=16):
    nll_sum=0.; ntok=0; ent_sum=0.
    for i in range(0,len(pairs),bs):
        ids_l=[];lab_l=[]
        for p,r in pairs[i:i+bs]:
            pi=tok.encode(p.strip(),add_special_tokens=False); ri=tok.encode(r.strip(),add_special_tokens=False)+[tok.eos_token_id]
            ri=ri[:max(1,MAXLEN-len(pi))]; ids=(pi+ri)[:MAXLEN]; lab=([-100]*len(pi)+ri)[:MAXLEN]; ids_l.append(ids); lab_l.append(lab)
        L=max(map(len,ids_l)); ids=torch.full((len(ids_l),L),tok.pad_token_id); labs=torch.full((len(ids_l),L),-100); att=torch.zeros((len(ids_l),L),dtype=torch.long)
        for k,(a,b) in enumerate(zip(ids_l,lab_l)): ids[k,:len(a)]=torch.tensor(a); labs[k,:len(b)]=torch.tensor(b); att[k,:len(a)]=1
        ids,labs,att=ids.cuda(),labs.cuda(),att.cuda()
        logits=model(input_ids=ids,attention_mask=att).logits[:,:-1].float(); tgt=labs[:,1:]; m=tgt.ne(-100)
        lp=F.log_softmax(logits,-1); nll=-lp.gather(-1,tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        ent=-(lp.exp()*lp).sum(-1)
        nll_sum+=nll[m].sum().item(); ent_sum+=ent[m].sum().item(); ntok+=m.sum().item()
    return nll_sum/ntok, ent_sum/ntok
rows=[]
for name,ck in CKPTS.items():
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16)
    if ck: model=PeftModel.from_pretrained(model,ck).merge_and_unload()
    model=model.cuda().eval()
    row={'checkpoint':name}
    for lab,pairs in sets.items():
        nll,ent=score(model,pairs); row[f'nll_{lab}']=round(nll,3); 
        if lab=='human_val': row['entropy_on_human_val']=round(ent,3)
    print(row,flush=True); rows.append(row)
    del model; torch.cuda.empty_cache()
out=pd.DataFrame(rows); out.to_csv('/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/fig/distill_nll.csv',index=False); print(out.to_string(index=False)); print('DONE')
