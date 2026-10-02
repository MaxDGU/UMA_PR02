#!/usr/bin/env python3
"""Arm C diagnostic: separate (merge-into-bf16 vs unmerged LoRA) x (greedy vs T=0.7 vs T=1.0) on SP2013 problems,
plus teacher-forced NLL on corpus rows for merged vs unmerged, plus LoRA delta norms. Arm A as reference."""
import sys, json, re, torch, pandas as pd, numpy as np, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0,'/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp')
from rescore_frac_dir import extract_final_answer, to_frac
TR='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication'
PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'
ARMS={'C':f'{TR}/qwen3_4b_C_all996_full_v10_e1_v10regen_lora/best','A':f'{TR}/qwen3_4b_p25regen_v10_e1_v10regen_lora/best'}
BASE='Qwen/Qwen3-4B-Base'; N=16
probs=pd.read_csv(f'{PS}/results/C_all996_full_distill.csv').drop_duplicates('problem')[['problem','correct_answer']]
corpus=pd.read_csv(f'{PS}/arms/A_paper25_from996.csv.gz',nrows=200000).sample(200,random_state=0)
tok=AutoTokenizer.from_pretrained(BASE); tok.padding_side='left'; tok.pad_token=tok.pad_token or tok.eos_token
def lora_norms(m):
    r=[]
    for n,mod in m.named_modules():
        if hasattr(mod,'lora_A') and 'default' in getattr(mod,'lora_A',{}):
            A=mod.lora_A['default'].weight.float(); B=mod.lora_B['default'].weight.float(); s=mod.scaling['default']
            W=mod.base_layer.weight.float(); d=(B@A)*s; r.append((d.norm()/W.norm()).item())
    return float(np.mean(r)), float(np.max(r))
@torch.no_grad()
def tf_nll(m):
    tot=0;n=0
    for p,y in zip(corpus.instruction_nl,corpus.response_nl):
        pi=tok.encode(str(p).strip(),add_special_tokens=False); ri=tok.encode(str(y).strip(),add_special_tokens=False)+[tok.eos_token_id]
        ids=torch.tensor([pi+ri]).cuda(); lo=m(ids).logits[0,len(pi)-1:-1].float()
        tot+=F.cross_entropy(lo,ids[0,len(pi):],reduction='sum').item(); n+=len(ri)
    return tot/n
@torch.no_grad()
def roll(m,temp):
    ok=0;tot=0;ex=None
    prompts=[f"Solve this fraction problem: {p}=?" for p in probs.problem]
    for p,c in zip(prompts,probs.correct_answer):
        enc=tok([p]*(1 if temp==0 else N),return_tensors='pt',padding=True).to('cuda')
        kw=dict(do_sample=False) if temp==0 else dict(do_sample=True,temperature=temp,top_p=0.95)
        out=m.generate(**enc,max_new_tokens=192,pad_token_id=tok.pad_token_id,**kw)
        for o in out:
            t=tok.decode(o[enc.input_ids.shape[1]:],skip_special_tokens=True); ex=ex or t
            cf=to_frac(c); ok+=int(cf is not None and cf in extract_final_answer(t)); tot+=1
    return round(100*ok/tot,1), ex[:160]
rows=[]
for arm,ck in ARMS.items():
    base=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16).cuda()
    pm=PeftModel.from_pretrained(base,ck).eval()
    mn,mx=lora_norms(pm); print(arm,'lora |dW|/|W| mean %.4f max %.4f'%(mn,mx),flush=True)
    r={'arm':arm,'dW_rel_mean':round(mn,4),'dW_rel_max':round(mx,4),'tfNLL_unmerged':round(tf_nll(pm),3)}
    for t in (0,0.7,1.0): r[f'unmerged_T{t}'],r[f'ex_unmerged_T{t}']=roll(pm,t); print(arm,'unmerged T',t,r[f'unmerged_T{t}'],repr(r[f'ex_unmerged_T{t}'][:100]),flush=True)
    mm=pm.merge_and_unload().eval(); r['tfNLL_merged']=round(tf_nll(mm),3)
    for t in (0,1.0): r[f'merged_T{t}'],r[f'ex_merged_T{t}']=roll(mm,t); print(arm,'merged T',t,r[f'merged_T{t}'],repr(r[f'ex_merged_T{t}'][:100]),flush=True)
    rows.append(r); del base,pm,mm; torch.cuda.empty_cache()
out=pd.DataFrame(rows); out.to_csv(f'{PS}/diag_C/diag_results.csv',index=False)
print(out[[c for c in out.columns if not c.startswith('ex_')]].to_string(index=False)); print('DONE')
