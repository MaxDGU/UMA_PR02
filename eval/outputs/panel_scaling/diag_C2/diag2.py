#!/usr/bin/env python3
"""Why does arm C's saved adapter fail on reload? Test reload precision variants and compare adapter norms with working runs."""
import sys, json, torch, pandas as pd, numpy as np, torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
TR='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication'; PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'
RUNS={'A':'qwen3_4b_p25regen_v10_e1_v10regen_lora','B1000':'qwen3_4b_B1000_sub_v10_e1_v10regen_lora','v8p1000':'qwen3_4b_p1000_v8_e1_v8scale2_lora','C':'qwen3_4b_C_all996_full_v10_e1_v10regen_lora'}
print('=== adapter norms (product |B@A|*scale / |W| needs base; here raw |A|,|B|)')
for k,r in RUNS.items():
    sd=load_file(f'{TR}/{r}/best/adapter_model.safetensors'); A=[v.float() for n,v in sd.items() if 'lora_A' in n]; B=[v.float() for n,v in sd.items() if 'lora_B' in n]
    print(f'  {k:8s} dtype {sorted({str(v.dtype) for v in sd.values()})} nonfinite {sum(int(~torch.isfinite(v.float()).all()) for v in sd.values())} |A| mean {np.mean([a.norm().item() for a in A]):.2f} |B| mean {np.mean([b.norm().item() for b in B]):.2f} |B| max {max(b.norm().item() for b in B):.2f}',flush=True)
BASE='Qwen/Qwen3-4B-Base'; tok=AutoTokenizer.from_pretrained(BASE)
rows=pd.read_csv(f'{PS}/arms/B_all996_sub520k.csv.gz',nrows=50000,usecols=['instruction_nl','response_nl']).sample(150,random_state=1)
@torch.no_grad()
def tf(m):
    tot=0;n=0
    for p,y in zip(rows.instruction_nl,rows.response_nl):
        pi=tok.encode(str(p).strip(),add_special_tokens=False); ri=tok.encode(str(y).strip(),add_special_tokens=False)+[tok.eos_token_id]
        ids=torch.tensor([pi+ri]).cuda(); lo=m(ids).logits[0,len(pi)-1:-1].float(); tot+=F.cross_entropy(lo,ids[0,len(pi):],reduction='sum').item(); n+=len(ri)
    return tot/n
@torch.no_grad()
def gen(m):
    e=tok(["Solve this fraction problem: 3/5 + 1/4=?"],return_tensors='pt').to('cuda'); o=m.generate(**e,max_new_tokens=60,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][e.input_ids.shape[1]:],skip_special_tokens=True)[:110]
print('=== reload variants: teacher-forced loss on 150 corpus rows (trainer reported val ~0.02-0.03) + greedy sample',flush=True)
for arm in ['B1000','C']:
    for base_dt,ad_dt,merge in [(torch.bfloat16,None,False),(torch.float32,None,False),(torch.float32,torch.float32,True),(torch.bfloat16,None,True)]:
        base=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=base_dt)
        m=PeftModel.from_pretrained(base,f'{TR}/{RUNS[arm]}/best')
        if ad_dt is not None: m=m.to(ad_dt)
        if merge: m=m.merge_and_unload()
        m=m.cuda().eval(); tag=f'{arm} base={str(base_dt).split(".")[-1]} merged={merge}'
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=(base_dt==torch.bfloat16)):
            l=tf(m); g=gen(m)
        print(f'  {tag:40s} loss {l:.4f} | {g!r}',flush=True); del m,base; torch.cuda.empty_cache()
print('DONE')
