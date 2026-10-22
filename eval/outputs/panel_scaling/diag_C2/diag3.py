import sys, torch, pandas as pd, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
TR='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication'; PS='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling'
sys.path.insert(0,TR); from train_transformer_hf import build_student_prompt, render_problem_for_prompt
RUNS={'B1000':'qwen3_4b_B1000_sub_v10_e1_v10regen_lora','v8p1000':'qwen3_4b_p1000_v8_e1_v8scale2_lora','C':'qwen3_4b_C_all996_full_v10_e1_v10regen_lora'}
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Base')
r=pd.read_csv(f'{PS}/arms/B_all996_sub520k.csv.gz',nrows=50000,usecols=['prob','g','d','rt_mu','ice','response_nl']).sample(150,random_state=1)
fmt={'training format (student tag)':[build_student_prompt(str(p),g,d,rt,ice) for p,g,d,rt,ice in zip(r.prob,r.g,r.d,r.rt_mu,r.ice)],
     'plain, spaced problem':[f"Solve this fraction problem: {render_problem_for_prompt(str(p))}=?" for p in r.prob]}
print('example training prompt:',repr(fmt['training format (student tag)'][0]))
@torch.no_grad()
def tf(m,prompts):
    tot=0;n=0
    for p,y in zip(prompts,r.response_nl):
        pi=tok.encode(p.strip(),add_special_tokens=False); ri=tok.encode(str(y).strip(),add_special_tokens=False)+[tok.eos_token_id]
        ids=torch.tensor([pi+ri]).cuda(); lo=m(ids).logits[0,len(pi)-1:-1].float(); tot+=F.cross_entropy(lo,ids[0,len(pi):],reduction='sum').item(); n+=len(ri)
    return tot/n
@torch.no_grad()
def gen(m,p):
    e=tok([p],return_tensors='pt').to('cuda'); o=m.generate(**e,max_new_tokens=60,do_sample=False,pad_token_id=tok.eos_token_id); return tok.decode(o[0][e.input_ids.shape[1]:],skip_special_tokens=True)[:100]
tagged=build_student_prompt('3/5+1/4',0.05,0.5,5,50)
for k,v in RUNS.items():
    m=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Base',torch_dtype=torch.bfloat16),f'{TR}/{v}/best').merge_and_unload().cuda().eval()
    for f,ps in fmt.items(): print(f'  {k:8s} {f:32s} loss {tf(m,ps):.4f}',flush=True)
    print(f'  {k:8s} greedy, tagged prompt: {gen(m,tagged)!r}',flush=True)
    del m; torch.cuda.empty_cache()
print('DONE')
