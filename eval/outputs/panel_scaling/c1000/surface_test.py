import sys, torch, pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
TR='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/results/transformer_replication'; C='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000'; JOB='/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc'
sys.path.insert(0,TR); sys.path.insert(0,f'{JOB}/tmp')
from train_transformer_hf import render_problem_for_prompt
from rescore_frac_dir import extract_final_answer, to_frac
h=pd.read_csv(f'{JOB}/emnlp-paper/tables/sp2013_fraction_cell_accuracy_profiles.csv'); h['cell']=(h.cell.str.lower().str[:3]+' '+h.cell.str.split().str[-1].str.upper()); href=h.set_index('cell')['Human']
probs=pd.read_csv(f'{C}/results/C1000drop_distill.csv').drop_duplicates('problem')[['problem','operation','denom_type','correct_answer']]
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Base'); tok.padding_side='left'; tok.pad_token=tok.pad_token or tok.eos_token
print('raw prompt   :',repr(f"Solve this fraction problem: {probs.problem.iloc[0]}=?")); print('spaced prompt:',repr(f"Solve this fraction problem: {render_problem_for_prompt(probs.problem.iloc[0])}=?"))
CK={'C distill':f'{TR}/qwen3_4b_C1000drop_v10_e1_v10regen_lora/best','C human-FT s42':f'{C}/hft_runs/C1000drop_seed42/best','A control distill':f'{TR}/qwen3_4b_Adrop_v10_e1_v10regen_lora/best'}
for name,ck in CK.items():
    m=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Base',torch_dtype=torch.bfloat16),ck).merge_and_unload().cuda().eval()
    for surf in ('raw','spaced'):
        torch.manual_seed(0); rows=[]
        for r in probs.itertuples():
            p=r.problem if surf=='raw' else render_problem_for_prompt(r.problem)
            enc=tok([f"Solve this fraction problem: {p}=?"]*64,return_tensors='pt',padding=True).to('cuda')
            with torch.no_grad(): out=m.generate(**enc,do_sample=True,temperature=1.0,top_p=0.95,max_new_tokens=192,pad_token_id=tok.pad_token_id)
            c=to_frac(r.correct_answer)
            for o in out: rows.append((r.operation.lower()[:3]+' '+r.denom_type.upper(), int(c is not None and c in extract_final_answer(tok.decode(o[enc.input_ids.shape[1]:],skip_special_tokens=True)))))
        d=pd.DataFrame(rows,columns=['cell','ok']); ca=d.groupby('cell').ok.mean()*100
        print(f'  {name:18s} {surf:6s} accuracy {d.ok.mean()*100:5.1f}%  MAE {float((ca-href.loc[ca.index]).abs().mean()):5.1f}',flush=True)
    del m; torch.cuda.empty_cache()
print('DONE')
