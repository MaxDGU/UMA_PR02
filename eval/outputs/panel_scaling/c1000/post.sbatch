#!/usr/bin/env bash
#SBATCH --job-name=c1000_post
#SBATCH --partition=ailab
#SBATCH --account=griffith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --array=0-6
#SBATCH --output=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000/logs/post_%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000/logs/post_%x_%A_%a.err
# task 0: distill-stage rollouts (SP2013, 256/problem) + NLL on held-out children; tasks 1-4: standard human-FT seeds 42-45;
# tasks 5-6: KL-anchored human-FT (beta 1.0) seeds 42-43. Protocols identical to arms A/B.
set -euo pipefail
ROOT=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft; TR=$ROOT/results/transformer_replication; C=/scratch/gpfs/GRIFFITHS/mg7411/llm_student/panel_scaling/c1000
SV=/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/tmp/seed_variance; PY=/home/mg7411/.conda/envs/gpt-analysis/bin/python
export HF_HOME=/home/mg7411/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DISTILL=$TR/qwen3_4b_${ARM}_v10_e1_v10regen_lora; i=$SLURM_ARRAY_TASK_ID; cd "$ROOT"
if [[ $i -eq 0 ]]; then
  [[ -f $C/results/${ARM}_distill.csv ]] || "$PY" "$SV/eval_seed_ckpt.py" --ckpt "$DISTILL/best" --base-model Qwen/Qwen3-4B-Base --domain fraction --out-csv "$C/results/${ARM}_distill.csv" --n-samples 256
  "$PY" - "$DISTILL/best" "$C/results/${ARM}_distill_child_nll.json" <<'PYEOF'
import sys, json, torch, pandas as pd, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
R='/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft'; sys.path.insert(0,R+'/results/transformer_replication')
from train_transformer_hf import render_problem_for_prompt
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Base'); m=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Base',torch_dtype=torch.bfloat16),sys.argv[1]).merge_and_unload().cuda().eval()
out={}
for split in ('train','val'):
    d=pd.read_csv(f'{R}/data/human_ft/data_{split}_nlp.csv'); tot=0; n=0
    for p,y in zip(d.prob,d.response_nl):
        pi=tok.encode(f"Solve this fraction problem: {render_problem_for_prompt(str(p))}=?",add_special_tokens=False); ri=(tok.encode(str(y).strip(),add_special_tokens=False)+[tok.eos_token_id])[:256-len(pi)]
        ids=torch.tensor([pi+ri]).cuda()
        with torch.no_grad(): lo=m(ids).logits[0,len(pi)-1:-1].float()
        tot+=F.cross_entropy(lo,ids[0,len(pi):],reduction='sum').item(); n+=len(ri)
    out[f'nll_human_{split}']=tot/n
json.dump(out,open(sys.argv[2],'w')); print(out)
PYEOF
  exit 0
fi
if [[ $i -le 4 ]]; then SEED=$((41+i)); KL=0.0; TAG=${ARM}_seed$SEED; else SEED=$((37+i)); KL=1.0; TAG=${ARM}_kl1.0_seed$SEED; fi
RUN=$C/hft_runs/$TAG; OUT=$C/results/${TAG}_dh.csv
[[ -d $RUN/best ]] || "$PY" human_ft/finetune_humandata.py --init_strategy checkpoint --model_path "$DISTILL" --checkpoint_subdir best --base_model_name Qwen/Qwen3-4B-Base \
  --train_csv data/human_ft/data_train_nlp.csv --val_csv data/human_ft/data_val_nlp.csv --output_dir "$RUN" --instruction_style plain \
  --epochs 1 --max_steps 0 --batch_size 2 --eval_batch_size 2 --grad_accum_steps 8 --lr 1e-4 --warmup_ratio 0.03 --max_length 256 --num_workers 2 \
  --amp_dtype bf16 --kl_beta $KL --best_by val_nll --log_every 5 --seed $SEED --no-use_lora --gradient_checkpointing --no-eval_sp2013 --no-eval_loaded_model --local_files_only
[[ -f $OUT ]] || "$PY" "$SV/eval_seed_ckpt.py" --ckpt "$RUN/best" --base-model Qwen/Qwen3-4B-Base --domain fraction --out-csv "$OUT" --n-samples 256
rm -rf "$RUN/final" "$RUN/last"; echo "DONE $TAG"
