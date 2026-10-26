# Paper runs: Qwen3-4B fraction model (Table 1)

Exact training record for the model behind the fractions column of Table 1 (distill + humanFT: MAE 5.2,
human-FT val NLL 1.361). Everything here was taken from the run directories and SLURM accounting on della.

| Stage | SLURM job | Checkpoint (under `results/transformer_replication/`) |
|---|---|---|
| Distillation | 7607804 (2026-05-02) | `qwen3_4b_panel25_default_base_e1_ailab_20260502_171918_4Bretry_lora/best` |
| Human FT | 7637206 (2026-05-03) | `qwen_humanft_from_distill_4b_ep1_20260503_124020/best` |

## Contents
* `submit_paper_runs.sh`: the two `sbatch` commands, copied verbatim from `sacct --format=SubmitLine`.
* `distill_run/`, `humanft_run/`: `train_args.json`, `history.json`, summary metrics, `adapter_config.json`,
  and the SLURM log of each run.

## Settings

**Distillation:**
* Data: 25 UMA learners, 511,315 rows (`data/distillation/uma_fraction_distillation_25`), 90/5/5 row split.
* Learners: one per (g, d) cell, with g in {.02,...,.10} and d in {.1,...,.9}; rt fixed at 5, ice fixed at 50.
* Model: Qwen3-4B-Base with LoRA (r 16, alpha 32, dropout 0.05).
* Training: 1 epoch (3,596 updates), batch 16 x 8 accumulation steps, lr 5e-4 cosine to 1e-6, 3% warmup, max_len 288, seed 42.
* Best checkpoint chosen by validation loss (0.0183).
* **Learner tag on every prompt** (`TRAIN_PROMPT_STUDENT_MODE=always`, dropout 0.0).

**Human FT:**
* Data: 288 Siegler 2011 training rows and 96 validation rows.
* Training: 1 epoch (18 updates), batch 2 x 8 accumulation steps, lr 1e-4, 3% warmup, seed 42.
* Best checkpoint chosen by validation NLL (1.3607).
* **No KL term** (`KL_BETA=0.0`).
* **Model:** it continues training the distilled LoRA adapter, with 33,030,144 of 4.06B parameters trainable.
  `USE_LORA=0` together with `INIT_STRATEGY=checkpoint` means "load the checkpoint's adapter as trainable".
  It does not mean full fine-tuning.
* Prompts are plain: no learner tag.

## Which code produced these runs
Use the main_max versions of `results/transformer_replication/train_transformer_hf.py`, `human_ft/finetune_humandata.py`
and the two `slurm_train_*.sbatch` wrappers. The exact trainer file from 2026-05-02 was not preserved. The evidence:

* **The trainer had `model.enable_input_require_grads()`.** The last committed version before the runs (local commit
  `a9823d7`, 2026-04-29, never pushed) lacks this call. In the same conda environment it crashes at the first
  backward pass ("element 0 of tensors does not require grad"): reproduced on 2026-10-26 with Qwen3-4B-Base and
  Qwen3-14B-Base.
  * The environment has not changed since October 2025 (torch 2.8.0, transformers 4.57.0, peft 0.17.1).
  * The May 4B and 8B logs show no such warning and trained normally.
  * The first 4B attempt on 2026-05-02 (job 7605429, log `slurm_logs/q34b_p25l_ailab_20260502_155943_7605429.err`)
    failed with exactly this error after 2 minutes.
  * The run in this folder is the retry ("4Bretry", job 7607804), started 3 hours later after the call was added.
* **The other differences on main_max don't change fraction runs.** They are an `--attn_implementation` flag whose
  default passes nothing, and whole-number prompt wording.
* **The human-FT trainer was unchanged.** `human_ft/finetune_humandata.py` has not changed since 2026-04-29.
* **The wrappers behave the same.** `slurm_train_qwen3_distill_1epoch.sbatch` only adds the attention flag (default
  eager). `slurm_train_qwen_humanft_lora.sbatch` only adds the optional `SAVE_EPOCH_*` flags.

## Known mismatches with the paper text
* The appendix says human FT uses early stopping with a maximum of five epochs. The reported model was trained
  for 1 epoch (18 updates). The best checkpoint was still chosen by validation NLL.
* The paper says distillation used the full 1,000-learner population. It used the 25 learners above
  (see `eval/outputs/panel_scaling/README.md`).
