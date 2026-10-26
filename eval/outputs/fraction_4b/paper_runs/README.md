# Paper runs: Qwen3-4B fraction model (Table 1)

Exact training record for the model behind the fractions column of Table 1 (distill + humanFT: MAE 5.2,
human-FT val NLL 1.361). Everything here was taken from the run directories and SLURM accounting on della.

| Stage | SLURM job | Checkpoint (under `results/transformer_replication/`) |
|---|---|---|
| Distillation | 7607804 (2026-05-02) | `qwen3_4b_panel25_default_base_e1_ailab_20260502_171918_4Bretry_lora/best` |
| Human FT | 7637206 (2026-05-03) | `qwen_humanft_from_distill_4b_ep1_20260503_124020/best` |

## Contents
* `submit_paper_runs.sh`: the two `sbatch` commands, copied verbatim from `sacct --format=SubmitLine`.
* `code_a9823d7/`: the trainers and SLURM wrappers as they were when the runs started (commit `a9823d7` on the
  local `feature/qwen-humanft-lora` branch, which was never pushed). The line numbers of warnings in both
  logs match these files.
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

## Differences from the current code on main_max
* `human_ft/finetune_humandata.py` and `slurm_train_qwen3_distill_1epoch.sbatch` are identical to these copies.
* `slurm_train_qwen_humanft_lora.sbatch` on main_max only adds the optional `SAVE_EPOCH_CHECKPOINTS` and
  `SAVE_EPOCH_LIST` flags. With those unset it runs the same command.
* `results/transformer_replication/train_transformer_hf.py` on main_max differs in three ways:
  * It adds an `--attn_implementation` flag. The default (`eager`) passes nothing, so behaviour is unchanged.
  * It adds whole-number prompt wording, which leaves fraction prompts unchanged.
  * It calls `model.enable_input_require_grads()` when LoRA is used with gradient checkpointing.
    **This last change affects training**, and the paper run predates it. Use `code_a9823d7/train_transformer_hf.py`
    to reproduce the paper exactly.

## Known mismatches with the paper text
* The appendix says human FT uses early stopping with a maximum of five epochs. The reported model was trained
  for 1 epoch (18 updates). The best checkpoint was still chosen by validation NLL.
* The paper says distillation used the full 1,000-learner population. It used the 25 learners above
  (see `eval/outputs/panel_scaling/README.md`).
