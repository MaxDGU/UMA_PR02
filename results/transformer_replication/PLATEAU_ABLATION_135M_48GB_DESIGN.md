# 135M Plateau Ablation Design for 48GB GPUs

This document proposes a focused ablation plan for the March 30
`135M params_always lr=1e-5 epochs=6` distillation plateau.

It is designed for a cluster with 48GB GPUs, so it treats the original
H200-style `batch_size=512` as a hardware-utilization choice rather than a
target to preserve. The default recommendation here is to use the largest batch
that fits on 48GB with `grad_accum_steps=1`, then study LR and scheduler
effects on top of that.

## 1. Working hypothesis

Current hypothesis:

1. The translated NLP supervision contains an "easy" mapping signal.
   - The model can quickly learn prompt/strategy-style surface structure.
   - This explains the rapid early loss drop in epoch 1.
2. The remaining error is dominated by harder reasoning/generalization.
   - This may require a lower effective LR, a different schedule shape, or a
     second-stage refinement run after the easy mapping is learned.

If that hypothesis is correct, then:

- train loss should drop quickly early,
- then flatten,
- but a lower-LR or better-shaped second-stage optimization path should recover
  at least some additional validation or SP2013 signal.

## 2. What we already know

From the local March 30 logs:

- The run does not get stuck because of resume.
- The plateau is already present by the first eval in epoch 2.
- The plateau starts around global update `43,707`.
- The final region is essentially flat:
  - train `~0.3066`
  - val `~0.3058`
  - test `~0.3070`
- SP2013 stays poor and collapses to zero late in training.

From the earlier March 7 no-params baseline:

- one epoch reached much lower loss:
  - train `0.1775`
  - val `0.1546`
- best logged SP2013 was `0.3125`

So the March 30 issue is not "the trainer cannot optimize at all." It is more
consistent with either:

- an objective/prompt mismatch, or
- an optimization path that is fine for the easy mapping but bad for the harder
  residual learning.

## 3. Code reality in the current repo

The current trainer now supports the knobs we need:

- `--grad_accum_steps`
- `--lr`
- `--lr_scheduler {cosine,linear,constant,constant_with_warmup}`
- `--warmup_ratio`
- `--warmup_steps`

The current March 30 run effectively used:

- `train_prompt_student_mode=always`
- `move_sp2013_rows_to_val=true`
- `lr=1e-5`
- `lr_scheduler=cosine`
- `warmup_ratio=0.03`

For new runs on this cluster, we will instead treat UMA distillation as a
midtraining-style continuation problem and use `lr_scheduler=linear` by
default.

The original Princeton launcher exposed `batch_size` but not
`grad_accum_steps`; for this cluster, prefer direct Python commands.

## 4. 48GB GPU operating point

Default recommendation:

- `grad_accum_steps=1`
- `amp_dtype=bf16`
- `max_length=192`
- use the largest stable per-device batch that fits

Working assumption:

- the old `batch_size=512` was mainly for GPU utilization,
- not something we should preserve at all costs.

That means smaller batches are not just acceptable here; they may actually be
useful, because noisier gradients could help the model continue improving after
the easy translation-pattern phase.

Recommended batch search order on 48GB:

- try `batch_size=512`
- if OOM, try `384`
- then `256`
- then `192`
- then `128`
- then `96`
- then `64`

Use `eval_batch_size` equal to the train batch at first. If eval OOMs while
train does not, lower only `eval_batch_size`.

Keep gradient accumulation off for the primary ablation unless:

- even `batch_size=64` is unstable, or
- we later explicitly want a separate "effective batch size" control.

## 5. Primary experiment goal

Keep the data and prompt condition fixed first:

- data: `synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz`
- prompt mode: `always`
- `move_sp2013_rows_to_val=true`

Only change:

- LR level
- LR schedule shape
- whether refinement starts from scratch or from an already-learned checkpoint

This isolates the optimization hypothesis before changing the objective.

## 6. Metrics to track

Primary metrics:

- `sp2013_acc`
- `val_loss`

Secondary metrics:

- `id_val_acc`
- `train_loss`
- `test_loss`

Interpretation rule:

- If train/val loss improve but `sp2013_acc` stays near zero, then the run is
  optimizing something but not the reasoning target we care about.
- If lower LR or a different schedule improves both val loss and SP2013, that
  supports the optimization-path hypothesis.
- If no optimizer change helps, the next likely culprit is the prompt/objective
  formulation itself.

## 7. Experimental strategy

### Phase 0: Fit-to-cluster smoke test

Goal:

- confirm the 48GB batch/accumulation setting,
- avoid wasting a full run on a memory mistake.

Run a short job with:

- `max_samples=131072`
- `epochs=1`
- `evals_per_epoch=2`
- `grad_accum_steps=1`
- `batch_size=<largest candidate that fits>`

If the initial candidate OOMs, step down the batch ladder above.

### Phase 1: Two-epoch optimizer screen

Goal:

- cheaply identify whether the plateau can be delayed or softened before
  committing to longer runs.

Why 2 epochs:

- the March 30 plateau is already obvious by early epoch 2,
- so 2 epochs are enough for a first-pass screen.

Recommended screen matrix:

1. `control_linear_1e5`
   - `lr=1e-5`
   - `lr_scheduler=linear`
2. `linear_5e6`
   - `lr=5e-6`
   - `lr_scheduler=linear`
3. `linear_3e6`
   - `lr=3e-6`
   - `lr_scheduler=linear`
4. `constwarm_5e6`
   - `lr=5e-6`
   - `lr_scheduler=constant_with_warmup`
5. `constwarm_3e6`
   - `lr=3e-6`
   - `lr_scheduler=constant_with_warmup`
6. `historical_cosine_1e5`
   - `lr=1e-5`
   - `lr_scheduler=cosine`

Keep fixed for all Phase 1 runs:

- `epochs=2`
- `warmup_ratio=0.03`
- `batch_size=<largest stable batch from Phase 0>`
- `eval_batch_size=<matching stable eval batch>`
- `grad_accum_steps=1`
- `train_prompt_student_mode=always`
- `move_sp2013_rows_to_val=true`
- `best_by=sp2013_acc`
- `eval_sp2013=true`
- `sp2013_use_param_grid=false`
- `eval_id_final_answer=true`

### Phase 2: Stage-2 refinement test

Goal:

- directly test the "easy mapping first, harder reasoning later" hypothesis.

Protocol:

1. Take the epoch-1-end checkpoint from the best Phase 1 control-style run.
2. Start a new run from those weights only:
   - set `--model_name` to that saved checkpoint directory,
   - keep `--resume_from none`
   - use a fresh output directory
3. Continue with a lower LR and a gentler schedule for 2 to 3 more epochs.

Recommended refinement variants:

1. `refine_linear_3e6`
   - start from epoch-1 checkpoint
   - `lr=3e-6`
   - `lr_scheduler=linear`
   - `warmup_steps=2000`
2. `refine_linear_1e6`
   - start from epoch-1 checkpoint
   - `lr=1e-6`
   - `lr_scheduler=linear`
   - `warmup_steps=2000`
3. `refine_constwarm_3e6`
   - start from epoch-1 checkpoint
   - `lr=3e-6`
   - `lr_scheduler=constant_with_warmup`
   - `warmup_steps=2000`

This is the strongest direct test of the hypothesis.

### Phase 3: Objective check if optimization ablations fail

Only do this if Phases 1 and 2 do not help.

Hold the best optimizer settings fixed, and compare prompt mode:

1. `always`
2. `dropout` with `dropout_prob=0.5`
3. `none`

This tests whether the plateau is actually caused by the prompt/translation
setup rather than the optimizer.

### Optional Phase 2.5: Batch-size sanity check

If one optimizer setting looks promising, do one small follow-up to see whether
batch size itself is part of the story.

Example:

1. best optimizer setting at `batch_size=<largest_fit>`, `grad_accum_steps=1`
2. same optimizer setting at roughly half that batch size, still
   `grad_accum_steps=1`

If the smaller batch improves SP2013 or avoids the flat region longer, then the
plateau is likely not only an LR issue; optimizer noise level is also part of
it.

## 8. Recommended command template

Direct Python form for this cluster:

```bash
python results/transformer_replication/train_transformer_hf.py \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --data_csv results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz \
  --sp2013_csv results/UMA_replication/sp2013.csv \
  --output_dir results/transformer_replication/<run_name> \
  --epochs 2 \
  --batch_size 128 \
  --eval_batch_size 128 \
  --grad_accum_steps 1 \
  --lr 5e-6 \
  --lr_scheduler linear \
  --warmup_ratio 0.03 \
  --max_length 192 \
  --amp_dtype bf16 \
  --evals_per_epoch 5 \
  --best_by sp2013_acc \
  --eval_sp2013 \
  --no-sp2013_use_param_grid \
  --eval_id_final_answer \
  --train_prompt_student_mode always \
  --train_prompt_student_dropout_prob 0.0 \
  --train_prompt_student_dropout_seed 0 \
  --move_sp2013_rows_to_val \
  --save_eval_checkpoints \
  --save_best_checkpoint \
  --no-save_final_checkpoint \
  --no-verify_saved_checkpoint_load \
  --preview_samples 0
```

Stage-2 refinement form:

```bash
python results/transformer_replication/train_transformer_hf.py \
  --model_name results/transformer_replication/<stage1_run>/eval_e01_u0036422 \
  --data_csv results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz \
  --sp2013_csv results/UMA_replication/sp2013.csv \
  --output_dir results/transformer_replication/<stage2_run> \
  --epochs 3 \
  --batch_size 128 \
  --eval_batch_size 128 \
  --grad_accum_steps 1 \
  --lr 3e-6 \
  --lr_scheduler linear \
  --warmup_steps 2000 \
  --max_length 192 \
  --amp_dtype bf16 \
  --evals_per_epoch 5 \
  --best_by sp2013_acc \
  --eval_sp2013 \
  --no-sp2013_use_param_grid \
  --eval_id_final_answer \
  --train_prompt_student_mode always \
  --train_prompt_student_dropout_prob 0.0 \
  --train_prompt_student_dropout_seed 0 \
  --move_sp2013_rows_to_val \
  --save_eval_checkpoints \
  --save_best_checkpoint \
  --no-save_final_checkpoint \
  --no-verify_saved_checkpoint_load \
  --preview_samples 0
```

## 9. Naming convention

Suggested run names:

- `ablate135_pa_linear1e5_e2_gb512`
- `ablate135_pa_linear5e6_e2_gb512`
- `ablate135_pa_linear3e6_e2_gb512`
- `ablate135_pa_constwarm5e6_e2_gb512`
- `ablate135_pa_constwarm3e6_e2_gb512`
- `ablate135_pa_histcos1e5_e2_gb512`
- `ablate135_pa_refine_linear3e6_frome1`
- `ablate135_pa_refine_constwarm3e6_frome1`

## 10. Decision rule after Phase 1

Move a config to Phase 2 only if at least one of these happens by epoch 2:

1. it pushes `sp2013_acc` clearly above the control, or
2. it keeps train loss moving after the control has already flattened, or
3. it lowers val loss without destroying ID-val accuracy.

If no Phase 1 run beats the control meaningfully, do not scale the optimizer
grid. Move directly to the prompt/objective ablation in Phase 3.

## 11. My recommendation

Recommended first pass:

1. Run the batch-fit smoke test with `grad_accum_steps=1`.
2. Fix the largest stable train batch.
3. Run the 6-way 2-epoch optimizer screen.
4. If one run looks better, do the stage-2 refinement test from the epoch-1
   checkpoint.

This gives us a strong read on your hypothesis without paying for many full
6-epoch runs up front, and it does not assume that matching the old `512`
effective batch is desirable.
