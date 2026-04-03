# Current Experiment Summary

This note summarizes the main questions we are actively testing in the UMA fraction-trace project as of March 25, 2026.

## Current Interpretation

- If longer human finetuning helps a lot, then human FT was at least partly undertuned.
- If `params_always` beats the existing `no_params` baseline, then hidden-parameter ambiguity was hurting distillation.
- If `params_dropout50_noprefix` beats both `no_params` and `params_always`, that would support the idea that mixed conditioning is a better bridge from synthetic distillation to human finetuning.
- If neither new prompt condition helps, then the next likely knobs are learning rate, checkpoint selection, or a more compressed student-state representation.

## Practical Next Steps

1. Finish the long human-finetune runs and evaluate the `best` checkpoints once.
2. Launch the new 2x3 distillation A/B grid by reusing the existing `synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz` dataset and rewriting prompts inside the trainer.
3. Compare:
   - existing `no_params` baseline
   - `params_always`
   - `params_dropout50_noprefix`
4. Use the strongest distilled checkpoints for the next round of human finetuning.

## Main Questions

1. Is human finetuning undertrained?
2. Are the distilled teacher models undertrained?
3. Or is the bigger issue the prompt format, because the same visible fraction problem can map to different traces and answers depending on hidden UMA parameters?

Our current working belief is:

- Human finetuning may have been somewhat undertuned when run for only a few epochs.
- The distillation models do **not** show strong evidence of being undertrained in the same way.
- The more important modeling issue is likely the synthetic prompt format: in the old `no_params` setup, the model sees the problem text but does **not** see the UMA state that partly determines the target trace.

## Experiment 1: Human Finetuning Undertuning Check

We are rerunning human finetuning with a longer schedule while keeping the default learning rate.

### Goal

Check whether the previous human finetuning runs were too short.

### Setup

- Start from the existing all1000 pretrained distilled models.
- Use the `last` distillation checkpoint as the starting checkpoint for human finetuning.
- Finetune for `50` epochs.
- Keep the default human-finetune learning rate at `1.5e-5`.
- During training, track validation NLL every epoch.
- Save `best` by minimum validation NLL.
- Do **not** run SP2013 sampling during training.
- After training, run one post-train SP2013 sampling eval on `best` with `1000` samples per prompt and no parameter grid.

### Why

The human dataset is small, so the optimization loop is cheap. The expensive part was repeatedly sampling SP2013 during training. The new workflow removes that repeated sampling cost while still preserving a clean `best` checkpoint by validation NLL.

## Experiment 2: Distillation Prompt-Conditioning A/B

We are testing whether making UMA parameters visible in the prompt helps distillation, and whether a mixed prompt regime transfers better to later human finetuning where params are unavailable.

### Baseline

- Reuse the existing March 7, 2026 `no_params` all1000 runs as the baseline.
- Do **not** rerun `no_params`.

### New Conditions

1. `params_always`
   - Every prompt includes the real student block:
     `<student> g ... d ... rt ... ice ... </student>`
   - The student block is placed **before** the fraction problem.

2. `params_dropout50_noprefix`
   - Each row deterministically either:
     - keeps the real student block, or
     - drops it entirely and falls back to the current human-like no-prefix prompt
   - Drop probability is `0.5`.
   - The keep/drop decision is deterministic from `source_uid` plus seed `0`.
   - We now implement this at trainer load time while reusing the existing `no_params` CSV, rather than regenerating full translated datasets.

### Grid

Run both new conditions on the same 2x3 grid as before:

- model sizes: `135m`, `360m`, `1.7b`
- init modes: `pretrained`, `scratch`

### Distillation Recipe

Keep the training recipe matched to the previous production all1000 runs:

- `2` total epochs
- `best_by=sp2013_acc`
- `eval_sp2013=true`
- `sp2013_use_param_grid=false`
- `evals_per_epoch=5`
- same model-specific resource, batch size, and chaining defaults as the existing all1000 launcher

### Why

The old `no_params` synthetic setup hides the UMA state from the model, even though that state can change the target trace and answer. Exposing params before the input may reduce this one-to-many ambiguity. The `dropout50_noprefix` condition is meant to preserve some robustness for the downstream human setting, where exact UMA parameters are not available.
