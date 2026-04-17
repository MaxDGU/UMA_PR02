# 135M Params-Always LR Ablation with UMA-Label SP2013 Selection

This note defines the current learning-rate ablation for SmolLM2-135M under the
following fixed condition:

- model: `HuggingFaceTB/SmolLM2-135M` pretrained
- data: `results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz`
- train prompt mode: `always`
- SP2013 rows excluded from train: `--move_sp2013_rows_to_val`
- model selection metric: `sp2013_acc` against UMA answers
- secondary diagnostic only: `sp2013_true_acc` against mathematically true answers

## Evaluation definition

We now score SP2013 in two ways:

1. `sp2013_acc`
   - compares model outputs to UMA `ans` labels from
     `results/UMA_replication/sp2013_eval/seed_1/sp2013_seed1_all_models.csv`
   - uses fixed subject profiles `(subjid, g, d, rt_mu, ice)`
   - this is the checkpoint-selection metric
2. `sp2013_true_acc`
   - compares model outputs to the mathematically correct answer from `prob`
   - this is logged for diagnosis only

The first 10 subject profiles were checked and match exactly between the
training corpus and the seed-1 SP2013 evaluation table, so the validation
condition is consistent with the training population.

## Why this grid

The goal of this ablation is to make the model fit the translated training data
better than the earlier runs, which appeared to flatten too early. The grid is:

- schedulers: `linear`, `cosine`
- learning rates: `5e-5`, `1e-5`, `5e-6`, `1e-6`

Interpretation:

- `5e-5`: aggressive upper-bound stress test; may diverge, but useful to test
  whether the earlier runs were simply too conservative for fitting
- `1e-5`: direct comparison point against the March 30 follow-up
- `5e-6`: moderate lower-LR alternative
- `1e-6`: very conservative fit-first setting

## Runtime choices

To keep the 8-job screen practical on 48GB L40 GPUs:

- batch size: `96`
- eval batch size: `96`
- gradient accumulation: `1`
- precision: `bf16`
- scheduler warmup: `2000` updates
- epochs: `1`
- evaluations per epoch: `5`
- ID final-answer eval: disabled
- periodic eval checkpoints: disabled
- final checkpoint save: disabled
- reload verification: disabled
- preview generations: disabled
- time limit per job: `48:00:00`

The reason for `epochs=1` is that with batch size `96`, one full pass over the
same dataset already implies about `512 / 96 ≈ 5.3x` more optimizer updates per
epoch than the older batch-512 setup. Using the March 30 run metadata as a
reference, batch `96` corresponds to about `194k` updates in one epoch, which
is already close to the earlier 6-epoch, batch-512 training horizon. For a
first LR screen, one full epoch is already substantial.

We use `5` evaluations within that epoch so checkpoint selection is meaningful:
the trainer can still save the best model by UMA-label `sp2013_acc` instead of
only scoring once at the very end.

## Submitted jobs

Submission script:

- `results/transformer_replication/submit_synth_unique_seed1_all1000_lr_ablation_135m_paramsalways_umaeval.sh`

Defaults used by that script:

- partition: `all`
- manifest:
  `results/transformer_replication/jobs_synth_unique_seed1_all1000_lr_ablation_135m_paramsalways_umaeval_<timestamp>.tsv`

Each run writes to:

- `results/transformer_replication/smol2_135m_s1000_paramsalways_umaeval_<scheduler>_<lr>_<timestamp>`

## Decision rule

Primary ranking:

1. higher `sp2013_acc` on UMA labels
2. lower `val_loss` as tie-breaker

Secondary diagnostics:

- lower `train_loss`
- lower `val_loss`
- higher `sp2013_true_acc`

If a run improves `train_loss` materially but not `sp2013_acc`, then it may be
fitting the trace distribution better without yet matching the UMA SP2013
outputs. If a run improves both, that is the cleanest sign that the optimizer
change helped.
