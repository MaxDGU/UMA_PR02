# March 30 135M Params-Always Investigation

This note records the local assets we can already use to investigate the
`135M params_always lr=1e-5 epochs=6` run, even without the original Princeton
checkpoint directories.

## Local assets available

- Code branch: `main_changho`
- Processed distillation CSV:
  `results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz`
- SP2013 eval CSV:
  `results/UMA_replication/sp2013.csv`
- Human FT CSVs:
  `results/human/data_train_nlp.csv`
  `results/human/data_val_nlp.csv`
- Reproduction note:
  `REPRODUCE_UMA_EXPERIMENTS_OTHER_CLUSTERS.md`
- March 30 run logs:
  `results/transformer_replication/logs/s1000_135_pa1e5v_p1_20260330_134456_6256713.out`
  `results/transformer_replication/logs/s1000_135_pa1e5v_p2_20260330_134456_6256714.out`
- Earlier baseline log:
  `results/transformer_replication/logs/s1000_135_pt_20260307_160837_5476287.out`

The partial Della copy also brought in `BBT/pipeline.py`, so the higher-level
pipeline wrapper now exists locally too.

## Known March 30 run identity

The logs show the March 30 follow-up actually wrote to:

`/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/smol2_135m_synth_unique_seed1_all1000_mincols_paramsalways_pretrained_sp2013noparam_tracev7_bestsp_lr1em5_e6_spval_20260330_134456`

Important config differences versus the March 7 baseline:

- prompt mode: `always`
- `move_sp2013_rows_to_val=true`
- `lr=1e-5`
- `epochs=6`
- train rows: `18,648,057`
- val rows: `1,051,931`
- test rows: `1,036,012`
- updates per epoch: `36,422`

## What the logs already show

- The run does not fail on resume. Stage `p2` resumes at
  `epoch=4, update_in_epoch=25599, global_update=134865`, and the plateau is
  already present before that point.
- Epoch 1 still learns substantially:
  - first eval at update `7,285`: `train=0.9524`, `val=0.3153`
  - end of epoch 1 at update `36,422`: `train=0.4367`, `val=0.3059`
- By the first eval in epoch 2, the run is already near its final loss:
  - update `43,707`: `train=0.3067`, `val=0.3059`
- From there it stays almost completely flat through the end of epoch 6:
  - final eval at update `218,532`: `train=0.3066`, `val=0.3058`, `test=0.3070`
- SP2013 is weak throughout and collapses to zero by late training:
  - best seen in the logs: `0.1250`
  - end of epoch 4 onward: `0.0000`

This means the main plateau starts around `44k` optimizer updates, not at the
resume boundary.

## Quick parser command

To regenerate a compact summary from the logs:

```bash
python results/transformer_replication/summarize_training_logs.py \
  results/transformer_replication/logs/s1000_135_pa1e5v_p1_20260330_134456_6256713.out \
  results/transformer_replication/logs/s1000_135_pa1e5v_p2_20260330_134456_6256714.out \
  results/transformer_replication/logs/s1000_135_pt_20260307_160837_5476287.out
```

If you want a CSV of eval points:

```bash
python results/transformer_replication/summarize_training_logs.py \
  results/transformer_replication/logs/s1000_135_pa1e5v_p1_20260330_134456_6256713.out \
  results/transformer_replication/logs/s1000_135_pa1e5v_p2_20260330_134456_6256714.out \
  results/transformer_replication/logs/s1000_135_pt_20260307_160837_5476287.out \
  --eval-csv results/transformer_replication/march30_vs_baseline_eval_points.csv \
  --summary-json results/transformer_replication/march30_vs_baseline_summary.json
```

## Most likely next investigation targets

1. Check whether `train_prompt_student_mode=always` changes the prompt/target
   semantics enough to make the model optimize an easier but unhelpful token
   distribution.
2. Compare a small batch of March 30 generations against the March 7 baseline
   on SP2013-style prompts, especially parseability and answer format.
3. Inspect whether the loss plateau is already reflected in token-level
   predictions on the training distribution, or whether this is mainly an
   evaluation mismatch.
