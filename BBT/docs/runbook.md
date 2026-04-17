# Runbook

## Local workflow

1. Dry-run:

```bash
python BBT/pipeline.py --dry-run
```

2. Full run:

```bash
python BBT/pipeline.py
```

Standard human workflow:
- `human_finetune` tracks validation NLL during training and saves `best` by minimum validation NLL.
- `human_eval` then runs one post-train sampling evaluation on the `best` checkpoint with `1000` samples per prompt and `use_param_grid=false`.

3. Partial rerun with external handoff:

```bash
python BBT/pipeline.py \
  --subpipelines distill_eval,human_finetune,human_eval \
  --set external_inputs.distill_run_dir="results/bbt/my_run/03_distill_train"
```

4. Human finetune + one-shot eval with a 50-epoch override:

```bash
python BBT/pipeline.py \
  --subpipelines human_finetune,human_eval \
  --set external_inputs.distill_run_dir="results/bbt/my_run/03_distill_train" \
  --set subpipelines.human_finetune.epochs=50
```

This keeps the default human-finetune LR at `1.5e-5`; only the epoch count is overridden.

## Slurm workflow

- Submit with dependency chain:

```bash
bash BBT/slurm/submit_pipeline.sh
```

- Run one subpipeline job script directly:

```bash
sbatch --export=ALL,SUBPIPELINE=distill_eval,CONFIG=BBT/configs/default_pipeline.json,RUN_DIR=/abs/run BBT/slurm/run_subpipeline.sbatch
```

## Resume strategy

- Keep `run_dir` fixed and rerun later subpipelines with `external_inputs.*` pointed to prior outputs.
- Upstream manifest files provide deterministic handoff paths.

## Troubleshooting

- Missing upstream input:
- set the corresponding `external_inputs.*` key.

- Checkpoint not found during eval:
- verify `<run_dir>/<checkpoint_subdir>` exists (`best` by default).

- Empty or invalid dataset:
- ensure `datasets` entries include correct `csv` and `problem_col`.
