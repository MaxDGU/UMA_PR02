# BBT (Experiment Subpipeline Pipeline)

`BBT/` organizes your workflow into experiment subpipelines instead of technical modules.

## Subpipelines

1. `uma_traces`:
- Generate UMA traces from saved UMA models.
- Output: `01_uma_traces/uma_traces.csv` + manifest.

2. `translation`:
- Translate UMA traces to NLP instruction/response format.
- Output: `02_translation/uma_traces_nlp.csv.gz` + manifest.

3. `distill_train`:
- Train distilled LM on translated traces (pretrained by default, optional scratch init).
- Output: `03_distill_train/` + manifest.

4. `distill_eval`:
- Sampling evaluation (non-greedy) on SP2013 and optional extra datasets.
- Output: `04_distill_eval/eval_samples.csv` + manifest.

5. `human_finetune`:
- Finetune distilled model on human train data.
- Output: `05_human_finetune/` + manifest.

6. `human_eval`:
- Evaluate held-out human NLL summary + sampling eval.
- Output: `06_human_eval/human_eval_metrics.json` + manifest.

## Quickstart

From repo root:

```bash
python BBT/pipeline.py --dry-run
```

Run all subpipelines:

```bash
python BBT/pipeline.py
```

Run selected subpipelines:

```bash
python BBT/pipeline.py \
  --subpipelines distill_train,distill_eval,human_finetune,human_eval \
  --set external_inputs.translated_csv="results/transformer_replication/synth_unique_subjid_0472_seeds1_20_nlp_no_params.pre_tracev6_20260303_154525.csv.gz"
```

From-scratch distillation run:

```bash
python BBT/pipeline.py \
  --subpipelines distill_train,distill_eval \
  --set subpipelines.distill_train.init_from_scratch=true
```

## Config

Default config path:

- `BBT/configs/default_pipeline.json`

Key sections:

- `experiment`
- `paths`
- `external_inputs`
- `subpipelines.*`

## Manifests

Every subpipeline writes `manifest.json` with:

- `subpipeline_name`
- `status`
- `started_at_utc`
- `finished_at_utc`
- `config_snapshot`
- `inputs`
- `primary_output`
- `secondary_outputs`
- `metrics_summary`
- `upstream_manifest`

See `BBT/docs/subpipeline_contracts.md` and `BBT/docs/config_reference.md` for details.
