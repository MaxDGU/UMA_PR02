# Corrected Decimal Fine-Tuning Rerun — 2026-07-28

This directory is the results-only snapshot of the corrected Qwen3-4B decimal
fine-tuning experiment frozen on 2026-07-28. It contains the final inference
rollouts and their participant-held-out evaluation, but not model checkpoints
or training data.

The `_07282026` suffix identifies the experiment freeze date. The rollout seed
is deliberately `20260729`, one day later than the profile-trace generation
seed (`20260728`), so evaluation samples cannot reuse the training-source
sampling stream.

## Scope

- Base model: `Qwen/Qwen3-4B-Base`
- Decimal problems: 12
- Training seeds: 42–46
- Rollouts: 120 per problem and seed
- Inference backend: vLLM
- Evaluation prompt: `Solve this decimal problem: <problem>=?`
- Sampling: temperature 1.0, top-p 0.95, at most 192 new tokens
- Evaluation uses plain prompts with no UMA profile conditioning.
- Human reference: 19 held-out participants, 228 responses
- Checkpoint selection used validation completion-token NLL; the test split
  was not used for training, hyperparameter selection, or checkpoint selection.

The six five-seed conditions are:

| Artifact key | Training condition |
|---|---|
| `direct_human_ft` | Direct answer-only human fine-tuning |
| `cpt_human_ft` | Cognitive-model distillation followed by answer-only human fine-tuning |
| `cpt_weighted_beta_0` | Importance-weighted self-training, KL weight 0 |
| `cpt_weighted_beta_0p01` | Importance-weighted self-training, KL weight 0.01 |
| `cpt_weighted_beta_0p1` | Importance-weighted self-training, KL weight 0.1 |
| `cpt_weighted_beta_0p5` | Importance-weighted self-training, KL weight 0.5 |

All weighted conditions use the same 12,000 profile-conditioned source traces,
converted to plain-prompt training examples before optimization. All four
pre-specified KL weights are reported; the held-out results were not used to
select a preferred beta.

## Directory Layout

```text
rerun_decimals_07282026/
├── README.md
├── snapshot_manifest.json
├── checksums.sha256
├── inference/
│   └── <condition>/seed_<42-46>.csv[.manifest.json]
└── evaluation/
    ├── summary.csv
    ├── per_run.csv
    ├── per_cell.csv
    ├── per_answer_nll.csv
    └── manifest.json
```

Each inference CSV has 1,440 rows (12 problems × 120 samples). Its adjacent
manifest records the checkpoint provenance, sampling configuration, row
count, output SHA-256, and the explicit
`"profile_conditioned_evaluation": false` assertion.

The evaluation tables contain:

- `summary.csv`: mean, standard deviation, and standard error across seeds;
- `per_run.csv`: accuracy, profile MAE, and categorical answer NLL by seed;
- `per_cell.csv`: problem-cell accuracy by run;
- `per_answer_nll.csv`: smoothed answer-category likelihood contributions;
- `manifest.json`: test-reference provenance, full matrix assertion, input
  inventory, and hashes of the four metric tables.

Paths inside the original manifests are archival absolute paths from the
experiment workspace. Integrity is path-independent: the recorded hashes refer
to file contents, and `checksums.sha256` uses paths relative to this snapshot.

## Main Results

Values are mean ± standard error over five training seeds. Accuracy and profile
MAE are in percentage points.

| Condition | Accuracy | Profile MAE ↓ | Answer NLL ↓ |
|---|---:|---:|---:|
| Direct human FT | 65.47 ± 1.52 | **6.91 ± 0.50** | 1.811 ± 0.022 |
| CPT human FT | 67.00 ± 1.86 | 7.48 ± 1.41 | **1.764 ± 0.019** |
| Weighted, β=0 | 85.78 ± 0.33 | 20.87 ± 0.33 | 1.956 ± 0.003 |
| Weighted, β=0.01 | 85.12 ± 0.64 | 20.21 ± 0.64 | 1.961 ± 0.003 |
| Weighted, β=0.1 | 85.24 ± 0.64 | 20.32 ± 0.64 | 1.955 ± 0.003 |
| Weighted, β=0.5 | 85.88 ± 0.42 | 20.96 ± 0.42 | 1.893 ± 0.014 |

The weighted variants are more accurate overall but do not improve the held-out
human accuracy-profile match. This snapshot records the full ablation rather
than selecting a beta after evaluation.

`summary.csv` also includes re-scored unchanged baselines (UMA, Centaur-70B,
Qwen3-4B Base, cognitive distillation only, and the selected Claude persona).
Their source rollouts are not duplicated here; the 30 newly generated
fine-tuning rollouts are the contents of `inference/`.

## Validation and Provenance

The complete 30-run matrix was revalidated and re-scored before publication.
The regenerated evaluation CSVs and manifest matched the archived files
byte-for-byte.

- Corrected training array: Slurm job `11709627`
- Matched fresh-seed inference: Slurm job `11709628`
- Held-out aggregation: Slurm job `11709629`
- Experiment-start repository commit:
  `a043142e8fcf01108855651b5d94ed05dbcb0060`
- Packaging base (`origin/main_max`):
  `020bc680ab8c9ec548ef544947a137d1f4ec53f9`
- Frozen human-test SHA-256:
  `f08d1a12443dc63a174d8214316779139365702f4c80ace875e3a40ffcbc5c3d`

Verify the published snapshot from this directory with:

```bash
sha256sum -c checksums.sha256
```

## Explicit Exclusions

This snapshot excludes:

- the earlier unconditioned weighted-KL pilot, which used the wrong trace
  cohort and is superseded;
- model adapters, optimizer state, and smoke-test checkpoints;
- profile-conditioned source traces and importance-weight training data;
- few-shot ICL, strategy-judge, persona-sweep, and Instruct-initialization
  artifacts;
- unchanged baseline inference files already stored elsewhere in the
  repository.
