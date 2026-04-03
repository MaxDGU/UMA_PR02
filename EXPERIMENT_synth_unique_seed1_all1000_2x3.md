# synth_unique_seed1_all1000 2x3 Smol experiment

This note makes the `135m / 360m / 1.7B` by `pretrained / scratch` plan explicit for the
`synth_unique_seed1_all1000` dataset. I did not find an existing dedicated doc for this exact run
grid on March 7, 2026, so this file is the explicit launch note.

As of the latest update, the launcher defaults to:

- `1` epoch
- `135m/360m`: `1` GPU and `24:00:00`
- `1.7B`: `2` GPUs and two chained `16:00:00` windows
- `10` resumable checkpoint saves per epoch

## Goal

Run a six-job grid on the translated all1000 UMA traces:

- models: `SmolLM2-135M`, `SmolLM2-360M`, `SmolLM2-1.7B`
- init modes: pretrained weights, random init from config
- dataset: `results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz`
- evaluation style: `BEST_BY=sp2013_acc`, `SP2013_USE_PARAM_GRID=0`

The launcher for this grid is:

- `results/transformer_replication/submit_synth_unique_seed1_all1000_2x3_ailab.sh`

## Follow-up: params-conditioning A/B

Reuse the existing March 7, 2026 `no_params` all1000 runs as the baseline. Do not rerun them for
this comparison.

We do not need new translated datasets for these conditions. The existing
`results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz`
already contains `source_uid`, `prob`, `g`, `d`, `rt_mu`, and `ice`, so the trainer can rewrite
`instruction_nl` at load time.

The two new conditions are implemented as trainer-side prompt modes:

- `params_always`
  - every loaded prompt becomes:
    `<student> g ... d ... rt ... ice ... </student>` followed by the fraction problem
- `params_dropout50_noprefix`
  - each row deterministically either:
    - keeps the plain no-prefix prompt from the existing CSV, or
    - rewrites to the student-prefixed prompt
  - the keep/add decision is keyed by `source_uid` plus seed `0`
  - the student prefix is added on roughly half the rows, so the remaining half stay exactly in the
    current no-prefix format

Train the new distillation grid with the A/B wrapper:

```bash
RUN_TS=<timestamp> \
bash results/transformer_replication/submit_synth_unique_seed1_all1000_params_ab_2x3_ailab.sh
```

That wrapper submits:

- `params_always`
- `params_dropout50_noprefix`

on the same `135m / 360m / 1.7B` by `pretrained / scratch` grid. Both conditions reuse the same
existing `no_params` CSV and differ only by the prompt-rewrite flags passed into
`train_transformer_hf.py`.

Recipe:

- `EPOCHS=2`
- `BEST_BY=sp2013_acc`
- `EVAL_SP2013=1`
- `SP2013_USE_PARAM_GRID=0`
- `EVALS_PER_EPOCH=5`
- `TRAIN_PROMPT_STUDENT_MODE=always` or `dropout`
- `TRAIN_PROMPT_STUDENT_DROPOUT_PROB=0.5`
- `TRAIN_PROMPT_STUDENT_DROPOUT_SEED=0`

## Default launch

Submit all six jobs:

```bash
bash results/transformer_replication/submit_synth_unique_seed1_all1000_2x3_ailab.sh
```

Dry-run the exact `sbatch` commands:

```bash
DRY_RUN=1 bash results/transformer_replication/submit_synth_unique_seed1_all1000_2x3_ailab.sh
```

Submit only part of the grid:

```bash
MODES=pretrained MODELS=135m,360m \
bash results/transformer_replication/submit_synth_unique_seed1_all1000_2x3_ailab.sh
```

Useful env overrides:

- `DATA_CSV`: alternate translated dataset path
- `RUN_TS`: stable timestamp if you want deterministic output names
- `PARTITION`: Slurm partition, default `ailab`
- `EPOCHS`: default `1`
- `TIME_135M`, `TIME_360M`, `TIME_1P7B`: per-model walltime requests
- `GPUS_135M`, `GPUS_360M`, `GPUS_1P7B`: per-model GPU counts
- `CHAIN_135M`, `CHAIN_360M`, `CHAIN_1P7B`: dependent job count per run
- `BATCH_135M`, `BATCH_360M`, `BATCH_1P7B`: per-model train batch sizes
- `EVAL_BATCH_135M`, `EVAL_BATCH_360M`, `EVAL_BATCH_1P7B`: per-model eval batch sizes
- `GC_135M`, `GC_360M`, `GC_1P7B`: per-model gradient checkpointing (`0/1`)
- `SAVE_LAST_EVERY_135M`, `SAVE_LAST_EVERY_360M`, `SAVE_LAST_EVERY_1P7B`: resumable checkpoint cadence
- `MANIFEST`: output TSV for submitted job ids

The launcher writes a manifest TSV by default:

- `results/transformer_replication/jobs_synth_unique_seed1_all1000_2x3_<timestamp>.tsv`

## Defaults used by the launcher

- `DATA_CSV=results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz`
- `BEST_BY=sp2013_acc`
- `SP2013_USE_PARAM_GRID=0`
- `RESUME_FROM=none`
- `EPOCHS=1`
- `TIME_135M=24:00:00`
- `TIME_360M=24:00:00`
- `TIME_1P7B=16:00:00`
- `GPUS_135M=1`
- `GPUS_360M=1`
- `GPUS_1P7B=2`
- `CHAIN_135M=1`
- `CHAIN_360M=1`
- `CHAIN_1P7B=2`
- `EVALS_PER_EPOCH=5`
- `EVAL_SP2013=1`
- `EVAL_ID_FINAL_ANSWER=1`
- `SAVE_BEST_CHECKPOINT=1`
- `SAVE_FINAL_CHECKPOINT=1`
- `SAVE_LAST_EVERY_*` unset by default and computed from dataset size, GPU count, and batch size

With the current default GPU counts, the launcher computes the default resumable checkpoint cadence
for `10` saves per epoch on this dataset:

- `135m`: `36,450` updates per epoch -> save every `3,645`
- `360m`: `72,900` updates per epoch -> save every `7,290`
- `1.7B`: `145,800` updates per epoch -> save every `14,580`

This is for the `last/` resumable checkpoint stream. If `SAVE_EVAL_CHECKPOINTS=1`, the trainer will
still create additional eval checkpoints.

Output directory naming:

- `smol2_<model>_synth_unique_seed1_all1000_mincols_<mode>_sp2013noparam_tracev6_bestsp_<timestamp>`

Job names:

- `s1000_135_pt`, `s1000_135_scr`
- `s1000_360_pt`, `s1000_360_scr`
- `s1000_1p7b_pt`, `s1000_1p7b_scr`

## Resume / continue training

Yes. The pipeline supports continuing training from the latest resumable checkpoint.

Important detail: the launcher uses `RUN_TS` in the output directory name, so a resume launch must
reuse the same `RUN_TS` as the original run.

For the current defaults, the launcher already chains the `1.7B` runs automatically:

- stage 1: fresh run (`RESUME_FROM=none`)
- stage 2: dependent continuation (`RESUME_FROM=auto`, `REQUIRE_RESUME=1`)

So the full 2x3 submit will automatically launch:

- `135m pretrained`: 1 job
- `135m scratch`: 1 job
- `360m pretrained`: 1 job
- `360m scratch`: 1 job
- `1.7B pretrained`: 2 dependent jobs
- `1.7B scratch`: 2 dependent jobs

Manual example: continue a finished `1`-epoch `1.7B pretrained` run to epoch `2` total:

```bash
RUN_TS=<original_timestamp> \
MODES=pretrained \
MODELS=1p7b \
RESUME_FROM=auto \
EPOCHS=2 \
bash results/transformer_replication/submit_synth_unique_seed1_all1000_2x3_ailab.sh
```

`EPOCHS` is the total target epoch count, not “extra epochs to add.”

## Runtime estimate

The best reference runs are the completed `subjid_0472` trace-v6 best-SP jobs on the smaller
dataset `results/transformer_replication/synth_unique_subjid_0472_seeds1_20_nlp_no_params.csv.gz`.

Observed baseline runs:

- `135m`: job `5317506`, elapsed `00:54:26`
- `360m`: job `5317507`, elapsed `01:13:38`
- `1.7B`: job `5318061`, elapsed `02:53:50`

These jobs all used:

- `rows total: 414,720`
- `rows train: 373,248`
- `rows val: 20,736`
- `rows test: 20,736`

The all1000 translated dataset has:

- `20,736,000` rows total

That is exactly `50x` the `subjid_0472` dataset size.

### Current default: 1 epoch

With the launcher now defaulting to `1` epoch, `1` GPU for `135m/360m`, and `2` GPUs for `1.7B`,
the direct `50x` extrapolation is:

| model | projected on all1000, current default |
| --- | --- |
| `135m` | `15h 07m` |
| `360m` | `20h 27m` |
| `1.7B` | `1d 00h 09m` |

The launcher therefore defaults to:

- `TIME_135M=24:00:00`
- `TIME_360M=24:00:00`
- `TIME_1P7B=16:00:00`

`1.7B` still projects slightly above one day, so the launcher chains two `16h` windows for that
model by default.

### Reference: 3 epochs

For the old `3`-epoch configuration, the same direct `50x` extrapolation is:

| model | observed on 472 | projected on all1000 |
| --- | --- | --- |
| `135m` | `00:54:26` | `1d 21h 21m` |
| `360m` | `01:13:38` | `2d 13h 21m` |
| `1.7B` | `02:53:50` | `6d 00h 51m` |

As of March 7, 2026, `scontrol show partition ailab` reported:

- `MaxTime=15-00:00:00`

So the projected jobs fit within the partition limit.

### Scratch vs pretrained runtime

There is no completed production scratch run for this exact grid yet. The code path exists via
`INIT_FROM_SCRATCH=1`, and the training loop is otherwise the same, so runtime should be very close
to pretrained. The difference should be startup noise, not epoch-scale time.

## Runtime reduction ideas

If the goal is “make `1.7B` fit in about one day,” the best remaining levers are:

- Raise `BATCH_1P7B` from `64` to `128` per GPU if it fits on H200.
  With `2` GPUs that would move global batch from `128` to `256`, cutting updates per epoch from
  `145,800` to `72,900`.
- Reduce training-time eval overhead with `EVALS_PER_EPOCH=1`.
  This is a smaller gain than increasing batch size, but it cuts repeated `val/test/SP2013/ID`
  evaluation during the epoch.
- Set `SAVE_EVAL_CHECKPOINTS=0` if you only need `last/`, `best/`, and `final/`.
  This reduces checkpoint I/O and directory churn.
- If runtime matters more than live SP2013 selection, disable in-training expensive evals:
  `EVAL_SP2013=0 EVAL_ID_FINAL_ANSWER=0 BEST_BY=val_loss`
  Then do SP2013 evaluation after training. This is the most aggressive non-batch runtime cut.
- If `BATCH_1P7B=128` fits comfortably, test `GC_1P7B=0`.
  Disabling gradient checkpointing can speed training, but only if memory headroom is real.

A practical “try to get 1.7B under one day” configuration is:

```bash
RUN_TS=<timestamp> \
MODES=pretrained \
MODELS=1p7b \
EPOCHS=1 \
GPUS_1P7B=2 \
BATCH_1P7B=128 \
EVAL_BATCH_1P7B=128 \
EVALS_PER_EPOCH=1 \
SAVE_EVAL_CHECKPOINTS=0 \
bash results/transformer_replication/submit_synth_unique_seed1_all1000_2x3_ailab.sh
```

Before committing to that, the safe probe is:

```bash
sbatch \
  --export=ALL,REPO_ROOT=/scratch/gpfs/BRENDEN/changho/UMA_PR02,CONDA_ENV=torch-env,MODEL_NAME=HuggingFaceTB/SmolLM2-1.7B,DATA_CSV=/scratch/gpfs/BRENDEN/changho/UMA_PR02/results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz,MAX_LENGTH=192,START_BATCH=64 \
  results/transformer_replication/slurm_probe_smollm2_batch_ailab.sbatch
```

## Memory note

`train_transformer_hf.py` is not streaming. It currently does:

- `pd.read_csv(args.data_csv, usecols=...)`
- full in-memory train/val/test DataFrames
- `tolist()` materialization for prompts and responses

For the `subjid_0472` dataset:

- the two-column pandas frame (`instruction_nl`, `response_nl`) was about `145 MiB` deep memory
- the same preprocessing path increased process RSS from about `0.139 GiB` to about `0.323 GiB`

Simple linear scaling from `414,720` rows to `20,736,000` rows suggests the text-table side is
still manageable on current job sizes, roughly:

- raw two-column DataFrame: about `7.2 GiB`
- rough in-memory preprocessing footprint: about `9-10 GiB`

This is consistent with the current per-model memory asks staying plausible:

- `135m`: `64G`
- `360m`: `80G`
- `1.7B`: `96G`

This estimate is about CPU-side text storage only. GPU memory behavior is unchanged by dataset
size; walltime is the main scaling problem here, not VRAM.

## Relevant files

- launcher: `results/transformer_replication/submit_synth_unique_seed1_all1000_2x3_ailab.sh`
- training sbatch files:
  - `results/transformer_replication/slurm_train_fraction2400_smol135_ailab.sbatch`
  - `results/transformer_replication/slurm_train_fraction2400_smol360_ailab.sbatch`
  - `results/transformer_replication/slurm_train_fraction2400_smol1p7b_ailab.sbatch`
- trainer:
  - `results/transformer_replication/train_transformer_hf.py`
- translated all1000 dataset:
  - `results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz`
- reference 472 dataset:
  - `results/transformer_replication/synth_unique_subjid_0472_seeds1_20_nlp_no_params.csv.gz`
