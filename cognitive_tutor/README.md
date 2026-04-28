# Cognitive Tutor Algebra Distillation

This folder contains the single-variable Algebra 2006-2007 pipeline used for
synthetic cognitive-model distillation.

The current distillation-ready artifact is the v3 CT-flow, human-mixed
translated synthetic dataset:

```text
cognitive_tutor/data/processed/algebra_2006_2007/
  translated_synthetic_algebra_v3_human_mixed/
```

It contains one million translated examples generated from 10,000 synthetic
single-variable equations crossed with the 100 UMA parameter students.

## Dataset

Tracked/shareable form:

```text
translated_synthetic_algebra_v3_human_mixed/parquet/
  train/part-*.parquet
  val/part-*.parquet
  test/part-*.parquet
  manifest.json
```

CSV form generated locally:

```text
singlevar_algebra_synth_v3_human_mixed_train.csv.gz
singlevar_algebra_synth_v3_human_mixed_val.csv.gz
singlevar_algebra_synth_v3_human_mixed_test.csv.gz
singlevar_algebra_synth_v3_human_mixed_audit.json
singlevar_algebra_synth_v3_human_mixed_preview.csv
```

Schema:

```text
source_uid, split, subjid, prob, equation_type, variable, answer, correct,
g, d, c, rt_mu, ice, instruction_nl, response_nl
```

The student parameters are retained as columns but are not exposed in the
default prompt. This supports both unconditioned distillation and later
student-conditioned prompt variants.

## Translation Surface

The default translated surface is `human_mixed`. It hides UMA internals
(`strategy`, `goals`, `exec`, raw rule traces, and KC labels) and emits only a
student-like solution attempt.

Correct example:

```text
Solve this algebra equation: y/36 = 5/18.

I start with y/36 = 5/18.
I clear the denominator, so it turns into y = 10.
So I get 10.
### answer: 10
```

Incorrect examples may stop at a current equation state:

```text
I start with (5y+5)/20 = -33.
I simplify that part and get 1/4*y+1/4 = -33.
I am not sure how to keep going from here, so I put 1/4*y+1/4 = -33.
### answer: 1/4*y+1/4 = -33
```

Or make a plausible wrong numeric guess:

```text
I write down (5y+5)/20 = -33.
I simplify that part and get 1/4*y+1/4 = -33.
I think the last step should give y = 133.
So I answer 133.
### answer: 133
```

## Rebuild Commands

Build the translated train/val/test CSV package:

```bash
python cognitive_tutor/build_translated_synthetic_algebra_dataset.py \
  --input-csv results/UMA_replication/algebra_uma_k100_v3_ct_flow/algebra_uma_synth_matched_master_10k_k100_v3_ct_flow_nlp_human_mixed.csv.gz \
  --out-dir cognitive_tutor/data/processed/algebra_2006_2007/translated_synthetic_algebra_v3_human_mixed \
  --prefix singlevar_algebra_synth_v3_human_mixed \
  --val-frac 0.02 \
  --test-frac 0.02 \
  --seed 20260428
```

Shard the package for GitHub/Neuronic transfer:

```bash
python cognitive_tutor/shard_translated_synthetic_algebra_parquet.py \
  --dataset-dir cognitive_tutor/data/processed/algebra_2006_2007/translated_synthetic_algebra_v3_human_mixed \
  --prefix singlevar_algebra_synth_v3_human_mixed \
  --rows-per-shard 50000 \
  --compression zstd
```

Load the Parquet shards:

```python
import pandas as pd

df = pd.read_parquet(
    "cognitive_tutor/data/processed/algebra_2006_2007/"
    "translated_synthetic_algebra_v3_human_mixed/parquet/train"
)
```

## Alignment Evaluation

The algebra analogue of fraction Add/Sub/Mul/Div ED/UD bins is equation type:

```text
linear_one_step
linear_two_step
linear_with_denominator
reciprocal_variable_denominator
linear_variable_both_sides
```

The implemented report covers:

1. Outcome alignment: `P(correct | equation_type)`.
2. Error-process alignment: stuck-stage distribution for incorrect traces.
3. Trace-shape alignment: visible step-count distribution.
4. Answer-type alignment: equation-state vs numeric vs variable-assignment
   wrong responses.

Run:

```bash
python cognitive_tutor/evaluate_algebra_alignment.py \
  --synthetic-translated-csv results/UMA_replication/algebra_uma_k100_v3_ct_flow/algebra_uma_synth_matched_master_10k_k100_v3_ct_flow_nlp_human_mixed.csv.gz \
  --out-dir cognitive_tutor/analysis/algebra_alignment_v3_human_mixed
```

Important outputs:

```text
cognitive_tutor/analysis/algebra_alignment_v3_human_mixed/
  accuracy_by_equation_type.csv
  stuck_stage_distribution.csv
  step_count_distribution.csv
  answer_type_distribution.csv
  distribution_tv_summary.csv
  alignment_summary.json
```

Current headline: accuracy and incorrect-trace stuck-stage distributions align
closely by equation type, while correct-trace step counts intentionally differ
because the synthetic translation verbalizes more intermediate work than the raw
Cognitive Tutor step logs.

## Neuronic

Example SLURM launcher:

```bash
sbatch cognitive_tutor/slurm/train_algebra_human_mixed_smol135_neuronic.sbatch
```

The script materializes the Parquet shards into a temporary CSV and runs
`results/transformer_replication/train_transformer_hf.py` with
`instruction_nl` and `response_nl`.

Useful overrides:

```bash
sbatch --export=ALL,CONDA_ENV=torch-env,MODEL_NAME=HuggingFaceTB/SmolLM2-135M,EPOCHS=3 \
  cognitive_tutor/slurm/train_algebra_human_mixed_smol135_neuronic.sbatch
```

Large raw Cognitive Tutor logs and raw simulator traces are not required for
Neuronic training. The sharded Parquet package plus this code is enough to run
the translated synthetic distillation experiment.
