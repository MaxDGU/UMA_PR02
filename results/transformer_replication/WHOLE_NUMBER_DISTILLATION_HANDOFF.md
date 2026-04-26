# Whole-Number Arithmetic Distillation Handoff

## Purpose

This note is a minimal handoff for adapting the current UMA-to-language
distillation pipeline to whole-number arithmetic. It is intended for a
collaborator who already has trained UMA student models and wants to reuse the
existing distillation and human-finetuning code with the smallest possible set
of domain-specific changes.

## Existing Reference Pipeline

The current distillation path has four stages.

1. Generate UMA symbolic traces from trained UMA students.
2. Translate each trace row into `instruction_nl` and `response_nl`.
3. Train a causal LM on the translated rows with `train_transformer_hf.py`.
4. Optionally fine-tune the distilled checkpoint on human rows with
   `finetune_humandata.py`.

The main files to read first are:

- `results/transformer_replication/translate_uma_traces_to_nlp.py`
- `results/transformer_replication/train_transformer_hf.py`
- `results/transformer_replication/finetune_humandata.py`
- `results/transformer_replication/translator_contract_spec.py`
- `results/transformer_replication/test_translator_contract.py`
- `BBT/README.md`
- `BBT/docs/runbook.md`
- `BBT/docs/subpipeline_contracts.md`

The `BBT/` wrapper is useful when you want an experiment directory with
manifests for each stage. The lower-level scripts are easier when you are still
debugging translation rules.

## Minimal Whole-Number Adaptation

### Step 1: Define the problem CSV

Create a held-out-safe whole-number problem set before generating traces. The
CSV should keep at least:

- `prob`: canonical surface form, such as `24+37` or `14*21`
- `operation`: `Add`, `Sub`, `Mul`, or whichever labels the evaluation uses
- optional structural labels, such as digit length or carry/borrow condition

TODO(Max): decide the six-cell or multi-cell taxonomy for whole-number
arithmetic before training. The decimal work used operation by operand-type
cells; whole-number arithmetic probably needs digit length and carry/borrow
structure instead.

### Step 2: Run UMA students on the problem set

The distillation trainer expects a trace CSV with the standard UMA columns:

- `subjid`
- `prob`
- `operation`
- `strategy`
- `goals`
- `exec`
- `answer`
- `correct` or `is_correct`
- optional student parameters such as `g`, `d`, `rt_mu`, `ice`
- optional `work`

For decimal XLSX students, the closest reference runner is:

```bash
python results/UMA_replication/run_xlsx_students_on_decimal_problemset.py \
  --models-dir fractionGPT/uma_trained_students \
  --student-csv results/UMA_replication/decimal_param_selection_k100/selected_params_gd_full_grid_optimized100.csv \
  --student-index 0 \
  --problem-csv results/UMA_replication/decimal_bss_matched_3600_nobss.csv \
  --out-csv /tmp/uma_trace_smoke.csv \
  --extract-work
```

TODO(Max): copy or adapt this runner for whole-number problems. The important
part is not the decimal grid logic; it is preserving the standard trace schema
so the translator/trainer do not need new data loaders.

### Step 3: Add whole-number translation rules

`translate_uma_traces_to_nlp.py` is deterministic and rule based. It already
knows fraction and decimal strategy tokens. For whole-number arithmetic, add a
new reasoning mode rather than changing an existing mode.

Recommended edits:

- Add a mode such as `whole_number_think_aloud_child` to
  `VALID_REASONING_MODES`.
- Add operation/strategy/goal/exec text for whole-number carry, borrow, and
  multi-digit multiplication tokens.
- Add a whole-number response builder parallel to the decimal think-aloud
  builder.
- Keep the final answer contract: every response should end with
  `### answer: <answer>`.
- Add examples to `test_translator_contract.py`.

TODO(Max): decide whether whole-number subtraction is in scope. If it is,
translation rules should describe borrow behavior explicitly and evaluation
should stratify by borrow/no-borrow.

TODO(Max): decide whether to expose UMA parameters in the stored CSV. For the
decimal runs we stored `student_prompt_mode=none` in the translated CSV and
added parameters later during training with `--train_prompt_student_mode
always`.

### Step 4: Build the NLP CSV

Once the trace CSV exists and the translation mode is implemented, run:

```bash
python results/transformer_replication/translate_uma_traces_to_nlp.py \
  --input-csv results/UMA_replication/whole_number_traces/uma_traces_whole_number.csv \
  --output-csv results/transformer_replication/whole_number_distill/uma_traces_whole_number_nlp.csv.gz \
  --no-include-sp2013-seeds \
  --reasoning-mode whole_number_think_aloud_child \
  --student-prompt-mode none
```

If the whole-number generator writes multiple shards, merge them before
translation or add a wrapper that translates each shard and concatenates the
translated outputs.

### Step 5: Train the distilled model

Use the same trainer as the fraction and decimal runs. Start with the 135M
model for fast iteration.

```bash
python results/transformer_replication/train_transformer_hf.py \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --data_csv results/transformer_replication/whole_number_distill/uma_traces_whole_number_nlp.csv.gz \
  --output_dir results/transformer_replication/whole_number_distill/smollm2_135m_params_always \
  --epochs 2 \
  --lr 5e-4 \
  --min_lr 1e-6 \
  --lr_scheduler_type linear \
  --batch_size 96 \
  --eval_batch_size 96 \
  --grad_accum_steps 1 \
  --max_length 256 \
  --val_frac 0.01 \
  --test_frac 0.01 \
  --split_by_problem \
  --train_prompt_student_mode none \
  --save_eval_checkpoints \
  --evals_per_epoch 1 \
  --loss_evals_per_epoch 1 \
  --save_best_checkpoint \
  --save_final_checkpoint
```

This first command keeps the translated `instruction_nl` column as the source
of truth. For a parameter-conditioned run, use
`--train_prompt_student_mode always` only after generalizing the trainer prompt
builder. In the current fraction-first trainer, that mode rewrites the prompt
surface and assumes fraction wording.

TODO(Max): add a trainer prompt noun such as `whole-number` if using
`--train_prompt_student_mode always`. The decimal branch did this with a
domain-specific prompt-kind flag; whole-number arithmetic should get the same
treatment before serious parameter-conditioned training.

TODO(Max): tune `max_length` after inspecting translated examples. Whole-number
vertical arithmetic may need less than 256 tokens for simple problems, but
multi-step multiplication with explicit carries can exceed the decimal trace
lengths.

## Evaluation Hooks

For fraction-style SP2013 eval, the trainer has built-in flags. For a new
whole-number evaluation set, the clean path is to copy the structure of
`eval_decimal_temp_grid.py` and replace:

- problem metadata loader
- cell taxonomy
- target human/UMA CSV paths
- answer normalization if the domain needs special formatting

TODO(Max): define the human target file before model selection. The decimal
report selected by six-cell MAG to human; a whole-number run should pick a
cell-wise metric before launching the grid.

## Human Fine-Tuning Reference

Human fine-tuning is separate from UMA distillation. The reference script is:

```bash
python results/transformer_replication/finetune_humandata.py \
  --model_path results/transformer_replication/whole_number_distill/smollm2_135m_params_always \
  --checkpoint_subdir best \
  --train_csv results/human/data_train_nlp.csv \
  --val_csv results/human/data_val_nlp.csv \
  --output_dir results/transformer_replication/whole_number_distill/smollm2_135m_humanft \
  --instruction_style plain \
  --epochs 3 \
  --lr 1.5e-5 \
  --batch_size 64 \
  --eval_batch_size 64
```

For whole-number human fine-tuning, replace `results/human/data_train_nlp.csv`
and `results/human/data_val_nlp.csv` with whole-number human rows. The script
can also initialize from a checkpoint, pretrained weights, or random weights
with `--init_strategy`.

TODO(Max): keep human fine-tuning data format aligned with the distillation
format: `prob`, `instruction_nl`, and `response_nl` are the minimum useful
columns. If subject or parameter conditioning is used, include `subjid` or
`g,d,rt_mu,ice` and set `--instruction_style` accordingly.

## Slurm / Pipeline Wrapper

The `BBT/` pipeline can run the stages with manifests:

```bash
python BBT/pipeline.py --dry-run
```

For whole-number arithmetic, the most useful BBT stages are:

```bash
python BBT/pipeline.py \
  --subpipelines translation,distill_train,distill_eval,human_finetune,human_eval \
  --set external_inputs.raw_trace_csv=results/UMA_replication/whole_number_traces/uma_traces_whole_number.csv \
  --set subpipelines.translation.reasoning_mode=whole_number_think_aloud_child \
  --set subpipelines.distill_train.model_name=HuggingFaceTB/SmolLM2-135M
```

TODO(Max): after whole-number translation is implemented, add a dedicated
`BBT/configs/whole_number_pipeline.json` rather than overloading the default
fraction/decimal config.

## Minimal Readiness Checklist

- [ ] Whole-number problem taxonomy is defined.
- [ ] UMA trace runner emits the standard trace schema.
- [ ] Translator has a new whole-number reasoning mode.
- [ ] Translator contract tests include carry, no-carry, borrow, and
      multiplication examples.
- [ ] Held-out human/UMA evaluation targets are fixed before model selection.
- [ ] Distillation command runs on a 100-row smoke CSV.
- [ ] Human fine-tuning command runs on a tiny train/val smoke split.
