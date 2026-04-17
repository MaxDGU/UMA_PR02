# Project Summary

This repository started as the reference codebase for **UMA** (the Unified Model of Arithmetic), a rule-based cognitive model of arithmetic reasoning. It has since grown into a larger research workspace for:

- training many simulated UMA students,
- collecting their fraction-solving traces,
- translating those traces into instruction/response text,
- distilling them into transformer language models,
- and then finetuning those distilled models on a small human reasoning dataset.

The top-level `README.md` still reflects the original UMA release. In practice, the newer distillation and finetuning work now lives mostly under `results/` and `BBT/`.

## What the project is doing

At a high level, the repo tries to answer a research question:

**Can a modern LM learn to imitate a symbolic cognitive model of fraction arithmetic, and then adapt toward real human reasoning with a small amount of human finetuning?**

The workflow is:

1. Build or load many UMA students with different latent parameters.
2. Run them on arithmetic problem sets and save their reasoning traces.
3. Convert those symbolic traces into natural-language style supervision.
4. Distill that supervision into a causal LM such as SmolLM2.
5. Finetune the distilled checkpoint on human fraction-solution data.
6. Evaluate on SP2013 fraction problems and held-out human data.

## Main layers of the repo

### 1. Original UMA simulator

`1. Model/` contains the original cognitive model:

- `uma.py`: UMA architecture, memory/state representation, context features, and rule machinery.
- `models.py`: the production-rule set for the full arithmetic model.
- `simulators.py`: training/testing utilities for running UMA on curricula and test sets.

This is the symbolic source model. It is not a neural LM. It generates the procedural reasoning behavior that the later transformer pipeline tries to imitate.

### 2. UMA replication and trace generation

`results/UMA_replication/` contains the practical tooling for large-scale synthetic data generation:

- `train_models_shard.py`: train shards of UMA students and save each trained model.
- `run_saved_models_on_problem_set.py`: load saved UMA models and run them on a chosen problem set to emit trace CSVs.
- `run_saved_models_on_human.py`: evaluate saved UMA models on human data.
- `generate_traces.py`, merge scripts, and many Slurm launchers: scale the process across clusters.

This layer is what turns the symbolic simulator into a large synthetic dataset.

Important latent student parameters used throughout the repo:

- `g`: decision noise
- `d`: error discount
- `rt_mu`: retrieval threshold
- `ice`: initial counting experience

These parameters matter because the same visible fraction problem can map to different traces depending on the hidden student state.

### 3. UMA-to-text translation and LM distillation

`results/transformer_replication/` contains the neural modeling work:

- `translate_uma_traces_to_nlp.py`: deterministically converts UMA trace rows into `instruction_nl` and `response_nl`.
- `train_transformer_hf.py`: distills translated traces into a Hugging Face causal LM.
- `finetune_humandata.py`: finetunes a distilled checkpoint on human data.
- notebooks, analysis exports, and Slurm submitters: compare runs and launch grids.

The translated prompts can include student parameters in three modes:

- `none`: only the problem text is shown
- `always`: prepend a `<student> ... </student>` block with `g/d/rt/ice`
- `dropout`: deterministically show the student block on only some rows

That prompt-conditioning choice is one of the main current research knobs in the repo.

### 4. BBT pipeline

`BBT/` is a newer orchestration layer that wraps the experiment into named subpipelines:

1. `uma_traces`
2. `translation`
3. `distill_train`
4. `distill_eval`
5. `human_finetune`
6. `human_eval`

Key files:

- `BBT/pipeline.py`: top-level orchestrator
- `BBT/configs/default_pipeline.json`: default experiment config
- `BBT/eval_sampling.py`: sampling-based SP2013 / dataset evaluation
- `BBT/src/*.py`: one wrapper per subpipeline

Each stage writes a `manifest.json`, so the pipeline is meant to support resumable, reproducible experiment handoffs.

## End-to-end data and training flow

### Synthetic data

The synthetic side of the project comes from many trained UMA students solving many fraction problems. One analysis artifact, `results/UMA_replication/synth_unique_seed1_all1000/accuracy_summary.json`, summarizes a **20,736,000-row** all-1000 synthetic evaluation set.

That summary reports:

- overall UMA accuracy: about **0.4307**
- multiplication easiest: about **0.6467**
- addition next: about **0.5505**
- subtraction harder: about **0.3329**
- division hardest: about **0.1928**

So the synthetic teacher itself has a strong, structured error profile rather than being uniformly correct.

### Translation

The text translation layer preserves source mapping columns like `source_uid`, adds natural-language strategy/goal/execution text, and ends each target with a parseable final-answer line:

`### answer: <answer>`

This lets the repo train LMs both on reasoning text and on explicit final-answer extraction.

### Distillation

The main distillation script is `results/transformer_replication/train_transformer_hf.py`. It supports:

- pretrained or scratch initialization
- SP2013 evaluation during training
- in-distribution held-out answer evaluation
- checkpoint selection by `val_loss`, `sp2013_acc`, or `id_val_acc`
- optional LoRA for larger models

### Human finetuning

The human adaptation stage uses:

- `results/human/data_train_nlp.csv`: **288** rows
- `results/human/data_val_nlp.csv`: **96** rows

`finetune_humandata.py` loads a distilled checkpoint, trains on the small human dataset, tracks validation NLL each epoch, and can optionally run SP2013 sampling during or after finetuning.

## What experiments seem most current

Two docs make the present direction fairly clear:

- `EXPERIMENT_humanft_and_prompt_conditioning_summary.md`
- `EXPERIMENT_synth_unique_seed1_all1000_2x3.md`

The main current questions are:

1. Is human finetuning undertrained if run for only a few epochs?
2. Is the bigger issue that the synthetic prompt hides the student state?
3. Does mixed conditioning (`params_dropout50_noprefix`) transfer better than always showing params?

The current working hypothesis in those docs is:

- human finetuning may have been somewhat undertuned,
- distillation itself may not be the main bottleneck,
- hidden-parameter ambiguity in the prompt format is likely a major issue.

## Representative current results

These are not “final paper numbers”; they are representative artifacts already present in the repo.

### Distillation baselines on the all1000 synthetic dataset

From the March 7, 2026 pretrained all1000 runs selected by SP2013 accuracy without a param grid:

- `135M`: SP2013 accuracy **0.375**
- `360M`: SP2013 accuracy **0.1875**
- `1.7B`: SP2013 accuracy **0.625**

So in the current artifacts, the 1.7B pretrained distilled model is the strongest all1000 baseline by a wide margin.

### Recent long human-finetune runs in BBT

From the March 25, 2026 BBT 50-epoch runs:

- `135M` human FT: validation NLL improved from **3.49** at epoch 1 to **1.65** best at epoch 43
- `360M` human FT: validation NLL improved from **2.64** at epoch 1 to **1.40** best at epoch 44

Those run directories currently contain the `05_human_finetune` artifacts and show clear held-out NLL improvement. I did not find completed downstream `06_human_eval` artifacts in those specific BBT run folders yet.

## Where to start reading

If you want the shortest path through the codebase:

1. `summary.md` (this file)
2. `results/README.md`
3. `BBT/README.md`
4. `results/transformer_replication/UMA_NLP_TRANSLATION_RULES.md`
5. `BBT/configs/default_pipeline.json`
6. `results/transformer_replication/train_transformer_hf.py`
7. `results/transformer_replication/finetune_humandata.py`

## One-sentence summary

This is a research repo that turns a symbolic fraction-arithmetic model (UMA) into large synthetic reasoning datasets, distills those datasets into transformer students, and then tests whether small-data human finetuning can move those students closer to real human fraction reasoning.
