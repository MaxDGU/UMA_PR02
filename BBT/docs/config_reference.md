# Config Reference

Config file: `BBT/configs/default_pipeline.json`

## Top-level

- `experiment.name`: run name prefix.
- `experiment.run_root`: root for pipeline runs.
- `experiment.seed`: default seed fallback.

- `paths.uma_models_dir`
- `paths.uma_problem_csv`
- `paths.sp2013_csv`
- `paths.human_train_csv`
- `paths.human_val_csv`

- `external_inputs.uma_trace_csv`
- `external_inputs.translated_csv`
- `external_inputs.distill_run_dir`
- `external_inputs.human_finetune_run_dir`

- `subpipelines.<name>.enabled`: enable/disable subpipeline.
- `subpipelines.<name>.output_dir`: output directory template.

## Notable subpipeline options

### `subpipelines.distill_train`
- `model_name`
- `init_from_scratch`
- training hyperparameters (`epochs`, `batch_size`, `lr`, etc.)
- eval toggles (`eval_sp2013`, `eval_id_final_answer`)

### `subpipelines.distill_eval` and `subpipelines.human_eval`
- `datasets`: list of `{name,csv,problem_col}`
- standard human-eval flow uses `checkpoint_subdir=best`
- sampling controls:
- `num_samples_per_prompt`
- `temperature`
- `top_p`
- `top_k`
- param grid controls:
- `use_param_grid`
- `sp2013_grid_g`
- `sp2013_grid_d`
- `sp2013_grid_rt`
- `sp2013_grid_ice`

For the default human-eval workflow, sampling runs once after training on the `best` checkpoint with `num_samples_per_prompt=1000` and `use_param_grid=false`.

### `subpipelines.human_finetune`
- `train_csv`
- `val_csv`
- `best_by` (default `val_nll`)
- `eval_sp2013` (default `false`)
- `eval_loaded_model` (default `false`)

The standard human-finetune workflow tracks validation NLL during training and leaves SP2013 sampling to the downstream `human_eval` stage.

Example experiment-specific override:
- `--set subpipelines.human_finetune.epochs=50`

This keeps the default human-finetune LR at `1.5e-5` unless you also override `subpipelines.human_finetune.lr`.
