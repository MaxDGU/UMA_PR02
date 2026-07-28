# Corrected Qwen3-4B Decimal Fine-Tuning Rerun

## Status

- Original protocol frozen: 2026-07-28
- Corrected profile-trace/KL-sweep protocol revised: 2026-07-28
- State: corrected v2 DAG submitted to AILab on 2026-07-28
- Repository commit at start: `a043142e8fcf01108855651b5d94ed05dbcb0060`
- Scope: Qwen3-4B only; no 8B jobs
- Scheduler: all corrected stages will use the `ailab` partition

The completed Direct human-FT and CPT human-FT runs remain valid. The first
Weighted-KL pilot used 1,000 unconditioned stochastic completions per problem
instead of one completion from each of 1,000 explicitly conditioned UMA
profiles. Its training, rollout, and aggregate artifacts are therefore
superseded and must not be used as results for the intended ablation.

## Unresolved Paper Follow-Up: Instruct Initialization

The decimal portion of the paper's Instruct-initialization ablation has not
been migrated to the participant-held-out protocol. Until it is rerun, leave
the existing appendix table, comparison figure, and accompanying text
unchanged but treat their decimal results as stale.

- The unadapted and cognitive-distillation-only Instruct rollouts can be
  rescored against the 19 held-out participants.
- Five corrected Direct human-FT checkpoints already exist under
  `results/finetuning/rerun_decimals/instruct/direct_human_ft/`, but still
  require matched rollout evaluation.
- Correct Instruct-initialized cognitive-distillation and CPT-LLM checkpoints
  are not currently available. Obtain them from Max before launching those
  conditions.
- Once all four conditions are available, regenerate the decimal column in
  the initialization table, the decimal panel of the Base-vs-Instruct figure,
  and the claim that the two final CPT-LLM models are nearly indistinguishable.

## Goal

Replace the old decimal Direct human-FT and distill-then-human-FT results with
participant-held-out results, and evaluate importance-weighted self-training
on distilled-model traces plus the KL objective described in the commented
Appendix ablation.

This rerun produces experimental artifacts for review. It does not update the
paper, Figure 3, Appendix text, or `main.pdf`.

## Frozen Data Protocol

- Source:
  `eval/outputs/neurips_reproduction/raw/decimal/decimal_human_responses.csv`
- Use all 92 participants and all 12 decimal problems.
- Sort unique participant IDs before splitting.
- Split participants with seed 42, stratified by the joint `(grade, school)`
  stratum:
  1. Hold out 19 participants for test.
  2. Hold out 18 participants from the remainder for validation.
  3. Use the remaining 55 participants for training.
- Expected valid rows:
  - train: 660
  - validation: 215 (one missing response is dropped after assignment)
  - test: 228
- Preserve the participant with the missing response in the assigned split;
  drop only the missing response.
- Every split must contain all 12 problems.
- Direct and CPT human-FT targets are answer-only:
  `My answer is <answer>.\n### answer: <answer>`.
- Prompts use the existing decimal instruction style.
- Test participants must not enter training targets, importance weights,
  validation, checkpoint selection, or hyperparameter selection.

The split builder must write exact participant lists, row/problem counts,
stratum counts, source/output SHA-256 hashes, and overlap assertions to a
machine-readable manifest.

## Frozen Model Protocol

Base model: `Qwen/Qwen3-4B-Base`.

CPT initialization:

- Repository: `MaxDGUPTA/uma-cognitive-llm-checkpoints`
- Revision: `cadb864b6542b4cadcd7afba079c00ffa9fcb12c`
- Subdirectory: `decimals/distill_lora`
- The old `decimals/distill_humanft` adapter is prohibited as an
  initialization.

Run five training seeds, 42 through 46, for each trained condition:

| Condition | Initialization | KL beta | LR | Epochs |
|---|---|---:|---:|---:|
| Direct human FT | Qwen3-4B-Base + fresh LoRA | — | `1e-4` | 5 |
| CPT human FT | decimal `distill_lora` | — | `1e-4` | 5 |
| CPT weighted self-training | decimal `distill_lora` | `0` | `1e-4` | 5 |
| CPT weighted self-training + KL | decimal `distill_lora` | `0.01` | `1e-4` | 5 |
| CPT weighted self-training + KL | decimal `distill_lora` | `0.1` | `1e-4` | 5 |
| CPT weighted self-training + KL | decimal `distill_lora` | `0.5` | `1e-4` | 5 |

Shared settings:

- LoRA rank 16, alpha 32, dropout 0.05
- targets: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Direct/CPT human FT: batch size 2; gradient accumulation 8
- weighted-KL CPT: batch size 16; gradient accumulation 1
- effective batch size 16 for every condition
- weight decay 0.01; warmup ratio 0.03
- maximum length 256
- bf16; gradient checkpointing for Direct/CPT human FT only
- checkpoint selection by minimum validation completion-token NLL
- no test-set evaluation during training

## Corrected Profile-Trace and Weighted-KL Protocol

On 2026-07-28, the shared optimization settings were updated to learning rate
`1e-4` and 5 epochs. The corrected experiment varies KL beta over
`{0, 0.01, 0.1, 0.5}` while holding every other weighted-training input and
setting fixed.

### Profile-conditioned trace generation

1. Freeze the downloaded decimal `distill_lora` checkpoint. Its archived
   training arguments record `train_prompt_student_mode=always`, so trace
   generation must expose the student parameters.
2. Construct the standard Cartesian UMA parameter grid:
   - `g = {0.01, 0.02, ..., 0.10}` (10 values);
   - `d = {0.1, 0.3, 0.5, 0.7, 0.9}` (5 values);
   - `rt = {3, 4, 5, 6}` (4 values);
   - `ice = {0, 25, 50, 75, 100}` (5 values).
   This produces exactly 1,000 profiles in deterministic lexicographic order,
   with stable profile IDs 0 through 999.
3. For each of the 12 human-data problems and each profile, generate exactly
   one complete response from:

   `<student> g <g> d <d> rt <rt> ice <ice> </student>`

   followed by the ordinary decimal problem prompt. Use vLLM, temperature
   `1.0`, top-p `0.95`, maximum 192 new tokens, and trace-generation seed
   `20260728`. The result is exactly 12,000 traces.
4. Retain profile ID and the four parameter values as provenance. Parse the
   generated final answer. The trace artifact and downstream training bundle
   must not contain or consume `correct_answer`, `is_correct`, or human `acc`.

Profile conditioning is used only to sample a diverse source trace for every
UMA profile. It is removed before weighted fine-tuning.

### Plain-prompt weighted training data

For every generated trace, set the training prompt to the unconditioned form:

`Solve this decimal problem: <problem>=?`

The generated reasoning response remains the completion target. The
profile-conditioned generation prompt is retained only as provenance in the
source trace artifact and its manifest, not in the plain-prompt weighted
training bundle or the model input used for fine-tuning.

For problem \(x\) and normalized generated answer \(a\), estimate:

- \(p_H(a\mid x)\) from the 55 training participants only;
- \(p_P(a\mid x)\) from the 1,000 profile-conditioned traces.

Numerically equivalent answers are merged and `Unparseable` is a category.
Let \(\mathcal A_x\) be the union of human and profile-trace answer categories.
With smoothing mass \(\alpha=0.05\):

`\tilde p(a|x) = 0.95 * p(a|x) + 0.05 / |\mathcal A_x|`

The raw and final per-trace weights are:

`r_i = min(5, \tilde p_H(a_i|x_i) / \tilde p_P(a_i|x_i))`

`w_i = r_i / mean_j(r_j)`

The same frozen 12,000-row, mean-one weighted dataset is shared by every beta
and training seed.

### KL-beta sweep

For each beta in `{0, 0.01, 0.1, 0.5}` and each seed 42--46, optimize on
plain prompts:

`weighted response CE + beta * KL(policy || frozen_distillation_reference)`

The trace weight applies to response-token CE only; KL is unweighted. Prompt
and padding tokens are masked. The frozen reference receives no gradients and
is evaluated on the same plain prompt and response tokens as the policy. For
beta `0`, do not load or run the reference model.

Each condition uses batch size 16, validation batch size 16, gradient
accumulation 1, bf16, no gradient checkpointing, five epochs, and checkpoint
selection by minimum NLL on the 215-response plain-prompt validation split.
There are 20 corrected weighted runs in total. All four beta values are
reported as an ablation; test results must not be used to select a preferred
beta.

## Frozen Evaluation Protocol

- Evaluate the selected checkpoint from every run.
- Training and evaluation prompts are plain and contain no profile parameters.
- Generate 120 rollouts per problem at temperature 1.0 and top-p 0.95.
- Use fixed evaluation seed `20260729`, which is distinct from trace-generation
  seed `20260728`.
- Re-evaluate the unchanged Direct and CPT human-FT checkpoints with evaluation
  seed `20260729` so every comparison uses matched sampling randomness. No
  retraining is required for those conditions.
- Human reference is the 19-participant test split only.
- Report per run and mean/standard error across training seeds:
  - overall accuracy;
  - per-cell accuracy;
  - accuracy-profile MAE;
  - categorical answer-level NLL.
- Categorical NLL follows the manuscript definition:
  - merge numerically equivalent answers;
  - add `Other` and `Unparseable`;
  - use symmetric Dirichlet smoothing with total prior mass one.
- Re-score compatible unchanged baselines, including distillation-only,
  against the same test-only human reference. Mark these as re-scored rather
  than newly trained.

## Output Layout

- Corrected profiles, traces, and weights:
  `data/human_ft/decimal_rerun/profile_weighted_v2/`
- Corrected weighted runs:
  `results/finetuning/rerun_decimals/profile_weighted_v2/kl_beta_<value>/seed_<seed>/`
- Corrected rollouts and metrics:
  `eval/outputs/rerun_decimals/profile_weighted_v2/`

Existing publication artifacts and the superseded unconditioned pilot must
not be overwritten.

## Required Validation

- Deterministic split and exact-count tests.
- Participant overlap and all-problem coverage tests.
- Weighted CE/KL unit tests:
  - unit weights and beta zero equal standard CE;
  - example weights scale completion loss correctly;
  - prompts and padding contribute no loss;
  - KL is finite and nonnegative;
  - reference parameters receive no gradients.
- Exact 1,000-row Cartesian profile-grid test with stable unique IDs.
- Exact 12 problems × 1,000 profiles trace-grid test.
- Trace prompts must contain the expected profile prefix; weighted-training
  prompts must contain no profile prefix.
- Profile parameters and generated answers must round-trip from trace artifact
  to the plain-prompt weighted bundle.
- `correct_answer`, `is_correct`, and human `acc` must be absent from the
  weighted-training input schema.
- Weight tests must reconstruct both smoothed per-problem distributions, the
  clip-at-five operation, and global mean-one normalization.
- Trace-generation and evaluation seeds must differ.
- Artifact validators must enforce the expected KL beta, batch protocol,
  profile-trace input hash, and all five completed epochs.
- Pinned adapter load smoke test.
- One-batch weighted training and checkpoint reload smoke test.
- Rollout parsing and categorical NLL tests.

## Corrected Execution Plan

No corrected jobs may be submitted until the new profile-grid, prompt-format,
weight-formula, and seed-separation tests pass locally.

The corrected DAG is:

1. Build and validate the deterministic 1,000-profile grid.
2. Generate and validate the 12,000 profile-conditioned source traces with
   vLLM.
3. Convert them to plain-prompt training examples and build the one shared
   importance-weight bundle from training-human responses.
4. Run one real-model save/reload smoke step for beta `0` and beta `0.5`.
5. Submit a 20-task training array:
   - tasks 0--4: beta `0`, seeds 42--46;
   - tasks 5--9: beta `0.01`, seeds 42--46;
   - tasks 10--14: beta `0.1`, seeds 42--46;
   - tasks 15--19: beta `0.5`, seeds 42--46.
6. Re-evaluate Direct and CPT human FT and evaluate all 20 corrected weighted
   checkpoints with plain prompts and seed `20260729`.
7. Validate the complete 30-run matrix, aggregate by condition/beta, and
   re-score unchanged compatible baselines against the frozen test humans.

Frozen `ailab` submission budgets, tightened using the completed pilot timing
logs:

- profile-conditioned vLLM trace generation: 20-minute `gpu-test`;
- profile-to-plain weight construction: 5-minute `gpu-test` utility job
  (the `ailab` partition rejects CPU-only allocations);
- beta-zero/beta-0.5 optimizer and reload smoke gate: 10-minute `gpu-test`;
- each corrected weighted training task: 20-minute `gpu-test`;
- evaluation: one 10-minute `gpu-test` job loading the base model once and
  swapping all 30 adapters;
- final validation and scoring: 5-minute `gpu-test` utility job for the same
  partition constraint.

The completed pilot indicates batch-16 training takes approximately 14
minutes per seed on an H200. The 20-minute training limit retains roughly a
five-minute margin while preserving the short-job QoS. This is only a
resource estimate; pilot metrics are not evidence for any beta condition.

## Run Log

| Stage | Job ID(s) | State | Notes |
|---|---|---|---|
| Corrected v2 implementation and local tests | — | complete | Added exact profile-grid/source validators, profile-to-plain bundle reconstruction, 30-run evaluator/scorer, source-data SHA validation, and a fresh namespace; 10 focused tests passed with 1 environment-dependent skip, plus Python and shell syntax checks |
| Real-model weighted/KL smoke | `11696606`, `11696661` | complete | One-step save/reload preflight completed in 1m58s; 20+20-update no-checkpointing throughput preflight completed in 2m41s with clean save/reload and finite NLL/KL; archived cluster telemetry reports 32.9% average GPU utilization and 19.5/140.4 GB peak GPU memory for `11696661` |
| Weighted/KL one-hour optimization benchmark | `11705558` | cancelled unstarted | Superseded when direct batch-16 production was authorized; no GPU time or artifacts consumed |
| Unconditioned pilot trace generation | `11691636` | complete; superseded | Generated 1,000 stochastic completions/problem without UMA profile prefixes; this is not the intended 1,000-profile trace cohort |
| Unconditioned pilot weight construction | `11696431` | complete; superseded | Internally valid for the wrong source cohort; must not initialize the corrected sweep |
| Direct human FT, seeds 42–46 | `11691723_[0-4]` | complete | All five 5-epoch runs completed and were independently revalidated; runtimes 7m41s--8m25s; best validation NLL range 0.18080--0.18447 |
| CPT human FT, seeds 42–46 | `11691723_[5-9]` | complete | All five 5-epoch runs completed and were jointly revalidated; runtimes 7m35s--7m50s; every best checkpoint was epoch 2, with validation NLL range 0.16645--0.17205 |
| Unconditioned pilot Weighted-KL beta 0.5, seeds 42–46 | `11705741_[10-14]` | complete; superseded | All five completed in 13m41s--14m53s, but trained on the wrong unconditioned source traces |
| Direct/CPT vLLM evaluation, seed 20260728 | `11691839` | complete; checkpoints remain valid | Ten rollout files validated, but the corrected comparison will regenerate them with matched evaluation seed `20260729` |
| Unconditioned pilot weighted evaluation | `11708553`, `11708554` | complete; superseded | Used the trace-generation seed on the same prompts; exact response overlap with source traces was 86.3%--90.6% for seeds 42--44 |
| Unconditioned pilot aggregation | `11691841` | complete; superseded | Completed successfully but aggregates the wrong weighted condition and must not be reported |
| Corrected profile-conditioned traces | `11709624` | pending | 12 problems × 1,000 profiles; 20-minute vLLM allocation |
| Corrected weighted bundle | `11709625` | dependency-pending | Reads only `prob` and `resp` from training humans; 5-minute utility allocation; `afterok:11709624` |
| Corrected optimizer smoke gate | `11709626` | dependency-pending | Beta 0, beta 0.5, and adapter reload; 10-minute allocation; `afterok:11709625` |
| Corrected beta sweep | `11709627_[0-19%3]` | dependency-pending | 4 betas × 5 seeds = 20 batch-16 tasks, each with a 20-minute allocation; `afterok:11709626` |
| Corrected fresh-seed evaluation | `11709628` | dependency-pending | One 10-minute plain-prompt vLLM job, rollout seed `20260729`, complete 30-run matrix; `afterok:11709627` |
| Corrected aggregation | `11709629` | dependency-pending | Five-minute held-out scoring job; `afterok:11709628` |

The scheduler audit immediately after submission confirmed partition `ailab`,
QoS `gpu-test`, one H200 request per task, the walltimes above, and the full
dependency chain. Expanding the array gives exactly 25 `gpu-test` records,
matching the per-user submit limit. The immutable submission manifest is
`results/finetuning/rerun_decimals/profile_weighted_v2/submission.tsv`.

To prioritize this rerun, pre-existing `ailab` jobs `11683010` and `11683011`
were placed on user hold and verified as `JobHeldUser` with priority zero.
Jobs in other partitions were not modified.

### Superseded Pilot Execution History

The remainder of this section records how the first pilot was executed and
debugged. It is retained for provenance and resource estimates only; it does
not override the corrected protocol above.

A final pre-run audit added the missing `enable_input_require_grads()` call
for checkpoint-initialized LoRA models under gradient checkpointing. This is
required for gradients to reach the trainable CPT adapter through the frozen
backbone. The focused test suite passed after the correction. The final
scorer was also smoke-tested against every unchanged baseline listed in the
scoring job; all five artifacts passed schema, 12-problem coverage, and
held-out scoring checks. Trace and evaluation validators additionally require
the fixed rollout seed and an exact output SHA-256 match. The weighting stage
validates trace/human input hashes, output hash, the 12-problem generation
grid, nonempty responses, finite positive weights, and mean-one
normalization; either stage regenerates its artifact automatically when its
validation gate fails.

For the three-hour weighted-KL budget, full training-set diagnostic sweeps
were disabled with `--no-eval_train_metrics`. The original custom-objective
path would otherwise add a complete policy-plus-reference pass over roughly
12,000 traces before training and after each of five epochs. These passes
serve logging only. All minibatch optimization is unchanged, and validation
NLL is still computed on the complete 215-row validation split after every
epoch for best-checkpoint selection.

The final scoring job now requires exactly the canonical 15-file rollout
matrix (three conditions by seeds 42--46), with every file passing the vLLM
artifact validator. Its manifest records the frozen held-out human-file hash
and hashes of all four output metric tables, so a partial or stale aggregation
cannot be accepted as completion.

The original training/scoring submissions (`11691547`, `11691548`, and
`11691549`) requested three days and were assigned the lower-priority
`gpu-medium` QOS. They were cancelled before starting and replaced by the
shorter jobs recorded above; the scientific configuration and dependencies
are otherwise unchanged. Those jobs were themselves cancelled before
starting when training and vLLM evaluation were separated: Direct/CPT
training now has a 59-minute budget, and weighted-KL training has a three-hour
budget. Evaluation is separated from training and uses two 59-minute vLLM
matrix jobs. Each matrix job loads the Qwen base once and swaps the relevant
LoRA adapters sequentially. It first requires training summaries proving that
all five epochs completed, and every rollout artifact is schema/count
validated before downstream scoring.

The original Transformers trace-generation job `11691545` was also cancelled
before starting and replaced by vLLM job `11691636`. The replacement serves
the same frozen LoRA adapter on the same cached Qwen3-4B base, uses the same
prompts and sampling parameters, and writes the same rollout schema. The
weight-building dependency was updated in place to point to `11691636`.

The first importance-weight job `11691546` reached the builder after its trace
dependency cleared but failed in one second: two training-human responses to
`0.41*0.31` were the raw string `1,271`, and the builder incorrectly rejected
unparseable human answers. The weighting protocol already defines an
Unparseable answer bucket, so the builder was corrected to place such human
responses in that bucket rather than guessing a numeric interpretation or
dropping the rows. A regression test and a full 12,000-row dry build passed.
Replacement job `11696431` then completed and validated the production bundle;
the pending weighted-training array dependency was updated in place from the
failed job to the replacement.

A real Qwen3-4B weighted/KL preflight (`11696606`) then exercised the exact
12,000-row production objective for one optimizer update, saved the adapter,
reloaded that saved checkpoint, and completed a second update. A 20+20-update
throughput run (`11696661`) showed that gradient checkpointing would put the
3,750-update production run marginally at the three-hour boundary, whereas
disabling this memory-only setting reduced amortized update time to about
1.5 seconds with peak H200 memory below 20 GB. Batch size 2, accumulation 8,
optimizer, schedule, loss, data order, seeds, learning rate, and epoch count
are unchanged. Because Slurm snapshots sbatch contents when a job is
submitted, never-started array `11691724` was cancelled and replaced by
`11696725`; weighted evaluation `11691840` was redirected to the replacement.

After the three-hour replacement received a next-day scheduler estimate, a
one-hour optimization path was authorized.
The completion loss now selects response-token positions before cross-entropy
and full-vocabulary policy/reference softmaxes instead of calculating those
quantities for prompt and padding positions that are discarded afterward.
A randomized regression test proves equivalence of the scalar loss, component
sums, and policy/reference gradients with the prior formula. On the production
traces, response tokens are 74.0% of padded batch-2 positions. The earlier
H200 smoke run peaked below 20 GB out of roughly 144 GB, so the user
authorized going directly to physical batch 16, gradient accumulation 1, and
validation batch 16 while retaining effective batch 16. Replacement array
`11705741_[10-14]` was submitted with a 59-minute `gpu-test` limit and
requeue enabled; its spooled script hash exactly matched the locally validated
script. Benchmark `11705558` and unstarted three-hour fallback `11696725`
were then cancelled with zero elapsed GPU time. The replacement jobs also
write elapsed time and PyTorch peak allocated CUDA memory to each training
summary. After seeds 42--44 completed, the unstarted five-seed weighted
evaluation `11691840` was replaced by early job `11708553` for those three
seeds and follow-up job `11708554` for seeds 45--46. Final scoring `11691841`
now requires both replacements, so early rollouts are available without
weakening the complete-result gate.

## Result Summary

The unconditioned beta-0.5 pilot is superseded and has no reportable result.
This section will be filled only after the corrected 30-run matrix completes:
5 Direct human-FT evaluations, 5 CPT human-FT evaluations, and 20
profile-source/plain-prompt weighted evaluations across the four KL betas.
