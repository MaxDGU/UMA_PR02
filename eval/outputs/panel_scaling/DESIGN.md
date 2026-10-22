# Panel-scaling study: full 996-learner distribution vs the 25/100-learner panels (fractions)

Goal: a direct, controlled comparison of distilling on the full UMA learner grid vs the paper's 25-learner panel,
using the exact paper recipe at every stage so the only thing that varies is the learner set (and, in one arm,
corpus size). Launched 2026-09-17; job ids in `chain_jobids.txt`.

## Held fixed across every arm
| Stage | Setting |
|---|---|
| Learners | saved grade-6 UMA models (`fractionGPT_repo/uma_trained_students`, 996 of 1000; 92/93/97/98 timed out in training) |
| Trace generation | `results/UMA_replication/run_saved_models_on_problem_set.py`, 20,720 published problems, writes `trace_steps_json` |
| Translation | one run of `translate_uma_traces_to_nlp.py` over all 996 learners: `trace_or_child`, `--row-filter none`, no student prefix |
| Distillation | Qwen3-4B-Base, LoRA r16, 1 epoch, lr 5e-4 cosine, max_len 288, effective batch 128, 90/5/5 row split, best by val loss |
| Human-FT | paper protocol: 288 Siegler-2011 train / 96 val rows, 1 epoch (18 updates), lr 1e-4, full FT, best by val NLL |
| Evaluation | SP2013 16 problems, 256 samples each, T=1.0 top_p=0.95, fixed mixed-number parser; 8-cell MAE vs children, accuracy, val NLL |
| Seeds | distill seed fixed; human-FT seeds 42, 43, 44, 45 for every arm (mean and SD reported) |

Everything an arm shares with the paper's 25-learner panel is verified: the 25 are a strict subset of the 996
with identical parameters, and rows for the 25 regenerated inside the 996 corpus are checked against the
standalone regeneration (arm A check file `arms/A_paper25_from996.csv.gz`).

## Arms
| Arm | Learners | Rows | Isolates |
|---|---|---|---|
| A  paper25 | the paper's 25 (even g, rt 5, ice 50) | 518K (all) | control; reproduces the paper (done: MAE 5.5-10.3 across seeds, NLL 1.36) |
| B  all996_sub | all 996 | 520 rows/learner = 518K | panel composition at the paper's corpus size |
| C  all996_full | all 996 | ~20.6M (all) | composition + corpus size (the "full distribution" arm) |
| D  rand25 (seeds 42, 43) | 25 drawn uniformly from the 996 | ~518K (all) | composition at equal learner COUNT: is the paper's mid-slice special, or is any 25 as good? |
| (done) balanced98 sub / full | 98 balanced learners | 466K / 1.83M | intermediate width; already run |

## Reading the outcome
* B vs A answers the user's question directly: does training on the full distribution (at fixed data size)
  degrade MAE/NLL relative to the mid-ability slice? Expected from the 98-learner result: yes, moderately.
* C vs B isolates corpus size under the fixed human-FT budget. If C is much worse than B, "full distribution
  collapses" is a human-FT-budget effect, not a composition effect, and must be reported as such.
* D vs A: if random 25-learner panels match the paper's 25, composition does not matter and B/C effects are size.
  If they are worse, the paper's panel choice is load-bearing and must be stated in the paper.
* Decision rule for the paper: report A, B, C, D means with seed SDs in one table; the reported result stands
  as "25-learner mid-ability panel"; the scaling table becomes a new finding either way.

## Cost / schedule
| Stage | Resource | Estimate |
|---|---|---|
| Regenerate 873 remaining learners | cpu partition, array %150, 1 core each | ~1,500 CPU-h, ~12-16 h wall |
| Translate 996 x 20,720 | 1 CPU job | ~7 h |
| Build arms | 1 CPU job | <1 h |
| Distill B, D42, D43 | 2 H100 each, 14 h limit | ~2-3 h each |
| Distill C | 8 H100, 2d20h limit | ~24 h |
| Human-FT x 4 seeds + rollouts, per arm | 1 H100 | ~2.5 h each |
| Score | CPU | minutes |
Disk: raw traces ~265 GB (970 GB free after the meta_geoclidean cleanup).

## Out of scope for this launch (needs a decision)
* Decimals: the 22M raw decimal pool exists but the decimal translator ("v12 middle-school think-aloud") is not
  in any repo copy on della; the 100-learner decimal corpus cannot be regenerated or widened until it is found.
* Human-FT budget sweep (epochs 1/3/5 on arms A, B, C) to test whether the size penalty is recoverable. Cheap
  (~35 min per run); recommended as the immediate follow-up once C finishes.

## Files
`panel_scaling/`: `subjids_*.txt`, `raw/` (996 x 20,720 traces), `uma_frac_all996_v10regen_nlp.csv.gz`,
`arms/`, `results/` (rollout CSVs, `summary.csv`), `hft_runs/`, `logs/`, `score_all.py`, `submit_chain.sh`.

## Run log
* 2026-09-18: 865/873 regeneration tasks completed; 8 learners (subjids 96, 195, 196, 290, 390, 495, 780, 905; mostly d=0.9,
  high rt, ice 0-25) exceeded the 6 h limit, which left the translate job in DependencyNeverSatisfied. Re-run as array
  14104584 with a 24 h limit; translate (14060790) re-pointed to it with `scontrol update Dependency=afterok:14104584`.
  The rest of the chain is unchanged.
* Population note: 29 of the 996 saved learners carry c=4.0 instead of the grid's c=5 (subjids 47-50, 77-91, 94-96, 99-100,
  196-200; all at g <= 0.02). None are in the paper's 25; three (87, 94, 199) are in the balanced 98. This is a property of
  the saved-model population, inherited by every corpus built from it. Small (3% of learners) but worth a footnote.

## Results (2026-09-23; arm C still queued, projected start 09-23 16:33)
See results/summary.csv. Size-matched arms (all ~518K rows, paper recipe, 4 human-FT seeds):
| arm | distill MAE / acc | D+H MAE (sd) / acc | val NLL |
|---|---|---|---|
| A paper25 regen | 10.3 / 60.0 | 7.6 (1.8) / 47.4 | 1.371 |
| B all996 size-matched | 9.5 / 53.9 | 18.7 (2.6) / 32.9 | 1.364 |
| D rand25 seed42 | 8.3 / 51.7 | 17.1 (1.1) / 34.5 | 1.386 |
| D rand25 seed43 | 12.7 / 58.6 | 8.9 (2.6) / 43.4 | 1.359 |
| balanced98 sub | 9.4 / 56.1 | 11.7 (1.7) / 40.1 | 1.395 |
Reading: at fixed corpus size, val NLL is flat (1.36-1.40) across every panel, so the human-FT stage matches child
TEXT equally well regardless of who was distilled; what moves is the accuracy profile (MAE), which tracks the
ability mix of the distilled panel. Panel composition matters a lot at equal learner count (two random 25-panels
differ by 8 MAE points); the paper's slice is at the good end of that distribution but not unique (seed 43 is close).

## Why arm B loses during human-FT (tests run 2026-10-01; kl_test/)
* Test 1 (kl_test/test1_strategy_consistency.txt): training data for B is only modestly less consistent on unequal-denominator
  +/- (common-denominator strategy 67% vs 74%). After distillation B's errors are a coherent child bug (add tops and bottoms
  separately, 30% vs 12%). After human-FT, 48% of B's answers match no recognizable procedure (A: 25%).
* Test 2 (kl_test/test2_results.txt, kl_per_run.csv): human-FT with KL(policy||own distill ckpt), seeds 42-43.
  | arm | kl_beta | MAE | UD +/- acc | words | val NLL |
  | A | 0 | 6.7 | 59 | 20 | 1.37 |   | A | 1.0 | 7.2 | 69 | 46 | 1.55 |
  | B | 0 | 16.9 | 24 | 20 | 1.36 |  | B | 1.0 | 5.6 | 51 | 43 | 1.59 |
  beta=1.0 removes the A-B gap (B 5.7 and 5.6 on both seeds; children UD +/- 54.8%). beta=0.1 barely helps.
  Cost: NLL on child text rises 1.36 -> 1.59 and responses stay UMA-length (~45 words vs children ~30).
* Conclusion: the full-distribution penalty is created by human-FT erasing step-by-step reasoning that B's common-denominator
  procedure depends on; anchoring human-FT to the distilled model recovers it. MAE/NLL trade-off is the open design question.

## Arm B1000 (2026-10-14; b1000/)
The 4 learners missing from the saved population (92, 93, 97, 98; g=.01, d=.9) were re-simulated through the full
grade 1-6 curriculum with the original fork code (c=4, matching their saved neighbors), replayed over the 20,720
problems (97, 98, 92 via 8-way problem chunks), translated with the paper recipe, and added to arm B's exact rows
(+520 each -> 520,000 rows). Same recipe, 2 GPUs, 5 val evals (best = final, val 0.0259).
| | B996 | B1000 |
|---|---|---|
| distill MAE / acc / child NLL | 9.5 / 53.9 / 3.28 | 7.7 / 53.2 / 3.40 |
| D+H MAE (4 seeds) | 18.7 +- 3.0 | 20.0 +- 2.0 (paired p = 0.12) |
| D+H acc | 32.9 +- 3.0 | 31.6 +- 2.0 |
| D+H val NLL | 1.364 +- 0.013 | 1.378 +- 0.012 |
Same per-cell signature (unequal-denominator add 17.0 vs 18.4). Adding the 4 learners does not change the result.

## CORRECTION (2026-10-17): arm C was not corrupted
diag_C2/diag3: arm C's saved checkpoint gives loss 0.0214 on training-format prompts (matches the trainer's 0.0218) and
coherent generations, but loss 0.955 and garbage on plain prompts. Cause: distillation used
TRAIN_PROMPT_STUDENT_MODE=always (every prompt prefixed with "<student> g .. d .. rt .. ice .. </student>"), while SP2013
evaluation and human-FT use plain prompts. After 145k updates the model became fully dependent on the tag. B1000 (3.7k
updates): 0.024 tagged vs 0.039 plain. Arm C's 0.2% accuracy and D+H MAE 50 are artifacts of this train/eval prompt
mismatch, not a checkpoint bug. The mismatch applies to every arm and grows with training length (likely contributes
to the balanced98-full NLL increase). Fix for the full-data rerun: TRAIN_PROMPT_STUDENT_MODE=dropout (0.5).

## Arm C, full corpus (2026-10-18; c1000/)
1000 learners x 20,720 problems = 20.72M rows (18.65M train, 145,688 updates, 8 GPUs, 21.9 h), prompt tag on 50% of rows
(TRAIN_PROMPT_STUDENT_MODE=dropout 0.5), 5 evals (val 0.0408 -> 0.0331 -> 0.0312 -> 0.0302 -> 0.0298, best = final).
Loss spike at updates ~7,200-8,800 (window loss 0.034 -> 0.73) at peak lr 5e-4; recovered by ~u30k.
EVALUATION FORMAT: eval_seed_ckpt.py writes problems unspaced ("3/5+1/5"); training uses spaced ("3/5 + 1/5"). Arm C is
format-sensitive (distill 14.9% / MAE 42.5 unspaced vs 54.1% / 7.9 spaced); short arms are not (A control 7.0 vs 8.0;
B1000 seed 42 18.7 vs 19.9). All arm C / control numbers below use spaced prompts (c1000/eval_spaced.py), 256 samples.
| | A control (25, tag 50%) | C (1000, all problems, tag 50%) |
|---|---|---|
| distill MAE / acc / child NLL | 10.4 / 57.2% / 3.14 | 7.2 / 53.3% / 12.13 |
| human-FT MAE (4 seeds) / NLL | 11.4 +- 2.3 / 1.367 | 13.0 +- 3.3 / 5.76 (p = 0.45) |
| KL human-FT MAE (2 seeds) / NLL | 5.8 +- 0.2 / 1.544 | 7.2 +- 0.5 / 6.55 (p = 0.12) |
Reading: on error profile (MAE) the full-data arm matches the 25-learner panel at every stage (best distill MAE of any arm),
but its text fit to children is far worse (NLL 5.8 vs 1.4 after human-FT): 40x distillation makes the model very confident
in UMA-style text and 18 human-FT updates cannot pull it back.

## PRIMARY METRIC: KL-anchored human-FT, 4 seeds each, spaced evaluation (2026-10-19; kl_primary/)
| arm | KL human-FT MAE | acc | child NLL |
|---|---|---|---|
| A: 25 learners, tag 100% | 8.1 +- 0.9 | 55.2% | 1.555 |
| A control: 25 learners, tag 50% | 6.1 +- 0.5 | 53.5% | 1.548 |
| B1000: 1000 x 520, tag 100% | 7.4 +- 1.1 | 46.6% | 1.630 |
| C: 1000 x 20,720, tag 50% | 7.3 +- 0.3 | 51.2% | 6.556 |
Welch: tag 50 vs 100 on A p=0.012; A vs B1000 (tag 100) p=0.34; control vs C (tag 50) p=0.011; B1000 vs C p=0.91.
Reading: under KL human-FT the "1000 learners doubles MAE" result disappears (B1000 7.4 vs A 8.1). Full data (C) gives no
MAE gain over B1000 and costs ~4x NLL on children's text. Tag 50% beats tag 100% for the 25-learner panel.

## Longer KL human-FT (10 epochs = 180 updates, kl 1.0, lr 1e-4; 2 seeds; spaced eval; longhft/) 2026-10-19
| | 25 learners MAE / NLL | 1,000 learners (full data) MAE / NLL |
|---|---|---|
| 1 epoch (4 seeds, earlier) | 6.1 / 1.55 | 7.3 / 6.56 |
| epoch 3 of 10 | 4.3 / 1.44 | 8.5 / 4.71 |
| epoch 10 of 10 | 5.7 / 1.56 | 6.2 / 4.26 |
| best by held-out child NLL | 4.9 (epochs 3-4) / 1.44 | 6.9 (epoch 9) / 4.26 |
Longer FT helps both arms; arm C's NLL falls 6.56 -> 4.26 but plateaus ~3x the 25-learner arm. 25 learners still better on both.

## Even longer KL human-FT for arm C (30 epochs = 540 updates, 2 seeds; longhft30/) 2026-10-21
| checkpoint | MAE | acc | child NLL |
|---|---|---|---|
| epoch 10 | 6.8 +- 1.8 | 44.8% | 3.95 |
| epoch 15 | 5.1 +- 0.0 | 47.0% | 3.90 |
| epoch 20 | 4.8 +- 0.0 | 47.1% | 3.86 |
| epoch 30 | 4.8 +- 0.2 | 47.0% | 3.84 |
| best by child NLL (epochs 18-19) | 5.4 | 46.4% | 3.82 |
MAE reaches the 25-learner arm's best (~4.3-4.9) by epoch 20 and holds; NLL plateaus ~3.84 (2.6x the 25-learner 1.44).
The KL anchor (beta 1.0) is now the binding constraint on style, not training length.
