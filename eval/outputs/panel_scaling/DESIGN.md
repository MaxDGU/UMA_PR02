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
