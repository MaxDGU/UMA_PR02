# Audit: arm B1000 vs arm B996 vs arm A (2026-10-15)

Consistent (verified from saved configs, corpora, rollouts, and the HF uploads):
* Distillation config identical across A / B996 / B1000 except evals_per_epoch and loss_evals_per_epoch (1 / 1 / 5).
  Same model, LoRA r16/a32, lr 5e-4 cosine, 1 epoch, seed 42, eff. batch 128 on 2 GPUs, max_len 288, 90/5/5 row split.
* Human-FT identical (288 train / 96 val Siegler-2011 rows, 1 epoch = 18 updates, lr 1e-4, full FT, best by val NLL,
  seeds 42-45). Seeds whose checkpoints were deleted (A, B996 seeds 43-45) only have summary files; their summaries match
  and they were launched by scripts with identical arguments.
* Corpora: same translator version, trace-step narration on 100% of rows, 0 arithmetic-mismatch rows, 0 SP2013 leakage,
  no student prefix. B996 and B1000 have exactly 520 rows per learner across all 20,720 problems. B1000 contains all
  517,920 B996 rows unchanged plus 2,080 rows from learners 92, 93, 97, 98 with the correct grid parameters.
* Evaluation: every rollout file has the same 16 SP2013 problems x 256 samples, same script and sampling (T=1.0,
  top_p=0.95), same fixed parser and human reference.
* HF uploads: B1000 distill train_args point to B1000_sub520k with 5 loss evals; history shows val loss falling at every
  check (best = final). The 4 human-FT seeds have seeds 42-45 and point to the B1000 distill checkpoint.

Flagged inconsistencies:
1. c parameter: the 4 regenerated learners use c=4 (matching their saved neighbors), but the paper states c=5 for the whole
   grid. Pre-existing: 29 saved learners already use c=4 (all in B996/B1000, none in arm A). Disclose in the methods.
2. Validation sets differ between B996 and B1000 (row-level random split over different row sets: 25,896 vs 26,000 rows),
   so their synthetic val losses (0.0244 vs 0.0259) are not on identical rows.
3. Evaluation frequency differs (1 vs 5 evals). Two identical arm B runs differing only in eval count reached 0.0244 vs
   0.0253 val loss, so training is not bit-reproducible; B1000 (0.0259) sits within ~0.0015 of that band.
4. Single-checkpoint distill MAE is noisy: B996 9.5 vs its own re-run ~10.0 (64 samples) vs B1000 7.7. Treat the distill
   MAE difference as run-to-run variation, not an effect of the 4 learners.
5. The 4 new learners are fresh stochastic simulations of those profiles (none existed before), and 3 of 4 were replayed in
   8 problem chunks: same distribution, different random stream than a single pass.
6. HF README row for the paper's fractions/distill_lora still says "996-learner UMA panel"; it was trained on 25 learners.
7. HF README points to GitHub eval/outputs/panel_scaling for results, but B1000 and validation-curve results are not on
   GitHub yet.
8. All adapter configs on HF (old and new) record base_model_name_or_path as a della path, not Qwen/Qwen3-4B-Base.

Addendum (2026-10-17): every distillation arm trains with the learner tag on 100% of prompts, while all evaluation and
human-FT prompts are plain. This is consistent across arms but becomes severe with long training (arm C).
