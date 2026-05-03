# Qwen3-4B Distill→Human-FT, SP2013 Human-Distribution Match

3-stage pipeline (pretrained → UMA distill LoRA → continue same adapter on the
288-row Siegler-Pyke human corpus), evaluated against the per-problem human
accuracy distribution from `data/siegler_fraction_human.csv`.

## Headline result

| | overall acc | MAE pp vs human | add | sub | mul | div |
|---|---|---|---|---|---|---|
| Human (target) | 0.516 | --- | --- | --- | --- | --- |
| 4B baseline (T=0.7, epoch 3) | 0.441 | 26.37 | 26.18 | 32.66 | 36.61 | 10.05 |
| **4B ep1 @ T=1.0** | **0.462** | **7.55** | **4.51** | 10.92 | **8.03** | 6.73 |

Two cheap interventions stack:

1. **Stop after 1 epoch of LoRA on the 288-row corpus** instead of training to
   the val-NLL minimum at epoch 3. SFT for 3+ epochs collapses the model onto
   single-mode responses; 1 epoch leaves the response distribution intact.
2. **Decode at T=1.0 instead of T=0.7.** With T=0.7 the policy is sharpened
   enough to suppress the diversity it actually has in its logits.

Either alone roughly halves MAE; together they cut it 3.5x (26.37 -> 7.55 pp)
and bring the model into striking distance of the prior SmolLM2-135M SFT+KL
baseline (~5.27 pp) on a much stronger backbone.

## Files

- `sp2013_per_problem_q3_4b_full_ep1_t10.csv` --- per-problem accuracy at the
  ep1 @ T=1.0 setting, with human accuracy and absolute error in pp.
- `sp2013_summary_q3_4b_full_ep1_t10.json` --- per-op MAE summary.
- `sp2013_per_problem_q3_4b_full_baseline_ep3_t07.csv` /
  `sp2013_summary_q3_4b_full_baseline_ep3_t07.json` --- same metrics at the
  pre-fix baseline (T=0.7, epoch 3) for contrast.
- `all_conditions_comparison.csv` --- 13-condition diagnostic sweep (4B + 8B,
  baseline / distill / full / temperature variants / epoch variants).
- `training_args.json` --- exact CLI configuration used for the human-FT stage.
- `training_history.json` --- per-epoch train/val NLL.
- `training_summary_metrics.json` --- final metrics from the trainer.

## Reproducing

Stage 1+2 (distill on `uma_fraction_distillation_25`, LoRA r=16/alpha=32 on
`q,k,v,o,gate,up,down_proj`, 1 epoch) --- see
`results/transformer_replication/QWEN_HUMANFT_LORA_RUNBOOK.md` (three-stage
variant section).

Stage 3 --- continue the distilled LoRA on the 288-row human corpus for **1
epoch** (not the runbook default of 3-5):

```bash
EPOCHS=1 LR=1e-4 KL_BETA=0.0 \
  PARTITION=ailab \
  PYTHON_BIN="$(command -v python)" \
  HF_HOME="$PWD/.cache/huggingface" \
  DISTILL_MANIFEST=results/transformer_replication/jobs_qwen3_distill_1epoch_<ts>.tsv \
  SIZES="4b" \
  bash results/transformer_replication/submit_qwen_humanft_lora_from_distill.sh
```

Eval --- generate 256 rollouts per SP2013 problem **at T=1.0** (not 0.7), score
final answers against ground truth, MAE the per-problem accuracy vs the
empirical Siegler distribution. Driver lives in the parent fractions repo at
`scripts/eval_qwen3_chained_sp2013.py` with `--temperature 1.0`.

## Caveat

This run trained with `KL_BETA=0.0` (pure SFT). The prior SmolLM2-135M result
that achieved 5.27 pp used SFT + KL anchor to the pre-FT distribution. Adding
the KL anchor to this 4B setup is the next obvious step toward closing the
remaining ~2 pp gap.

Grading uses Python `fractions.Fraction` equality (so 5/10 = 1/2 is correct),
but does NOT yet handle mixed-number outputs like "1 2/5" --- if the model
produces those, this eval would under-count. See the recent main_max regrade
commits for the equivalence-aware parser.
