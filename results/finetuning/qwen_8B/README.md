# Qwen3-8B Distill→Human-FT, SP2013 Human-Distribution Match

3-stage pipeline (pretrained → UMA distill LoRA → continue same adapter on the
288-row Siegler-Pyke human corpus), evaluated against the per-problem human
accuracy distribution from `data/siegler_fraction_human.csv`.

Sister run to the Qwen3-4B sweep in `results/finetuning/qwen_4B/`. Same
recipe (1 epoch human-FT, decode at T=1.0); only the base model size differs.

## Headline result

| | overall acc | MAE pp vs human | add | sub | mul | div |
|---|---|---|---|---|---|---|
| Human (target) | 0.516 | --- | --- | --- | --- | --- |
| 8B baseline (T=0.7, epoch 3) | 0.471 | 23.69 | 26.57 | 23.87 | 33.97 | 10.36 |
| **8B ep1 @ T=1.0** | **0.484** | **8.37** | 8.52 | 11.51 | **6.04** | 7.42 |

Same two cheap interventions as the 4B sister run stack: stop after 1 epoch of
LoRA on the 288-row corpus, and decode at T=1.0 instead of T=0.7. Together
they cut MAE from 23.69 -> 8.37 pp.

## 4B vs 8B: capacity-prior tradeoff

The 4B run reaches 7.55 pp MAE with the same recipe; the 8B run reaches 8.37
pp. The ~0.8 pp gap is concentrated on UD addition / subtraction problems
where the 8B's stronger prior over-corrects --- it executes the
common-denominator algorithm too reliably to mimic human noise.

| problem | human | 4B (ep1@T=1.0) | 8B (ep1@T=1.0) |
|---|---|---|---|
| 2/3+3/5 (UD add) | 0.51 | 0.49 (matches) | 0.64 (over by 13 pp) |
| 3/5-1/4 (UD sub) | 0.55 | 0.56 (matches) | 0.66 (over by 11 pp) |
| 3/5+1/5 (ED add) | 0.79 | 0.80 (matches) | 0.71 (under by 8 pp) |

8B is also fractionally better than 4B on multiplication MAE (6.04 vs 8.03)
because mul has fewer "human-conceptual-failure" modes for the prior to
preserve; the gap concentrates in operations where humans struggle most.

## Things tried that did NOT help on 8B

- **Higher temperature (T=1.1)**: MAE went 8.37 -> 11.21 pp. T pushes wrongness
  uniformly across problems, but 8B's residual error is asymmetric (over on UD,
  under on hard ED). T can't selectively de-confidence the right problems.
- **Two epochs instead of one (ep2)**: MAE went 8.37 -> 14.46 pp. Easy
  problems erode faster than hard ones with extra SFT.
- **KL anchor against distill (β=1, β=10)**: MAE went 8.37 -> 12.33 -> 17.24
  pp. KL pulls the model *toward* the distill behaviour --- exactly the
  over-correct execution we needed to undo. The SmolLM2 SFT+KL recipe assumes
  the KL ref has human-like noise; ours doesn't.

See `all_conditions_comparison.csv` for the full 16-condition diagnostic.

## Files

- `sp2013_per_problem_q3_8b_full_ep1_t10.csv` --- per-problem accuracy at the
  ep1 @ T=1.0 setting, with human accuracy and absolute error in pp.
- `sp2013_summary_q3_8b_full_ep1_t10.json` --- per-op MAE summary.
- `sp2013_per_problem_q3_8b_full_baseline_ep3_t07.csv` /
  `sp2013_summary_q3_8b_full_baseline_ep3_t07.json` --- same metrics at the
  pre-fix baseline (T=0.7, epoch 3) for contrast.
- `all_conditions_comparison.csv` --- 16-condition diagnostic sweep
  (4B + 8B, baseline / distill / full / temperature / epoch / KL variants).
- `training_args.json` --- exact CLI configuration used for the human-FT stage.
- `training_history.json` --- per-epoch train/val NLL.
- `training_summary_metrics.json` --- final metrics from the trainer.

## Reproducing

Stage 1+2 (distill on `uma_fraction_distillation_25`, LoRA r=16/alpha=32 on
`q,k,v,o,gate,up,down_proj`, 1 epoch) --- see
`results/transformer_replication/QWEN_HUMANFT_LORA_RUNBOOK.md` (three-stage
variant section).

Stage 3 --- continue the distilled LoRA on the 288-row human corpus for **1
epoch**:

```bash
EPOCHS=1 LR=1e-4 KL_BETA=0.0 \
  PARTITION=ailab \
  PYTHON_BIN="$(command -v python)" \
  HF_HOME="$PWD/.cache/huggingface" \
  DISTILL_MANIFEST=results/transformer_replication/jobs_qwen3_distill_1epoch_<ts>.tsv \
  SIZES="8b" \
  bash results/transformer_replication/submit_qwen_humanft_lora_from_distill.sh
```

Eval --- 256 rollouts per SP2013 problem **at T=1.0**, score final answers,
MAE the per-problem accuracy vs the empirical Siegler distribution. Driver in
the parent fractions repo at `scripts/eval_qwen3_chained_sp2013.py` with
`--temperature 1.0`.

## Caveat

Grading uses Python `fractions.Fraction` equality (5/10 = 1/2 is correct), but
does NOT yet handle mixed-number outputs like "1 2/5". See the recent main_max
regrade commits for the equivalence-aware parser.

8B specifically may also benefit from the equivalence-aware regrade if it
emits more mixed-number outputs than 4B (untested here).
