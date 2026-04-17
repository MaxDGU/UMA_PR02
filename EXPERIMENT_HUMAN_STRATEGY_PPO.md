# Human Strategy-Matching PPO

## Question
We study whether a distilled fraction-solver can be trained to match the human strategy distribution on the same problem prompts. The target is not a single human label per rollout. The target is a problem-conditional distribution over five coarse strategy families:
\[
\mathcal{C} = \{\mathrm{KDON}, \mathrm{CDON}, \mathrm{ONOD}, \mathrm{ICDM}, \mathrm{CROP}\}.
\]

## Setup
We freeze the existing 160k strategy classifier at [best](/n/fs/cogai/cs1095/UMA_PR02/results/transformer_replication/strategy_classifier_distilbert_n160000_e3_final/best) and use it to convert human responses into posterior targets. For each human response \(x_{p,i}\) on problem \(p\), the classifier returns \(s_\phi(c \mid p, x_{p,i})\). We define the human target by posterior averaging:
\[
h_p(c) = \frac{1}{n_p}\sum_{i=1}^{n_p} s_\phi(c \mid p, x_{p,i}).
\]
On April 10, 2026, the combined human train+val set contains exactly 8 unique problems with \(n_p=48\) responses per problem, and [sp2013_human.csv](/n/fs/cogai/cs1095/UMA_PR02/sp2013_human.csv) contains 16 unique problems with 0 overlap with those 8.

## Method
We start from the distilled checkpoints [135m_final](/n/fs/cogai/cs1095/UMA_PR02/distilled/135m_final), [360m_final](/n/fs/cogai/cs1095/UMA_PR02/distilled/360m_final), and [1p7b_final](/n/fs/cogai/cs1095/UMA_PR02/distilled/1p7b_final). We train a LoRA-adapted policy with PPO, a learned value head, and a frozen reference-model KL penalty. The user’s “350m” target is mapped to the existing 360M checkpoint.

For each PPO update, we sample 4 unique human-train problems and generate \(k=32\) sampled rollouts per problem. The frozen classifier scores each rollout, and we average those rollout posteriors to obtain the model-side strategy distribution:
\[
q_p(c) = \frac{1}{k}\sum_{j=1}^{k} s_\phi(c \mid p, y_{p,j}).
\]

We compare \(q_p\) against \(h_p\) with one of three losses:
\[
D_{\mathrm{TV}}(h_p, q_p)=\frac12\sum_c |h_p(c)-q_p(c)|,
\]
\[
D_{\mathrm{KL}}(h_p \parallel q_p),
\]
with \(\varepsilon=10^{-6}\) smoothing on both distributions, and
\[
D_{\mathrm{W}}(h_p, q_p),
\]
where the Wasserstein cost is 0 on the diagonal, 1 for \(\mathrm{KDON}\leftrightarrow\mathrm{CDON}\) and \(\mathrm{ONOD}\leftrightarrow\mathrm{CROP}\), and 2 for all other off-diagonal pairs.

We assign one group score per problem:
\[
r_p = -D(h_p, q_p),
\]
subtract the prompt-batch mean as a baseline, attach that centered scalar to all rollouts from the same problem, and combine it with the tokenwise KL penalty to the frozen reference model. The main runs are reward-only PPO. We do not add a teacher-forced NLL term.

## Evaluation
We report two settings.

Setting. ID evaluation uses the same 8 human-train problems. For each problem, we generate 128 sampled rollouts, estimate the rollout distribution \(q_p\), and report the mean TV, KL, and Wasserstein distances to \(h_p\). We also report final-answer accuracy from generated answers.

Setting. OOD evaluation uses the 16 unique problems in [sp2013_human.csv](/n/fs/cogai/cs1095/UMA_PR02/sp2013_human.csv). We report final-answer accuracy overall and by operation. We do not report OOD strategy divergence in the main experiment because `sp2013_human.csv` uses a different native strategy label space.

## Artifacts
The human target artifact is [human_strategy_targets_classifier160k.json](/n/fs/cogai/cs1095/UMA_PR02/results/human/human_strategy_targets_classifier160k.json). Each PPO run writes:

- `train_args.json`
- `history.csv` and `history.json`
- `best_id/`
- `best_id_eval_summary.json`
- `final/`
- `final_eval/eval_summary.json`
- `final_eval/id_per_problem.csv`
- `final_eval/id_rollouts.csv`
- `final_eval/ood_per_problem.csv`
- `final_eval/ood_rollouts.csv`

The run naming convention is:
\[
\texttt{human\_strategy\_ppo\_<model>\_<loss>\_<timestamp>}.
\]

## Launches
Command. Build the frozen human targets:

```bash
python results/transformer_replication/build_human_strategy_targets.py \
  --classifier_ckpt results/transformer_replication/strategy_classifier_distilbert_n160000_e3_final/best \
  --train_csv results/human/data_train_nlp.csv \
  --val_csv results/human/data_val_nlp.csv \
  --sp2013_human_csv sp2013_human.csv \
  --output_json results/human/human_strategy_targets_classifier160k.json \
  --local_files_only
```

Command. Run one PPO experiment:

```bash
python results/transformer_replication/train_human_strategy_ppo.py \
  --init_model_dir distilled/135m_final \
  --classifier_ckpt results/transformer_replication/strategy_classifier_distilbert_n160000_e3_final/best \
  --human_target_json results/human/human_strategy_targets_classifier160k.json \
  --sp2013_human_csv sp2013_human.csv \
  --loss tv \
  --rollouts_per_prompt 32 \
  --prompt_batch_size 4 \
  --max_new_tokens 128 \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 50 \
  --total_updates 100 \
  --local_files_only
```

Command. Evaluate a saved checkpoint:

```bash
python results/transformer_replication/eval_human_strategy_ppo.py \
  --model_path results/transformer_replication/human_strategy_ppo_<run>/final \
  --classifier_ckpt results/transformer_replication/strategy_classifier_distilbert_n160000_e3_final/best \
  --human_target_json results/human/human_strategy_targets_classifier160k.json \
  --sp2013_human_csv sp2013_human.csv \
  --local_files_only
```

## Grid
Protocol. The main comparison is a 3 x 3 grid:

- `distilled/135m_final` x `{tv, kl, wasserstein}`
- `distilled/360m_final` x `{tv, kl, wasserstein}`
- `distilled/1p7b_final` x `{tv, kl, wasserstein}`

We keep the rollout budget, seed, and fixed-length training protocol aligned across the 9 runs.

## Checks
Finding. The implementation includes unit checks for zero-divergence identities and for the fixed Wasserstein cost matrix. A minimal end-to-end smoke run completed successfully at [human_strategy_ppo_smoke_135m_min](/n/fs/cogai/cs1095/UMA_PR02/results/transformer_replication/human_strategy_ppo_smoke_135m_min), and the standalone eval CLI completed successfully at [human_strategy_eval_smoke_cli](/n/fs/cogai/cs1095/UMA_PR02/results/transformer_replication/human_strategy_eval_smoke_cli).
