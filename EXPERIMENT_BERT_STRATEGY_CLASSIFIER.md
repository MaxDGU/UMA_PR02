# BERT Strategy Classifier

## Question

Can we train a discriminative text classifier that recovers the **coarse strategy family** used in UMA-translated fraction solutions, and then use that classifier to estimate strategy distributions in distilled-model rollouts?

The operational target is:

\[
y \in \{\mathrm{KDON}, \mathrm{CDON}, \mathrm{ONOD}, \mathrm{ICDM}, \mathrm{CROP}\}.
\]

We collapse the original UMA labels as follows:

- `KDON_AS`, `KDON_OG` \(\rightarrow\) `KDON`
- `CDON_AS`, `CDON_OG` \(\rightarrow\) `CDON`
- `ONOD_M`, `ONOD_OG` \(\rightarrow\) `ONOD`
- `ICDM_D`, `ICDM_OG` \(\rightarrow\) `ICDM`
- `CROP_M` \(\rightarrow\) `CROP`

`OTHER` is dropped in the main experiment.

## Formal Setup

Let each translated training example be a pair \((x_i, y_i)\), where \(x_i\) is text derived from a UMA trace and \(y_i\) is the collapsed strategy family.

The main input variants are:

\[
x_i^{(\mathrm{resp})} = \texttt{response\_nl}_i
\]

and

\[
x_i^{(\mathrm{prob+resp})} = \texttt{"Problem: "}\; p_i \; \texttt{"=?\\nResponse:\\n"} \; \texttt{response\_nl}_i.
\]

The classifier is a pretrained encoder \(f_{\theta}\) with a softmax head:

\[
p_{\theta}(y \mid x) = \mathrm{softmax}(W h_{\theta}(x) + b).
\]

We train by minimizing weighted cross-entropy:

\[
\mathcal{L}(\theta) = - \sum_{i=1}^{N} w_{y_i} \log p_{\theta}(y_i \mid x_i),
\]

where \(w_c\) is inverse-frequency reweighting computed from the training split.

## Why This Experiment

The earlier rule-matching check suggested that literal pattern matching is too sparse on translated rationales and too brittle on actual distilled generations. A classifier trained on labeled translated data is the simplest next test.

The key uncertainty is not only separability, but also **generalization**. If the classifier only memorizes problem templates, it will overstate how well we can infer strategy from new rollouts.

## Main Protocol

Dataset:

- input CSV: `results/transformer_replication/synth_unique_seed1_all1000_nlp_no_params_mincols.csv.gz`
- expected fields: `prob`, `strategy`, `response_nl`, and optionally `instruction_nl`, `source_uid`

Main split:

- split unit: `prob`
- train / val / test: `0.8 / 0.1 / 0.1`

Why this split:

- the same problem appears many times across different UMA students
- row-random splitting would let the classifier see the same problem form in both train and test
- problem-held-out evaluation is the better approximation to rollout transfer

Main metrics:

- macro-F1
- balanced accuracy
- plain accuracy
- confusion matrix

Primary model:

- `google-bert/bert-base-uncased`

Primary text setting:

- `text_mode=prob_response`
- `drop_answer_line=true`

The answer line is removed in the main setting because it may create shortcuts from final answer patterns to strategy family. We should still run an ablation with the answer line kept.

## First Run Grid

The first grid should stay small and isolate the main methodological choices.

Axis 1: input text

- `response`
- `prob_response`

Axis 2: split

- `row`
- `problem`

Axis 3: answer-line handling

- `drop_answer_line=true`
- `drop_answer_line=false`

This gives a compact \(2 \times 2 \times 2\) design.

Interpretation target:

- if `row` is much better than `problem`, the classifier is relying heavily on problem-template leakage
- if `prob_response` clearly beats `response`, the problem statement is important context for strategy inference
- if keeping the answer line boosts performance sharply, the classifier may be using answer artifacts rather than genuine rationale structure

## Initial Commands

Smoke run on a deterministic subsample:

```bash
python results/transformer_replication/train_strategy_classifier.py \
  --model_name prajjwal1/bert-tiny \
  --text_mode prob_response \
  --label_mode family \
  --split_mode problem \
  --drop_answer_line \
  --sample_frac 0.002 \
  --epochs 1 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --max_length 256
```

Main first run:

```bash
python results/transformer_replication/train_strategy_classifier.py \
  --model_name google-bert/bert-base-uncased \
  --text_mode prob_response \
  --label_mode family \
  --split_mode problem \
  --drop_answer_line \
  --epochs 3 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --max_length 256 \
  --scheduler linear
```

## Empirical Scaling Study

Question.
What is a reasonable training size for the classifier, measured by held-out problem-level generalization rather than by training loss alone?

Setup.
We fix a single problem-held-out validation/test split from the prepared pool and vary only the number of training examples \(N\). The empirical scaling curve is

\[
m(N) = \mathrm{MacroF1}_{\mathrm{test}}(N).
\]

We then choose the smallest \(N\) that reaches a target fraction \(\tau\) of the best observed test score:

\[
N^{\star} = \min \left\{N : m(N) \ge \tau \cdot \max_{N'} m(N') \right\}.
\]

For the main sweep we use \(\tau = 0.95\), `distilbert-base-uncased`, `text_mode=prob_response`, `split_mode=problem`, `drop_answer_line=true`, and one training epoch. This isolates data-scaling effects before we spend extra compute on longer optimization.

Method.
We prepared a fixed 5% pool from the translated dataset, giving 825,557 train examples, 106,500 validation examples, and 105,313 test examples. From that fixed pool we trained on nested train subsets of size `10k`, `40k`, `160k`, and `640k`.

Finding.
The curve rises sharply up to `160k` and then saturates. The jump from `10k` to `40k` is large, the jump from `40k` to `160k` is still substantial, and the jump from `160k` to `640k` is negligible.

| Train examples | Test macro-F1 | Test accuracy | Test balanced accuracy |
| --- | ---: | ---: | ---: |
| 10,000 | 0.5731 | 0.6969 | 0.7700 |
| 40,000 | 0.7632 | 0.8640 | 0.8848 |
| 160,000 | 0.8779 | 0.9514 | 0.9261 |
| 640,000 | 0.8792 | 0.9532 | 0.9367 |

Implication.
`160k` is the practical knee. It reaches \(0.8779 / 0.8792 = 99.84\%\) of the best one-epoch test macro-F1 while using one quarter of the largest train set. Under the \(\tau = 0.95\) rule, the recommended training size is therefore `160k`.

## Proper Final Run At The Recommended Size

Setup.
After selecting `160k`, we trained a fuller run with the same data protocol and three epochs, again selecting the checkpoint by validation macro-F1.

Finding.
The best checkpoint occurred at epoch 2. This final `160k` run slightly outperformed every one-epoch sweep point, including the `640k` run.

| Run | Best epoch | Test macro-F1 | Test accuracy | Test balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| `160k`, 1 epoch | 1 | 0.8779 | 0.9514 | 0.9261 |
| `640k`, 1 epoch | 1 | 0.8792 | 0.9532 | 0.9367 |
| `160k`, 3 epochs | 2 | 0.8922 | 0.9580 | 0.9317 |

Implication.
For the next rollout-labeling experiments, `160k` examples with early stopping around epoch 2 is a sensible default. More data than that did not materially improve macro-F1 in the fixed one-epoch sweep, while modest extra optimization at `160k` did.

## Recorded Artifacts

- Scaling summary JSON: `results/transformer_replication/strategy_scale_summary.json`
- Scaling summary CSV: `results/transformer_replication/strategy_scale_summary.csv`
- Scaling plot: `results/transformer_replication/strategy_scale_summary.png`
- Recommended-size final run: `results/transformer_replication/strategy_classifier_distilbert_n160000_e3_final`

## Expected Outputs

Each run should write:

- `prepared/prepared_manifest.json`
- `train_args.json`
- `label_space.json`
- `history.json`
- `best/`
- `last/`
- `best_val_metrics.json`
- `best_val_confusion.csv`
- `test_metrics.json`
- `test_confusion.csv`

## Success Criterion

The experiment is useful if the classifier achieves strong **problem-held-out** macro-F1 and produces a confusion matrix that is interpretable enough to trust when applied to rollout text.

The practical threshold is not “near-perfect.” The practical threshold is:

- stable enough to compare distributions across models,
- not obviously driven by answer leakage,
- and clearly better than the current regex baseline.

## Risks

Risk 1:
The classifier may learn problem-form regularities rather than reasoning cues.

Risk 2:
The translated gold rationales may be cleaner than actual distilled generations, so rollout-time performance may fall.

Risk 3:
Some families may remain confusable even with the problem text, especially when the generated rationale drifts toward generic arithmetic narration.

## Follow-Up If Promising

If this works on translated UMA text, the next step is to score:

1. UMA gold translated responses
2. distilled-model generations on the same prompts
3. human-finetuned generations

Then we can compare empirical strategy distributions directly rather than relying on surface rules.
