# Prompt ablations — LLM baselines (regraded with mixed-number equivalence)

## Headline

After regrading the model responses with a parser that treats
**mixed numbers and improper fractions as equivalent** (`12/5 == 2 2/5 == 1 \frac{4}{15}`),
**all three LLMs are essentially saturated on SP2013 fractions** under
every prompt cell:

| Model | acc, original parser | acc, equivalence-aware | flips |
|---|---|---|---|
| gpt-4.1-mini | 90.46% | **100.00%** | 824 |
| gemini-2.5-flash | 86.69% | **99.95%** | 573 |
| claude-sonnet-4 | 99.44% | **100.00%** | 24 |

The earlier "below-average persona beats high-performing" effect
(−7.5pp on gpt-4.1-mini, +9.4pp inverted on gemini) **was a parser
artifact, not a model behavior**: the high-performing persona
preferentially converts `12/5 → 2 2/5` ("more polished" final form), and
the original regex captured the trailing `2/5` substring as the answer.
The same explains the format-marginal "show-reasoning loses" effect.

## Design

3 × 3 × 3 grid (27 cells per dataset):

| Axis | Values |
|---|---|
| `grade` | 6th, 7th, 8th |
| `performance` | below-average, average, high-performing |
| `format` | direct, show-reasoning, cot |

System prompt template:
```
You are a student in {grade} grade who is {performance} at math,
solving {fraction|decimal} arithmetic problems. {format suffix}
Do not use a calculator. End with 'Answer: <answer>'.
```

Datasets: SP2013 (16 fractions), BSS2021 (12 decimals).
Samples: n=20 (gpt-4.1-mini), n=10 (gemini, claude). temperature=1.0.

## Files

| Pattern | Description |
|---|---|
| `{model}_fraction_marginals.png` | Original parser, SP2013 marginals |
| `{model}_fraction_marginals_v2.png` | Side-by-side: original parser vs equivalence-aware (the "v2" plots) |
| `{model}_decimal_marginals.png` | BSS2021 marginals (saturated for all models) |
| `{model}_prompts_and_responses.csv` | All raw rows: system_prompt, problem, model_response, parsed_answer, is_correct |
| `{model}_fraction_regraded.csv` | Same fraction rows + `is_correct_v2`, `parsed_answer_v2` |

## Decimal (BSS2021): saturated for all 3 models

All three saturate at ≥99.7% on every cell with the original parser
already; mixed-number rules don't apply. The original 71.7% baseline
was a different parser artifact (regex matched the *first* number after
"answer:" in `Final answer: 12.3 + 5.6 = 17.9`, capturing `12.3`
instead of `17.9`).

## Implication

The LLM baseline on both SP2013 and BSS2021 is **~100% under any
reasonable prompting**, against ~50% for humans and ~55% for UMA.
The gap is not an artifact of unfortunate prompting, persona choice, or
format — it's the LLM correctly solving these problems. The earlier
"prompt-sensitivity" finding was a parser story, not a model story.
