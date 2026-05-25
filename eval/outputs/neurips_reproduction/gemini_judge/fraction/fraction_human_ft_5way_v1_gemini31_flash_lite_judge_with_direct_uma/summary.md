# Human-Likeness Alignment Preference Evaluation

- plan: `fraction_human_ft_5way`
- prompt_version: `human_likeness_preference_v5`
- backend: `requests`
- judge_model: `gemini-3.1-flash-lite-preview`
- samples_per_problem: 100
- preference_sets: 6
- rows: 4800
- parse_success_rate: 1.0

The run compares human-likeness preferences over fraction written solutions.
Entities: distilled_panel25, gemini_2p5_flash, gpt_4p1_mini, human, uma_nlp.

## Human Rendering Check

- human_judged_rows: 2400
- human_preference_rate: 0.8275
- human_on_a_count: 800
- human_on_b_count: 800
- human_preference_rate_when_on_a: 0.9825
- human_preference_rate_when_on_b: 0.84
- human_side_bias_gap: 0.14250000000000007
