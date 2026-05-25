# Human-Likeness Alignment Preference Evaluation

- plan: `fraction_human_ft_5way`
- prompt_version: `human_likeness_preference_v5`
- backend: `gemini_batch`
- judge_model: `gemini-3.1-flash-lite-preview`
- samples_per_problem: 100
- preference_sets: 2
- rows: 1600
- parse_success_rate: 1.0

The run compares human-likeness preferences over fraction written solutions.
Entities: centaur_70b, distilled_panel25, human.

## Human Rendering Check

- human_judged_rows: 800
- human_preference_rate: 0.845
- human_on_a_count: 400
- human_on_b_count: 400
- human_preference_rate_when_on_a: 0.965
- human_preference_rate_when_on_b: 0.725
- human_side_bias_gap: 0.24
