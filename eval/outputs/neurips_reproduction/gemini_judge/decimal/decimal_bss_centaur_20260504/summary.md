# Human-Likeness Alignment Preference Evaluation

- plan: `decimal_bss_anchor_v1`
- prompt_version: `human_likeness_preference_v5`
- backend: `gemini_batch`
- judge_model: `gemini-3.1-flash-lite-preview`
- samples_per_problem: 100
- preference_sets: 2
- rows: 2400
- parse_success_rate: 1.0

The run compares human-likeness preferences over decimal written solutions.
Entities: centaur_70b, human, qwen3_0p6b_mag6h.

## Human Rendering Check

- human_judged_rows: 1200
- human_preference_rate: 0.9366666666666666
- human_on_a_count: 600
- human_on_b_count: 600
- human_preference_rate_when_on_a: 0.96
- human_preference_rate_when_on_b: 0.9133333333333333
- human_side_bias_gap: 0.046666666666666634
