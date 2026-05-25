# Human-Likeness Alignment Preference Evaluation

- plan: `decimal_bss_anchor_v1`
- prompt_version: `human_likeness_preference_v5`
- backend: `gemini_sync`
- judge_model: `gemini-3.1-flash-lite-preview`
- samples_per_problem: 100
- preference_sets: 9
- rows: 10800
- parse_success_rate: 1.0

The run compares human-likeness preferences over decimal written solutions.
Entities: claude_sonnet_4, gemini_2p5_flash, gpt_4p1_mini, human, qwen3_0p6b_mag6h, uma_nlp.

## Human Rendering Check

- human_judged_rows: 6000
- human_preference_rate: 0.754
- human_on_a_count: 3000
- human_on_b_count: 3000
- human_preference_rate_when_on_a: 0.882
- human_preference_rate_when_on_b: 0.626
- human_side_bias_gap: 0.256
