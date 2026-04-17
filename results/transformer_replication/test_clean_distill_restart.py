import json
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate_translator_automatic_metrics as auto_metrics
import train_transformer_hf as trainer
import translate_uma_traces_to_nlp as translator


def test_trainer_accepts_legacy_colon_and_new_slash():
    assert trainer.render_problem_for_prompt("3/5:1/5") == "3/5 / 1/5"
    assert trainer.render_problem_for_prompt("3/5 / 1/5") == "3/5 / 1/5"
    assert trainer.compute_correct_answer("3/5:1/5") == "3"
    assert trainer.compute_correct_answer("3/5 / 1/5") == "3"
    assert trainer.op_from_prob("3/5:1/5") == "/"
    assert trainer.op_from_prob("3/5 / 1/5") == "/"


def test_translator_clean_child_ignores_false_trace_claims():
    bad_trace = json.dumps(
        [
            {
                "i": 1,
                "rule": "bogus",
                "main": {"op": "+", "op1": "2", "op2": "2", "ans": "3"},
                "subs": [],
            }
        ]
    )
    record = {
        "prob": "1/2+1/2",
        "operation": "+",
        "strategy": "ONOD_OG",
        "goals": "operate_nums operate_dens",
        "exec": "",
        "answer": "1",
        "correct": "1",
        "is_correct": "1",
        "g": "0.01",
        "d": "0.1",
        "rt_mu": "3",
        "ice": "0",
        "trace_steps_json": bad_trace,
    }
    prompt_cfg = translator.StudentPromptConfig(mode="none", drop_prob=0.0, drop_seed=0)
    out = translator.translate_record(
        record,
        include_outcome_text=False,
        student_prompt_config=prompt_cfg,
        reasoning_mode="clean_child",
        surface_hidden_trace_steps=False,
    )
    assert out["instruction_nl"] == "Solve this fraction problem: 1/2 + 1/2=?"
    assert "I took 2 plus 2 and got 3" not in out["response_nl"]
    assert "### correctness:" not in out["response_nl"]
    assert "### answer: 1" in out["response_nl"]


def test_apply_row_filter_correct_exec_and_answer_is_subset():
    frame = pd.DataFrame(
        [
            {
                "reasoning_quality_flags": "",
                "reasoning_unverified_claims": 0,
                "is_correct": 1,
            },
            {
                "reasoning_quality_flags": "arith_claim_mismatch",
                "reasoning_unverified_claims": 0,
                "is_correct": 1,
            },
            {
                "reasoning_quality_flags": "",
                "reasoning_unverified_claims": 1,
                "is_correct": 1,
            },
            {
                "reasoning_quality_flags": "",
                "reasoning_unverified_claims": 0,
                "is_correct": 0,
            },
        ]
    )

    correct_exec, stats_exec = translator.apply_row_filter(frame, "correct_exec")
    correct_both, stats_both = translator.apply_row_filter(frame, "correct_exec_and_answer")

    assert len(correct_exec) == 2
    assert stats_exec["rows_kept"] == 2
    assert len(correct_both) == 1
    assert stats_both["rows_kept"] == 1
    assert len(correct_both) <= len(correct_exec)


def test_translator_cdon_ud_add_sub_are_not_false_mismatches():
    cases = [
        ("3/5+1/4", "+", "17/20"),
        ("2/3+3/5", "+", "19/15"),
        ("3/5-1/4", "-", "7/20"),
        ("2/3-3/5", "-", "1/15"),
    ]

    for prob, operation, answer in cases:
        reasoning, quality = translator.build_child_reasoning(
            prob=prob,
            operation=operation,
            strategy_code="CDON_AS",
            goals=["convert_CD"],
            exec_rules=[],
            answer=answer,
        )

        assert "arith_claim_mismatch" not in quality["reasoning_quality_flags"]
        assert quality["reasoning_unverified_claims"] == 0
        assert quality["reasoning_verified_claims"] == 1
        assert answer in reasoning


def test_translator_cdon_simplified_answer_counts_as_verified():
    reasoning, quality = translator.build_child_reasoning(
        prob="1/6+1/3",
        operation="+",
        strategy_code="CDON_AS",
        goals=["convert_CD", "simplify_fraction"],
        exec_rules=[],
        answer="1/2",
    )

    assert "arith_claim_mismatch" not in quality["reasoning_quality_flags"]
    assert quality["reasoning_unverified_claims"] == 0
    assert quality["reasoning_verified_claims"] == 1
    assert "3/6" in reasoning
    assert "1/2" in reasoning


def test_translator_mentions_lcm_rewrite_and_direct_division():
    reasoning, quality = translator.build_child_reasoning(
        prob="2/7:2/3",
        operation=":",
        strategy_code="CDON_OG",
        goals=["convert_CD_LCM", "get_LCM", "convert_fra_to_den", "check_simplify"],
        exec_rules=["div_calculator"],
        answer="1/21",
    )

    assert "least common multiple of the denominators, 21" in reasoning
    assert "I rewrote 2/7 as 6/21 and 2/3 as 14/21" in reasoning
    assert "I worked out that division directly" in reasoning
    assert quality["reasoning_unverified_claims"] == 0


def test_translator_makes_convert_cd_explicit_with_denominator():
    reasoning, quality = translator.build_child_reasoning(
        prob="1/6+1/3",
        operation="+",
        strategy_code="CDON_AS",
        goals=["convert_CD", "pass_den", "operate_nums"],
        exec_rules=[],
        answer="1/2",
    )

    assert "I found a common denominator, 6, first" in reasoning
    assert "I changed both fractions to use denominator 6" in reasoning
    assert quality["reasoning_unverified_claims"] == 0


def test_translator_mentions_gcd_and_denied_div_to_mul():
    reasoning, quality = translator.build_child_reasoning(
        prob="1/2*7/2",
        operation="*",
        strategy_code="ICDM_OG",
        goals=["invert_op2", "operate_dens", "operate_nums", "check_simplify", "get_GCD", "simplify_fraction"],
        exec_rules=["div_to_mul_denied", "div_calculator", "div_LbS"],
        answer="1/4",
    )

    assert "I thought about flipping the second fraction and changing it to multiplication, but I kept it in this form" in reasoning
    assert "I worked out that division directly" in reasoning
    assert "greatest common factor" in reasoning
    assert quality["reasoning_unverified_claims"] == 0


def test_translator_distinguishes_second_fraction_from_random_flip_in_division():
    reasoning_goal, _ = translator.build_child_reasoning(
        prob="1/2:1/3",
        operation=":",
        strategy_code="ICDM_D",
        goals=["invert_op2", "div_to_mul", "operate_dens", "operate_nums"],
        exec_rules=[],
        answer="3/2",
    )
    assert "I flipped the second fraction, 1/3, to 3/1 and changed it to multiplication" in reasoning_goal
    assert "I flipped one of the fractions" not in reasoning_goal

    reasoning_exec, _ = translator.build_child_reasoning(
        prob="1/2:1/3",
        operation=":",
        strategy_code="ICDM_OG",
        goals=["div_to_mul", "operate_dens", "operate_nums"],
        exec_rules=["invert_rand"],
        answer="3/2",
    )
    assert "I flipped one of the fractions, 1/3, to 3/1 and changed it to multiplication" in reasoning_exec
    assert "I flipped the second fraction" not in reasoning_exec


def test_translator_mentions_invert_rand_for_icdm_non_division():
    reasoning, _ = translator.build_child_reasoning(
        prob="1/4*1/3",
        operation="*",
        strategy_code="ICDM_OG",
        goals=["operate_dens", "operate_nums", "skip_simplify"],
        exec_rules=["invert_rand", "div_to_mul_denied"],
        answer="4/3",
    )

    assert "I worked on the top numbers and on the bottom numbers separately and got 4/3" in reasoning
    assert "I flipped one of the fractions" in reasoning


def test_translator_can_surface_hidden_internal_steps_when_enabled():
    reasoning, _ = translator.build_child_reasoning(
        prob="1/2+1/2",
        operation="+",
        strategy_code="ONOD_OG",
        goals=["operate_nums", "operate_dens", "skip_simplify"],
        exec_rules=["invert_fail", "acc_skip", "acc_extra"],
        answer="1/1",
        surface_hidden_trace_steps=True,
    )

    assert "I decided not to simplify the fraction any further" in reasoning
    assert "I tried to flip a fraction, but I did not change it" in reasoning
    assert "I moved on without updating the running answer" in reasoning
    assert "I made an extra running-answer update" in reasoning


def test_translator_keeps_hidden_internal_steps_off_by_default():
    reasoning, _ = translator.build_child_reasoning(
        prob="1/2+1/2",
        operation="+",
        strategy_code="ONOD_OG",
        goals=["operate_nums", "operate_dens", "skip_simplify"],
        exec_rules=["invert_fail", "acc_skip", "acc_extra"],
        answer="1/1",
    )

    assert "I decided not to simplify the fraction any further" not in reasoning
    assert "I tried to flip a fraction, but I did not change it" not in reasoning
    assert "I moved on without updating the running answer" not in reasoning
    assert "I made an extra running-answer update" not in reasoning


def test_auto_metric_extractors_recover_new_surface_forms():
    text = (
        "I found the least common multiple of the denominators, 21, first. "
        "I rewrote 2/7 as 6/21 and 2/3 as 14/21. "
        "I thought about flipping the second fraction and changing it to multiplication, but I kept it in this form. "
        "I worked out that division directly. "
        "I checked for a greatest common factor and then used it to simplify it."
    )

    assert auto_metrics.infer_strategy_family(text) == "ICDM"
    assert {"convert_CD_LCM", "get_LCM", "convert_fra_to_den", "get_GCD", "simplify_fraction"} <= auto_metrics.infer_goal_tokens(text)
    assert "convert_CD" not in auto_metrics.infer_goal_tokens(text)
    assert {"div_to_mul_denied", "div_calculator"} <= auto_metrics.infer_exec_tokens(text)

    text_onod = "I applied divided by to top numbers and bottom numbers separately. So I got 1/3.5."
    assert {"operate_nums", "operate_dens"} <= auto_metrics.infer_goal_tokens(text_onod)

    text_icdm = "I flipped the second fraction, 1/3, to 3/1 and changed it to multiplication. Then I did 1 times 3 and got 3, and 2 times 1 and got 2."
    assert {"operate_nums", "operate_dens", "invert_op2", "div_to_mul"} <= auto_metrics.infer_goal_tokens(text_icdm)
    assert "invert_rand" not in auto_metrics.infer_exec_tokens(text_icdm)

    text_rand = "I flipped one of the fractions, 1/3, to 3/1 and changed it to multiplication."
    assert {"div_to_mul"} <= auto_metrics.infer_goal_tokens(text_rand)
    assert {"invert_rand"} <= auto_metrics.infer_exec_tokens(text_rand)

    text_denied = (
        "I worked on the top numbers and on the bottom numbers separately and got 8/3. "
        "I thought about flipping the second fraction and changing it to multiplication, but I kept it in this form."
    )
    inferred_goals = auto_metrics.infer_goal_tokens(text_denied)
    assert {"operate_nums", "operate_dens", "invert_op2"} <= inferred_goals
    assert "div_to_mul" not in inferred_goals

    text_cdon_omit = (
        "I found a common denominator, 4, first. "
        "Then I combined the top numbers with plus and kept 4 on the bottom. "
        "I changed the bottom numbers and kept the top numbers the same."
    )
    inferred_cdon_omit = auto_metrics.infer_goal_tokens(text_cdon_omit)
    assert {"pass_den", "operate_nums"} <= inferred_cdon_omit
    assert "convert_CD" not in inferred_cdon_omit

    text_cdon_exact = (
        "I found a common denominator, 6, first. "
        "I changed both fractions to use denominator 6. "
        "Then I took 1 plus 2 and got 3, and I kept 6 on the bottom."
    )
    assert {"convert_CD", "pass_den", "operate_nums"} <= auto_metrics.infer_goal_tokens(text_cdon_exact)

    text_div_combo = "I divided the bigger number by the smaller number and kept only the whole-number part."
    assert {"div_LbS_drop_rem"} <= auto_metrics.infer_exec_tokens(text_div_combo)
    assert "div_LbS" not in auto_metrics.infer_exec_tokens(text_div_combo)
    assert "div_drop_rem" not in auto_metrics.infer_exec_tokens(text_div_combo)

    text_div_parts = "I divided the bigger number by the smaller number. I used the whole-number part after dividing."
    inferred_div_parts = auto_metrics.infer_exec_tokens(text_div_parts)
    assert {"div_LbS", "div_drop_rem"} <= inferred_div_parts
    assert "div_LbS_drop_rem" not in inferred_div_parts

    text_hidden = (
        "I decided not to simplify the fraction any further. "
        "I tried to flip a fraction, but I did not change it. "
        "I moved on without updating the running answer. "
        "I made an extra running-answer update."
    )
    assert {"skip_simplify"} <= auto_metrics.infer_goal_tokens(text_hidden)
    assert {"invert_fail", "acc_skip", "acc_extra"} <= auto_metrics.infer_exec_tokens(text_hidden)


def test_sp2013_uma_tuple_metrics_exact_match_fixture(tmp_path):
    target_csv = tmp_path / "sp2013_targets.csv"
    pd.DataFrame(
        [
            {"prob": "3/5+1/5", "g": 0.01, "d": 0.1, "rt_mu": 3, "ice": 0, "ans": "4/5"},
            {"prob": "3/5 / 1/5", "g": 0.01, "d": 0.1, "rt_mu": 3, "ice": 0, "ans": "3"},
        ]
    ).to_csv(target_csv, index=False)

    out_df = pd.DataFrame(
        [
            {
                "prob": "3/5+1/5",
                "op": "+",
                "g": 0.01,
                "d": 0.1,
                "rt_mu": 3,
                "ice": 0,
                "pred_answer": "4/5",
                "pred_answer_label": "4/5",
                "pred_answer_parseable": 1,
                "is_correct_true": 1,
            },
            {
                "prob": "3/5 / 1/5",
                "op": "/",
                "g": 0.01,
                "d": 0.1,
                "rt_mu": 3,
                "ice": 0,
                "pred_answer": "3",
                "pred_answer_label": "3",
                "pred_answer_parseable": 1,
                "is_correct_true": 1,
            },
        ]
    )

    target_df = trainer.load_sp2013_target_frame(str(target_csv))
    metrics, merged = trainer.compute_sp2013_uma_tuple_metrics(out_df, target_df)

    assert len(merged) == 2
    assert metrics["primary_name"] == "teacher_tuple_acc_all"
    assert metrics["primary_higher_is_better"] is True
    assert metrics["teacher_acc_all"] == 1.0
    assert metrics["true_acc_all"] == 1.0


def test_sp2013_uma_distribution_metrics_tv_fixture(tmp_path):
    target_csv = tmp_path / "sp2013_distribution_targets.csv"
    pd.DataFrame(
        [
            {"prob": "3/5+1/5", "g": 0.01, "d": 0.1, "rt_mu": 3, "ice": 0, "ans": "4/5"},
            {"prob": "3/5+1/5", "g": 0.02, "d": 0.1, "rt_mu": 3, "ice": 0, "ans": "4/5"},
            {"prob": "3/5+1/5", "g": 0.03, "d": 0.1, "rt_mu": 3, "ice": 0, "ans": "1"},
            {"prob": "3/5+1/5", "g": 0.04, "d": 0.1, "rt_mu": 3, "ice": 0, "ans": "1"},
        ]
    ).to_csv(target_csv, index=False)
    target_df = trainer.load_sp2013_target_frame(str(target_csv))

    exact_out_df = pd.DataFrame(
        [
            {"prob": "3/5+1/5", "op": "+", "pred_answer_label": "4/5", "pred_answer_parseable": 1, "is_correct_true": 1},
            {"prob": "3/5+1/5", "op": "+", "pred_answer_label": "4/5", "pred_answer_parseable": 1, "is_correct_true": 1},
            {"prob": "3/5+1/5", "op": "+", "pred_answer_label": "1", "pred_answer_parseable": 1, "is_correct_true": 1},
            {"prob": "3/5+1/5", "op": "+", "pred_answer_label": "1", "pred_answer_parseable": 1, "is_correct_true": 1},
        ]
    )
    mismatch_out_df = pd.DataFrame(
        [
            {"prob": "3/5+1/5", "op": "+", "pred_answer_label": "4/5", "pred_answer_parseable": 1, "is_correct_true": 1},
            {"prob": "3/5+1/5", "op": "+", "pred_answer_label": "4/5", "pred_answer_parseable": 1, "is_correct_true": 1},
            {"prob": "3/5+1/5", "op": "+", "pred_answer_label": "4/5", "pred_answer_parseable": 1, "is_correct_true": 1},
            {"prob": "3/5+1/5", "op": "+", "pred_answer_label": "4/5", "pred_answer_parseable": 1, "is_correct_true": 1},
        ]
    )

    exact_metrics = trainer.compute_sp2013_uma_distribution_metrics(exact_out_df, target_df)
    mismatch_metrics = trainer.compute_sp2013_uma_distribution_metrics(mismatch_out_df, target_df)

    assert exact_metrics["primary_name"] == "distribution_tv_mean"
    assert exact_metrics["primary_higher_is_better"] is False
    assert exact_metrics["distribution_tv_mean"] == 0.0
    assert mismatch_metrics["distribution_tv_mean"] > 0.0
