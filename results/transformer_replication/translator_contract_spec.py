"""
Logic-first contract cases for the UMA -> NLP translator.

These fixtures are intentionally symbolic and synthetic: they are chosen to
exercise translator branches and token-to-language mappings directly, without
depending on the empirical data distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ChildReasoningCase:
    name: str
    prob: str
    operation: str
    strategy_code: str
    goals: Tuple[str, ...]
    exec_rules: Tuple[str, ...]
    answer: str
    expected_substrings: Tuple[str, ...]
    forbidden_substrings: Tuple[str, ...] = ()
    expected_verified_claims: int | None = None
    expected_unverified_claims: int | None = None
    surface_hidden_trace_steps: bool = False


@dataclass(frozen=True)
class DetailReasoningCase:
    name: str
    goals: Tuple[str, ...]
    exec_rules: Tuple[str, ...]
    expected_sentences: Tuple[str, ...]
    surface_hidden_trace_steps: bool = False


@dataclass(frozen=True)
class IgnoredRuleCase:
    name: str
    prob: str
    operation: str
    strategy_code: str
    goals: Tuple[str, ...]
    exec_rules: Tuple[str, ...]
    answer: str
    forbidden_substrings: Tuple[str, ...]
    expected_answer_suffix: str = ""
    surface_hidden_trace_steps: bool = False


STRATEGY_BRANCH_CASES: Tuple[ChildReasoningCase, ...] = (
    ChildReasoningCase(
        name="cdon_add",
        prob="1/6+1/3",
        operation="+",
        strategy_code="CDON_AS",
        goals=("convert_CD", "pass_den", "operate_nums"),
        exec_rules=(),
        answer="1/2",
        expected_substrings=(
            "I found a common denominator, 6, first",
            "I changed both fractions to use denominator 6",
            "Then I took 1 plus 2 and got 3, and I kept 6 on the bottom",
        ),
        expected_verified_claims=1,
        expected_unverified_claims=0,
    ),
    ChildReasoningCase(
        name="kdon_add",
        prob="1/4+2/4",
        operation="+",
        strategy_code="KDON_AS",
        goals=("pass_den", "operate_nums"),
        exec_rules=(),
        answer="3/4",
        expected_substrings=(
            "I took 1 plus 2 and got 3, and I kept 4 on the bottom",
        ),
        expected_verified_claims=1,
        expected_unverified_claims=0,
    ),
    ChildReasoningCase(
        name="onod_add",
        prob="1/2+1/3",
        operation="+",
        strategy_code="ONOD_OG",
        goals=("operate_dens", "operate_nums"),
        exec_rules=(),
        answer="2/5",
        expected_substrings=(
            "I took 1 plus 1 and got 2, and 2 plus 3 and got 5",
        ),
        expected_verified_claims=2,
        expected_unverified_claims=0,
    ),
    ChildReasoningCase(
        name="crop_mul",
        prob="1/2*1/3",
        operation="*",
        strategy_code="CROP_M",
        goals=("operate_dens", "operate_nums"),
        exec_rules=(),
        answer="3/2",
        expected_substrings=(
            "I crossed them and did 1 times 3 to get 3, and 2 times 1 to get 2",
        ),
        expected_verified_claims=2,
        expected_unverified_claims=0,
    ),
    ChildReasoningCase(
        name="icdm_division",
        prob="1/2:1/3",
        operation=":",
        strategy_code="ICDM_D",
        goals=("invert_op2", "div_to_mul", "operate_dens", "operate_nums"),
        exec_rules=(),
        answer="3/2",
        expected_substrings=(
            "I flipped the second fraction, 1/3, to 3/1 and changed it to multiplication",
            "Then I did 1 times 3 and got 3, and 2 times 1 and got 2",
        ),
        expected_verified_claims=2,
        expected_unverified_claims=0,
    ),
    ChildReasoningCase(
        name="kdon_division_decimal_answer",
        prob="1/3:9/3",
        operation=":",
        strategy_code="KDON_OG",
        goals=("pass_den", "operate_nums"),
        exec_rules=("div_calculator",),
        answer="0.111/3",
        expected_substrings=(
            "I took 1 divided by 9 and got 0.111, and I kept 3 on the bottom",
        ),
        forbidden_substrings=("I worked with 1/3 and 9/3",),
        expected_verified_claims=1,
        expected_unverified_claims=0,
    ),
    ChildReasoningCase(
        name="onod_division_decimal_answer",
        prob="1/2:2/2",
        operation=":",
        strategy_code="ONOD_OG",
        goals=("operate_dens", "operate_nums"),
        exec_rules=("div_calculator",),
        answer="0.5/1",
        expected_substrings=(
            "I took 1 divided by 2 and got 0.5, and 2 divided by 2 and got 1",
        ),
        forbidden_substrings=("I worked with 1/2 and 2/2",),
        expected_verified_claims=2,
        expected_unverified_claims=0,
    ),
    ChildReasoningCase(
        name="cdon_division_decimal_answer",
        prob="1/2:1/5",
        operation=":",
        strategy_code="CDON_OG",
        goals=("convert_CD", "pass_den", "operate_nums"),
        exec_rules=("div_calculator",),
        answer="2.5/10",
        expected_substrings=(
            "I found a common denominator, 10, first",
            "I changed both fractions to use denominator 10",
            "Then I took 5 divided by 2 and got 2.5, and I kept 10 on the bottom",
        ),
        forbidden_substrings=("I worked with 1/2 and 1/5",),
        expected_verified_claims=1,
        expected_unverified_claims=0,
    ),
    ChildReasoningCase(
        name="cdon_omit_nums_branch",
        prob="1/2+2/4",
        operation="+",
        strategy_code="CDON_AS",
        goals=("pass_den", "operate_nums"),
        exec_rules=("convert_CD_omit_nums",),
        answer="3/8",
        expected_substrings=(
            "Then I combined the top numbers with plus and kept 4 on the bottom",
            "I changed the bottom numbers and kept the top numbers the same",
        ),
    ),
    ChildReasoningCase(
        name="cdon_multiply_pass_den",
        prob="1/2*5/8",
        operation="*",
        strategy_code="CDON_OG",
        goals=("pass_den", "operate_nums", "check_simplify", "get_GCD"),
        exec_rules=(),
        answer="5/16",
        expected_substrings=(
            "I found a common denominator, 8, first",
            "Then I took 1 times 5 and got 5, and I kept 16 on the bottom",
        ),
        expected_verified_claims=1,
        expected_unverified_claims=0,
    ),
)


DETAIL_REASONING_CASES: Tuple[DetailReasoningCase, ...] = (
    DetailReasoningCase(
        name="goal_check_simplify",
        goals=("check_simplify",),
        exec_rules=(),
        expected_sentences=("I checked if it could be simplified",),
    ),
    DetailReasoningCase(
        name="goal_check_simplify_with_gcd",
        goals=("check_simplify", "get_GCD"),
        exec_rules=(),
        expected_sentences=("I checked for a greatest common factor",),
    ),
    DetailReasoningCase(
        name="goal_simplify_with_gcd",
        goals=("simplify_fraction", "get_GCD"),
        exec_rules=(),
        expected_sentences=("I checked for a greatest common factor and then used it to simplify it",),
    ),
    DetailReasoningCase(
        name="exec_fact_and_error_rules",
        goals=(),
        exec_rules=(
            "add_fact",
            "sub_fact",
            "mul_fact",
            "convert_CD_omit_nums",
            "invert_rand",
            "div_to_mul_denied",
            "div_calculator",
            "sub_LbS",
            "div_LbS_drop_rem",
        ),
        expected_sentences=(
            "I used an addition fact to get that result",
            "I used a subtraction fact to get that result",
            "I used a multiplication fact to get that result",
            "I changed the bottom numbers and kept the top numbers the same",
            "I flipped one of the fractions",
            "I thought about changing it to multiplication, but I kept it in this form",
            "I worked out that division directly",
            "I subtracted the smaller number from the bigger number",
            "I divided the bigger number by the smaller number and kept only the whole-number part",
        ),
    ),
    DetailReasoningCase(
        name="exec_division_parts_are_separate_when_tokens_are_separate",
        goals=(),
        exec_rules=(
            "div_LbS",
            "div_drop_rem",
        ),
        expected_sentences=(
            "I divided the bigger number by the smaller number",
            "I used the whole-number part after dividing",
        ),
    ),
    DetailReasoningCase(
        name="surfaced_internal_steps",
        goals=("skip_simplify",),
        exec_rules=("invert_fail", "acc_skip", "acc_extra"),
        expected_sentences=(
            "I decided not to simplify the fraction any further",
            "I tried to flip a fraction, but I did not change it",
            "I moved on without updating the running answer",
            "I made an extra running-answer update",
        ),
        surface_hidden_trace_steps=True,
    ),
)


INTERACTION_CASES: Tuple[ChildReasoningCase, ...] = (
    ChildReasoningCase(
        name="cdon_lcm_rewrite_division",
        prob="2/7:2/3",
        operation=":",
        strategy_code="CDON_OG",
        goals=("convert_CD_LCM", "get_LCM", "convert_fra_to_den", "check_simplify"),
        exec_rules=("div_calculator",),
        answer="1/21",
        expected_substrings=(
            "I found the least common multiple of the denominators, 21, first",
            "I rewrote 2/7 as 6/21 and 2/3 as 14/21",
            "I worked out that division directly",
            "I checked if it could be simplified",
        ),
        expected_unverified_claims=0,
    ),
    ChildReasoningCase(
        name="icdm_nondv_invert_and_gcd",
        prob="1/4*1/3",
        operation="*",
        strategy_code="ICDM_OG",
        goals=("invert_op2", "operate_dens", "operate_nums", "check_simplify", "get_GCD"),
        exec_rules=("invert_rand", "div_to_mul_denied"),
        answer="4/3",
        expected_substrings=(
            "I worked on the top numbers and on the bottom numbers separately and got 4/3",
            "I flipped one of the fractions",
            "I thought about flipping the second fraction and changing it to multiplication, but I kept it in this form",
            "I checked for a greatest common factor",
        ),
        expected_unverified_claims=0,
    ),
    ChildReasoningCase(
        name="icdm_division_random_flip_surface",
        prob="1/2:1/3",
        operation=":",
        strategy_code="ICDM_OG",
        goals=("div_to_mul", "operate_dens", "operate_nums"),
        exec_rules=("invert_rand",),
        answer="3/2",
        expected_substrings=(
            "I flipped one of the fractions, 1/3, to 3/1 and changed it to multiplication",
            "Then I did 1 times 3 and got 3, and 2 times 1 and got 2",
        ),
        forbidden_substrings=(
            "I flipped the second fraction",
        ),
        expected_unverified_claims=0,
    ),
    ChildReasoningCase(
        name="cdon_rewrite_with_decimal_answer",
        prob="1/4:5/7",
        operation=":",
        strategy_code="CDON_OG",
        goals=("convert_CD_LCM", "get_LCM", "convert_fra_to_den", "pass_den", "operate_nums"),
        exec_rules=("div_calculator",),
        answer="0.35/28",
        expected_substrings=(
            "I found the least common multiple of the denominators, 28, first",
            "I rewrote 1/4 as 7/28 and 5/7 as 20/28",
            "Then I took 7 divided by 20 and got 0.35, and I kept 28 on the bottom",
        ),
        forbidden_substrings=("I worked with 1/4 and 5/7",),
        expected_unverified_claims=0,
    ),
)


IGNORED_RULE_CASES: Tuple[IgnoredRuleCase, ...] = (
    IgnoredRuleCase(
        name="ignored_internal_tokens",
        prob="1/2+1/2",
        operation="+",
        strategy_code="ONOD_OG",
        goals=("skip_simplify",),
        exec_rules=("invert_fail",),
        answer="1",
        forbidden_substrings=(
            "skip simplification",
            "did not simplify",
            "invert fail",
            "failed to invert",
        ),
        expected_answer_suffix="### answer: 1",
    ),
)
