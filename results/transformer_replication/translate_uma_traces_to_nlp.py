#!/usr/bin/env python3
"""
Translate UMA symbolic traces into deterministic NLP-style training text.

This script is intended as a preprocessing step before transformer training.
It preserves source trace fields, appends natural-language columns, and can
assemble one output dataset from multiple UMA CSV sources.

By default it combines:
- results/UMA_replication/uma_traces_all.csv
- results/UMA_replication/sp2013_eval/seed_{1..20}/sp2013_seed{seed}_all_models.csv

Example:
  python results/transformer_replication/translate_uma_traces_to_nlp.py \
    --output-csv results/transformer_replication/uma_traces_all_nlp.csv.gz

  python results/transformer_replication/translate_uma_traces_to_nlp.py \
    --no-include-sp2013-seeds \
    --input-csv results/UMA_replication/uma_traces_all.csv \
    --output-csv results/transformer_replication/uma_traces_all_nlp.csv.gz
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UMA_INPUT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "UMA_replication", "uma_traces_all.csv"))
DEFAULT_SP2013_EVAL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "UMA_replication", "sp2013_eval"))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "uma_traces_all_nlp.csv.gz")
DEFAULT_SP2013_SEEDS = "1-20"
BASE_TRANSLATION_VERSION = "uma_nlp_rules_v10_clean_child_label_alignment"
VALID_STUDENT_PROMPT_MODES = ("none", "always", "dropout")
VALID_REASONING_MODES = ("trace_or_child", "clean_child")
VALID_ROW_FILTER_MODES = ("none", "correct_exec", "correct_exec_and_answer")
DEFAULT_STUDENT_PARAMS_DROP_PROB = 0.5
DEFAULT_STUDENT_PARAMS_DROP_SEED = 0


OPERATION_TEXT: Dict[str, str] = {
    "+": "addition",
    "-": "subtraction",
    "*": "multiplication",
    ":": "division",
}

OPERATION_WORD: Dict[str, str] = {
    "+": "plus",
    "-": "minus",
    "*": "times",
    ":": "divided by",
}


STRATEGY_TEXT: Dict[str, str] = {
    "KDON_AS": "I combine the numerators and keep one denominator (KDON for addition or subtraction)",
    "KDON_OG": "I combine the numerators and keep one denominator, even outside the usual KDON scope",
    "CDON_AS": "I convert both fractions to a common denominator, then I combine numerators and keep that denominator",
    "CDON_OG": "I use a common-denominator conversion and then combine numerators across operations",
    "ONOD_M": "I apply the operation to numerators and denominators separately (ONOD for multiplication)",
    "ONOD_OG": "I apply the operation separately to numerators and denominators",
    "CROP_M": "I cross-operate between numerator and denominator positions",
    "ICDM_D": "I invert a divisor and convert division to multiplication",
    "ICDM_OG": "I use an invert-and-convert approach where applicable before operating",
    "OTHER": "I use a mixed or uncategorized strategy",
}


STRATEGY_CODES: List[str] = [
    "KDON_AS",
    "KDON_OG",
    "CDON_AS",
    "CDON_OG",
    "ONOD_M",
    "ONOD_OG",
    "CROP_M",
    "ICDM_D",
    "ICDM_OG",
]


GOAL_TEXT: Dict[str, str] = {
    "operate_nums": "I set a goal to operate on the numerators",
    "operate_dens": "I set a goal to operate on the denominators",
    "pass_den": "I set a goal to pass a denominator directly into the answer",
    "convert_CD": "I plan to convert operands to a common denominator",
    "convert_CD_LCM": "I plan to use the least common denominator during conversion",
    "get_LCM": "I compute the least common multiple of the denominators",
    "convert_fra_to_den": "I rewrite a fraction to match a target denominator",
    "invert_op2": "I set a goal to invert the second operand",
    "div_to_mul": "I set a goal to turn division into multiplication",
    "check_simplify": "I check whether the final fraction should be simplified",
    "skip_simplify": "I decide to skip simplification",
    "get_GCD": "I compute the greatest common divisor of numerator and denominator",
    "simplify_fraction": "I simplify the fraction with the GCD",
}


EXEC_TEXT: Dict[str, str] = {
    "add_fact": "I retrieve an addition fact",
    "sub_fact": "I retrieve a subtraction fact",
    "mul_fact": "I retrieve a multiplication fact",
    "div_calculator": "I perform the division directly (calculator-style)",
    "convert_CD_omit_nums": "I change denominators but forget to scale numerators",
    "invert_rand": "I invert a random operand instead of the intended one",
    "invert_fail": "I fail to invert an operand",
    "div_to_mul_denied": "I abandon the division-to-multiplication conversion",
    "acc_skip": "I advance an accumulation step without updating the running total",
    "acc_extra": "I add an extra accumulation action",
    "sub_LbS": "I subtract the smaller value from the larger value",
    "div_LbS": "I divide the larger value by the smaller value",
    "div_LbS_drop_rem": "I divide larger by smaller and drop any remainder",
    "div_drop_rem": "I divide and drop any remainder",
}

NUMERIC_TOKEN_RE = r"[+\-]?(?:\d+(?:/\d+)?|\d*\.\d+)"
BINARY_PROB_RE = re.compile(rf"\s*({NUMERIC_TOKEN_RE})\s*([+\-*/:])\s*({NUMERIC_TOKEN_RE})\s*")

# These rule tokens are intentionally omitted from student-facing narratives.
# Rationale: they are internal/meta states that real students would usually not
# verbalize explicitly in think-aloud explanations.
IGNORED_EXEC_RULES_IN_REASONING = {
    "invert_fail",
}
IGNORED_GOAL_RULES_IN_REASONING = {
    "skip_simplify",
}
SURFACED_HIDDEN_GOAL_SENTENCES: Dict[str, str] = {
    "skip_simplify": "I decided not to simplify the fraction any further",
}
SURFACED_HIDDEN_EXEC_SENTENCES: Dict[str, str] = {
    "invert_fail": "I tried to flip a fraction, but I did not change it",
    "acc_skip": "I moved on without updating the running answer",
    "acc_extra": "I made an extra running-answer update",
}


CANONICAL_COLUMNS: List[str] = [
    "subjid",
    "prob",
    "operation",
    "strategy",
    "goals",
    "exec",
    "answer",
    "correct",
    "is_correct",
    "g",
    "d",
    "rt_mu",
    "ice",
]

MAPPING_COLUMNS: List[str] = [
    "source_dataset",
    "source_file",
    "source_seed",
    "source_row_idx",
    "source_uid",
]


@dataclass(frozen=True)
class SourceSpec:
    path: str
    dataset: str
    seed: Optional[int]


@dataclass(frozen=True)
class StudentPromptConfig:
    mode: str
    drop_prob: float
    drop_seed: int


NLP_OUTPUT_COLUMNS: List[str] = [
    "operation_nl",
    "student_prompt_mode",
    "student_params_visible",
    "student_nl",
    "strategy_nl",
    "goals_nl",
    "exec_nl",
    "instruction_nl",
    "response_nl",
    "reasoning_quality_flags",
    "reasoning_verified_claims",
    "reasoning_unverified_claims",
    "translation_version",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate UMA traces into NLP-style text.")
    parser.add_argument(
        "--input-csv",
        type=str,
        action="append",
        default=[],
        help="Input UMA CSV path (repeatable). Globs are allowed.",
    )
    parser.add_argument(
        "--include-default-uma-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Include default UMA traces file ({DEFAULT_UMA_INPUT}).",
    )
    parser.add_argument(
        "--include-sp2013-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include SP2013 inference outputs from seed-level all_models CSV files.",
    )
    parser.add_argument(
        "--sp2013-eval-dir",
        type=str,
        default=DEFAULT_SP2013_EVAL_DIR,
        help="Directory containing SP2013 seed folders (seed_1..seed_20).",
    )
    parser.add_argument(
        "--sp2013-seeds",
        type=str,
        default=DEFAULT_SP2013_SEEDS,
        help="Seed list/ranges to include from sp2013_eval (e.g. '1-20' or '1,3,5-8').",
    )
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT, help="Path for NLP output CSV/CSV.GZ.")
    parser.add_argument("--chunksize", type=int, default=100000, help="Rows per processing chunk.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick test runs.")
    parser.add_argument(
        "--minimal-columns",
        action="store_true",
        help="Write mapping + canonical UMA fields + NLP fields instead of all source columns.",
    )
    parser.add_argument(
        "--include-outcome-text",
        action="store_true",
        help="Append correctness text to response_nl.",
    )
    parser.add_argument(
        "--student-prompt-mode",
        type=str,
        choices=VALID_STUDENT_PROMPT_MODES,
        default=None,
        help="How to condition prompts on student params: none, always, or deterministic dropout.",
    )
    parser.add_argument(
        "--student-params-drop-prob",
        type=float,
        default=DEFAULT_STUDENT_PARAMS_DROP_PROB,
        help="Row-level probability of dropping the student block when --student-prompt-mode=dropout.",
    )
    parser.add_argument(
        "--student-params-drop-seed",
        type=int,
        default=DEFAULT_STUDENT_PARAMS_DROP_SEED,
        help="Seed mixed into source_uid hashing for deterministic student-block dropout.",
    )
    parser.add_argument(
        "--include-student-params",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy alias for student prompt mode: true -> always, false -> none.",
    )
    parser.add_argument(
        "--reasoning-mode",
        type=str,
        choices=VALID_REASONING_MODES,
        default="trace_or_child",
        help="How to build reasoning text: keep trace-step claims when available, or always use clean child reasoning.",
    )
    parser.add_argument(
        "--row-filter",
        type=str,
        choices=VALID_ROW_FILTER_MODES,
        default="none",
        help="Optional post-translation row filter for cleaned distillation datasets.",
    )
    parser.add_argument(
        "--surface-hidden-trace-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, verbalize internal UMA steps that are hidden in the default humanlike trace.",
    )
    return parser.parse_args()


def safe_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    txt = str(value).strip()
    if txt.lower() == "nan":
        return default
    return txt if txt else default


def resolve_student_prompt_config(args: argparse.Namespace) -> StudentPromptConfig:
    requested_mode = safe_text(args.student_prompt_mode, default="")
    legacy_include = args.include_student_params

    if requested_mode == "":
        mode = "always" if legacy_include is not False else "none"
    else:
        mode = requested_mode
        if legacy_include is not None:
            legacy_mode = "always" if legacy_include else "none"
            if mode != legacy_mode:
                raise ValueError(
                    "Conflicting student prompt settings: "
                    f"--student-prompt-mode {mode} does not match legacy "
                    f"--{'include' if legacy_include else 'no-include'}-student-params."
                )

    drop_prob = float(args.student_params_drop_prob)
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError(f"--student-params-drop-prob must be in [0, 1], got {drop_prob}.")

    return StudentPromptConfig(
        mode=mode,
        drop_prob=drop_prob,
        drop_seed=int(args.student_params_drop_seed),
    )


def safe_float(value: object) -> Optional[float]:
    text = safe_text(value, default="")
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def split_tokens(value: object) -> List[str]:
    text = safe_text(value, default="")
    if text == "":
        return []
    return [tok for tok in text.split() if tok]


def detect_operation(prob: str, op_from_col: str) -> str:
    op = safe_text(op_from_col, default="")
    if op == "/":
        op = ":"
    if op in OPERATION_TEXT:
        return op
    parsed = parse_binary_problem_parts(prob)
    if parsed is not None:
        return parsed[1]
    if re.search(r"\s/\s", str(prob)):
        return ":"
    return "?"


def parse_fraction(text: str) -> Optional[Tuple[str, str]]:
    t = safe_text(text, default="")
    match = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", t)
    if not match:
        return None
    return match.group(1), match.group(2)


def split_fraction_text(text: str) -> Optional[Tuple[str, str]]:
    t = safe_text(text, default="")
    if "/" not in t:
        return None
    left, right = t.split("/", 1)
    left = left.strip()
    right = right.strip()
    if left == "" or right == "":
        return None
    return left, right


def numeric_text_tolerance(text: object) -> float:
    token = safe_text(text, default="")
    if token == "":
        return 1e-9
    if "." in token:
        decimals = len(token.split(".", 1)[1])
        return 0.5 * (10 ** (-decimals)) + 1e-9
    return 1e-9


def numeric_text_matches(expected: Optional[float], observed_text: object) -> Optional[bool]:
    observed = safe_float(observed_text)
    if expected is None or observed is None:
        return None
    return abs(expected - observed) <= numeric_text_tolerance(observed_text)


def parse_binary_problem_parts(prob: str) -> Optional[Tuple[str, str, str]]:
    p = safe_text(prob, default="")
    match = BINARY_PROB_RE.fullmatch(p)
    if not match:
        return None
    left = match.group(1).strip()
    op = match.group(2).strip()
    right = match.group(3).strip()
    if op == "/":
        op = ":"
    return left, op, right


def parse_binary_problem(prob: str, operation: str) -> Tuple[str, str, Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
    p = safe_text(prob, default="")
    if operation not in OPERATION_TEXT:
        return p, "", None, None
    parsed = parse_binary_problem_parts(p)
    if parsed is None:
        return p, "", None, None
    left, parsed_op, right = parsed
    if parsed_op != operation:
        return p, "", None, None
    return left, right, parse_fraction(left), parse_fraction(right)


def render_problem_for_prompt(prob: str, operation: str) -> str:
    left_txt, right_txt, _, _ = parse_binary_problem(prob, operation)
    if left_txt and right_txt:
        op_surface = f" {operation} "
        return f"{left_txt}{op_surface}{right_txt}"
    return safe_text(prob, default="?")


def is_known_math_token(value: object) -> bool:
    token = safe_text(value, default="")
    return token.lower() not in {"", "?", "none", "nan", "null"}


def normalize_trace_problem(problem_obj: object) -> Dict[str, str]:
    if not isinstance(problem_obj, dict):
        return {"op": "?", "op1": "?", "op2": "?", "ans": "?"}
    return {
        "op": safe_text(problem_obj.get("op"), default="?"),
        "op1": safe_text(problem_obj.get("op1"), default="?"),
        "op2": safe_text(problem_obj.get("op2"), default="?"),
        "ans": safe_text(problem_obj.get("ans"), default="?"),
    }


def parse_trace_steps(value: object) -> List[Dict[str, object]]:
    raw = safe_text(value, default="")
    if raw == "":
        return []

    try:
        payload = json.loads(raw)
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    steps: List[Dict[str, object]] = []
    for step_obj in payload:
        if not isinstance(step_obj, dict):
            continue

        main = normalize_trace_problem(step_obj.get("main"))
        subs_raw = step_obj.get("subs", [])
        subs: List[Dict[str, str]] = []
        if isinstance(subs_raw, list):
            for sub_obj in subs_raw:
                subs.append(normalize_trace_problem(sub_obj))

        steps.append(
            {
                "i": step_obj.get("i"),
                "rule": safe_text(step_obj.get("rule"), default=""),
                "main": main,
                "subs": subs,
            }
        )

    return steps


def build_trace_claim(problem_obj: Dict[str, str]) -> str:
    op = safe_text(problem_obj.get("op"), default="?")
    op1 = safe_text(problem_obj.get("op1"), default="?")
    op2 = safe_text(problem_obj.get("op2"), default="?")
    ans = safe_text(problem_obj.get("ans"), default="?")

    if not is_known_math_token(ans):
        return ""

    if op in OPERATION_WORD and is_known_math_token(op1) and is_known_math_token(op2):
        return f"I took {op1} {OPERATION_WORD[op]} {op2} and got {ans}"
    if is_known_math_token(op) and is_known_math_token(op1) and is_known_math_token(op2):
        return f"I worked on {op1} {op} {op2} and got {ans}"
    if is_known_math_token(op1) and is_known_math_token(op2):
        return f"I worked with {op1} and {op2} and got {ans}"
    return f"I got {ans}"


def collect_trace_claims(trace_steps: List[Dict[str, object]], max_claims: int = 8) -> List[str]:
    claims: List[str] = []
    seen = set()

    for step in trace_steps:
        step_items: List[Dict[str, str]] = []
        main = step.get("main")
        if isinstance(main, dict):
            step_items.append(main)  # type: ignore[arg-type]
        subs = step.get("subs", [])
        if isinstance(subs, list):
            for sub in subs:
                if isinstance(sub, dict):
                    step_items.append(sub)  # type: ignore[arg-type]

        for item in step_items:
            claim = build_trace_claim(item)
            if claim == "" or claim in seen:
                continue
            seen.add(claim)
            claims.append(claim)
            if len(claims) >= max_claims:
                return claims

    return claims


def apply_int_operation(
    lhs: int,
    rhs: int,
    operation: str,
    prefer_larger_first: bool = False,
    drop_remainder: bool = False,
) -> Optional[int]:
    if operation == "+":
        return lhs + rhs
    if operation == "-":
        if prefer_larger_first:
            return max(lhs, rhs) - min(lhs, rhs)
        return lhs - rhs
    if operation == "*":
        return lhs * rhs
    if operation == ":":
        num, den = (lhs, rhs)
        if prefer_larger_first:
            num, den = max(lhs, rhs), min(lhs, rhs)
        if den == 0:
            return None
        if drop_remainder:
            return int(num / den)
        if num % den == 0:
            return num // den
        return None
    return None


def apply_numeric_operation(
    lhs: int,
    rhs: int,
    operation: str,
    prefer_larger_first: bool = False,
    drop_remainder: bool = False,
) -> Optional[float]:
    if operation != ":":
        out = apply_int_operation(
            lhs,
            rhs,
            operation,
            prefer_larger_first=prefer_larger_first,
            drop_remainder=drop_remainder,
        )
        return None if out is None else float(out)

    num, den = (lhs, rhs)
    if prefer_larger_first:
        num, den = max(lhs, rhs), min(lhs, rhs)
    if den == 0:
        return None
    if drop_remainder:
        return float(int(num / den))
    return num / den


def render_rule_sequence(tokens: List[str], rule_text: Dict[str, str], unknown_prefix: str) -> str:
    if not tokens:
        return "No explicit rule was logged."

    seen: Dict[str, int] = {}
    steps: List[str] = []
    for token in tokens:
        seen[token] = seen.get(token, 0) + 1
        phrase = rule_text.get(token, f"I apply UMA {unknown_prefix} rule {token}")
        if seen[token] > 1:
            phrase = f"{phrase} again (repeat {seen[token]})"
        steps.append(f"{phrase}.")
    return " ".join(steps)


def to_optional_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = safe_text(value, default="").lower()
    if text in {"1", "1.0", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n"}:
        return False
    return None


def build_outcome_sentence(record: Dict[str, object]) -> str:
    is_correct = to_optional_bool(record.get("is_correct"))
    correct = safe_text(record.get("correct"), default="?")
    answer = safe_text(record.get("answer"), default="?")

    if is_correct is True:
        return f"correct (matches {correct})"
    if is_correct is False:
        return f"incorrect (expected {correct}, got {answer})"
    return "unknown"


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def format_numeric_token(value: object, fallback: str = "unk") -> str:
    val = safe_float(value)
    if val is None:
        return fallback
    if val.is_integer():
        return str(int(val))
    text = f"{val:.10f}".rstrip("0").rstrip(".")
    return text if text != "" else fallback


def build_student_block(record: Dict[str, object]) -> str:
    g_value = format_numeric_token(record.get("g"))
    d_value = format_numeric_token(record.get("d"))
    rt_value = format_numeric_token(record.get("rt_mu"))
    ice_value = format_numeric_token(record.get("ice"))
    return f"<student> g {g_value} d {d_value} rt {rt_value} ice {ice_value} </student>"


def stable_hash_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)


def student_drop_key(record: Dict[str, object]) -> str:
    source_uid = safe_text(record.get("source_uid"), default="")
    if source_uid != "":
        return source_uid
    return "::".join(
        [
            safe_text(record.get("source_file"), default=""),
            safe_text(record.get("source_row_idx"), default=""),
            safe_text(record.get("prob"), default="?"),
            safe_text(record.get("answer"), default="?"),
        ]
    )


def student_params_visible(record: Dict[str, object], prompt_cfg: StudentPromptConfig) -> bool:
    if prompt_cfg.mode == "always":
        return True
    if prompt_cfg.mode == "none":
        return False

    key = f"{prompt_cfg.drop_seed}:{student_drop_key(record)}"
    return stable_hash_fraction(key) >= prompt_cfg.drop_prob


def translation_version(*, surface_hidden_trace_steps: bool) -> str:
    if surface_hidden_trace_steps:
        return f"{BASE_TRANSLATION_VERSION}_with_hidden_steps"
    return BASE_TRANSLATION_VERSION


def prepare_reasoning_rules(
    goals: List[str],
    exec_rules: List[str],
    *,
    surface_hidden_trace_steps: bool,
) -> Tuple[List[str], List[str]]:
    if surface_hidden_trace_steps:
        return list(goals), list(exec_rules)
    exec_rules_for_narration = [r for r in exec_rules if r not in IGNORED_EXEC_RULES_IN_REASONING]
    goals_for_narration = [g for g in goals if g not in IGNORED_GOAL_RULES_IN_REASONING]
    return goals_for_narration, exec_rules_for_narration


def build_reasoning_details(
    goals_for_narration: List[str],
    exec_rules_for_narration: List[str],
    *,
    strategy_code: str = "",
    operation: str = "",
    surface_hidden_trace_steps: bool = False,
) -> List[str]:
    detail_sentences: List[str] = []
    if "add_fact" in exec_rules_for_narration:
        detail_sentences.append("I used an addition fact to get that result")
    if "sub_fact" in exec_rules_for_narration:
        detail_sentences.append("I used a subtraction fact to get that result")
    if "mul_fact" in exec_rules_for_narration:
        detail_sentences.append("I used a multiplication fact to get that result")
    if "convert_CD_omit_nums" in exec_rules_for_narration:
        detail_sentences.append("I changed the bottom numbers and kept the top numbers the same")
    if "invert_rand" in exec_rules_for_narration and not (strategy_code in {"ICDM_D", "ICDM_OG"} and operation == ":"):
        detail_sentences.append("I flipped one of the fractions")
    if "div_to_mul_denied" in exec_rules_for_narration:
        if "invert_op2" in goals_for_narration:
            detail_sentences.append("I thought about flipping the second fraction and changing it to multiplication, but I kept it in this form")
        else:
            detail_sentences.append("I thought about changing it to multiplication, but I kept it in this form")
    if "div_calculator" in exec_rules_for_narration:
        detail_sentences.append("I worked out that division directly")
    if "sub_LbS" in exec_rules_for_narration:
        detail_sentences.append("I subtracted the smaller number from the bigger number")
    if "div_LbS_drop_rem" in exec_rules_for_narration:
        detail_sentences.append("I divided the bigger number by the smaller number and kept only the whole-number part")
    if "div_LbS" in exec_rules_for_narration:
        detail_sentences.append("I divided the bigger number by the smaller number")
    if "div_drop_rem" in exec_rules_for_narration:
        detail_sentences.append("I used the whole-number part after dividing")
    if "simplify_fraction" in goals_for_narration:
        if "get_GCD" in goals_for_narration:
            detail_sentences.append("I checked for a greatest common factor and then used it to simplify it")
        else:
            detail_sentences.append("Then I simplified it")
    elif "check_simplify" in goals_for_narration:
        if "get_GCD" in goals_for_narration:
            detail_sentences.append("I checked for a greatest common factor")
        else:
            detail_sentences.append("I checked if it could be simplified")
    if surface_hidden_trace_steps:
        for goal in goals_for_narration:
            sentence = SURFACED_HIDDEN_GOAL_SENTENCES.get(goal)
            if sentence is not None:
                detail_sentences.append(sentence)
        for rule in exec_rules_for_narration:
            sentence = SURFACED_HIDDEN_EXEC_SENTENCES.get(rule)
            if sentence is not None:
                detail_sentences.append(sentence)
    return detail_sentences


def build_trace_reasoning(
    prob: str,
    goals: List[str],
    exec_rules: List[str],
    answer: str,
    trace_steps: List[Dict[str, object]],
    *,
    surface_hidden_trace_steps: bool = False,
) -> Tuple[str, Dict[str, object]]:
    goals_for_narration, exec_rules_for_narration = prepare_reasoning_rules(
        goals,
        exec_rules,
        surface_hidden_trace_steps=surface_hidden_trace_steps,
    )

    quality_flags: Set[str] = set()
    if any(
        tag in exec_rules_for_narration
        for tag in {"convert_CD_omit_nums", "invert_rand", "sub_LbS", "div_LbS", "div_LbS_drop_rem", "div_drop_rem", "acc_skip", "acc_extra"}
    ):
        quality_flags.add("has_exec_error_rule")

    claims = collect_trace_claims(trace_steps, max_claims=8)
    sentences = dedupe_keep_order(
        claims
        + build_reasoning_details(
            goals_for_narration,
            exec_rules_for_narration,
            surface_hidden_trace_steps=surface_hidden_trace_steps,
        )
    )

    if not sentences:
        left_txt, right_txt, _, _ = parse_binary_problem(prob, detect_operation(prob, ""))
        if left_txt and right_txt:
            sentences.append(f"I worked with {left_txt} and {right_txt} and got {answer}")
        else:
            sentences.append(f"I worked it out and got {answer}")

    if answer not in {"", "?"}:
        last = sentences[-1].lower()
        if answer not in last:
            sentences.append(f"So I got {answer}")

    reasoning = ". ".join(sentence.rstrip(".") for sentence in sentences if sentence.strip()) + "."
    return (
        reasoning,
        {
            "reasoning_quality_flags": ",".join(sorted(quality_flags)),
            "reasoning_verified_claims": len(claims),
            "reasoning_unverified_claims": 0,
        },
    )


def build_child_reasoning(
    prob: str,
    operation: str,
    strategy_code: str,
    goals: List[str],
    exec_rules: List[str],
    answer: str,
    *,
    surface_hidden_trace_steps: bool = False,
) -> Tuple[str, Dict[str, object]]:
    goals_for_narration, exec_rules_for_narration = prepare_reasoning_rules(
        goals,
        exec_rules,
        surface_hidden_trace_steps=surface_hidden_trace_steps,
    )

    op_word = OPERATION_WORD.get(operation, "with")
    left_txt, right_txt, left_frac, right_frac = parse_binary_problem(prob, operation)
    ans_frac = parse_fraction(answer)
    ans_text_frac = split_fraction_text(answer)
    ans_num = ans_frac[0] if ans_frac else (ans_text_frac[0] if ans_text_frac else safe_text(answer, default="?"))
    ans_den = ans_frac[1] if ans_frac else (ans_text_frac[1] if ans_text_frac else None)
    ans_num_int = int(ans_num) if ans_frac else None
    ans_den_int = int(ans_den) if ans_frac else None

    has_larger_first = ("sub_LbS" in exec_rules_for_narration) or ("div_LbS" in exec_rules_for_narration) or ("div_LbS_drop_rem" in exec_rules_for_narration)
    has_drop_remainder = ("div_drop_rem" in exec_rules_for_narration) or ("div_LbS_drop_rem" in exec_rules_for_narration)

    quality_flags: Set[str] = set()
    verified_claims = 0
    unverified_claims = 0

    def note_check_result(matched: Optional[bool]) -> None:
        nonlocal verified_claims, unverified_claims
        if matched is True:
            verified_claims += 1
            return
        unverified_claims += 1
        if matched is False:
            quality_flags.add("arith_claim_mismatch")
        else:
            quality_flags.add("unverified_numeric_claim")

    main_sentences: List[str] = []
    if strategy_code in {"ICDM_D", "ICDM_OG"} and operation == ":" and left_frac and right_frac and ans_den is not None:
        a, b = left_frac
        c, d = right_frac
        if "invert_rand" in exec_rules_for_narration:
            main_sentences.append(f"I flipped one of the fractions, {c}/{d}, to {d}/{c} and changed it to multiplication")
        elif "invert_op2" in goals_for_narration:
            main_sentences.append(f"I flipped the second fraction, {c}/{d}, to {d}/{c} and changed it to multiplication")
        else:
            main_sentences.append(f"I flipped {c}/{d} to {d}/{c} and changed it to multiplication")
        top_prod = int(a) * int(d)
        bot_prod = int(b) * int(c)
        if ans_num_int is not None and ans_den_int is not None and top_prod == ans_num_int and bot_prod == ans_den_int:
            note_check_result(True)
            note_check_result(True)
            main_sentences.append(f"Then I did {a} times {d} and got {ans_num}, and {b} times {c} and got {ans_den}")
        else:
            note_check_result(False if ans_num_int is not None and ans_den_int is not None else None)
            note_check_result(False if ans_num_int is not None and ans_den_int is not None else None)
            main_sentences.append("Then I multiplied across to form a new top and bottom")
    elif strategy_code in {"ICDM_D", "ICDM_OG"} and left_frac and right_frac and ans_den is not None:
        main_sentences.append(
            f"I worked on the top numbers and on the bottom numbers separately and got {ans_num}/{ans_den}"
        )
    elif strategy_code.startswith("CDON"):
        common_den = ans_den or (left_frac[1] if left_frac else "?")
        common_den_int: Optional[int] = None
        if left_frac and right_frac:
            b_val = int(left_frac[1])
            d_val = int(right_frac[1])
            if b_val != 0 and d_val != 0:
                common_den_int = (abs(b_val * d_val)) // math.gcd(abs(b_val), abs(d_val))
                common_den = str(common_den_int)
        if "get_LCM" in goals_for_narration:
            main_sentences.append(f"I found the least common multiple of the denominators, {common_den}, first")
        elif "convert_CD_LCM" in goals_for_narration:
            main_sentences.append(f"I used the least common denominator, {common_den}, first")
        else:
            main_sentences.append(f"I found a common denominator, {common_den}, first")
        if left_frac and right_frac:
            a, b = left_frac
            c, d = right_frac
            left_conv_num: Optional[int] = None
            right_conv_num: Optional[int] = None
            if common_den_int is not None:
                left_conv_num = int(a) * (common_den_int // int(b))
                right_conv_num = int(c) * (common_den_int // int(d))
            if (
                "convert_CD" in goals_for_narration
                and "convert_fra_to_den" not in goals_for_narration
                and common_den_int is not None
            ):
                main_sentences.append(f"I changed both fractions to use denominator {common_den}")
            if "convert_fra_to_den" in goals_for_narration and common_den_int is not None:
                if "convert_CD_omit_nums" in exec_rules_for_narration:
                    main_sentences.append(f"I tried to change both fractions to use denominator {common_den}")
                elif left_conv_num is not None and right_conv_num is not None:
                    main_sentences.append(
                        f"I rewrote {a}/{b} as {left_conv_num}/{common_den} and {c}/{d} as {right_conv_num}/{common_den}"
                    )
            lhs_num = int(a)
            rhs_num = int(c)
            if operation in {"+", "-", ":"} and common_den_int is not None:
                lhs_num *= common_den_int // int(b)
                rhs_num *= common_den_int // int(d)
            num_result = apply_int_operation(
                lhs_num,
                rhs_num,
                operation,
                prefer_larger_first=has_larger_first and operation in {"-", ":"},
                drop_remainder=has_drop_remainder and operation == ":",
            )
            numeric_result = apply_numeric_operation(
                lhs_num,
                rhs_num,
                operation,
                prefer_larger_first=has_larger_first and operation in {"-", ":"},
                drop_remainder=has_drop_remainder and operation == ":",
            )
            answer_match: Optional[bool] = None
            if num_result is not None:
                if operation in {"+", "-"} and common_den_int is not None and ans_den_int not in {None, 0} and ans_num_int is not None:
                    answer_match = (num_result * ans_den_int) == (ans_num_int * common_den_int)
                elif ans_num_int is not None:
                    answer_match = num_result == ans_num_int
            if answer_match is None:
                answer_match = numeric_text_matches(numeric_result, ans_num)
            if answer_match is True:
                note_check_result(True)
                explicit_keep_den = (
                    "pass_den" in goals_for_narration
                    and (
                        operation in {"+", "-", ":"}
                        or (operation == "*" and ans_den is not None)
                    )
                )
                keep_den = common_den if operation in {"+", "-", ":"} else ans_den
                if (
                    operation in {"+", "-"}
                    and common_den_int is not None
                    and ans_den_int not in {None, 0}
                    and ans_num_int is not None
                    and (num_result != ans_num_int or common_den_int != ans_den_int)
                ):
                    if explicit_keep_den and keep_den is not None and num_result is not None:
                        main_sentences.append(
                            f"Then I took {lhs_num} {op_word} {rhs_num} and got {num_result}, and I kept {keep_den} on the bottom"
                        )
                    else:
                        main_sentences.append(
                            f"Then I took {lhs_num} {op_word} {rhs_num} and got {num_result}/{common_den}"
                        )
                elif explicit_keep_den and keep_den is not None:
                    main_sentences.append(
                        f"Then I took {lhs_num} {op_word} {rhs_num} and got {ans_num}, and I kept {keep_den} on the bottom"
                    )
                else:
                    main_sentences.append(f"Then I took {lhs_num} {op_word} {rhs_num} and got {ans_num}")
            else:
                if operation == ":":
                    main_sentences.append(f"Then I worked on the top numbers and kept {common_den} on the bottom")
                else:
                    note_check_result(answer_match)
                    main_sentences.append(f"Then I combined the top numbers with {op_word} and kept {common_den} on the bottom")
    elif strategy_code.startswith("KDON"):
        if left_frac and right_frac and ans_den is not None:
            a, b = left_frac
            c, d = right_frac
            keep_den = b if b == d else ans_den
            num_result = apply_int_operation(
                int(a),
                int(c),
                operation,
                prefer_larger_first=has_larger_first and operation in {"-", ":"},
                drop_remainder=has_drop_remainder and operation == ":",
            )
            numeric_result = apply_numeric_operation(
                int(a),
                int(c),
                operation,
                prefer_larger_first=has_larger_first and operation in {"-", ":"},
                drop_remainder=has_drop_remainder and operation == ":",
            )
            answer_match: Optional[bool] = None
            if ans_num_int is not None and num_result is not None:
                answer_match = num_result == ans_num_int
            if answer_match is None:
                answer_match = numeric_text_matches(numeric_result, ans_num)
            if answer_match is True:
                note_check_result(True)
                main_sentences.append(f"I took {a} {op_word} {c} and got {ans_num}, and I kept {keep_den} on the bottom")
            else:
                note_check_result(answer_match)
                main_sentences.append(f"I combined the top numbers and kept {keep_den} on the bottom")
    elif strategy_code.startswith("ONOD"):
        if left_frac and right_frac and ans_den is not None:
            a, b = left_frac
            c, d = right_frac
            top_result = apply_int_operation(
                int(a),
                int(c),
                operation,
                prefer_larger_first=has_larger_first and operation in {"-", ":"},
                drop_remainder=has_drop_remainder and operation == ":",
            )
            bottom_result = apply_int_operation(
                int(b),
                int(d),
                operation,
                prefer_larger_first=has_larger_first and operation in {"-", ":"},
                drop_remainder=has_drop_remainder and operation == ":",
            )
            top_numeric = apply_numeric_operation(
                int(a),
                int(c),
                operation,
                prefer_larger_first=has_larger_first and operation in {"-", ":"},
                drop_remainder=has_drop_remainder and operation == ":",
            )
            bottom_numeric = apply_numeric_operation(
                int(b),
                int(d),
                operation,
                prefer_larger_first=has_larger_first and operation in {"-", ":"},
                drop_remainder=has_drop_remainder and operation == ":",
            )
            top_match: Optional[bool] = None
            bot_match: Optional[bool] = None
            if ans_num_int is not None and top_result is not None:
                top_match = top_result == ans_num_int
            if ans_den_int is not None and bottom_result is not None:
                bot_match = bottom_result == ans_den_int
            if top_match is None:
                top_match = numeric_text_matches(top_numeric, ans_num)
            if bot_match is None:
                bot_match = numeric_text_matches(bottom_numeric, ans_den)
            if top_match is True and bot_match is True:
                note_check_result(True)
                note_check_result(True)
                main_sentences.append(f"I took {a} {op_word} {c} and got {ans_num}, and {b} {op_word} {d} and got {ans_den}")
            else:
                note_check_result(top_match)
                note_check_result(bot_match)
                main_sentences.append(f"I applied {op_word} to top numbers and bottom numbers separately")
    elif strategy_code == "CROP_M":
        if left_frac and right_frac and ans_den is not None:
            a, b = left_frac
            c, d = right_frac
            top_prod = int(a) * int(d)
            bot_prod = int(b) * int(c)
            if ans_num_int is not None and ans_den_int is not None and top_prod == ans_num_int and bot_prod == ans_den_int:
                note_check_result(True)
                note_check_result(True)
                main_sentences.append(f"I crossed them and did {a} times {d} to get {ans_num}, and {b} times {c} to get {ans_den}")
            else:
                note_check_result(False if ans_num_int is not None and ans_den_int is not None else None)
                note_check_result(False if ans_num_int is not None and ans_den_int is not None else None)
                main_sentences.append("I crossed the fractions and multiplied to form a new top and bottom")

    if not main_sentences:
        if left_frac and right_frac:
            main_sentences.append(f"I worked with {left_txt} and {right_txt} and got {answer}")
        else:
            main_sentences.append(f"I worked it out and got {answer}")

    detail_sentences = build_reasoning_details(
        goals_for_narration,
        exec_rules_for_narration,
        strategy_code=strategy_code,
        operation=operation,
        surface_hidden_trace_steps=surface_hidden_trace_steps,
    )

    if any(
        tag in exec_rules_for_narration
        for tag in {"convert_CD_omit_nums", "invert_rand", "sub_LbS", "div_LbS", "div_LbS_drop_rem", "div_drop_rem", "acc_skip", "acc_extra"}
    ):
        quality_flags.add("has_exec_error_rule")

    sentences = dedupe_keep_order(main_sentences + detail_sentences)
    if answer not in {"", "?"}:
        last = sentences[-1].lower()
        if answer not in last:
            sentences.append(f"So I got {answer}")

    reasoning = ". ".join(sentence.rstrip(".") for sentence in sentences if sentence.strip()) + "."
    quality = {
        "reasoning_quality_flags": ",".join(sorted(quality_flags)),
        "reasoning_verified_claims": verified_claims,
        "reasoning_unverified_claims": unverified_claims,
    }
    return reasoning, quality


def infer_strategy_from_flags(chunk: pd.DataFrame) -> pd.Series:
    present = [code for code in STRATEGY_CODES if code in chunk.columns]
    if not present:
        return pd.Series("OTHER", index=chunk.index, dtype="object")

    flags = chunk[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    gt0 = flags > 0
    has_any = gt0.any(axis=1)
    first = gt0.idxmax(axis=1)
    inferred = pd.Series("OTHER", index=chunk.index, dtype="object")
    inferred.loc[has_any] = first.loc[has_any]
    return inferred


def missing_value_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    cleaned = text.str.strip()
    return text.isna() | cleaned.isna() | (cleaned == "") | cleaned.str.lower().isin(["nan", "none"])


def canonicalize_chunk(chunk: pd.DataFrame, source: SourceSpec, row_offset: int) -> pd.DataFrame:
    out = chunk.copy()

    n_rows = len(out)
    out["source_dataset"] = source.dataset
    out["source_file"] = source.path
    out["source_seed"] = source.seed if source.seed is not None else pd.NA
    out["source_row_idx"] = range(row_offset, row_offset + n_rows)
    out["source_uid"] = out["source_file"].astype(str) + "::" + out["source_row_idx"].astype(str)

    for column in CANONICAL_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA

    text_columns = ["prob", "operation", "strategy", "goals", "exec", "answer", "correct"]
    for column in text_columns:
        out[column] = out[column].astype("string")

    if "ans" in out.columns:
        missing_answer = missing_value_mask(out["answer"])
        out.loc[missing_answer, "answer"] = out.loc[missing_answer, "ans"].astype("string")

    strategy_missing = missing_value_mask(out["strategy"])
    if strategy_missing.any():
        inferred = infer_strategy_from_flags(out)
        out.loc[strategy_missing, "strategy"] = inferred.loc[strategy_missing]

    for column in ["goals", "exec"]:
        missing = missing_value_mask(out[column])
        if missing.any():
            out.loc[missing, column] = ""

    op_missing_or_bad = missing_value_mask(out["operation"]) | (~out["operation"].astype(str).str.strip().isin(OPERATION_TEXT.keys()))
    if op_missing_or_bad.any():
        probs = out.loc[op_missing_or_bad, "prob"].astype(str)
        out.loc[op_missing_or_bad, "operation"] = probs.map(lambda p: detect_operation(p, ""))

    if "key" in out.columns:
        missing_correct = missing_value_mask(out["correct"])
        out.loc[missing_correct, "correct"] = out.loc[missing_correct, "key"].astype("string")

    if "acc" in out.columns:
        missing_is_correct = missing_value_mask(out["is_correct"])
        out.loc[missing_is_correct, "is_correct"] = out.loc[missing_is_correct, "acc"]

    return out


def translate_record(
    record: Dict[str, object],
    include_outcome_text: bool,
    student_prompt_config: StudentPromptConfig,
    reasoning_mode: str,
    surface_hidden_trace_steps: bool,
) -> Dict[str, object]:
    prob = safe_text(record.get("prob"), default="?")
    strategy_code = safe_text(record.get("strategy"), default="OTHER")
    if strategy_code == "":
        strategy_code = "OTHER"

    goals = split_tokens(record.get("goals"))
    exec_rules = split_tokens(record.get("exec"))
    answer = safe_text(record.get("answer"), default="?")
    operation = detect_operation(prob, safe_text(record.get("operation"), default=""))
    op_text = OPERATION_TEXT.get(operation, "unknown operation")
    prompt_prob = render_problem_for_prompt(prob, operation)

    strategy_nl = STRATEGY_TEXT.get(strategy_code, f"I use an uncataloged strategy pattern ({strategy_code})")
    goals_nl = render_rule_sequence(goals, GOAL_TEXT, "goal")
    exec_nl = render_rule_sequence(exec_rules, EXEC_TEXT, "execution")
    params_visible = student_params_visible(record, student_prompt_config)
    student_nl = build_student_block(record) if params_visible else ""
    if params_visible:
        instruction_nl = f"{student_nl}\nSolve this fraction problem: {prompt_prob}=?"
    else:
        instruction_nl = f"Solve this fraction problem: {prompt_prob}=?"
    trace_steps = parse_trace_steps(record.get("trace_steps_json"))
    if reasoning_mode == "clean_child":
        reasoning_nl, quality = build_child_reasoning(
            prob=prob,
            operation=operation,
            strategy_code=strategy_code,
            goals=goals,
            exec_rules=exec_rules,
            answer=answer,
            surface_hidden_trace_steps=surface_hidden_trace_steps,
        )
    elif len(trace_steps) > 0:
        reasoning_nl, quality = build_trace_reasoning(
            prob=prob,
            goals=goals,
            exec_rules=exec_rules,
            answer=answer,
            trace_steps=trace_steps,
            surface_hidden_trace_steps=surface_hidden_trace_steps,
        )
    else:
        reasoning_nl, quality = build_child_reasoning(
            prob=prob,
            operation=operation,
            strategy_code=strategy_code,
            goals=goals,
            exec_rules=exec_rules,
            answer=answer,
            surface_hidden_trace_steps=surface_hidden_trace_steps,
        )
    response_lines = [reasoning_nl, f"### answer: {answer}"]
    if include_outcome_text:
        response_lines.append(f"### correctness: {build_outcome_sentence(record)}")
    response_nl = "\n".join(response_lines)

    return {
        "operation_nl": op_text,
        "student_prompt_mode": student_prompt_config.mode,
        "student_params_visible": params_visible,
        "student_nl": student_nl,
        "strategy_nl": strategy_nl,
        "goals_nl": goals_nl,
        "exec_nl": exec_nl,
        "instruction_nl": instruction_nl,
        "response_nl": response_nl,
        "reasoning_quality_flags": quality["reasoning_quality_flags"],
        "reasoning_verified_claims": quality["reasoning_verified_claims"],
        "reasoning_unverified_claims": quality["reasoning_unverified_claims"],
        "translation_version": translation_version(surface_hidden_trace_steps=surface_hidden_trace_steps),
    }


def translate_chunk(
    chunk: pd.DataFrame,
    include_outcome_text: bool,
    student_prompt_config: StudentPromptConfig,
    reasoning_mode: str,
    surface_hidden_trace_steps: bool,
) -> pd.DataFrame:
    records = chunk.to_dict(orient="records")
    translated = [
        translate_record(
            record,
            include_outcome_text=include_outcome_text,
            student_prompt_config=student_prompt_config,
            reasoning_mode=reasoning_mode,
            surface_hidden_trace_steps=surface_hidden_trace_steps,
        )
        for record in records
    ]
    return pd.DataFrame(translated)


def truthy_mask(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "1.0", "true", "t", "yes", "y"})


def apply_row_filter(frame: pd.DataFrame, row_filter: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if row_filter == "none" or len(frame) == 0:
        return frame, {
            "rows_in": int(len(frame)),
            "rows_kept": int(len(frame)),
            "rows_dropped": 0,
        }

    quality_flags = frame["reasoning_quality_flags"].fillna("").astype(str)
    unverified = pd.to_numeric(frame["reasoning_unverified_claims"], errors="coerce").fillna(0).astype(int)
    keep_mask = (unverified == 0) & (~quality_flags.str.contains("arith_claim_mismatch", regex=False))

    if row_filter == "correct_exec_and_answer":
        keep_mask &= truthy_mask(frame["is_correct"])

    out = frame.loc[keep_mask].reset_index(drop=True)
    return out, {
        "rows_in": int(len(frame)),
        "rows_kept": int(len(out)),
        "rows_dropped": int(len(frame) - len(out)),
    }


def parse_seed_ranges(seed_expr: str) -> List[int]:
    text = safe_text(seed_expr, default="")
    if text == "":
        return []

    seeds = set()
    for part in text.split(","):
        token = part.strip()
        if token == "":
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if start > end:
                start, end = end, start
            for value in range(start, end + 1):
                seeds.add(value)
        else:
            seeds.add(int(token))

    return sorted(seeds)


def classify_dataset(path: str) -> Tuple[str, Optional[int]]:
    base = os.path.basename(path)
    match = re.fullmatch(r"sp2013_seed(\d+)_all_models\.csv", base)
    if match:
        return "sp2013_eval_seed", int(match.group(1))
    if "uma_traces_all" in base:
        return "uma_traces_all", None
    return "uma_trace_custom", None


def expand_input_paths(raw_paths: List[str]) -> List[str]:
    expanded: List[str] = []
    for raw in raw_paths:
        token = safe_text(raw, default="")
        if token == "":
            continue
        matches = sorted(glob.glob(token))
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(token)
    return expanded


def build_source_specs(args: argparse.Namespace) -> List[SourceSpec]:
    specs: List[SourceSpec] = []

    if args.include_default_uma_traces:
        specs.append(SourceSpec(path=DEFAULT_UMA_INPUT, dataset="uma_traces_all", seed=None))

    for path in expand_input_paths(args.input_csv):
        abs_path = os.path.abspath(path)
        dataset, seed = classify_dataset(abs_path)
        specs.append(SourceSpec(path=abs_path, dataset=dataset, seed=seed))

    if args.include_sp2013_seeds:
        sp_root = os.path.abspath(args.sp2013_eval_dir)
        for seed in parse_seed_ranges(args.sp2013_seeds):
            path = os.path.join(sp_root, f"seed_{seed}", f"sp2013_seed{seed}_all_models.csv")
            specs.append(SourceSpec(path=os.path.abspath(path), dataset="sp2013_eval_seed", seed=seed))

    dedup: Dict[str, SourceSpec] = {}
    for spec in specs:
        dedup[spec.path] = spec

    final_specs = list(dedup.values())
    if not final_specs:
        raise ValueError("No input sources were resolved. Check --input-csv/seed options.")

    missing = [spec.path for spec in final_specs if not os.path.exists(spec.path)]
    if missing:
        raise FileNotFoundError(
            "Missing input source files:\n" + "\n".join(f"  - {path}" for path in missing)
        )

    return final_specs


def collect_union_source_columns(sources: List[SourceSpec]) -> List[str]:
    ordered: Dict[str, None] = {}
    for source in sources:
        header_cols = pd.read_csv(source.path, nrows=0).columns.tolist()
        for column in header_cols:
            ordered.setdefault(column, None)

    for column in MAPPING_COLUMNS + CANONICAL_COLUMNS:
        ordered.setdefault(column, None)

    return list(ordered.keys())


def build_base_output_columns(all_source_columns: List[str], minimal_columns: bool) -> List[str]:
    if minimal_columns:
        return MAPPING_COLUMNS + CANONICAL_COLUMNS

    base = list(all_source_columns)
    for column in MAPPING_COLUMNS + CANONICAL_COLUMNS:
        if column not in base:
            base.append(column)
    return base


def main() -> None:
    args = parse_args()
    student_prompt_config = resolve_student_prompt_config(args)
    output_csv = os.path.abspath(args.output_csv)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sources = build_source_specs(args)
    all_source_columns = collect_union_source_columns(sources)
    base_columns = build_base_output_columns(all_source_columns, minimal_columns=args.minimal_columns)

    print("=" * 70)
    print("UMA -> NLP TRACE TRANSLATION")
    print("=" * 70)
    print(f"output: {output_csv}")
    print(f"chunksize: {args.chunksize}")
    print(f"max_rows: {args.max_rows if args.max_rows is not None else 'ALL'}")
    print(f"minimal_columns: {args.minimal_columns}")
    print(f"include_outcome_text: {args.include_outcome_text}")
    print(f"student_prompt_mode: {student_prompt_config.mode}")
    print(f"student_params_drop_prob: {student_prompt_config.drop_prob}")
    print(f"student_params_drop_seed: {student_prompt_config.drop_seed}")
    print(f"reasoning_mode: {args.reasoning_mode}")
    print(f"row_filter: {args.row_filter}")
    if args.include_student_params is not None:
        print(f"legacy_include_student_params: {args.include_student_params}")
    print(f"surface_hidden_trace_steps: {args.surface_hidden_trace_steps}")
    print(f"translation_version: {translation_version(surface_hidden_trace_steps=args.surface_hidden_trace_steps)}")
    print(f"input_sources: {len(sources)}")
    for idx, source in enumerate(sources, start=1):
        seed_text = f", seed={source.seed}" if source.seed is not None else ""
        print(f"  [{idx}] {source.dataset}{seed_text} -> {source.path}")
    print("")

    wrote_header = False
    total_rows = 0
    total_rows_kept = 0

    for source_idx, source in enumerate(sources, start=1):
        source_rows = 0
        reader = pd.read_csv(source.path, chunksize=args.chunksize)

        for chunk_idx, chunk in enumerate(reader, start=1):
            if args.max_rows is not None and total_rows >= args.max_rows:
                break

            if args.max_rows is not None:
                remaining = args.max_rows - total_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()

            normalized = canonicalize_chunk(chunk, source=source, row_offset=source_rows)
            source_rows += len(normalized)

            translated = translate_chunk(
                normalized,
                include_outcome_text=args.include_outcome_text,
                student_prompt_config=student_prompt_config,
                reasoning_mode=args.reasoning_mode,
                surface_hidden_trace_steps=args.surface_hidden_trace_steps,
            )
            base = normalized.reindex(columns=base_columns).reset_index(drop=True)
            out_chunk = pd.concat([base, translated], axis=1)
            out_chunk, filter_stats = apply_row_filter(out_chunk, args.row_filter)

            if len(out_chunk) == 0:
                total_rows += len(normalized)
                print(
                    f"[source {source_idx}/{len(sources)} chunk {chunk_idx}] "
                    f"filtered out all {filter_stats['rows_in']:,} rows "
                    f"(source_total {source_rows:,}, global_seen {total_rows:,})"
                )
                continue

            mode = "w" if not wrote_header else "a"
            out_chunk.to_csv(output_csv, index=False, mode=mode, header=not wrote_header, compression="infer")

            wrote_header = True
            total_rows += len(normalized)
            total_rows_kept += len(out_chunk)
            print(
                f"[source {source_idx}/{len(sources)} chunk {chunk_idx}] "
                f"kept {len(out_chunk):,}/{filter_stats['rows_in']:,} rows "
                f"(source_total {source_rows:,}, global_seen {total_rows:,}, global_kept {total_rows_kept:,})"
            )

        if args.max_rows is not None and total_rows >= args.max_rows:
            break

    if not wrote_header:
        pd.DataFrame(columns=base_columns + NLP_OUTPUT_COLUMNS).to_csv(
            output_csv,
            index=False,
            mode="w",
            header=True,
            compression="infer",
        )

    print("")
    print(f"Done. Total translated rows seen: {total_rows:,}")
    print(f"Done. Total translated rows kept: {total_rows_kept:,}")
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
