#!/usr/bin/env python3
"""
Translate UMA symbolic traces into deterministic NLP-style training text.

This script is intended as a preprocessing step before transformer training.
It preserves the source trace fields and appends natural-language columns.

Example:
  python results/transformer_replication/translate_uma_traces_to_nlp.py \
    --input-csv results/UMA_replication/uma_traces_all.csv \
    --output-csv results/transformer_replication/uma_traces_all_nlp.csv.gz
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "UMA_replication", "uma_traces_all.csv"))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "uma_traces_all_nlp.csv.gz")
TRANSLATION_VERSION = "uma_nlp_rules_v2_child_style"


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

# These rule tokens are intentionally omitted from student-facing narratives.
# Rationale: they are internal/meta states that real students would usually not
# verbalize explicitly in think-aloud explanations.
IGNORED_EXEC_RULES_IN_REASONING = {
    "invert_fail",
}
IGNORED_GOAL_RULES_IN_REASONING = {
    "skip_simplify",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate UMA traces into NLP-style text.")
    parser.add_argument("--input-csv", type=str, default=DEFAULT_INPUT, help="Path to UMA trace CSV.")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT, help="Path for NLP output CSV/CSV.GZ.")
    parser.add_argument("--chunksize", type=int, default=100000, help="Rows per processing chunk.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick test runs.")
    parser.add_argument(
        "--minimal-columns",
        action="store_true",
        help="Write only key metadata columns + NLP columns instead of all source columns.",
    )
    parser.add_argument(
        "--include-outcome-text",
        action="store_true",
        help="Append correctness text to response_nl.",
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


def split_tokens(value: object) -> List[str]:
    text = safe_text(value, default="")
    if text == "":
        return []
    return [tok for tok in text.split() if tok]


def detect_operation(prob: str, op_from_col: str) -> str:
    op = safe_text(op_from_col, default="")
    if op in OPERATION_TEXT:
        return op
    if "*" in prob:
        return "*"
    if ":" in prob:
        return ":"
    if "+" in prob:
        return "+"
    if "-" in prob:
        return "-"
    return "?"


def parse_fraction(text: str) -> Optional[Tuple[str, str]]:
    t = safe_text(text, default="")
    match = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", t)
    if not match:
        return None
    return match.group(1), match.group(2)


def parse_binary_problem(prob: str, operation: str) -> Tuple[str, str, Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
    p = safe_text(prob, default="")
    if operation not in OPERATION_TEXT:
        return p, "", None, None
    parts = p.split(operation, 1)
    if len(parts) != 2:
        return p, "", None, None
    left = parts[0].strip()
    right = parts[1].strip()
    return left, right, parse_fraction(left), parse_fraction(right)


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
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
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


def build_child_reasoning(
    prob: str,
    operation: str,
    strategy_code: str,
    goals: List[str],
    exec_rules: List[str],
    answer: str,
) -> str:
    # Hide selected internal UMA tokens from the narrated trace.
    exec_rules_for_narration = [r for r in exec_rules if r not in IGNORED_EXEC_RULES_IN_REASONING]
    goals_for_narration = [g for g in goals if g not in IGNORED_GOAL_RULES_IN_REASONING]

    op_word = OPERATION_WORD.get(operation, "with")
    left_txt, right_txt, left_frac, right_frac = parse_binary_problem(prob, operation)
    ans_frac = parse_fraction(answer)
    ans_num = ans_frac[0] if ans_frac else safe_text(answer, default="?")
    ans_den = ans_frac[1] if ans_frac else None

    main_sentences: List[str] = []
    if strategy_code in {"ICDM_D", "ICDM_OG"} and operation == ":" and left_frac and right_frac and ans_den is not None:
        a, b = left_frac
        c, d = right_frac
        main_sentences.append(f"I flipped {c}/{d} to {d}/{c} and changed it to multiplication")
        main_sentences.append(f"Then I did {a} times {d} and got {ans_num}, and {b} times {c} and got {ans_den}")
    elif strategy_code.startswith("CDON"):
        common_den = ans_den or (left_frac[1] if left_frac else "?")
        main_sentences.append(f"I found a common denominator, {common_den}, first")
        if left_frac and right_frac and ans_den is not None:
            a, _ = left_frac
            c, _ = right_frac
            main_sentences.append(f"Then I took {a} {op_word} {c} and got {ans_num}")
    elif strategy_code.startswith("KDON"):
        if left_frac and right_frac and ans_den is not None:
            a, b = left_frac
            c, d = right_frac
            keep_den = b if b == d else ans_den
            main_sentences.append(f"I took {a} {op_word} {c} and got {ans_num}, and I kept {keep_den} on the bottom")
    elif strategy_code.startswith("ONOD"):
        if left_frac and right_frac and ans_den is not None:
            a, b = left_frac
            c, d = right_frac
            main_sentences.append(f"I took {a} {op_word} {c} and got {ans_num}, and {b} {op_word} {d} and got {ans_den}")
    elif strategy_code == "CROP_M":
        if left_frac and right_frac and ans_den is not None:
            a, b = left_frac
            c, d = right_frac
            main_sentences.append(f"I crossed them and did {a} times {d} to get {ans_num}, and {b} times {c} to get {ans_den}")

    if not main_sentences:
        if left_frac and right_frac:
            main_sentences.append(f"I worked with {left_txt} and {right_txt} and got {answer}")
        else:
            main_sentences.append(f"I worked it out and got {answer}")

    detail_sentences: List[str] = []
    if "convert_CD_omit_nums" in exec_rules_for_narration:
        detail_sentences.append("I changed the bottom numbers and kept the top numbers the same")
    if "invert_rand" in exec_rules_for_narration and strategy_code not in {"ICDM_D", "ICDM_OG"}:
        detail_sentences.append("I flipped one of the fractions")
    if "sub_LbS" in exec_rules_for_narration:
        detail_sentences.append("I subtracted the smaller number from the bigger number")
    if "div_LbS" in exec_rules_for_narration or "div_LbS_drop_rem" in exec_rules_for_narration:
        detail_sentences.append("I divided the bigger number by the smaller number")
    if "div_drop_rem" in exec_rules_for_narration or "div_LbS_drop_rem" in exec_rules_for_narration:
        detail_sentences.append("I used the whole-number part after dividing")
    if "simplify_fraction" in goals_for_narration:
        detail_sentences.append("Then I simplified it")
    elif "check_simplify" in goals_for_narration:
        detail_sentences.append("I checked if it could be simplified")

    sentences = dedupe_keep_order(main_sentences + detail_sentences)
    if answer not in {"", "?"}:
        last = sentences[-1].lower()
        if answer not in last:
            sentences.append(f"So I got {answer}")

    return ". ".join(sentence.rstrip(".") for sentence in sentences if sentence.strip()) + "."


def translate_record(record: Dict[str, object], include_outcome_text: bool) -> Dict[str, str]:
    prob = safe_text(record.get("prob"), default="?")
    strategy_code = safe_text(record.get("strategy"), default="OTHER")
    if strategy_code == "":
        strategy_code = "OTHER"

    goals = split_tokens(record.get("goals"))
    exec_rules = split_tokens(record.get("exec"))
    answer = safe_text(record.get("answer"), default="?")
    operation = detect_operation(prob, safe_text(record.get("operation"), default=""))
    op_text = OPERATION_TEXT.get(operation, "unknown operation")

    strategy_nl = STRATEGY_TEXT.get(strategy_code, f"I use an uncataloged strategy pattern ({strategy_code})")
    goals_nl = render_rule_sequence(goals, GOAL_TEXT, "goal")
    exec_nl = render_rule_sequence(exec_rules, EXEC_TEXT, "execution")
    student_nl = ""

    instruction_nl = f"Solve this fraction problem: {prob}=?"
    reasoning_nl = build_child_reasoning(
        prob=prob,
        operation=operation,
        strategy_code=strategy_code,
        goals=goals,
        exec_rules=exec_rules,
        answer=answer,
    )
    response_lines = [reasoning_nl, f"### answer: {answer}"]
    if include_outcome_text:
        response_lines.append(f"### correctness: {build_outcome_sentence(record)}")
    response_nl = "\n".join(response_lines)

    return {
        "operation_nl": op_text,
        "student_nl": student_nl,
        "strategy_nl": strategy_nl,
        "goals_nl": goals_nl,
        "exec_nl": exec_nl,
        "instruction_nl": instruction_nl,
        "response_nl": response_nl,
        "translation_version": TRANSLATION_VERSION,
    }


def translate_chunk(chunk: pd.DataFrame, include_outcome_text: bool) -> pd.DataFrame:
    records = chunk.to_dict(orient="records")
    translated = [translate_record(record, include_outcome_text=include_outcome_text) for record in records]
    return pd.DataFrame(translated)


def output_columns(chunk: pd.DataFrame, minimal_columns: bool) -> pd.DataFrame:
    if not minimal_columns:
        return chunk.reset_index(drop=True)

    keep = ["subjid", "prob", "operation", "strategy", "goals", "exec", "answer", "correct", "is_correct", "g", "d", "rt_mu", "ice"]
    present = [col for col in keep if col in chunk.columns]
    return chunk[present].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_csv = os.path.abspath(args.input_csv)
    output_csv = os.path.abspath(args.output_csv)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("UMA -> NLP TRACE TRANSLATION")
    print("=" * 70)
    print(f"input: {input_csv}")
    print(f"output: {output_csv}")
    print(f"chunksize: {args.chunksize}")
    print(f"max_rows: {args.max_rows if args.max_rows is not None else 'ALL'}")
    print(f"minimal_columns: {args.minimal_columns}")
    print(f"include_outcome_text: {args.include_outcome_text}")
    print("")

    reader = pd.read_csv(input_csv, chunksize=args.chunksize)
    wrote_header = False
    total_rows = 0

    for chunk_idx, chunk in enumerate(reader, start=1):
        if args.max_rows is not None and total_rows >= args.max_rows:
            break
        if args.max_rows is not None:
            remaining = args.max_rows - total_rows
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

        translated = translate_chunk(chunk, include_outcome_text=args.include_outcome_text)
        base = output_columns(chunk, minimal_columns=args.minimal_columns)
        out_chunk = pd.concat([base, translated], axis=1)

        mode = "w" if not wrote_header else "a"
        out_chunk.to_csv(output_csv, index=False, mode=mode, header=not wrote_header, compression="infer")

        wrote_header = True
        total_rows += len(chunk)
        print(f"[chunk {chunk_idx}] wrote {len(chunk):,} rows (total {total_rows:,})")

    print("")
    print(f"Done. Total translated rows: {total_rows:,}")
    print(f"Translation version: {TRANSLATION_VERSION}")


if __name__ == "__main__":
    main()
