#!/usr/bin/env python3
"""Build problem-attempt trace-firstshot datasets from Cognitive Tutor logs.

The output mirrors the lightweight human NLP CSV shape used elsewhere in this
repo, but the correctness label is derived from tutor step-level first attempts:
an attempt is correct only when every logged step has Correct First Attempt == 1.
For firstshot-correct attempts, the response target is a conservatively derived
final answer when one can be computed. For non-firstshot attempts, the response
target is the first step where the student got stuck, and the visible trace is
truncated at that step.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_DATASET = "algebra_2006_2007"
DEFAULT_INPUT_ROOT = Path("cognitive_tutor/data/extracted")
DEFAULT_OUT_DIR = Path("cognitive_tutor/data/processed/algebra_2006_2007")
DEFAULT_SPLITS = ("train", "master")

OUTPUT_COLUMNS = [
    "subjid",
    "prob",
    "problem_hierarchy",
    "problem_name",
    "problem_view",
    "resp",
    "target_kind",
    "acc",
    "num_steps",
    "num_original_steps",
    "final_observed_step",
    "derived_final_answer",
    "first_wrong_step_index",
    "first_wrong_step",
    "first_wrong_incorrects",
    "first_wrong_hints",
    "source_rows",
    "trace",
    "strategy",
    "instruction_nl",
    "response_nl",
]

REQUIRED_COLUMNS = [
    "Row",
    "Anon Student Id",
    "Problem Hierarchy",
    "Problem Name",
    "Problem View",
    "Step Name",
    "Step Start Time",
    "First Transaction Time",
    "Correct First Attempt",
    "Incorrects",
    "Hints",
]


@dataclass(frozen=True)
class StepRecord:
    row_id: int
    step_name: str
    step_start_time: str
    first_transaction_time: str
    correct_first_attempt: str
    incorrects: int
    hints: int


def safe_int(value: object, default: int = 0) -> int:
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def step_sort_key(step: StepRecord) -> tuple[int, str, str, int]:
    """Sort by timestamp when present, otherwise by source row id."""
    if step.step_start_time:
        return (0, step.step_start_time, step.first_transaction_time, step.row_id)
    if step.first_transaction_time:
        return (0, step.first_transaction_time, step.first_transaction_time, step.row_id)
    return (1, "", "", step.row_id)


def is_math_like_step(step_name: str) -> bool:
    """Conservative math-like predicate used for the clean subset."""
    text = str(step_name).strip()
    if not any(ch.isdigit() for ch in text):
        return False
    lowered = text.lower()
    return "sqrt" in lowered or any(marker in text for marker in ("=", "+", "-", "*", "/", "^"))


def normalize_math_text(text: str) -> str:
    expr = str(text).strip()
    expr = expr.replace("^", "**")
    expr = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", expr)
    expr = re.sub(r"(?<=\))(?=[A-Za-z0-9(])", "*", expr)
    expr = re.sub(r"(?<=[A-Za-z])(?=\d)", "*", expr)
    expr = re.sub(r"(?<![A-Za-z])([A-Za-z])(?=\()", r"\1*", expr)
    return expr


NumericValue = Fraction | float


def combine_numeric(left: NumericValue, right: NumericValue, op: type[ast.operator]) -> NumericValue:
    if isinstance(left, float) or isinstance(right, float):
        left_f = float(left)
        right_f = float(right)
        if op is ast.Add:
            return left_f + right_f
        if op is ast.Sub:
            return left_f - right_f
        if op is ast.Mult:
            return left_f * right_f
        if op is ast.Div:
            return left_f / right_f
        if op is ast.Pow:
            return left_f**right_f
    else:
        if op is ast.Add:
            return left + right
        if op is ast.Sub:
            return left - right
        if op is ast.Mult:
            return left * right
        if op is ast.Div:
            if right == 0:
                raise ZeroDivisionError
            return left / right
        if op is ast.Pow:
            if right.denominator != 1:
                return float(left) ** float(right)
            exponent = int(right)
            if abs(exponent) > 12:
                return float(left) ** exponent
            return left**exponent
    raise ValueError(f"Unsupported operator: {op}")


def eval_numeric_ast(node: ast.AST) -> NumericValue:
    if isinstance(node, ast.Expression):
        return eval_numeric_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value)) if isinstance(node.value, float) else Fraction(node.value)
    if isinstance(node, ast.UnaryOp):
        value = eval_numeric_ast(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        return combine_numeric(eval_numeric_ast(node.left), eval_numeric_ast(node.right), type(node.op))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1 and not node.keywords:
            return math.sqrt(float(eval_numeric_ast(node.args[0])))
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


def eval_numeric_expr(text: str) -> NumericValue | None:
    normalized = normalize_math_text(text)
    if re.sub(r"sqrt", "", normalized, flags=re.IGNORECASE).isalpha():
        return None
    if re.search(r"[A-Za-z]", re.sub(r"sqrt", "", normalized, flags=re.IGNORECASE)):
        return None
    if not re.fullmatch(r"[0-9\s+\-*/().]*sqrt[0-9\s+\-*/().]*|[0-9\s+\-*/().]+", normalized):
        return None
    try:
        return eval_numeric_ast(ast.parse(normalized, mode="eval"))
    except Exception:
        return None


def format_numeric_value(value: NumericValue | None) -> str:
    if value is None:
        return ""
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    if math.isfinite(value):
        if abs(value - round(value)) < 1e-10:
            return str(int(round(value)))
        return f"{value:.12g}"
    return ""


@lru_cache(maxsize=200_000)
def derive_final_answer(step_name: str) -> str:
    """Conservatively derive a final answer from a math-like final step."""
    text = str(step_name).strip()
    if not is_math_like_step(text) or "?" in text:
        return ""
    normalized = normalize_math_text(text)
    if normalized.count("=") != 1:
        return format_numeric_value(eval_numeric_expr(normalized))

    lhs_text, rhs_text = [piece.strip() for piece in normalized.split("=", 1)]
    variable = r"[A-Za-z]"
    if re.fullmatch(variable, lhs_text):
        return format_numeric_value(eval_numeric_expr(rhs_text))
    if re.fullmatch(variable, rhs_text):
        return format_numeric_value(eval_numeric_expr(lhs_text))

    match = re.fullmatch(rf"({variable})/(.+)", rhs_text)
    if match:
        lhs_value = eval_numeric_expr(lhs_text)
        denominator = eval_numeric_expr(match.group(2))
        if lhs_value is not None and denominator is not None:
            return format_numeric_value(combine_numeric(lhs_value, denominator, ast.Mult))
    match = re.fullmatch(rf"({variable})\\*(.+)", rhs_text)
    if match:
        lhs_value = eval_numeric_expr(lhs_text)
        coefficient = eval_numeric_expr(match.group(2))
        if lhs_value is not None and coefficient not in (None, 0):
            return format_numeric_value(combine_numeric(lhs_value, coefficient, ast.Div))
    match = re.fullmatch(rf"(.+)/({variable})", rhs_text)
    if match:
        lhs_value = eval_numeric_expr(lhs_text)
        numerator = eval_numeric_expr(match.group(1))
        if lhs_value not in (None, 0) and numerator is not None:
            return format_numeric_value(combine_numeric(numerator, lhs_value, ast.Div))

    match = re.fullmatch(rf"(.+)/({variable})", lhs_text)
    if match:
        rhs_value = eval_numeric_expr(rhs_text)
        numerator = eval_numeric_expr(match.group(1))
        if rhs_value not in (None, 0) and numerator is not None:
            return format_numeric_value(combine_numeric(numerator, rhs_value, ast.Div))
    match = re.fullmatch(rf"({variable})/(.+)", lhs_text)
    if match:
        rhs_value = eval_numeric_expr(rhs_text)
        denominator = eval_numeric_expr(match.group(2))
        if rhs_value is not None and denominator is not None:
            return format_numeric_value(combine_numeric(rhs_value, denominator, ast.Mult))
    match = re.fullmatch(rf"({variable})\\*(.+)", lhs_text)
    if match:
        rhs_value = eval_numeric_expr(rhs_text)
        coefficient = eval_numeric_expr(match.group(2))
        if rhs_value is not None and coefficient not in (None, 0):
            return format_numeric_value(combine_numeric(rhs_value, coefficient, ast.Div))

    lhs_value = eval_numeric_expr(lhs_text)
    rhs_value = eval_numeric_expr(rhs_text)
    if lhs_value is not None and rhs_value is not None and float(lhs_value) == float(rhs_value):
        return format_numeric_value(rhs_value)
    return ""


def build_strategy(step_names: Sequence[str]) -> str:
    return "\n".join(f"Step {idx}: {step}" for idx, step in enumerate(step_names, start=1))


def build_output_row(
    key: tuple[str, str, str, str],
    steps: Sequence[StepRecord],
) -> dict[str, object]:
    subjid, problem_hierarchy, problem_name, problem_view = key
    ordered_steps = sorted(steps, key=step_sort_key)
    original_step_names = [step.step_name for step in ordered_steps]
    final_observed_step = original_step_names[-1] if original_step_names else ""
    acc = int(bool(ordered_steps) and all(step.correct_first_attempt == "1" for step in ordered_steps))
    derived_final_answer = derive_final_answer(final_observed_step) if acc else ""
    first_wrong_index = 0
    first_wrong_step = ""
    first_wrong_incorrects = 0
    first_wrong_hints = 0
    for idx, step in enumerate(ordered_steps, start=1):
        if step.correct_first_attempt != "1":
            first_wrong_index = idx
            first_wrong_step = step.step_name
            first_wrong_incorrects = step.incorrects
            first_wrong_hints = step.hints
            break

    if acc:
        if derived_final_answer:
            resp = derived_final_answer
            target_kind = "derived_final_answer"
        else:
            resp = final_observed_step
            target_kind = "final_observed_step"
    else:
        resp = first_wrong_step
        target_kind = "stuck_step"

    visible_steps = ordered_steps if acc else ordered_steps[:first_wrong_index]
    step_names = [step.step_name for step in visible_steps]
    source_rows = [step.row_id for step in visible_steps]
    strategy = build_strategy(step_names)
    instruction_nl = f"Solve this algebra problem: {problem_name}."
    if acc and target_kind == "derived_final_answer":
        target_line = f"### answer: {resp}"
    elif acc:
        target_line = f"### final observed step: {resp}"
    else:
        target_line = f"### answer: {resp}"
    response_nl = f"{strategy}\n{target_line}" if strategy else target_line
    return {
        "subjid": subjid,
        "prob": problem_name,
        "problem_hierarchy": problem_hierarchy,
        "problem_name": problem_name,
        "problem_view": problem_view,
        "resp": resp,
        "target_kind": target_kind,
        "acc": acc,
        "num_steps": len(visible_steps),
        "num_original_steps": len(ordered_steps),
        "final_observed_step": final_observed_step,
        "derived_final_answer": derived_final_answer,
        "first_wrong_step_index": first_wrong_index,
        "first_wrong_step": first_wrong_step,
        "first_wrong_incorrects": first_wrong_incorrects,
        "first_wrong_hints": first_wrong_hints,
        "source_rows": json.dumps(source_rows, separators=(",", ":")),
        "trace": json.dumps(step_names, ensure_ascii=False, separators=(",", ":")),
        "strategy": strategy,
        "instruction_nl": instruction_nl,
        "response_nl": response_nl,
    }


def quantiles(values: Iterable[int]) -> dict[str, float | None]:
    seq = sorted(int(v) for v in values)
    if not seq:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p99": None,
            "max": None,
            "mean": None,
        }

    def pick(p: float) -> float:
        if len(seq) == 1:
            return float(seq[0])
        idx = p * (len(seq) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return float(seq[lo])
        weight = idx - lo
        return float(seq[lo] * (1.0 - weight) + seq[hi] * weight)

    return {
        "min": float(seq[0]),
        "p25": pick(0.25),
        "median": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "p99": pick(0.99),
        "max": float(seq[-1]),
        "mean": float(statistics.fmean(seq)),
    }


def read_attempts(path: Path) -> tuple[dict[tuple[str, str, str, str], list[StepRecord]], int]:
    attempts: dict[tuple[str, str, str, str], list[StepRecord]] = defaultdict(list)
    row_count = 0
    with path.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")

        for row in reader:
            row_count += 1
            key = (
                row["Anon Student Id"].strip(),
                row["Problem Hierarchy"].strip(),
                row["Problem Name"].strip(),
                row["Problem View"].strip(),
            )
            attempts[key].append(
                StepRecord(
                    row_id=safe_int(row["Row"], default=row_count),
                    step_name=row["Step Name"].strip(),
                    step_start_time=row["Step Start Time"].strip(),
                    first_transaction_time=row["First Transaction Time"].strip(),
                    correct_first_attempt=row["Correct First Attempt"].strip(),
                    incorrects=safe_int(row["Incorrects"]),
                    hints=safe_int(row["Hints"]),
                )
            )
    return dict(attempts), row_count


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    num_steps = [safe_int(row["num_steps"]) for row in rows]
    acc_values = [safe_int(row["acc"]) for row in rows]
    target_kind_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["target_kind"])
        target_kind_counts[key] = target_kind_counts.get(key, 0) + 1
    return {
        "attempts": int(len(rows)),
        "accuracy_rate": float(statistics.fmean(acc_values)) if acc_values else None,
        "firstshot_correct_attempts": int(sum(acc_values)),
        "target_kind_counts": target_kind_counts,
        "derived_final_answer_attempts": int(sum(1 for row in rows if str(row["derived_final_answer"]).strip())),
        "step_count_quantiles": quantiles(num_steps),
    }


def build_split(dataset: str, split: str, input_root: Path, out_dir: Path) -> dict[str, object]:
    input_path = input_root / dataset / f"{dataset}_{split}.txt"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input split: {input_path}")

    attempts, input_rows = read_attempts(input_path)
    complete_rows = [
        build_output_row(key, steps)
        for key, steps in sorted(attempts.items(), key=lambda item: min(step.row_id for step in item[1]))
    ]
    math_only_rows = [
        row
        for row in complete_rows
        if row["num_steps"] and all(is_math_like_step(step) for step in json.loads(str(row["trace"])))
    ]

    complete_path = out_dir / f"trace_firstshot_{split}.csv"
    math_only_path = out_dir / f"trace_firstshot_math_only_{split}.csv"
    write_rows(complete_path, complete_rows)
    write_rows(math_only_path, math_only_rows)

    complete_summary = summarize_rows(complete_rows)
    math_only_summary = summarize_rows(math_only_rows)
    return {
        "input_path": str(input_path),
        "input_rows": int(input_rows),
        "complete_output": str(complete_path),
        "math_only_output": str(math_only_path),
        "complete": complete_summary,
        "math_only": {
            **math_only_summary,
            "retained_attempt_rate": (
                math_only_summary["attempts"] / complete_summary["attempts"]
                if complete_summary["attempts"]
                else None
            ),
        },
    }


def build_trace_firstshot_outputs(
    dataset: str = DEFAULT_DATASET,
    splits: Sequence[str] = DEFAULT_SPLITS,
    input_root: Path = DEFAULT_INPUT_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = {
        split: build_split(dataset=dataset, split=split, input_root=input_root, out_dir=out_dir)
        for split in splits
    }
    audit = {
        "dataset": dataset,
        "definition": (
            "One row is one student/problem-hierarchy/problem-name/problem-view attempt. "
            "acc is 1 iff every original step has Correct First Attempt == 1. "
            "For acc=1 rows, resp is a conservatively derived final answer when available, "
            "otherwise the final observed step. For acc=0 rows, resp is the first step "
            "with Correct First Attempt != 1."
        ),
        "input_root": str(input_root),
        "out_dir": str(out_dir),
        "splits": split_summaries,
        "columns": OUTPUT_COLUMNS,
        "math_only_predicate": (
            "All original step names in the attempt contain at least one digit and "
            "at least one of =, +, -, *, /, ^, or sqrt."
        ),
    }
    audit_path = out_dir / "trace_firstshot_audit.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    audit["audit_path"] = str(audit_path)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build trace-firstshot problem-attempt datasets from Cognitive Tutor logs."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_trace_firstshot_outputs(
        dataset=args.dataset,
        splits=args.splits,
        input_root=args.input_root,
        out_dir=args.out_dir,
    )
    for split, summary in audit["splits"].items():
        complete = summary["complete"]
        math_only = summary["math_only"]
        print(
            f"{split}: {complete['attempts']} attempts, "
            f"acc={complete['accuracy_rate']:.4f}, "
            f"math_only={math_only['attempts']} attempts "
            f"({math_only['retained_attempt_rate']:.4f})",
            flush=True,
        )
    print(f"audit_json: {audit['audit_path']}", flush=True)


if __name__ == "__main__":
    main()
