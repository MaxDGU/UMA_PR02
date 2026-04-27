#!/usr/bin/env python3
"""Build human fine-tuning CSVs containing only numerically correct responses."""

from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from train_transformer_hf import answers_match, compute_correct_answer, op_from_prob  # type: ignore  # noqa: E402
except (ImportError, SyntaxError):
    # Keep this CLI usable from lightweight login-node Python environments. These
    # functions intentionally mirror train_transformer_hf's numeric answer logic.
    NUMERIC_TOKEN_RE = r"[+\-]?(?:\d+(?:/\d+)?|\d*\.\d+)"
    PROB_RE = re.compile(rf"\s*({NUMERIC_TOKEN_RE})\s*([+\-*/:])\s*({NUMERIC_TOKEN_RE})\s*")

    def canonicalize_division_op(op: str) -> str:
        return ":" if str(op).strip() == "/" else str(op).strip()

    def display_operation_symbol(op: str) -> str:
        return "/" if canonicalize_division_op(op) == ":" else canonicalize_division_op(op)

    def parse_binary_problem(prob: str) -> Optional[Tuple[str, str, str]]:
        match = PROB_RE.fullmatch(str(prob).strip())
        if match is None:
            return None
        return match.group(1), canonicalize_division_op(match.group(2)), match.group(3)

    def op_from_prob(prob: str) -> str:
        parsed = parse_binary_problem(prob)
        if parsed is None:
            return "?"
        return display_operation_symbol(parsed[1])

    def parse_numeric_token(token: str) -> Fraction:
        text = str(token).strip()
        if "/" in text:
            num_s, den_s = text.split("/", 1)
            return Fraction(int(num_s), int(den_s))
        return Fraction(text)

    def compute_correct_answer(prob: str) -> str:
        parsed = parse_binary_problem(prob)
        if parsed is None:
            return "?"
        left_s, op, right_s = parsed
        try:
            left = parse_numeric_token(left_s)
            right = parse_numeric_token(right_s)
            if op == "+":
                ans = left + right
            elif op == "-":
                ans = left - right
            elif op == "*":
                ans = left * right
            elif op == ":":
                if right == 0:
                    return "?"
                ans = left / right
            else:
                return "?"
            return str(ans.numerator) if ans.denominator == 1 else f"{ans.numerator}/{ans.denominator}"
        except Exception:
            return "?"

    def normalize_answer(ans: str) -> str:
        text = str(ans).strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*/\s*", "/", text)
        text = text.rstrip(".,;:!?")
        return text.strip()

    def answer_to_fraction(ans: str) -> Optional[Fraction]:
        text = normalize_answer(ans)
        if text == "" or text in {"?", "nan", "None"}:
            return None
        try:
            mixed = re.fullmatch(r"([+-]?\d+)\s+(\d+)/(\d+)", text)
            if mixed is not None:
                whole = int(mixed.group(1))
                num = int(mixed.group(2))
                den = int(mixed.group(3))
                if den == 0:
                    return None
                frac = Fraction(num, den)
                return Fraction(whole) - frac if whole < 0 else Fraction(whole) + frac
            if "/" in text:
                a, b = text.split("/", 1)
                den = Fraction(b)
                if den == 0:
                    return None
                return Fraction(a) / den
            return Fraction(text)
        except Exception:
            return None

    def answer_to_float(ans: str) -> Optional[float]:
        frac = answer_to_fraction(ans)
        if frac is None:
            return None
        return float(frac)

    def answers_match(pred: str, correct: str, tol: float = 1e-6) -> bool:
        pred_val = answer_to_float(pred)
        correct_val = answer_to_float(correct)
        if pred_val is None or correct_val is None:
            return False
        return abs(pred_val - correct_val) < tol


TRAIN_OUT_NAME = "data_train_nlp_correct.csv"
VAL_OUT_NAME = "data_val_nlp_correct.csv"
AUDIT_OUT_NAME = "human_correct_only_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter human NLP train/val CSVs down to rows whose final answer is numerically correct."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("results/human/data_train_nlp.csv"),
        help="Input human train CSV with at least prob and resp columns.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("results/human/data_val_nlp.csv"),
        help="Input human validation CSV with at least prob and resp columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/human/correct_only"),
        help="Directory for filtered CSVs and the audit JSON.",
    )
    return parser.parse_args()


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [col for col in ("prob", "resp") if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")


def mark_correctness(df: pd.DataFrame) -> pd.DataFrame:
    marked = df.copy()
    marked["_correct_answer"] = marked["prob"].map(compute_correct_answer)
    marked["_operation"] = marked["prob"].map(op_from_prob)
    marked["_is_correct_final_answer"] = [
        bool(answers_match(resp, correct)) if correct != "?" else False
        for resp, correct in zip(marked["resp"], marked["_correct_answer"])
    ]
    return marked


def summarize_marked(marked: pd.DataFrame) -> dict[str, Any]:
    total_rows = int(len(marked))
    kept_rows = int(marked["_is_correct_final_answer"].sum())
    dropped_rows = total_rows - kept_rows
    by_operation: dict[str, dict[str, Any]] = {}

    for op, group in marked.groupby("_operation", dropna=False, sort=True):
        op_key = str(op)
        op_total = int(len(group))
        op_kept = int(group["_is_correct_final_answer"].sum())
        by_operation[op_key] = {
            "total_rows": op_total,
            "kept_rows": op_kept,
            "dropped_rows": op_total - op_kept,
            "kept_rate": float(op_kept / op_total) if op_total else 0.0,
        }

    return {
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "dropped_rows": dropped_rows,
        "kept_rate": float(kept_rows / total_rows) if total_rows else 0.0,
        "by_operation": by_operation,
    }


def filter_split(input_csv: Path, output_csv: Path) -> dict[str, Any]:
    df = read_csv_flexible(input_csv)
    require_columns(df, input_csv)
    marked = mark_correctness(df)
    filtered = marked.loc[marked["_is_correct_final_answer"], df.columns].copy()
    filtered.to_csv(output_csv, index=False)

    summary = summarize_marked(marked)
    summary["input_csv"] = str(input_csv)
    summary["output_csv"] = str(output_csv)
    return summary


def build_correct_only_outputs(train_csv: Path, val_csv: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / TRAIN_OUT_NAME
    val_out = out_dir / VAL_OUT_NAME
    audit_out = out_dir / AUDIT_OUT_NAME

    audit = {
        "definition": "Rows are kept when resp is numerically equivalent to the exact answer computed from prob.",
        "splits": {
            "train": filter_split(train_csv, train_out),
            "val": filter_split(val_csv, val_out),
        },
    }
    with audit_out.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    audit["audit_json"] = str(audit_out)
    return audit


def main() -> None:
    args = parse_args()
    audit = build_correct_only_outputs(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        out_dir=args.out_dir,
    )
    for split_name, split in audit["splits"].items():
        print(
            f"{split_name}: kept {split['kept_rows']}/{split['total_rows']} "
            f"rows -> {split['output_csv']}",
            flush=True,
        )
    print(f"audit_json: {audit['audit_json']}", flush=True)


if __name__ == "__main__":
    main()
