"""Extract and validate fraction strategy codes from reasoning traces.

The default workflow validates a Gemini extraction prompt against the human
fraction fine-tuning data in ``data/human_ft``. The module is also usable on
model rollout CSVs by passing ``--input_csv`` and the relevant column names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.strategy_mapping import (
    CANONICAL_STRATEGY_CODEBOOK,
    CANONICAL_STRATEGY_CODES,
    map_human_fraction_code_to_canonical,
)

STRATEGY_CODES: Tuple[str, ...] = CANONICAL_STRATEGY_CODES
PROMPT_VERSION = "fraction_strategy_extraction_v7"
DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_BATCH_INLINE_LIMIT_BYTES = 20 * 1024 * 1024
DEFAULT_DECIMAL_CSV = "fractionGPT/decimals/bss2021_human_responses.csv"
DEFAULT_OUT_DIR = EVAL_DIR / "results" / "procedural_alignment" / "fraction_strategy_extraction"
DEFAULT_GEMINI_INPUT_USD_PER_MILLION = 0.10
DEFAULT_GEMINI_OUTPUT_USD_PER_MILLION = 0.40
EVIDENCE_CHOICES: Tuple[str, ...] = ("auto", "trace", "work", "combined")
RESOLVED_EVIDENCE_CHOICES: Tuple[str, ...] = ("trace", "work", "combined")

STRATEGY_CODEBOOK: Mapping[str, str] = CANONICAL_STRATEGY_CODEBOOK
DECIMAL_STRATEGY_CODES: Tuple[str, ...] = ("AS", "M", "BOTH", "O")
DECIMAL_STRATEGY_CODEBOOK: Mapping[str, str] = {
    "AS": (
        "Addition-style decimal strategy. The student aligns decimal points, "
        "pads with zeros, or places the answer decimal as in aligned operands."
    ),
    "M": (
        "Multiplication-style decimal strategy. The student ignores decimal "
        "points at first, aligns rightmost digits, multiplies whole-number-like "
        "digits, or places the decimal by counting decimal places."
    ),
    "BOTH": "The work shows evidence of both addition-style and multiplication-style decimal strategies.",
    "O": "Other, unclear, no visible work, or insufficient evidence.",
}

ANSWER_PATTERN_HINTS: Mapping[Tuple[str, str], str] = {
    ("3/5+1/2", "4/7"): "ONOD",
    ("3/5-1/2", "2/3"): "ONOD",
    ("3/5*1/2", "3/10"): "ONOD",
    ("3/5*2/5", "6/25"): "ONOD",
    ("3/5+2/5", "5/10"): "ONOD",
    ("3/5-2/5", "1/1"): "ONOD",
    ("3/5:1/2", "3/2"): "ONOD",
    ("3/5:2/5", "1/1"): "ONOD",
    ("3/5+1/2", "11/10"): "CDON",
    ("3/5+1/2", "1 1/10"): "CDON",
    ("3/5-1/2", "1/10"): "CDON",
    ("3/5*1/2", "30/10"): "CDON",
    ("3/5*2/5", "150/25"): "CDON",
    ("3/5+2/5", "5/5"): "KDON",
    ("3/5+2/5", "1"): "KDON",
}

CANONICAL_FEW_SHOT_EXAMPLES: Tuple[Mapping[str, str], ...] = (
    {
        "strategy_code": "KDON",
        "problem": "3/5+2/5",
        "answer": "5/5",
        "reasoning_trace": "I added 3 plus 2 to get 5 and kept the denominator 5.",
    },
    {
        "strategy_code": "CDON",
        "problem": "3/5+1/2",
        "answer": "11/10",
        "reasoning_trace": "I changed 3/5 to 6/10 and 1/2 to 5/10, then added 6 plus 5 and kept 10.",
    },
    {
        "strategy_code": "ONOD",
        "problem": "3/5+1/2",
        "answer": "4/7",
        "reasoning_trace": "I added 3 plus 1 to get 4 and 5 plus 2 to get 7.",
    },
    {
        "strategy_code": "OTHER",
        "problem": "3/5+1/2",
        "answer": "11/20",
        "reasoning_trace": "I changed 3/5 to 6/10 and 1/2 to 5/10, then added 6 plus 5 and 10 plus 10.",
    },
    {
        "strategy_code": "ICDM",
        "problem": "3/5:1/2",
        "answer": "6/5",
        "reasoning_trace": "I flipped 1/2 to 2/1, then multiplied 3 times 2 and 5 times 1.",
    },
    {
        "strategy_code": "CROP",
        "problem": "3/5:1/2",
        "answer": "6/5",
        "reasoning_trace": "I cross multiplied 3 times 2 for the top and 5 times 1 for the bottom.",
    },
    {
        "strategy_code": "OTHER",
        "problem": "3/5*1/2",
        "answer": "3/4",
        "reasoning_trace": "I was not sure what to do, so I just guessed.",
    },
    {
        "strategy_code": "OTHER",
        "problem": "3/5:1/2",
        "answer": "1.2",
        "reasoning_trace": "I changed the fractions to decimals and divided them.",
    },
    {
        "strategy_code": "KDON",
        "problem": "3/5-2/5",
        "answer": "1/5",
        "reasoning_trace": "The denominators were already the same, so I did 3 minus 2 and kept 5.",
    },
    {
        "strategy_code": "CDON",
        "problem": "3/5*1/2",
        "answer": "30/10",
        "reasoning_trace": "I made 3/5 into 6/10 and 1/2 into 5/10, then multiplied 6 times 5 and kept 10.",
    },
    {
        "strategy_code": "ONOD",
        "problem": "3/5-2/5",
        "answer": "1/5",
        "reasoning_trace": "I did 3 minus 2 to get 1, and I thought 5 minus 5 was 0 but typed 5 for the denominator.",
    },
    {
        "strategy_code": "CROP",
        "problem": "3/5+1/2",
        "answer": "6/5",
        "reasoning_trace": "I added 5 and 1 for the top number, then added 2 and 3 for the bottom number.",
    },
)

RESPONSE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "strategy_code": {"type": "string", "enum": list(STRATEGY_CODES)},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["strategy_code", "confidence", "evidence"],
}


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    title: str
    prompt_version: str
    strategy_codes: Tuple[str, ...]
    codebook: Mapping[str, str]
    normalize_code: Callable[[Any], Optional[str]]
    user_template: str

    @property
    def response_schema(self) -> Mapping[str, Any]:
        if self.name == "decimal":
            return {
                "type": "object",
                "properties": {
                    "addition_style_evidence": {"type": "boolean"},
                    "multiplication_style_evidence": {"type": "boolean"},
                    "insufficient_evidence": {"type": "boolean"},
                    "direct_procedure_evidence": {"type": "boolean"},
                    "strategy_inferred_from_answer": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": [
                    "addition_style_evidence",
                    "multiplication_style_evidence",
                    "insufficient_evidence",
                    "direct_procedure_evidence",
                    "strategy_inferred_from_answer",
                    "confidence",
                    "evidence",
                ],
            }
        return {
            "type": "object",
            "properties": {
                "strategy_code": {"type": "string", "enum": list(self.strategy_codes)},
                "confidence": {"type": "number"},
                "evidence": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["strategy_code", "confidence", "evidence"],
        }


FRACTION_USER_TEMPLATE = """Problem: {problem}
Student final answer: {answer}
Diagnostic answer-pattern hint: {optional_hint}
Reasoning trace:
{reasoning_trace}

Classify the reasoning trace with one allowed strategy code. Return only JSON."""

DECIMAL_USER_TEMPLATE = """Problem: {problem}
Operation: {operation}
Operand 1 shown in problem: {op1}
Operand 2 shown in problem: {op2}
Student final answer: {answer}

Visible-work derived summary:
{work_summary}

Coded visible-work cue summary:
{coded_cue_summary}

Visible written work:
Written-work layout: {work_layout_description}
Operand 1 as written: {work_operand_1}
Operand 2 as written: {work_operand_2}
Answer as written: {work_answer}

Flag the visible decimal strategy cues. Return only JSON."""

DECIMAL_TRACE_USER_TEMPLATE = """Problem: {problem}
Operation: {operation}
Operand 1 shown in problem: {op1}
Operand 2 shown in problem: {op2}
Student final answer: {answer}

Reasoning trace:
{reasoning_trace}

Flag the described decimal strategy cues, including any described operand-alignment
and answer decimal-placement cues. Return only JSON."""

DECIMAL_COMBINED_USER_TEMPLATE = """Problem: {problem}
Operation: {operation}
Operand 1 shown in problem: {op1}
Operand 2 shown in problem: {op2}
Student final answer: {answer}

Reasoning trace:
{reasoning_trace}

Visible-work derived summary:
{work_summary}

Coded visible-work cue summary:
{coded_cue_summary}

Visible written work:
Written-work layout: {work_layout_description}
Operand 1 as written: {work_operand_1}
Operand 2 as written: {work_operand_2}
Answer as written: {work_answer}

Flag the student's decimal strategy cues, including any described or visible
operand-alignment and answer decimal-placement cues. Return only JSON."""


def repo_root() -> Path:
    return ROOT_DIR


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def normalize_code(value: Any) -> Optional[str]:
    return map_human_fraction_code_to_canonical(value)


def normalize_decimal_code(value: Any) -> Optional[str]:
    text = clean_text(value).upper()
    if not text:
        return None
    aliases = {
        "ADD": "AS",
        "ADDITION": "AS",
        "ADDITION_STYLE": "AS",
        "ADDITION-STYLE": "AS",
        "MUL": "M",
        "MULTIPLICATION": "M",
        "MULTIPLICATION_STYLE": "M",
        "MULTIPLICATION-STYLE": "M",
        "OTHER": "O",
        "NONE": "O",
        "UNCLEAR": "O",
    }
    text = aliases.get(text, text)
    return text if text in DECIMAL_STRATEGY_CODES else None


def parse_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    text = clean_text(value).lower()
    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return True
    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return False
    return None


def derive_decimal_code_from_cues(
    addition_style_evidence: Optional[bool],
    multiplication_style_evidence: Optional[bool],
    insufficient_evidence: Optional[bool],
) -> Optional[str]:
    if addition_style_evidence is None or multiplication_style_evidence is None:
        return None
    if insufficient_evidence is None:
        return None
    if insufficient_evidence:
        return "O"
    if addition_style_evidence and multiplication_style_evidence:
        return "BOTH"
    if addition_style_evidence:
        return "AS"
    if multiplication_style_evidence:
        return "M"
    return "O"


def decimal_work_layout_description(value: Any) -> str:
    text = clean_text(value)
    if text in {"0", "0.0"}:
        return "bare answer only"
    if text in {"1", "1.0"}:
        return "vertical column work"
    if text in {"2", "2.0"}:
        return "other written layout"
    return ""


def decimal_work_alignment_description(value: Any) -> str:
    text = clean_text(value)
    if text in {"0", "0.0"}:
        return "no clear operand alignment"
    if text in {"1", "1.0"}:
        return "decimal points aligned (addition-style alignment cue)"
    if text in {"2", "2.0"}:
        return "rightmost digits aligned (multiplication-style alignment cue)"
    if text in {"3", "3.0"}:
        return "other or unclear operand alignment"
    return ""


def decimal_dplace_description(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    normalized = text.lower()
    if normalized == "add":
        return "answer decimal placed as in aligned operands (addition-style placement cue)"
    if normalized == "mul":
        return "answer decimal placed by counting decimal digits (multiplication-style placement cue)"
    if normalized == "none":
        return "no clear answer decimal-placement cue"
    return ""


def build_decimal_coded_cue_summary(row: Mapping[str, Any]) -> str:
    alignment = clean_text(row.get("work_alignment_description", ""))
    placement = clean_text(row.get("decimal_placement_description", ""))
    if not alignment and not placement:
        return ""
    lines = [
        f"- Operand alignment cue: {alignment or 'not provided'}",
        f"- Answer decimal-placement cue: {placement or 'not provided'}",
    ]
    return "\n".join(lines)


def parse_decimal_number(value: Any) -> Optional[Decimal]:
    text = clean_text(value).replace(",", "")
    if not text:
        return None
    if text.startswith("."):
        text = f"0{text}"
    if text.endswith("."):
        text = text[:-1]
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def decimal_place_count(value: Any) -> Optional[int]:
    text = clean_text(value).replace(",", "")
    if not text:
        return None
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def digits_only(value: Any) -> str:
    return re.sub(r"\D", "", clean_text(value))


def significant_digits_int(value: Any) -> Optional[int]:
    digits = digits_only(value)
    if not digits:
        return None
    stripped = digits.lstrip("0").rstrip("0")
    if not stripped:
        stripped = "0"
    return int(stripped)


def parse_problem_operands(
    problem: Any,
    fallback_op1: Any = "",
    fallback_op2: Any = "",
) -> Tuple[str, str]:
    text = clean_text(problem)
    for separator in ("+", "*"):
        if separator in text:
            left, right = text.split(separator, 1)
            return clean_text(left), clean_text(right)
    return clean_text(fallback_op1), clean_text(fallback_op2)


def decimal_value_matches_any(value: Any, candidates: Sequence[Any]) -> bool:
    parsed = parse_decimal_number(value)
    if parsed is None:
        return False
    for candidate in candidates:
        candidate_parsed = parse_decimal_number(candidate)
        if candidate_parsed is not None and parsed == candidate_parsed:
            return True
    return False


def decimal_padding_summary(work_value: Any, problem_values: Sequence[Any]) -> str:
    work_text = clean_text(work_value)
    if not work_text:
        return ""
    work_number = parse_decimal_number(work_text)
    work_places = decimal_place_count(work_text)
    if work_number is None or work_places is None:
        return f"{work_text} is not a parseable decimal number"
    for problem_value in problem_values:
        problem_text = clean_text(problem_value)
        problem_number = parse_decimal_number(problem_text)
        problem_places = decimal_place_count(problem_text)
        if (
            problem_number is not None
            and problem_places is not None
            and work_number == problem_number
            and work_places > problem_places
        ):
            return f"{work_text} preserves {problem_text} with added trailing zeros"
    if not decimal_value_matches_any(work_text, problem_values):
        return f"{work_text} does not preserve the value of either displayed operand"
    return ""


def format_bool_feature(value: Optional[bool]) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def build_decimal_work_summary(row: Mapping[str, Any]) -> str:
    problem_op1, problem_op2 = parse_problem_operands(
        row.get("problem", ""),
        row.get("op1", ""),
        row.get("op2", ""),
    )
    work_op1 = clean_text(row.get("work_operand_1", ""))
    work_op2 = clean_text(row.get("work_operand_2", ""))
    work_answer = clean_text(row.get("work_answer", ""))
    problem_values = [problem_op1, problem_op2]
    problem_places = [
        decimal_place_count(problem_op1),
        decimal_place_count(problem_op2),
    ]
    work_places = [
        decimal_place_count(work_op1),
        decimal_place_count(work_op2),
    ]
    answer_places = decimal_place_count(work_answer)
    padding_notes = [
        note
        for note in (
            decimal_padding_summary(work_op1, problem_values),
            decimal_padding_summary(work_op2, problem_values),
        )
        if note
    ]
    work_sig_1 = significant_digits_int(work_op1)
    work_sig_2 = significant_digits_int(work_op2)
    answer_sig = significant_digits_int(work_answer)
    digit_add_match: Optional[bool] = None
    digit_mul_match: Optional[bool] = None
    if work_sig_1 is not None and work_sig_2 is not None and answer_sig is not None:
        digit_add_match = (work_sig_1 + work_sig_2) == answer_sig
        digit_mul_match = (work_sig_1 * work_sig_2) == answer_sig
    work_num_1 = parse_decimal_number(work_op1)
    work_num_2 = parse_decimal_number(work_op2)
    answer_num = parse_decimal_number(work_answer)
    numeric_add_match: Optional[bool] = None
    numeric_mul_match: Optional[bool] = None
    if work_num_1 is not None and work_num_2 is not None and answer_num is not None:
        numeric_add_match = (work_num_1 + work_num_2) == answer_num
        numeric_mul_match = (work_num_1 * work_num_2) == answer_num
    max_work_places = max([place for place in work_places if place is not None], default=None)
    answer_matches_max_work_places = (
        answer_places == max_work_places
        if answer_places is not None and max_work_places is not None
        else None
    )

    lines = [
        (
            "Displayed operands parsed from problem: "
            f"{problem_op1 or '(blank)'} and {problem_op2 or '(blank)'}; "
            f"decimal places: {problem_places[0]}, {problem_places[1]}."
        ),
        (
            "Written operand decimal places: "
            f"{work_places[0]}, {work_places[1]}; "
            f"written answer decimal places: {answer_places}."
        ),
        (
            "Value-preserving / value-changing operand notes: "
            f"{'; '.join(padding_notes) if padding_notes else 'none'}."
        ),
        (
            "Significant-digit relation from written operands to written answer: "
            f"addition match={format_bool_feature(digit_add_match)}, "
            f"multiplication match={format_bool_feature(digit_mul_match)}."
        ),
        (
            "Numeric relation from written operands to written answer: "
            f"addition match={format_bool_feature(numeric_add_match)}, "
            f"multiplication match={format_bool_feature(numeric_mul_match)}."
        ),
        (
            "Answer decimal places equal the max written operand decimal places: "
            f"{format_bool_feature(answer_matches_max_work_places)}."
        ),
    ]
    return "\n".join(f"- {line}" for line in lines)


FRACTION_PROFILE = StrategyProfile(
    name="fraction",
    title="Fraction Strategy Extraction Run",
    prompt_version=PROMPT_VERSION,
    strategy_codes=STRATEGY_CODES,
    codebook=STRATEGY_CODEBOOK,
    normalize_code=normalize_code,
    user_template=FRACTION_USER_TEMPLATE,
)

DECIMAL_PROFILE = StrategyProfile(
    name="decimal",
    title="Decimal Strategy Extraction Run",
    prompt_version="decimal_strategy_extraction_v15_answer_informed_trace_guess_guard",
    strategy_codes=DECIMAL_STRATEGY_CODES,
    codebook=DECIMAL_STRATEGY_CODEBOOK,
    normalize_code=normalize_decimal_code,
    user_template=DECIMAL_USER_TEMPLATE,
)

PROFILES: Mapping[str, StrategyProfile] = {
    FRACTION_PROFILE.name: FRACTION_PROFILE,
    DECIMAL_PROFILE.name: DECIMAL_PROFILE,
}


def get_profile(domain: str) -> StrategyProfile:
    try:
        return PROFILES[domain]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy domain: {domain}") from exc


def default_evidence_mode(profile: StrategyProfile) -> str:
    if profile.name == "decimal":
        return "work"
    return "trace"


def resolve_evidence_mode(
    args: argparse.Namespace,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> str:
    requested = getattr(args, "evidence", "auto")
    mode = default_evidence_mode(profile) if requested == "auto" else requested
    if mode not in RESOLVED_EVIDENCE_CHOICES:
        raise ValueError(f"Unknown evidence mode: {requested}")
    if profile.name == "fraction" and mode != "trace":
        raise ValueError(
            "The fraction profile currently supports only trace evidence. "
            "Use --evidence trace or --evidence auto."
        )
    return mode


def read_csv_with_encoding(path: Path, encoding: str, **read_csv_kwargs: Any) -> pd.DataFrame:
    if encoding != "auto":
        return pd.read_csv(path, encoding=encoding, **read_csv_kwargs)
    for candidate in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=candidate, **read_csv_kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", **read_csv_kwargs)


def load_human_split(
    path: Path,
    split: str,
    encoding: str = "cp1252",
    profile: StrategyProfile = FRACTION_PROFILE,
) -> pd.DataFrame:
    df = read_csv_with_encoding(path, encoding)
    required = {"prob", "resp", "strategy", "code"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "row_id": [f"{split}:{idx:05d}" for idx in range(len(df))],
            "source_split": split,
            "source_row": list(range(len(df))),
            "subjid": df.get("subjid", pd.Series([""] * len(df))).map(clean_text),
            "problem": df["prob"].map(clean_text),
            "answer": df["resp"].map(clean_text),
            "reasoning_trace": df["strategy"].map(clean_text),
            "raw_gold_code": df["code"].map(clean_text),
            "gold_code": df["code"].map(profile.normalize_code),
        }
    )
    if "arithCorrect" in df.columns:
        out["arithCorrect"] = df["arithCorrect"]
    return out


def decimal_gold_flags(
    row: Mapping[str, Any],
) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    strat_add = parse_optional_bool(row.get("strat_add"))
    strat_mul = parse_optional_bool(row.get("strat_mul"))
    if strat_add is None or strat_mul is None:
        return None, None, None
    return strat_add, strat_mul, not (strat_add or strat_mul)


def decimal_gold_code(row: Mapping[str, Any]) -> Optional[str]:
    strat_add, strat_mul, insufficient = decimal_gold_flags(row)
    if strat_add is None or strat_mul is None or insufficient is None:
        return None
    if strat_add and strat_mul:
        return "BOTH"
    if strat_add:
        return "AS"
    if strat_mul:
        return "M"
    return "O"


def load_decimal_split(path: Path, split: str, encoding: str = "auto") -> pd.DataFrame:
    df = read_csv_with_encoding(path, encoding, dtype=str, keep_default_na=False)
    required = {
        "prob",
        "operation",
        "op1",
        "op2",
        "resp",
        "work_operand_1",
        "work_operand_2",
        "work_answer",
        "strat_add",
        "strat_mul",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required decimal columns: {sorted(missing)}")
    gold_flags = [decimal_gold_flags(row) for row in df.to_dict("records")]
    gold_codes = [decimal_gold_code(row) for row in df.to_dict("records")]
    out = pd.DataFrame(
        {
            "row_id": [f"{split}:{idx:05d}" for idx in range(len(df))],
            "source_split": split,
            "source_row": list(range(len(df))),
            "subjid": df.get("subjid", pd.Series([""] * len(df))).map(clean_text),
            "problem": df["prob"].map(clean_text),
            "operation": df["operation"].map(clean_text),
            "op1": df["op1"].map(clean_text),
            "op2": df["op2"].map(clean_text),
            "answer": df["resp"].map(clean_text),
            "work_operand_1": df["work_operand_1"].map(clean_text),
            "work_operand_2": df["work_operand_2"].map(clean_text),
            "work_answer": df["work_answer"].map(clean_text),
            "work_layout_description": df.get(
                "work_layout", pd.Series([""] * len(df))
            ).map(decimal_work_layout_description),
            "work_alignment_description": df.get(
                "work_alignment", pd.Series([""] * len(df))
            ).map(decimal_work_alignment_description),
            "decimal_placement_description": df.get(
                "dplace", pd.Series([""] * len(df))
            ).map(decimal_dplace_description),
            "gold_addition_style_evidence": [flags[0] for flags in gold_flags],
            "gold_multiplication_style_evidence": [flags[1] for flags in gold_flags],
            "gold_insufficient_evidence": [flags[2] for flags in gold_flags],
            "raw_gold_code": gold_codes,
            "gold_code": gold_codes,
        }
    )
    return out


def load_human_targets(
    args: argparse.Namespace,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = load_human_split(Path(args.train_csv), "train", args.human_encoding, profile)
    val = load_human_split(Path(args.val_csv), "val", args.human_encoding, profile)
    if args.split == "train":
        targets = train
    elif args.split == "val":
        targets = val
    elif args.split == "all":
        targets = pd.concat([train, val], ignore_index=True)
    else:
        raise ValueError(f"Unknown split: {args.split}")
    return targets.reset_index(drop=True), train.reset_index(drop=True)


def load_decimal_targets(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    targets = load_decimal_split(Path(args.decimal_csv), "decimal", args.input_encoding)
    return targets.reset_index(drop=True), pd.DataFrame()


def load_generic_targets(
    args: argparse.Namespace,
    profile: StrategyProfile = FRACTION_PROFILE,
    evidence_mode: str = "trace",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(args.input_csv)
    df = read_csv_with_encoding(path, args.input_encoding)
    required_columns = [args.problem_col]
    if evidence_mode in ("trace", "combined"):
        required_columns.append(args.trace_col)
    if profile.name == "decimal" and evidence_mode in ("work", "combined"):
        required_columns.extend(
            [
                args.work_operand_1_col,
                args.work_operand_2_col,
                args.work_answer_col,
            ]
        )
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"{path} is missing required column: {column}")

    if args.answer_col and args.answer_col not in df.columns:
        raise ValueError(f"{path} is missing answer column: {args.answer_col}")
    if args.id_col and args.id_col not in df.columns:
        raise ValueError(f"{path} is missing id column: {args.id_col}")
    optional_decimal_cols = (
        args.operation_col,
        args.op1_col,
        args.op2_col,
    )
    for column in optional_decimal_cols:
        if column and column not in df.columns:
            raise ValueError(f"{path} is missing decimal context column: {column}")

    id_values = (
        df[args.id_col].map(clean_text).tolist()
        if args.id_col
        else [f"input:{idx:05d}" for idx in range(len(df))]
    )
    gold_values: List[Optional[str]]
    if args.gold_code_col and args.gold_code_col in df.columns:
        gold_values = df[args.gold_code_col].map(profile.normalize_code).tolist()
        raw_gold_values = df[args.gold_code_col].map(clean_text).tolist()
    else:
        gold_values = [None] * len(df)
        raw_gold_values = [None] * len(df)

    out_data: Dict[str, Any] = {
        "row_id": id_values,
        "source_split": "input",
        "source_row": list(range(len(df))),
        "subjid": "",
        "problem": df[args.problem_col].map(clean_text),
        "answer": (
            df[args.answer_col].map(clean_text)
            if args.answer_col
            else pd.Series([""] * len(df))
        ),
        "reasoning_trace": (
            df[args.trace_col].map(clean_text)
            if evidence_mode in ("trace", "combined")
            else pd.Series([""] * len(df))
        ),
        "raw_gold_code": raw_gold_values,
        "gold_code": gold_values,
    }
    if profile.name == "decimal":
        gold_flags = [decimal_cues_from_code(code) for code in gold_values]
        work_alignment_description = pd.Series([""] * len(df))
        decimal_placement_description = pd.Series([""] * len(df))
        if evidence_mode in ("work", "combined"):
            if "work_alignment" in df.columns:
                work_alignment_description = df["work_alignment"].map(
                    decimal_work_alignment_description
                )
            elif "work_alignment_description" in df.columns:
                work_alignment_description = df["work_alignment_description"].map(clean_text)
            if "dplace" in df.columns:
                decimal_placement_description = df["dplace"].map(decimal_dplace_description)
            elif "decimal_placement_description" in df.columns:
                decimal_placement_description = df[
                    "decimal_placement_description"
                ].map(clean_text)
        out_data.update(
            {
                "operation": (
                    df[args.operation_col].map(clean_text)
                    if args.operation_col
                    else pd.Series([""] * len(df))
                ),
                "op1": (
                    df[args.op1_col].map(clean_text)
                    if args.op1_col
                    else pd.Series([""] * len(df))
                ),
                "op2": (
                    df[args.op2_col].map(clean_text)
                    if args.op2_col
                    else pd.Series([""] * len(df))
                ),
                "work_operand_1": (
                    df[args.work_operand_1_col].map(clean_text)
                    if evidence_mode in ("work", "combined")
                    else pd.Series([""] * len(df))
                ),
                "work_operand_2": (
                    df[args.work_operand_2_col].map(clean_text)
                    if evidence_mode in ("work", "combined")
                    else pd.Series([""] * len(df))
                ),
                "work_answer": (
                    df[args.work_answer_col].map(clean_text)
                    if evidence_mode in ("work", "combined")
                    else pd.Series([""] * len(df))
                ),
                "work_alignment_description": work_alignment_description,
                "decimal_placement_description": decimal_placement_description,
                "gold_addition_style_evidence": [flags[0] for flags in gold_flags],
                "gold_multiplication_style_evidence": [flags[1] for flags in gold_flags],
                "gold_insufficient_evidence": [flags[2] for flags in gold_flags],
            }
        )
    out = pd.DataFrame(out_data)
    needs_human_few_shots = (
        profile.name == "fraction"
        and args.few_shot_per_code > 0
        and args.few_shot_preset == "human_shortest"
    )
    train = (
        load_human_split(Path(args.train_csv), "train", args.human_encoding, profile)
        if needs_human_few_shots
        else pd.DataFrame()
    )
    return out.reset_index(drop=True), train.reset_index(drop=True)


def load_targets(
    args: argparse.Namespace,
    profile: StrategyProfile = FRACTION_PROFILE,
    evidence_mode: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if evidence_mode == "auto":
        evidence_mode = resolve_evidence_mode(args, profile)
    if args.input_csv:
        return load_generic_targets(args, profile, evidence_mode)
    if profile.name == "decimal":
        if evidence_mode != "work":
            raise ValueError(
                "The built-in decimal BSS CSV has visible-work fields but no "
                "natural-language traces. Use --evidence work/auto, or pass "
                "--input_csv with --evidence trace for trace extraction."
            )
        return load_decimal_targets(args)
    return load_human_targets(args, profile)


def sample_targets(
    df: pd.DataFrame,
    args: argparse.Namespace,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> pd.DataFrame:
    sampled = df.copy()
    if args.sample_per_code > 0 and "gold_code" in sampled.columns:
        pieces = []
        for code in profile.strategy_codes:
            group = sampled[sampled["gold_code"] == code]
            if group.empty:
                continue
            pieces.append(
                group.sample(
                    n=min(args.sample_per_code, len(group)),
                    random_state=args.seed,
                    replace=False,
                )
            )
        if pieces:
            sampled = pd.concat(pieces, ignore_index=True)
    if args.max_rows > 0 and len(sampled) > args.max_rows:
        sampled = sampled.sample(n=args.max_rows, random_state=args.seed, replace=False)
    return sampled.sort_values(["source_split", "source_row"]).reset_index(drop=True)


def truncate_text(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"


def select_few_shot_examples(
    train_df: pd.DataFrame,
    examples_per_code: int,
    seed: int,
    max_chars: int,
    preset: str = "human_shortest",
    profile: StrategyProfile = FRACTION_PROFILE,
) -> List[Mapping[str, str]]:
    if examples_per_code <= 0:
        return []
    if profile.name != "fraction":
        return []
    if preset == "canonical":
        examples = []
        for code in profile.strategy_codes:
            code_examples = [
                ex for ex in CANONICAL_FEW_SHOT_EXAMPLES if ex["strategy_code"] == code
            ][:examples_per_code]
            examples.extend(code_examples)
        return examples

    examples: List[Mapping[str, str]] = []
    for code in profile.strategy_codes:
        group = train_df[train_df["gold_code"] == code]
        if group.empty:
            continue
        group = group.assign(_trace_len=group["reasoning_trace"].str.len()).sort_values(
            ["_trace_len", "source_row"]
        )
        group = group.head(examples_per_code)
        for _, row in group.iterrows():
            examples.append(
                {
                    "problem": clean_text(row["problem"]),
                    "answer": clean_text(row["answer"]),
                    "reasoning_trace": truncate_text(
                        clean_text(row["reasoning_trace"]), max_chars
                    ),
                    "strategy_code": clean_text(row["gold_code"]),
                }
            )
    return examples


def codebook_text(profile: StrategyProfile = FRACTION_PROFILE) -> str:
    return "\n".join(
        f"- `{code}`: {profile.codebook[code]}" for code in profile.strategy_codes
    )


def decimal_cue_examples(evidence_mode: str) -> str:
    if evidence_mode != "work":
        return ""
    return """
Calibration examples:

Example A
Problem: 0.415+52
Visible written work:
Coded visible-work cue summary:
- Operand alignment cue: decimal points aligned (addition-style alignment cue)
- Answer decimal-placement cue: answer decimal placed as in aligned operands (addition-style placement cue)
Written-work layout: vertical column work
Operand 1 as written: 52.000
Operand 2 as written: 0.415
Answer as written: 52.415
Correct JSON: {"addition_style_evidence": true, "multiplication_style_evidence": false, "insufficient_evidence": false, "confidence": 0.9, "evidence": "whole-number operand padded with zeros for decimal alignment"}

Example B
Problem: 2.4*1.2
Visible written work:
Coded visible-work cue summary:
- Operand alignment cue: decimal points aligned (addition-style alignment cue)
- Answer decimal-placement cue: answer decimal placed by counting decimal digits (multiplication-style placement cue)
Written-work layout: vertical column work
Operand 1 as written: 2.400
Operand 2 as written: 1.200
Answer as written: 2.8800
Correct JSON: {"addition_style_evidence": true, "multiplication_style_evidence": true, "insufficient_evidence": false, "confidence": 0.85, "evidence": "decoded alignment cue is addition-style and answer placement cue is multiplication-style"}

Example C
Problem: 2.3*0.4
Visible written work:
Coded visible-work cue summary:
- Operand alignment cue: rightmost digits aligned (multiplication-style alignment cue)
- Answer decimal-placement cue: answer decimal placed by counting decimal digits (multiplication-style placement cue)
Written-work layout: vertical column work
Operand 1 as written: 23
Operand 2 as written: 4
Answer as written: .92
Correct JSON: {"addition_style_evidence": false, "multiplication_style_evidence": true, "insufficient_evidence": false, "confidence": 0.9, "evidence": "decimal points removed, whole-number multiplication, decimal placed in answer"}

Example D
Problem: 2.3*0.13
Visible written work:
Coded visible-work cue summary:
- Operand alignment cue: decimal points aligned (addition-style alignment cue)
- Answer decimal-placement cue: answer decimal placed by counting decimal digits (multiplication-style placement cue)
Written-work layout: vertical column work
Operand 1 as written: 2.30
Operand 2 as written: 0.13
Answer as written: 0.299
Correct JSON: {"addition_style_evidence": true, "multiplication_style_evidence": true, "insufficient_evidence": false, "confidence": 0.85, "evidence": "operand padded in decimal form and answer reflects multiplication-style decimal placement"}

Example E
Problem: 2.46+4.1
Visible written work:
Coded visible-work cue summary:
- Operand alignment cue: decimal points aligned (addition-style alignment cue)
- Answer decimal-placement cue: answer decimal placed by counting decimal digits (multiplication-style placement cue)
Written-work layout: vertical column work
Operand 1 as written: 4.10
Operand 2 as written: 2.46
Answer as written: 0.656
Correct JSON: {"addition_style_evidence": true, "multiplication_style_evidence": true, "insufficient_evidence": false, "confidence": 0.8, "evidence": "trailing-zero padding supports addition-style evidence; answer decimal placement supports multiplication-style evidence"}

Example F
Problem: 0.415+52
Visible written work:
Coded visible-work cue summary:
- Operand alignment cue: other or unclear operand alignment
- Answer decimal-placement cue: answer decimal placed as in aligned operands (addition-style placement cue)
Written-work layout: vertical column work
Operand 1 as written: 0.415
Operand 2 as written: 0.520
Answer as written: 0.935
Correct JSON: {"addition_style_evidence": false, "multiplication_style_evidence": false, "insufficient_evidence": true, "confidence": 0.75, "evidence": "placement-only cue is not enough because 52 was transformed to 0.520 rather than padded to 52.000"}

Example G
Problem: 2.4*1.2
Visible written work:
Coded visible-work cue summary:
- Operand alignment cue: rightmost digits aligned (multiplication-style alignment cue)
- Answer decimal-placement cue: no clear answer decimal-placement cue
Written-work layout: vertical column work
Operand 1 as written: 24
Operand 2 as written: 12
Answer as written: 7.0
Correct JSON: {"addition_style_evidence": false, "multiplication_style_evidence": true, "insufficient_evidence": false, "confidence": 0.75, "evidence": "decoded rightmost-digit alignment cue is multiplication-style evidence"}

Example H
Problem: 14*0.21
Visible written work:
Coded visible-work cue summary:
- Operand alignment cue: no clear operand alignment
- Answer decimal-placement cue: answer decimal placed by counting decimal digits (multiplication-style placement cue)
Written-work layout: vertical column work
Operand 1 as written: 0.210
Operand 2 as written: 1.400
Answer as written: 0.800
Correct JSON: {"addition_style_evidence": false, "multiplication_style_evidence": false, "insufficient_evidence": true, "confidence": 0.75, "evidence": "placement-only cue is not enough because both operands were transformed into different decimal values"}
"""


def format_few_shots(examples: Sequence[Mapping[str, str]]) -> str:
    if not examples:
        return ""
    chunks = ["\nFew-shot examples:"]
    for idx, example in enumerate(examples, start=1):
        chunks.append(
            "\n".join(
                [
                    f"Example {idx}",
                    f"Problem: {example['problem']}",
                    f"Student final answer: {example['answer']}",
                    "Reasoning trace:",
                    example["reasoning_trace"],
                    f"Correct JSON: {{\"strategy_code\": \"{example['strategy_code']}\"}}",
                ]
            )
        )
    return "\n\n".join(chunks)


def build_fraction_system_prompt(examples: Sequence[Mapping[str, str]] = ()) -> str:
    base = f"""You are classifying the procedure used in a student's fraction reasoning trace.
Use exactly one strategy code from the codebook below. Your goal is to recover
the closest listed fraction strategy, not to judge whether the arithmetic
or procedure is mathematically valid. Students often describe messy, incomplete,
or wrong procedures; do not mark a trace as OTHER just because the arithmetic is
wrong, odd, or overgeneralized. Prefer explicit procedural evidence in the
reasoning trace, but use the final answer as strong supporting evidence when
the trace is terse, vague, or only partly explains the work.

Allowed strategy codes:
{codebook_text(FRACTION_PROFILE)}

Decision rules:
- OTHER is a last resort. Choose the closest non-OTHER strategy whenever the
  trace contains enough procedural evidence for KDON, CDON, ONOD, ICDM, or CROP.
- If a trace only says "I guessed", "just guessed", or "I did not know so I
  put ...", use `OTHER`. If the same trace also contains a recognizable
  procedure, common-denominator work, cross-operation, or diagnostic final
  answer pattern, classify that attempted procedure instead of defaulting to
  `OTHER`.
- Do not use `OTHER` only because the student picked the wrong operation, made
  an arithmetic mistake, used a strategy outside its standard operation, or gave
  a messy/incomplete explanation.
- Match the student's attempted or resulting procedure pattern, even when the
  explanation is broken. For these fixed fraction tasks, final answers like `4/7`,
  `2/3`, `3/10`, `6/25`, `5/10`, or `1/1` often indicate `ONOD`; answers like
  `11/10`, `1 1/10`, `1/10`, `30/10`, or `150/25` often indicate `CDON`.
  Do not let these answer hints override a clear conflicting trace.
- Some user prompts include a diagnostic answer-pattern hint for these fixed
  fraction tasks. Use that hint when the reasoning trace is vague, incomplete,
  or contains uncertainty/guessing language, but ignore it if the trace clearly
  describes a different strategy.
- If a trace says "added the fractions together" and the answer matches
  numerator-plus-numerator and denominator-plus-denominator, use `ONOD`.
- If the trace mentions common denominator, lowest common denominator, LCM, or
  changing one or both fractions to a shared denominator, prefer `CDON` when the
  trace then combines, adds, subtracts, multiplies, or divides the numerators or
  top numbers while keeping that denominator. This remains `CDON` even if the
  common denominator or numerator conversion is arithmetically wrong.
- If a trace mentions finding common denominators first and then operating only
  on numerators while keeping the denominator, use `CDON`.
- If the fractions already have the same denominator and the trace only says
  the denominators are the same/common/equal before operating on numerators, use
  `KDON` rather than `CDON`; no actual denominator conversion was attempted.
- If a trace first finds common denominators but then operates on both
  numerators and denominators, use `OTHER` because this rare hybrid has no
  direct listed strategy code. Typical examples are `6/10` and `5/10` followed by
  `6+5` for the numerator and `10+10` for the denominator, or final answers
  like `11/20`, `15/20`, or `10/20` after common-denominator conversion.
- Use `KDON` when the student operates on numerator or top-number values and
  explicitly keeps, reuses, carries over, or chooses one operand denominator or
  the common denominator. If they mention a denominator operation but then say
  they kept or picked an existing/common denominator, prefer `KDON`.
- Use `ONOD` when the student explicitly operates on both numerator values and
  denominator values to form the answer denominator, even if the operation,
  arithmetic, or simplification is wrong.
- Use `ONOD` when the student says they operated on a denominator, cancelled a
  denominator, or got a denominator result such as zero or one, even if they
  later typed a different denominator because zero was not allowed.
- For division traces, distinguish inverse-then-operate (`ICDM`) from direct
  cross-operation (`CROP`) when the explanation says so. If it only says it
  flipped a fraction and multiplied, use `ICDM`.
- `CROP` is not limited to division wording. Use `CROP` whenever the top answer
  combines a numerator from one fraction with a denominator from the other, and
  the bottom answer combines the remaining cross pair.
- Use `ICDM` for a coherent inverse-then-operate procedure. If a non-division
  trace only says the student switched a fraction and changed the operation but
  gives no coherent inverse-then-operate procedure, use `OTHER`.
- Use `OTHER` for guessed, idiosyncratic, conceptual, decimal-conversion, or
  incomplete reasoning that cannot be mapped to the listed fraction procedures.

Return only JSON with keys:
- strategy_code: one of "KDON", "CDON", "ONOD", "ICDM", "CROP", "OTHER"
- confidence: number from 0 to 1
- evidence: short phrase from the trace or answer
- notes: optional short note
"""
    return base + format_few_shots(examples)


def build_decimal_system_prompt(
    examples: Sequence[Mapping[str, str]] = (),
    evidence_mode: str = "work",
) -> str:
    del examples
    if evidence_mode == "trace":
        evidence_description = "strategy described in a student's decimal arithmetic reasoning trace"
        procedure_description = "procedure stated in the trace"
        evidence_rules = """- Prefer explicit procedural wording in the trace over the final answer alone.
- If the trace explicitly says the student guessed, was not sure, did not know,
  made no procedure, or otherwise disclaims having a strategy, do not infer a
  strategy from the final answer. Set both evidence flags false,
  `insufficient_evidence=true`, `direct_procedure_evidence=false`, and
  `strategy_inferred_from_answer=false`, even if the final answer resembles an
  AS or M outcome.
- When the trace is terse, incomplete, or only shows a bare equation or answer,
  use the student's final answer as supporting evidence for the likely AS/M
  strategy. Do not let final-answer evidence override clear conflicting
  procedural wording in the trace.
- For addition problems, an answer that preserves the decimal-aligned place
  values of the operands supports AS when the trace gives no stronger cue; an
  answer that looks like whole-number digit addition followed by decimal
  placement supports M. For example, `5.61+23 -> 5.84` or
  `0.415+52 -> 0.467` supports M rather than AS because it resembles adding
  right-aligned digit strings and then placing a decimal.
- For multiplication problems, an answer that fits whole-number-like
  multiplication followed by decimal-place placement supports M when the trace
  gives no stronger cue.
- Do not infer BOTH from a bare final answer unless the answer gives independent
  evidence for both AS and M, which is rare. With final-answer evidence alone,
  prefer a single AS/M cue or O for insufficient evidence.
- Infer the same cue dimensions used for visible work: operand alignment and
  answer decimal placement. Language such as "lined up the decimals",
  "decimal points under each other", "added zeros to line them up", or
  "brought the decimal down" is AS evidence.
- Use AS when the trace says the student lined up decimal points, padded with
  zeros for alignment, or placed the answer decimal straight down from aligned
  operands.
- Language such as "ignored the decimals", "removed the decimal points",
  "lined up the last digits", "multiplied like whole numbers", or "counted
  decimal places" is M evidence.
- Use M when the trace says the student ignored/removed decimal points, aligned
  rightmost digits, multiplied whole-number-like digits, or counted total
  decimal places to place the answer.
- For multiplication traces, do not treat ordinary setup, vertical formatting,
  addition of partial products, or alignment of partial-product rows as AS
  evidence by itself. Partial-product addition is part of multiplication work.
  Mark AS true only when the trace aligns the original decimal operands by
  decimal point, pads original operands for decimal alignment, or places the
  answer decimal by bringing it straight down from aligned original operands.
- Even when partial products are written as decimals, padded with zeros, or
  aligned for summing (for example, adding 0.32 and 0.640 inside a
  multiplication solution), do not count that as AS evidence. If the only
  AS-like cue is adding or aligning partial products while the trace also uses
  multiplication-style decimal placement or decimal-place counting, set
  addition_style_evidence=false and multiplication_style_evidence=true.
- Ignore tentative or rejected approaches after a self-correction unless the
  final procedure uses them.
- Use BOTH when the trace contains both AS and M cues, even if one cue seems
  more important.
- If a trace mentions only answer decimal placement while also describing
  operand transformations that change the original values, treat that placement
  cue cautiously; it may be insufficient without a clear alignment or
  whole-number-like operation cue.
- Use O when the trace only gives a guess, has no usable procedure, and the
  final answer does not provide a recognizable AS/M cue.
- Do not use correctness as the strategy label."""
    elif evidence_mode == "combined":
        evidence_description = (
            "student's decimal arithmetic strategy from a reasoning trace and visible written work"
        )
        procedure_description = "procedure stated in the trace and/or visible in the written work"
        evidence_rules = """- Prefer explicit procedural wording in the trace; use visible written work as
  supporting evidence when the trace is vague, incomplete, or absent.
- Use decoded visible-work cue summaries when provided. They describe operand
  alignment and answer decimal placement; they are evidence features, not target
  strategy labels.
- In a coded visible-work cue summary, any cue explicitly marked
  "addition-style" should normally set `addition_style_evidence` true, and any
  cue explicitly marked "multiplication-style" should normally set
  `multiplication_style_evidence` true. Cues saying "no clear" or "other or
  unclear" do not set either flag by themselves.
- Be cautious with answer decimal-placement cues when they appear without a
  matching operand-alignment cue. If the visible operands were changed into
  different decimal values rather than padded/preserved, placement alone may be
  insufficient evidence for AS or M.
- Infer the same operand-alignment and answer decimal-placement cues from trace
  language when the trace describes them.
- Use AS for decimal-point alignment, trailing-zero padding for alignment,
  bringing the decimal straight down, or trace language describing those moves.
- Use M for rightmost digit alignment, removing/ignoring decimal points,
  whole-number multiplication work, counting total decimal places, or trace
  language describing those moves.
- For multiplication traces, do not treat ordinary setup, vertical formatting,
  addition of partial products, or alignment of partial-product rows as AS
  evidence by itself. Partial-product addition is part of multiplication work.
  Mark AS true only when the trace aligns the original decimal operands by
  decimal point, pads original operands for decimal alignment, or places the
  answer decimal by bringing it straight down from aligned original operands.
- Even when partial products are written as decimals, padded with zeros, or
  aligned for summing (for example, adding 0.32 and 0.640 inside a
  multiplication solution), do not count that as AS evidence. If the only
  AS-like cue is adding or aligning partial products while the trace also uses
  multiplication-style decimal placement or decimal-place counting, set
  addition_style_evidence=false and multiplication_style_evidence=true.
- Ignore tentative or rejected approaches after a self-correction unless the
  final procedure uses them.
- Use BOTH when either the trace or the written work shows both AS and M cues, or
  when trace and written-work evidence clearly point to different AS/M
  procedures with no stronger source.
- Use O when neither source reveals AS or M evidence.
- Do not use correctness as the strategy label."""
    else:
        evidence_description = "visible written strategy in a student's decimal arithmetic work"
        procedure_description = "procedure visible in the written work"
        evidence_rules = """- Prefer written-work evidence, but use the written final answer to identify the
  procedure visible in the work. Do not score mathematical correctness.
- The written operands and answer preserve the student's number strings where
  available, but exact spacing may be lost. Infer cues from decimal points,
  trailing zeros, removed decimal points, layout description, and whether the
  answer digits fit an aligned-decimal or whole-number-like procedure.
- Some prompts include a derived summary computed only from the problem and
  visible written strings. Use it as arithmetic/string evidence; it is not a
  human code and may be incomplete when the written work is ambiguous.
- Some prompts include a coded visible-work cue summary decoded from worksheet
  features. Use these operand-alignment and answer decimal-placement cues as
  direct evidence. They are cue descriptions, not the target AS/M flags, and
  they may be sufficient even when digit-relation summaries are ambiguous.
- In a coded visible-work cue summary, any cue explicitly marked
  "addition-style" should normally set `addition_style_evidence` true, and any
  cue explicitly marked "multiplication-style" should normally set
  `multiplication_style_evidence` true. Cues saying "no clear" or "other or
  unclear" do not set either flag by themselves.
- Be cautious with answer decimal-placement cues when they appear without a
  matching operand-alignment cue. If the visible operands were changed into
  different decimal values rather than padded/preserved, placement alone may be
  insufficient evidence for AS or M. For example, changing 52 into 0.520 is not
  the same as padding 52 to 52.000 for alignment.
- Treat AS and M as independent cue flags. Do not choose the stronger cue; mark
  both flags true when both cues are present.
- Do not mark AS true merely because operands or answers contain decimal
  points. Require addition-style evidence: decimal-point alignment, trailing-zero
  padding that preserves the original place value for alignment, or answer
  decimal placement that is carried down from aligned operands.
- Use AS when a whole-number operand is padded for decimal alignment, when a
  shorter decimal is padded to match a longer decimal for addition-style column
  work, or when the answer decimal is placed as in aligned operands.
- Use M when the work removes/ignores decimal points, uses whole-number-like
  digit strings, aligns rightmost digits, or the answer looks like a digit-wise
  operation followed by decimal placement/counting. For example, an addition
  problem such as 0.415 + 52 giving 0.467 is M evidence because it resembles
  415 + 52 = 467 with a decimal placed afterward.
- Do not mark M true solely because decimal points were stripped; the answer
  must fit a recognizable whole-number-like operation or multiplication-style
  decimal placement.
- For multiplication problems, preserving decimal-form operands is not by
  itself AS evidence. Mark BOTH only when there is an independent AS cue such as
  alignment-style padding or addition-style decimal placement, plus independent
  M evidence such as whole-number multiplication or multiplication-style decimal
  placement.
- Use O when the work is blank, only a bare answer, or the written operands and
  answer do not coherently reveal either an AS or M cue. Do not infer M solely
  from stripped decimal points if the answer digits do not fit a recognizable
  whole-number-like operation.
- Do not use correctness as the strategy label."""

    return f"""You are detecting the two human-coded BSS decimal strategy flags for the {evidence_description}.
Your goal is to recover the {procedure_description}, not to judge whether the
answer is correct. Return the two independent human-style cue flags; the
evaluation pipeline scores those flags directly and also derives a compatibility
strategy code from them.

Derived compatibility strategy codes:
{codebook_text(DECIMAL_PROFILE)}

Decision rules:
{evidence_rules}
- Set `addition_style_evidence` to true exactly when the work would receive the
  human BSS addition-style flag: addition-style decimal alignment or
  addition-style decimal placement evidence is present.
- Set `multiplication_style_evidence` to true exactly when the work would
  receive the human BSS multiplication-style flag: multiplication-style alignment,
  whole-number-like operation, or multiplication-style decimal placement
  evidence is present.
- Set `insufficient_evidence` to true only when neither AS nor M evidence is
  present or the evidence is too unclear to classify. This corresponds to the
  derived O case when both human flags are false.
- Both evidence flags may be true. This corresponds to the derived BOTH case.
- Both evidence flags may be false. This corresponds to the derived O case.
- The derived compatibility label is AS for addition-only evidence, M for
  multiplication-only evidence, BOTH when both cues are present, and O when
  evidence is insufficient or neither cue is present.
- If both cue flags are true, `insufficient_evidence` must be false.
- Set `direct_procedure_evidence` to true when the trace or visible work itself
  contains an AS/M procedural cue. Set it to false when the label is based only
  on final-answer evidence or when there is insufficient evidence.
- Set `strategy_inferred_from_answer` to true when the trace or visible work
  lacks direct AS/M procedural evidence and the student's final answer is used
  to infer the likely AS/M flags. Set it to false when direct procedural
  evidence is present or when the result is O for insufficient evidence.
- Do not output `strategy_code`; output the cue flags instead.

Return only JSON with keys:
- addition_style_evidence: boolean
- multiplication_style_evidence: boolean
- insufficient_evidence: boolean
- direct_procedure_evidence: boolean
- strategy_inferred_from_answer: boolean
- confidence: number from 0 to 1
- evidence: short phrase naming the cue
- notes: optional short note
{decimal_cue_examples(evidence_mode)}
"""


def build_system_prompt(
    examples: Sequence[Mapping[str, str]] = (),
    profile: StrategyProfile = FRACTION_PROFILE,
    evidence_mode: str = "auto",
) -> str:
    if evidence_mode == "auto":
        evidence_mode = default_evidence_mode(profile)
    if profile.name == "decimal":
        return build_decimal_system_prompt(examples, evidence_mode)
    return build_fraction_system_prompt(examples)


def build_fraction_user_prompt(row: Mapping[str, Any], max_trace_chars: int) -> str:
    trace = truncate_text(clean_text(row["reasoning_trace"]), max_trace_chars)
    problem = clean_text(row["problem"])
    answer = clean_text(row.get("answer", ""))
    lines = [
        f"Problem: {problem}",
        f"Student final answer: {answer}",
    ]
    answer_hint = ANSWER_PATTERN_HINTS.get((problem, answer))
    if answer_hint:
        lines.append(
            "Diagnostic answer-pattern hint: for this fixed fraction task, "
            f"this final answer is often associated with `{answer_hint}` when "
            "the trace is vague or incomplete."
        )
    lines.extend(
        [
            "Reasoning trace:",
            trace,
            "",
            "Classify the reasoning trace with one allowed strategy code. Return only JSON.",
        ]
    )
    return "\n".join(lines)


def build_decimal_user_prompt(
    row: Mapping[str, Any],
    max_trace_chars: int,
    evidence_mode: str = "work",
) -> str:
    lines = [
        f"Problem: {clean_text(row.get('problem', ''))}",
        f"Operation: {clean_text(row.get('operation', ''))}",
        f"Operand 1 shown in problem: {clean_text(row.get('op1', ''))}",
        f"Operand 2 shown in problem: {clean_text(row.get('op2', ''))}",
        f"Student final answer: {clean_text(row.get('answer', ''))}",
    ]
    if evidence_mode in ("trace", "combined"):
        lines.extend(
            [
                "",
                "Reasoning trace:",
                truncate_text(
                    clean_text(row.get("reasoning_trace", "")),
                    max_trace_chars,
                ),
            ]
        )
    if evidence_mode in ("work", "combined"):
        lines.extend(["", "Visible written work:"])
        lines.extend(
            [
                "Derived summary from visible strings:",
                build_decimal_work_summary(row),
                "",
            ]
        )
        coded_cue_summary = build_decimal_coded_cue_summary(row)
        if coded_cue_summary:
            lines.extend(
                [
                    "Coded visible-work cue summary:",
                    coded_cue_summary,
                    "",
                ]
            )
        work_layout_description = clean_text(row.get("work_layout_description", ""))
        if work_layout_description:
            lines.append(f"Written-work layout: {work_layout_description}")
        lines.extend(
            [
                f"Operand 1 as written: {clean_text(row.get('work_operand_1', ''))}",
                f"Operand 2 as written: {clean_text(row.get('work_operand_2', ''))}",
                f"Answer as written: {clean_text(row.get('work_answer', ''))}",
            ]
        )
    classifier_text = (
        (
            "Flag the described decimal strategy cues, including any described "
            "operand-alignment and answer decimal-placement cues. Return only JSON."
        )
        if evidence_mode == "trace"
        else (
            "Flag the student's decimal strategy cues, including any described or "
            "visible operand-alignment and answer decimal-placement cues. Return only JSON."
        )
        if evidence_mode == "combined"
        else "Flag the visible decimal strategy cues. Return only JSON."
    )
    lines.extend(["", classifier_text])
    return "\n".join(lines)


def build_user_prompt(
    row: Mapping[str, Any],
    max_trace_chars: int,
    profile: StrategyProfile = FRACTION_PROFILE,
    evidence_mode: str = "auto",
) -> str:
    if evidence_mode == "auto":
        evidence_mode = default_evidence_mode(profile)
    if profile.name == "decimal":
        return build_decimal_user_prompt(row, max_trace_chars, evidence_mode)
    return build_fraction_user_prompt(row, max_trace_chars)


def user_prompt_template(
    profile: StrategyProfile = FRACTION_PROFILE,
    evidence_mode: str = "auto",
) -> str:
    if evidence_mode == "auto":
        evidence_mode = default_evidence_mode(profile)
    if profile.name != "decimal":
        return profile.user_template
    if evidence_mode == "trace":
        return DECIMAL_TRACE_USER_TEMPLATE
    if evidence_mode == "combined":
        return DECIMAL_COMBINED_USER_TEMPLATE
    return profile.user_template


def request_key(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def make_request_record(
    row: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    args: argparse.Namespace,
    profile: StrategyProfile = FRACTION_PROFILE,
    evidence_mode: str = "trace",
) -> Dict[str, Any]:
    key_payload = {
        "prompt_version": profile.prompt_version,
        "domain": profile.name,
        "evidence": evidence_mode,
        "model": args.model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "thinking_budget": args.thinking_budget,
        "thinking_level": args.thinking_level,
    }
    key = request_key(key_payload)
    return {
        "request_key": key,
        "prompt_version": profile.prompt_version,
        "domain": profile.name,
        "evidence": evidence_mode,
        "model": args.model,
        "row_id": row["row_id"],
        "source_split": row["source_split"],
        "source_row": int(row["source_row"]),
        "gold_code": row.get("gold_code"),
        "system_instruction": system_prompt,
        "user_prompt": user_prompt,
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_candidate(text: str) -> Optional[Mapping[str, Any]]:
    text = strip_code_fence(text)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, Mapping) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, Mapping) else None
    except json.JSONDecodeError:
        return None


def normalize_confidence(value: Any) -> Optional[float]:
    try:
        confidence = float(value)
        if math.isnan(confidence):
            return None
        if confidence < 0:
            return 0.0
        if confidence > 1:
            return 1.0
        return confidence
    except (TypeError, ValueError):
        return None


def empty_prediction_payload() -> Dict[str, Any]:
    return {
        "pred_code": None,
        "confidence": None,
        "evidence": "",
        "notes": "",
        "addition_style_evidence": None,
        "multiplication_style_evidence": None,
        "insufficient_evidence": None,
        "direct_procedure_evidence": None,
        "strategy_inferred_from_answer": None,
        "parse_success": False,
    }


def decimal_cues_from_code(code: Optional[str]) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    if code == "AS":
        return True, False, False
    if code == "M":
        return False, True, False
    if code == "BOTH":
        return True, True, False
    if code == "O":
        return False, False, True
    return None, None, None


def normalize_model_payload(
    payload: Optional[Mapping[str, Any]],
    profile: StrategyProfile = FRACTION_PROFILE,
) -> Dict[str, Any]:
    if payload is None:
        return empty_prediction_payload()
    confidence = normalize_confidence(payload.get("confidence"))
    evidence = clean_text(payload.get("evidence", ""))
    notes = clean_text(payload.get("notes", ""))
    if profile.name == "decimal":
        add = parse_optional_bool(
            payload.get(
                "addition_style_evidence",
                payload.get("as_evidence", payload.get("addition_evidence")),
            )
        )
        mul = parse_optional_bool(
            payload.get(
                "multiplication_style_evidence",
                payload.get("m_evidence", payload.get("multiplication_evidence")),
            )
        )
        insufficient = parse_optional_bool(
            payload.get(
                "insufficient_evidence",
                payload.get("unclear_or_insufficient", payload.get("insufficient")),
            )
        )
        direct_procedure = parse_optional_bool(
            payload.get(
                "direct_procedure_evidence",
                payload.get("direct_strategy_evidence", payload.get("trace_procedure_evidence")),
            )
        )
        answer_inferred = parse_optional_bool(
            payload.get(
                "strategy_inferred_from_answer",
                payload.get("answer_informed_inference", payload.get("inferred_from_answer")),
            )
        )
        code = derive_decimal_code_from_cues(add, mul, insufficient)
        if code is None:
            code = profile.normalize_code(
                payload.get("strategy_code", payload.get("code", payload.get("label")))
            )
            add, mul, insufficient = decimal_cues_from_code(code)
        return {
            "pred_code": code,
            "confidence": confidence,
            "evidence": evidence,
            "notes": notes,
            "addition_style_evidence": add,
            "multiplication_style_evidence": mul,
            "insufficient_evidence": insufficient,
            "direct_procedure_evidence": direct_procedure,
            "strategy_inferred_from_answer": answer_inferred,
            "parse_success": code in profile.strategy_codes,
        }

    code = profile.normalize_code(
        payload.get("strategy_code", payload.get("code", payload.get("label")))
    )
    return {
        "pred_code": code,
        "confidence": confidence,
        "evidence": evidence,
        "notes": notes,
        "addition_style_evidence": None,
        "multiplication_style_evidence": None,
        "insufficient_evidence": None,
        "direct_procedure_evidence": None,
        "strategy_inferred_from_answer": None,
        "parse_success": code in profile.strategy_codes,
    }


def regex_bool_field(text: str, field: str) -> Optional[bool]:
    match = re.search(
        rf'"{re.escape(field)}"\s*:\s*("?)(true|false|yes|no|0|1)\1',
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return parse_optional_bool(match.group(2))


def parse_model_response(
    text: str,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> Dict[str, Any]:
    parsed = normalize_model_payload(extract_json_candidate(text), profile)
    if parsed["parse_success"]:
        return parsed
    if profile.name == "decimal":
        recovered_cues: Dict[str, Any] = {}
        for field in (
            "addition_style_evidence",
            "multiplication_style_evidence",
            "insufficient_evidence",
        ):
            value = regex_bool_field(text, field)
            if value is not None:
                recovered_cues[field] = value
        if recovered_cues:
            confidence_match = re.search(r'"confidence"\s*:\s*([01](?:\.\d+)?)', text)
            if confidence_match:
                recovered_cues["confidence"] = confidence_match.group(1)
            recovered_parsed = normalize_model_payload(recovered_cues, profile)
            if recovered_parsed["parse_success"]:
                return recovered_parsed
    code_alternatives = "|".join(re.escape(code) for code in profile.strategy_codes)
    match = re.search(
        rf'"strategy_code"\s*:\s*"?({code_alternatives}|10|20|[1-6])"?',
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return parsed
    recovered: Dict[str, Any] = {"strategy_code": match.group(1)}
    confidence_match = re.search(r'"confidence"\s*:\s*([01](?:\.\d+)?)', text)
    if confidence_match:
        recovered["confidence"] = confidence_match.group(1)
    return normalize_model_payload(recovered, profile)


def response_to_raw_payload(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_json_dict"):
        return response.to_json_dict()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return {"repr": repr(response)}


def state_name(value: Any) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if name:
        return str(name)
    return str(value)


def batch_error_text(error: Any) -> str:
    if not error:
        return ""
    if isinstance(error, str):
        return error
    if isinstance(error, Mapping):
        for key in ("message", "details", "code"):
            if key in error:
                return clean_text(error[key])
        return json.dumps(error, ensure_ascii=False, sort_keys=True)
    message = getattr(error, "message", None)
    if message:
        return clean_text(message)
    return repr(error)


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def usage_value(usage: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return 0


def summarize_gemini_cache_usage(
    cache_dir: Path,
    input_usd_per_million: float,
    output_usd_per_million: float,
) -> Dict[str, Any]:
    cache_files = sorted(cache_dir.glob("*.json"))
    prompt_tokens = 0
    candidate_tokens = 0
    total_tokens = 0
    usage_record_count = 0
    for path in cache_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_response = payload.get("raw_response") or {}
        if not isinstance(raw_response, Mapping):
            continue
        usage = raw_response.get("usage_metadata") or raw_response.get("usageMetadata") or {}
        if not isinstance(usage, Mapping):
            continue
        usage_record_count += 1
        prompt_tokens += usage_value(usage, "prompt_token_count", "promptTokenCount")
        candidate_tokens += usage_value(
            usage,
            "candidates_token_count",
            "candidatesTokenCount",
        )
        total_tokens += usage_value(usage, "total_token_count", "totalTokenCount")

    input_usd = prompt_tokens / 1_000_000 * input_usd_per_million
    output_usd = candidate_tokens / 1_000_000 * output_usd_per_million
    return {
        "cache_file_count": len(cache_files),
        "usage_record_count": usage_record_count,
        "prompt_tokens": prompt_tokens,
        "candidate_tokens": candidate_tokens,
        "total_tokens": total_tokens,
        "input_usd_per_million": input_usd_per_million,
        "output_usd_per_million": output_usd_per_million,
        "estimated_input_usd": input_usd,
        "estimated_output_usd": output_usd,
        "estimated_total_usd": input_usd + output_usd,
        "pricing_note": (
            "Estimate uses CLI rates; update --gemini_input_usd_per_million "
            "and --gemini_output_usd_per_million if model pricing changes."
        ),
    }


def write_gemini_usage_summary(out_dir: Path, args: argparse.Namespace) -> None:
    usage = summarize_gemini_cache_usage(
        out_dir / "cache",
        args.gemini_input_usd_per_million,
        args.gemini_output_usd_per_million,
    )
    (out_dir / "gemini_usage.json").write_text(
        json.dumps(usage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_oracle_backend(
    targets: pd.DataFrame,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> pd.DataFrame:
    rows = []
    for _, row in targets.iterrows():
        if profile.name == "decimal":
            add = parse_optional_bool(row.get("gold_addition_style_evidence"))
            mul = parse_optional_bool(row.get("gold_multiplication_style_evidence"))
            insufficient = parse_optional_bool(row.get("gold_insufficient_evidence"))
            if add is None or mul is None or insufficient is None:
                add, mul, insufficient = decimal_cues_from_code(row.get("gold_code"))
        else:
            add, mul, insufficient = None, None, None
        rows.append(
            {
                "row_id": row["row_id"],
                "pred_code": row.get("gold_code"),
                "confidence": 1.0,
                "evidence": "oracle backend",
                "notes": "",
                "addition_style_evidence": add,
                "multiplication_style_evidence": mul,
                "insufficient_evidence": insufficient,
                "direct_procedure_evidence": None,
                "strategy_inferred_from_answer": None,
                "parse_success": row.get("gold_code") in profile.strategy_codes,
                "raw_response_text": "",
                "request_error": "",
            }
        )
    return pd.DataFrame(rows)


def run_predictions_backend(
    targets: pd.DataFrame,
    predictions_path: Path,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> pd.DataFrame:
    pred_df = read_csv_with_encoding(predictions_path, "auto")
    if "row_id" not in pred_df.columns:
        raise ValueError(f"{predictions_path} must include a row_id column")
    code_col = None
    for candidate in ("pred_code", "strategy_code", "code", "label"):
        if candidate in pred_df.columns:
            code_col = candidate
            break
    cue_cols = [
        "addition_style_evidence",
        "multiplication_style_evidence",
        "insufficient_evidence",
    ]
    provenance_cols = [
        "direct_procedure_evidence",
        "strategy_inferred_from_answer",
    ]
    has_cue_cols = all(column in pred_df.columns for column in cue_cols)
    if code_col is None and not (profile.name == "decimal" and has_cue_cols):
        raise ValueError(
            f"{predictions_path} must include one of pred_code, strategy_code, code, label"
            " or decimal cue columns"
        )
    pred_df = pred_df.copy()
    if code_col is not None:
        pred_df["pred_code"] = pred_df[code_col].map(profile.normalize_code)
    else:
        pred_df["pred_code"] = [
            derive_decimal_code_from_cues(
                parse_optional_bool(row["addition_style_evidence"]),
                parse_optional_bool(row["multiplication_style_evidence"]),
                parse_optional_bool(row["insufficient_evidence"]),
            )
            for row in pred_df.to_dict("records")
        ]
    if profile.name == "decimal":
        for idx, column in enumerate(cue_cols):
            if column not in pred_df.columns:
                pred_df[column] = [
                    decimal_cues_from_code(code)[idx]
                    for code in pred_df["pred_code"].tolist()
                ]
            else:
                pred_df[column] = pred_df[column].map(parse_optional_bool)
    else:
        for column in cue_cols:
            if column not in pred_df.columns:
                pred_df[column] = None
    for column in provenance_cols:
        if column not in pred_df.columns:
            pred_df[column] = None
        else:
            pred_df[column] = pred_df[column].map(parse_optional_bool)
    pred_df["parse_success"] = pred_df["pred_code"].isin(profile.strategy_codes)
    for column in ("confidence", "evidence", "notes", "raw_response_text", "request_error"):
        if column not in pred_df.columns:
            pred_df[column] = ""
    merged = targets[["row_id"]].merge(
        pred_df[
            [
                "row_id",
                "pred_code",
                "confidence",
                "evidence",
                "notes",
                "addition_style_evidence",
                "multiplication_style_evidence",
                "insufficient_evidence",
                "direct_procedure_evidence",
                "strategy_inferred_from_answer",
                "parse_success",
                "raw_response_text",
                "request_error",
            ]
        ],
        on="row_id",
        how="left",
    )
    merged["parse_success"] = merged["parse_success"].fillna(False)
    return merged


def make_gemini_client(args: argparse.Namespace) -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "Gemini backend requires google-genai. Install it in the active "
            "environment before using --backend gemini."
        ) from exc
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key, http_options={"api_version": args.api_version})


def make_gemini_config(
    args: argparse.Namespace,
    system_prompt: str,
    sample_idx: int,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> Any:
    from google.genai import types

    config_kwargs: Dict[str, Any] = {
        "system_instruction": system_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "seed": args.seed + sample_idx,
        "response_mime_type": "application/json",
        "response_schema": profile.response_schema,
    }
    if args.thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=args.thinking_level
        )
    else:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=args.thinking_budget
        )
    return types.GenerateContentConfig(**config_kwargs)


def make_batch_config_dict(
    args: argparse.Namespace,
    system_prompt: str,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "response_mime_type": "application/json",
        "response_schema": profile.response_schema,
    }
    if args.thinking_level:
        config["thinking_config"] = {"thinking_level": args.thinking_level}
    else:
        config["thinking_config"] = {"thinking_budget": args.thinking_budget}
    return config


def build_batch_inline_requests(
    request_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> List[Dict[str, Any]]:
    inline_requests = []
    for req in request_records:
        inline_requests.append(
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": str(req["user_prompt"])}],
                    }
                ],
                "config": make_batch_config_dict(
                    args,
                    str(req["system_instruction"]),
                    profile,
                ),
                "metadata": {
                    "row_id": str(req["row_id"]),
                    "request_key": str(req["request_key"]),
                    "prompt_version": str(req["prompt_version"]),
                    "domain": str(req.get("domain", profile.name)),
                    "evidence": str(req.get("evidence", "")),
                    "source_split": str(req["source_split"]),
                    "source_row": str(req["source_row"]),
                },
            }
        )
    return inline_requests


def batch_payload_size_bytes(inline_requests: Sequence[Mapping[str, Any]]) -> int:
    return len(json.dumps(inline_requests, ensure_ascii=False).encode("utf-8"))


def text_from_generate_response(response: Any) -> str:
    if not response:
        return ""
    text = getattr(response, "text", None)
    if text:
        return clean_text(text)
    if isinstance(response, Mapping):
        text = response.get("text")
        if text:
            return clean_text(text)
        candidates = response.get("candidates") or []
        if candidates:
            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )
            return "".join(clean_text(part.get("text", "")) for part in parts)
    return ""


def metadata_from_inline_response(inline_response: Any) -> Dict[str, str]:
    metadata = {}
    if isinstance(inline_response, Mapping):
        metadata = inline_response.get("metadata") or {}
    else:
        metadata = getattr(inline_response, "metadata", None) or {}
    return {str(key): clean_text(value) for key, value in dict(metadata).items()}


def response_from_inline_response(inline_response: Any) -> Any:
    if isinstance(inline_response, Mapping):
        return inline_response.get("response")
    return getattr(inline_response, "response", None)


def error_from_inline_response(inline_response: Any) -> Any:
    if isinstance(inline_response, Mapping):
        return inline_response.get("error")
    return getattr(inline_response, "error", None)


def parse_batch_inline_responses(
    inline_responses: Sequence[Any],
    request_records: Sequence[Mapping[str, Any]],
    profile: StrategyProfile = FRACTION_PROFILE,
) -> pd.DataFrame:
    request_by_row_id = {str(req["row_id"]): req for req in request_records}
    rows = []
    for idx, inline_response in enumerate(inline_responses):
        fallback_req = request_records[idx] if idx < len(request_records) else {}
        metadata = metadata_from_inline_response(inline_response)
        row_id = metadata.get("row_id") or clean_text(fallback_req.get("row_id", ""))
        request_error = batch_error_text(error_from_inline_response(inline_response))
        response_text = ""
        if not request_error:
            response_text = text_from_generate_response(
                response_from_inline_response(inline_response)
            )
        parsed = parse_model_response(response_text, profile)
        rows.append(
            {
                "row_id": row_id,
                "pred_code": parsed["pred_code"],
                "confidence": parsed["confidence"],
                "evidence": parsed["evidence"],
                "notes": parsed["notes"],
                "addition_style_evidence": parsed["addition_style_evidence"],
                "multiplication_style_evidence": parsed["multiplication_style_evidence"],
                "insufficient_evidence": parsed["insufficient_evidence"],
                "direct_procedure_evidence": parsed["direct_procedure_evidence"],
                "strategy_inferred_from_answer": parsed["strategy_inferred_from_answer"],
                "parse_success": parsed["parse_success"],
                "raw_response_text": response_text,
                "request_error": request_error,
                "request_key": metadata.get(
                    "request_key",
                    clean_text(request_by_row_id.get(row_id, fallback_req).get("request_key", "")),
                ),
            }
        )
    seen = {row["row_id"] for row in rows}
    for req in request_records:
        if str(req["row_id"]) in seen:
            continue
        rows.append(
            {
                "row_id": str(req["row_id"]),
                "pred_code": None,
                "confidence": None,
                "evidence": "",
                "notes": "",
                "addition_style_evidence": None,
                "multiplication_style_evidence": None,
                "insufficient_evidence": None,
                "direct_procedure_evidence": None,
                "strategy_inferred_from_answer": None,
                "parse_success": False,
                "raw_response_text": "",
                "request_error": "Missing batch response.",
                "request_key": str(req["request_key"]),
            }
        )
    return pd.DataFrame(rows)


def estimate_batch_cost(
    inline_requests: Sequence[Mapping[str, Any]],
    row_count: int,
    output_tokens_per_row: int,
) -> Dict[str, Any]:
    # A rough planning estimate. Tokenization varies by model, but 4 chars/token is
    # close enough for choosing between Standard and Batch on short text requests.
    request_chars = sum(len(json.dumps(req, ensure_ascii=False)) for req in inline_requests)
    estimated_input_tokens = request_chars / 4.0
    estimated_output_tokens = row_count * output_tokens_per_row
    return {
        "method": "rough_chars_per_token",
        "chars_per_token": 4.0,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "batch_input_usd": estimated_input_tokens / 1_000_000 * 0.125,
        "batch_output_usd": estimated_output_tokens / 1_000_000 * 0.75,
        "batch_total_usd": (
            estimated_input_tokens / 1_000_000 * 0.125
            + estimated_output_tokens / 1_000_000 * 0.75
        ),
    }


def extract_inline_responses_from_job(batch_job: Any, client: Any, out_dir: Path) -> List[Any]:
    dest = getattr(batch_job, "dest", None)
    if not dest and isinstance(batch_job, Mapping):
        dest = batch_job.get("dest")
    if not dest:
        return []

    inlined_responses = (
        dest.get("inlined_responses")
        if isinstance(dest, Mapping)
        else getattr(dest, "inlined_responses", None)
    )
    if inlined_responses:
        return list(inlined_responses)

    file_name = dest.get("file_name") if isinstance(dest, Mapping) else getattr(dest, "file_name", None)
    if not file_name:
        return []

    file_content = client.files.download(file=file_name)
    if isinstance(file_content, bytes):
        text = file_content.decode("utf-8")
    else:
        text = str(file_content)
    (out_dir / "batch_results.jsonl").write_text(text, encoding="utf-8")
    responses = []
    for line in text.splitlines():
        if line.strip():
            responses.append(json.loads(line))
    return responses


def run_gemini_batch_backend(
    targets: pd.DataFrame,
    request_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> pd.DataFrame:
    inline_requests = build_batch_inline_requests(request_records, args, profile)
    payload_size = batch_payload_size_bytes(inline_requests)
    if payload_size > args.batch_inline_limit_bytes:
        raise ValueError(
            "Inline Gemini Batch payload is too large: "
            f"{payload_size} bytes > {args.batch_inline_limit_bytes} bytes. "
            "Split the run or add file-input batch support before submitting."
        )

    (out_dir / "batch_inline_requests.json").write_text(
        json.dumps(inline_requests, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    batch_manifest = {
        "backend": "gemini_batch",
        "domain": profile.name,
        "evidence": (
            str(request_records[0].get("evidence", "")) if request_records else ""
        ),
        "model": args.model,
        "row_count": int(len(targets)),
        "request_count": len(inline_requests),
        "payload_size_bytes": payload_size,
        "inline_limit_bytes": args.batch_inline_limit_bytes,
        "thinking_level": args.thinking_level,
        "thinking_budget": args.thinking_budget if not args.thinking_level else None,
        "estimated_cost": estimate_batch_cost(
            inline_requests, len(inline_requests), args.max_output_tokens
        ),
    }
    (out_dir / "batch_manifest.json").write_text(
        json.dumps(batch_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    client = make_gemini_client(args)
    if args.batch_job_name:
        batch_job = client.batches.get(name=args.batch_job_name)
    else:
        display_name = args.batch_display_name or f"{profile.name}-strategy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        batch_job = client.batches.create(
            model=args.model,
            src=inline_requests,
            config={"display_name": display_name},
        )

    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    start_time = time.time()
    while state_name(getattr(batch_job, "state", None)) not in terminal_states:
        elapsed = time.time() - start_time
        (out_dir / "batch_job.json").write_text(
            json.dumps(response_to_raw_payload(batch_job), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"Batch job {getattr(batch_job, 'name', args.batch_job_name)} "
            f"state={state_name(getattr(batch_job, 'state', None))} elapsed={elapsed:.0f}s",
            flush=True,
        )
        if elapsed > args.batch_timeout_seconds:
            break
        time.sleep(args.batch_poll_seconds)
        batch_job = client.batches.get(name=batch_job.name)

    (out_dir / "batch_job.json").write_text(
        json.dumps(response_to_raw_payload(batch_job), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    final_state = state_name(getattr(batch_job, "state", None))
    if final_state != "JOB_STATE_SUCCEEDED":
        request_error = f"Batch job ended with state {final_state}."
        job_error = batch_error_text(getattr(batch_job, "error", None))
        if job_error:
            request_error = f"{request_error} {job_error}"
        return pd.DataFrame(
            [
                {
                    "row_id": str(req["row_id"]),
                    "pred_code": None,
                    "confidence": None,
                    "evidence": "",
                    "notes": "",
                    "addition_style_evidence": None,
                    "multiplication_style_evidence": None,
                    "insufficient_evidence": None,
                    "direct_procedure_evidence": None,
                    "strategy_inferred_from_answer": None,
                    "parse_success": False,
                    "raw_response_text": "",
                    "request_error": request_error,
                    "request_key": str(req["request_key"]),
                }
                for req in request_records
            ]
        )

    inline_responses = extract_inline_responses_from_job(batch_job, client, out_dir)
    return parse_batch_inline_responses(inline_responses, request_records, profile)


def run_gemini_backend(
    targets: pd.DataFrame,
    request_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> pd.DataFrame:
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_or_needed = []
    needs_client = False
    for req in request_records:
        path = cache_path(cache_dir, str(req["request_key"]))
        cached_or_needed.append(path)
        if args.force or not path.exists():
            needs_client = True

    client = make_gemini_client(args) if needs_client else None
    rows = []
    for sample_idx, (target_row, req, path) in enumerate(
        zip(targets.to_dict("records"), request_records, cached_or_needed)
    ):
        if path.exists() and not args.force:
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            assert client is not None
            config = make_gemini_config(
                args,
                str(req["system_instruction"]),
                sample_idx,
                profile,
            )
            response_text = ""
            raw_payload = None
            last_error = ""
            for attempt in range(1, args.max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=args.model,
                        contents=str(req["user_prompt"]),
                        config=config,
                    )
                    raw_payload = response_to_raw_payload(response)
                    response_text = getattr(response, "text", "") or ""
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt >= args.max_retries:
                        break
                    time.sleep(args.retry_base_seconds * (2 ** (attempt - 1)))
            payload = {
                "request": req,
                "raw_response": raw_payload,
                "response_text": response_text,
                "request_error": last_error,
            }
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        parsed = parse_model_response(
            clean_text(payload.get("response_text", "")),
            profile,
        )
        rows.append(
            {
                "row_id": target_row["row_id"],
                "pred_code": parsed["pred_code"],
                "confidence": parsed["confidence"],
                "evidence": parsed["evidence"],
                "notes": parsed["notes"],
                "addition_style_evidence": parsed["addition_style_evidence"],
                "multiplication_style_evidence": parsed["multiplication_style_evidence"],
                "insufficient_evidence": parsed["insufficient_evidence"],
                "direct_procedure_evidence": parsed["direct_procedure_evidence"],
                "strategy_inferred_from_answer": parsed["strategy_inferred_from_answer"],
                "parse_success": parsed["parse_success"],
                "raw_response_text": clean_text(payload.get("response_text", "")),
                "request_error": clean_text(payload.get("request_error", "")),
            }
        )
    return pd.DataFrame(rows)


def label_counts(
    series: pd.Series,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> Dict[str, int]:
    counts = series.value_counts(dropna=True).to_dict()
    return {code: int(counts.get(code, 0)) for code in profile.strategy_codes}


def distribution_rows(
    gold_counts: Mapping[str, int],
    pred_counts: Mapping[str, int],
    profile: StrategyProfile = FRACTION_PROFILE,
) -> List[Dict[str, Any]]:
    gold_total = sum(gold_counts.values())
    pred_total = sum(pred_counts.values())
    rows = []
    for code in profile.strategy_codes:
        gold_prop = gold_counts[code] / gold_total if gold_total else 0.0
        pred_prop = pred_counts[code] / pred_total if pred_total else 0.0
        rows.append(
            {
                "strategy_code": code,
                "gold_count": int(gold_counts[code]),
                "pred_count": int(pred_counts[code]),
                "gold_prop": gold_prop,
                "pred_prop": pred_prop,
                "abs_prop_diff": abs(gold_prop - pred_prop),
            }
        )
    return rows


def macro_f1_rows(
    df: pd.DataFrame,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> Tuple[List[Dict[str, Any]], float, float]:
    rows = []
    f1_all = []
    f1_present = []
    for code in profile.strategy_codes:
        gold = df["gold_code"] == code
        pred = df["pred_code"] == code
        tp = int((gold & pred).sum())
        fp = int((~gold & pred).sum())
        fn = int((gold & ~pred).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int(gold.sum())
        row = {
            "strategy_code": code,
            "support": support,
            "predicted": int(pred.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        rows.append(row)
        f1_all.append(f1)
        if support > 0:
            f1_present.append(f1)
    return rows, sum(f1_all) / len(f1_all), (sum(f1_present) / len(f1_present) if f1_present else 0.0)


def optional_bool_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([None] * len(df), index=df.index)
    return df[column].map(parse_optional_bool)


def compute_decimal_binary_metrics(enriched: pd.DataFrame) -> Dict[str, Any]:
    flag_specs = [
        (
            "strat_add",
            "gold_addition_style_evidence",
            "addition_style_evidence",
        ),
        (
            "strat_mul",
            "gold_multiplication_style_evidence",
            "multiplication_style_evidence",
        ),
    ]
    gold_add = optional_bool_column(enriched, "gold_addition_style_evidence")
    gold_mul = optional_bool_column(enriched, "gold_multiplication_style_evidence")
    pred_add = optional_bool_column(enriched, "addition_style_evidence")
    pred_mul = optional_bool_column(enriched, "multiplication_style_evidence")

    valid_joint = gold_add.notna() & gold_mul.notna()
    parsed_joint = pred_add.notna() & pred_mul.notna()
    joint_correct = (
        valid_joint
        & pred_add.eq(gold_add)
        & pred_mul.eq(gold_mul)
    )
    valid_joint_count = int(valid_joint.sum())
    primary_accuracy = (
        float(joint_correct.sum() / valid_joint_count)
        if valid_joint_count
        else None
    )

    per_flag = []
    for flag_name, gold_col, pred_col in flag_specs:
        gold = optional_bool_column(enriched, gold_col)
        pred = optional_bool_column(enriched, pred_col)
        valid = gold.notna()
        valid_count = int(valid.sum())
        gold_true = valid & gold.eq(True)
        gold_false = valid & gold.eq(False)
        pred_true = valid & pred.eq(True)
        pred_false = valid & pred.eq(False)
        tp = int((gold_true & pred_true).sum())
        fp = int((gold_false & pred_true).sum())
        fn = int((gold_true & ~pred_true).sum())
        tn = int((gold_false & pred_false).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / valid_count if valid_count else None
        per_flag.append(
            {
                "flag": flag_name,
                "gold_column": gold_col,
                "pred_column": pred_col,
                "valid_gold_count": valid_count,
                "support": int(gold_true.sum()),
                "predicted_positive": int(pred_true.sum()),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return {
        "primary_metric": "binary_joint_accuracy",
        "primary_accuracy": primary_accuracy,
        "binary_joint_valid_gold_count": valid_joint_count,
        "binary_joint_parse_success_count": int((valid_joint & parsed_joint).sum()),
        "per_flag": per_flag,
    }


def load_reference_counts(
    path: Optional[str],
    encoding: str,
    profile: StrategyProfile = FRACTION_PROFILE,
) -> Optional[Dict[str, int]]:
    if not path:
        return None
    df = read_csv_with_encoding(Path(path), encoding)
    if profile.name == "decimal" and {"strat_add", "strat_mul"}.issubset(df.columns):
        return label_counts(
            pd.Series([decimal_gold_code(row) for row in df.to_dict("records")]),
            profile,
        )
    if "code" not in df.columns:
        raise ValueError(f"{path} must include a code column")
    return label_counts(df["code"].map(profile.normalize_code), profile)


def compute_metrics(
    enriched: pd.DataFrame,
    reference_counts: Optional[Mapping[str, int]] = None,
    profile: StrategyProfile = FRACTION_PROFILE,
    evidence_mode: str = "trace",
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    has_gold = enriched["gold_code"].notna().any()
    valid_pred = enriched["pred_code"].isin(profile.strategy_codes)
    pred_counts = label_counts(enriched.loc[valid_pred, "pred_code"], profile)

    has_reference = reference_counts is not None or has_gold
    if reference_counts is None and has_gold:
        reference_counts = label_counts(enriched["gold_code"], profile)

    if reference_counts is None:
        reference_counts = {code: 0 for code in profile.strategy_codes}

    distribution = pd.DataFrame(distribution_rows(reference_counts, pred_counts, profile))
    tvd = 0.5 * float(distribution["abs_prop_diff"].sum()) if has_reference else None

    if has_gold:
        enriched = enriched.copy()
        enriched["correct"] = enriched["gold_code"] == enriched["pred_code"]
        accuracy = float(enriched["correct"].mean()) if len(enriched) else 0.0
        per_code, macro_f1_all, macro_f1_present = macro_f1_rows(enriched, profile)
        confusion = pd.crosstab(
            enriched["gold_code"],
            enriched["pred_code"].where(valid_pred, "INVALID"),
            dropna=False,
        )
        for row_code in profile.strategy_codes:
            if row_code not in confusion.index:
                confusion.loc[row_code] = 0
        for col_code in (*profile.strategy_codes, "INVALID"):
            if col_code not in confusion.columns:
                confusion[col_code] = 0
        confusion = confusion.loc[
            list(profile.strategy_codes),
            [*profile.strategy_codes, "INVALID"],
        ]
    else:
        accuracy = None
        macro_f1_all = None
        macro_f1_present = None
        per_code = []
        confusion = pd.DataFrame()

    metrics = {
        "prompt_version": profile.prompt_version,
        "domain": profile.name,
        "evidence": evidence_mode,
        "row_count": int(len(enriched)),
        "has_gold": bool(has_gold),
        "parse_success_count": int(valid_pred.sum()),
        "parse_success_rate": float(valid_pred.mean()) if len(enriched) else 0.0,
        "accuracy": accuracy,
        "macro_f1_all_codes": macro_f1_all,
        "macro_f1_present_codes": macro_f1_present,
        "distribution_tvd_to_reference": tvd,
        "reference_counts": {
            code: int(reference_counts[code]) for code in profile.strategy_codes
        },
        "pred_counts": pred_counts,
        "per_code": per_code,
    }
    if profile.name == "decimal":
        binary_metrics = compute_decimal_binary_metrics(enriched)
        metrics.update(
            {
                "primary_metric": binary_metrics["primary_metric"],
                "primary_accuracy": binary_metrics["primary_accuracy"],
                "binary_joint_valid_gold_count": binary_metrics[
                    "binary_joint_valid_gold_count"
                ],
                "binary_joint_parse_success_count": binary_metrics[
                    "binary_joint_parse_success_count"
                ],
                "decimal_binary_metrics": binary_metrics["per_flag"],
            }
        )
    return metrics, distribution, confusion, enriched


def write_prompt_file(
    out_dir: Path,
    system_prompt: str,
    examples: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
    profile: StrategyProfile = FRACTION_PROFILE,
    evidence_mode: str = "trace",
) -> None:
    content = "\n".join(
        [
            f"# {profile.prompt_version}",
            "",
            f"- domain: `{profile.name}`",
            f"- evidence: `{evidence_mode}`",
            f"- model default: `{args.model}`",
            f"- few-shot examples: {len(examples)}",
            f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## System Instruction",
            "",
            system_prompt,
            "",
            "## User Message Template",
            "",
            "```text",
            user_prompt_template(profile, evidence_mode),
            "```",
            "",
        ]
    )
    (out_dir / "prompt.md").write_text(content, encoding="utf-8")


def write_summary(
    out_dir: Path,
    metrics: Optional[Mapping[str, Any]],
    args: argparse.Namespace,
    profile: StrategyProfile = FRACTION_PROFILE,
    evidence_mode: str = "trace",
) -> None:
    lines = [
        f"# {profile.title}",
        "",
        f"- domain: `{profile.name}`",
        f"- evidence: `{evidence_mode}`",
        f"- backend: `{args.backend}`",
        f"- model: `{args.model}`",
        f"- prompt_version: `{profile.prompt_version}`",
        f"- output_dir: `{out_dir}`",
        "",
    ]
    if metrics is not None:
        tvd = metrics["distribution_tvd_to_reference"]
        tvd_text = f"{tvd:.4f}" if tvd is not None else "n/a"
        primary_metric = metrics.get("primary_metric")
        primary_accuracy = metrics.get("primary_accuracy")
        primary_accuracy_text = (
            f"{primary_accuracy:.4f}" if isinstance(primary_accuracy, (int, float)) else "n/a"
        )
        lines.extend(
            [
                "## Metrics",
                "",
                f"- rows: {metrics['row_count']}",
                f"- parse_success_rate: {metrics['parse_success_rate']:.4f}",
                *(
                    [
                        f"- primary_metric: `{primary_metric}`",
                        f"- primary_accuracy: {primary_accuracy_text}",
                    ]
                    if primary_metric
                    else []
                ),
                f"- accuracy: {metrics['accuracy'] if metrics['accuracy'] is not None else 'n/a'}",
                f"- distribution_tvd_to_reference: {tvd_text}",
                "",
            ]
        )
    lines.extend(
        [
            "## Files",
            "",
            "- `prompt.md`: exact system prompt and user-message template",
            "- `requests.jsonl`: cached request payloads, one per row",
            "- `predictions.csv`: extracted labels, when a prediction backend was run",
            "- `metrics.json`: accuracy and distribution metrics, when labels or a reference exist",
            "- `strategy_distribution.csv`: reference versus extracted strategy distribution",
            "- `confusion_matrix.csv`: gold-by-predicted table when gold labels are available",
            "- `decimal_binary_metrics.csv`: per-flag decimal BSS metrics when `--domain decimal` is used",
            "- `gemini_usage.json`: token counts and estimated cost for sync Gemini runs",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and run domain-aware strategy extraction for procedural alignment."
    )
    parser.add_argument(
        "--domain",
        choices=sorted(PROFILES),
        default="fraction",
        help="Strategy label domain. Fraction is the backward-compatible default.",
    )
    parser.add_argument(
        "--evidence",
        choices=EVIDENCE_CHOICES,
        default="auto",
        help=(
            "Evidence source for classification. auto uses trace for fraction "
            "and visible work for decimal."
        ),
    )
    parser.add_argument("--train_csv", default=str(EVAL_DIR / "data" / "human_ft" / "data_train.csv"))
    parser.add_argument("--val_csv", default=str(EVAL_DIR / "data" / "human_ft" / "data_val.csv"))
    parser.add_argument("--decimal_csv", default=DEFAULT_DECIMAL_CSV)
    parser.add_argument("--human_encoding", default="cp1252")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")

    parser.add_argument(
        "--input_csv",
        default="",
        help="Optional model rollout CSV. If set, classify this file instead of human split rows.",
    )
    parser.add_argument("--input_encoding", default="auto")
    parser.add_argument("--problem_col", default="prob")
    parser.add_argument("--trace_col", default="response_nl")
    parser.add_argument("--answer_col", default="resp")
    parser.add_argument(
        "--operation_col",
        default="",
        help="Optional decimal operation column for --input_csv prompts.",
    )
    parser.add_argument(
        "--op1_col",
        default="",
        help="Optional first decimal operand column for --input_csv prompts.",
    )
    parser.add_argument(
        "--op2_col",
        default="",
        help="Optional second decimal operand column for --input_csv prompts.",
    )
    parser.add_argument(
        "--work_operand_1_col",
        default="work_operand_1",
        help="Decimal visible-work operand 1 column for --evidence work/combined.",
    )
    parser.add_argument(
        "--work_operand_2_col",
        default="work_operand_2",
        help="Decimal visible-work operand 2 column for --evidence work/combined.",
    )
    parser.add_argument(
        "--work_answer_col",
        default="work_answer",
        help="Decimal visible-work answer column for --evidence work/combined.",
    )
    parser.add_argument("--gold_code_col", default="code")
    parser.add_argument("--id_col", default="")
    parser.add_argument(
        "--reference_human_csv",
        default="",
        help="Optional raw human CSV with code column for distribution comparison.",
    )

    parser.add_argument(
        "--backend",
        choices=["requests", "oracle", "predictions", "gemini", "gemini_batch"],
        default="requests",
        help=(
            "requests writes prompts only; oracle tests metrics for free; "
            "gemini calls the sync API; gemini_batch uses async Batch API."
        ),
    )
    parser.add_argument("--predictions_csv", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_version", default="v1beta")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_output_tokens", type=int, default=128)
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=0,
        help="Gemini 2.5 thinking budget. 0 disables thinking; -1 uses dynamic thinking.",
    )
    parser.add_argument(
        "--thinking_level",
        default="",
        help="Gemini 3 thinking level override, such as low or minimal.",
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_base_seconds", type=float, default=2.0)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="Ignore Gemini cache entries.")
    parser.add_argument(
        "--gemini_input_usd_per_million",
        type=float,
        default=DEFAULT_GEMINI_INPUT_USD_PER_MILLION,
        help="Input-token price used for gemini_usage.json cost estimates.",
    )
    parser.add_argument(
        "--gemini_output_usd_per_million",
        type=float,
        default=DEFAULT_GEMINI_OUTPUT_USD_PER_MILLION,
        help="Output-token price used for gemini_usage.json cost estimates.",
    )
    parser.add_argument(
        "--batch_poll_seconds",
        type=float,
        default=30.0,
        help="Seconds between Gemini Batch job status polls.",
    )
    parser.add_argument(
        "--batch_timeout_seconds",
        type=float,
        default=24 * 60 * 60,
        help="Maximum seconds to wait for a Gemini Batch job.",
    )
    parser.add_argument(
        "--batch_inline_limit_bytes",
        type=int,
        default=DEFAULT_BATCH_INLINE_LIMIT_BYTES,
        help="Maximum inline Batch API request payload size.",
    )
    parser.add_argument(
        "--batch_display_name",
        default="",
        help="Optional Gemini Batch display name.",
    )
    parser.add_argument(
        "--batch_job_name",
        default="",
        help="Optional existing Gemini Batch job name to poll instead of submitting.",
    )

    parser.add_argument("--few_shot_per_code", type=int, default=0)
    parser.add_argument(
        "--few_shot_preset",
        choices=["human_shortest", "canonical"],
        default="human_shortest",
        help="Few-shot example source. canonical avoids noisy human utterances.",
    )
    parser.add_argument("--few_shot_max_chars", type=int, default=260)
    parser.add_argument("--max_trace_chars", type=int, default=900)
    parser.add_argument("--sample_per_code", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--out_dir",
        default=str(DEFAULT_OUT_DIR),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    profile = get_profile(args.domain)
    evidence_mode = resolve_evidence_mode(args, profile)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets, train_df = load_targets(args, profile, evidence_mode)
    targets = sample_targets(targets, args, profile)
    examples = select_few_shot_examples(
        train_df,
        args.few_shot_per_code,
        args.seed,
        args.few_shot_max_chars,
        args.few_shot_preset,
        profile,
    )
    system_prompt = build_system_prompt(examples, profile, evidence_mode)
    write_prompt_file(out_dir, system_prompt, examples, args, profile, evidence_mode)

    request_records: List[Mapping[str, Any]] = []
    for row in targets.to_dict("records"):
        user_prompt = build_user_prompt(row, args.max_trace_chars, profile, evidence_mode)
        request_records.append(
            make_request_record(row, system_prompt, user_prompt, args, profile, evidence_mode)
        )
    write_jsonl(out_dir / "requests.jsonl", request_records)

    (out_dir / "targets.csv").write_text(targets.to_csv(index=False), encoding="utf-8")

    if args.backend == "requests":
        write_summary(out_dir, None, args, profile, evidence_mode)
        print(f"Wrote {len(request_records)} requests to {out_dir / 'requests.jsonl'}")
        return 0

    if args.backend == "oracle":
        predictions = run_oracle_backend(targets, profile)
    elif args.backend == "predictions":
        if not args.predictions_csv:
            parser.error("--backend predictions requires --predictions_csv")
        predictions = run_predictions_backend(targets, Path(args.predictions_csv), profile)
    elif args.backend == "gemini":
        predictions = run_gemini_backend(targets, request_records, args, out_dir, profile)
    elif args.backend == "gemini_batch":
        predictions = run_gemini_batch_backend(
            targets,
            request_records,
            args,
            out_dir,
            profile,
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    if args.backend == "gemini":
        write_gemini_usage_summary(out_dir, args)

    enriched = targets.merge(predictions, on="row_id", how="left")
    reference_counts = load_reference_counts(
        args.reference_human_csv or None,
        args.human_encoding,
        profile,
    )
    metrics, distribution, confusion, enriched = compute_metrics(
        enriched,
        reference_counts,
        profile,
        evidence_mode,
    )

    enriched.to_csv(out_dir / "predictions.csv", index=False)
    distribution.to_csv(out_dir / "strategy_distribution.csv", index=False)
    if profile.name == "decimal" and metrics.get("decimal_binary_metrics") is not None:
        pd.DataFrame(metrics["decimal_binary_metrics"]).to_csv(
            out_dir / "decimal_binary_metrics.csv",
            index=False,
        )
    if not confusion.empty:
        confusion.to_csv(out_dir / "confusion_matrix.csv")
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary(out_dir, metrics, args, profile, evidence_mode)

    print(
        "Completed "
        f"{args.backend}: rows={metrics['row_count']} "
        f"parse_success={metrics['parse_success_rate']:.3f} "
        f"accuracy={metrics['accuracy'] if metrics['accuracy'] is not None else 'n/a'} "
        f"tvd={metrics['distribution_tvd_to_reference'] if metrics['distribution_tvd_to_reference'] is not None else 'n/a'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
