"""Shared utilities for reproducible NeurIPS arithmetic evaluations."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from procedural_alignment import frontier_arithmetic_baseline as fab
from procedural_alignment.strategy_mapping import map_sp2013_strat_to_family

EVAL_DIR = ROOT_DIR / "eval"
DEFAULT_FRONTIER_OUTPUTS_DIR = EVAL_DIR / "outputs"
DEFAULT_CENTAUR_FRACTION_CSV = ROOT_DIR / "Centaur" / "runs" / "centaur_vllm_fraction_s100" / "outputs.csv"
DEFAULT_CENTAUR_DECIMAL_CSV = ROOT_DIR / "Centaur" / "runs" / "centaur_vllm_decimal_s100" / "outputs.csv"
DEFAULT_UMA_CSV = (
    ROOT_DIR
    / "results"
    / "procedural_alignment"
    / "uma_cognitive_model_sp2013_bss2021_20260516"
    / "uma_cognitive_model_sp2013_bss2021_trials.csv.gz"
)
DEFAULT_FRACTION_HUMAN_CSV = ROOT_DIR / "data" / "siegler_fraction_human.csv"
CENTAUR_SOURCE_KEY = "centaur_70b"

FRACTION_DOMAINS = ("fraction",)
DECIMAL_DOMAINS = ("decimal",)
ALL_DOMAINS = ("fraction", "decimal")
OPERATION_LABELS = {
    "+": "Add",
    "-": "Sub",
    "*": "Mul",
    ":": "Div",
    "/": "Div",
    "÷": "Div",
    "add": "Add",
    "addition": "Add",
    "sub": "Sub",
    "subtract": "Sub",
    "subtraction": "Sub",
    "mul": "Mul",
    "mult": "Mul",
    "multiply": "Mul",
    "multiplication": "Mul",
    "div": "Div",
    "divide": "Div",
    "division": "Div",
    "Add": "Add",
    "Sub": "Sub",
    "Mul": "Mul",
    "Div": "Div",
}
DECIMAL_PROBLEM_RE = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*([+*])\s*(-?\d+(?:\.\d+)?)\s*")


@dataclass(frozen=True)
class SourceFrame:
    source_key: str
    source_family: str
    source_model: str
    domain: str
    path: Path
    frame: pd.DataFrame


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", clean_text(text)).strip("_").lower()
    return slug or "source"


def resolve_path(value: str | Path, root: Path = ROOT_DIR) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def domains_from_arg(value: str) -> tuple[str, ...]:
    if value == "both":
        return ALL_DOMAINS
    if value not in ALL_DOMAINS:
        raise ValueError(f"Unknown domain: {value}")
    return (value,)


def fraction_to_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def canonical_fraction_problem(problem: Any) -> str:
    left, op, right = fab.parse_fraction_problem(clean_text(problem))
    op_text = ":" if op == ":" else op
    return f"{fraction_to_text(left)}{op_text}{fraction_to_text(right)}"


def canonical_decimal_problem(problem: Any) -> str:
    match = DECIMAL_PROBLEM_RE.fullmatch(clean_text(problem))
    if not match:
        raise ValueError(f"Could not parse decimal problem: {problem!r}")
    left, op, right = match.groups()
    return f"{left}{op}{right}"


def canonical_problem(problem: Any, domain: str) -> str:
    if domain == "fraction":
        return canonical_fraction_problem(problem)
    if domain == "decimal":
        return canonical_decimal_problem(problem)
    raise ValueError(f"Unknown domain: {domain}")


def decimal_problem_parts(problem: str) -> tuple[str, str, str]:
    match = DECIMAL_PROBLEM_RE.fullmatch(clean_text(problem))
    if not match:
        raise ValueError(f"Could not parse decimal problem: {problem!r}")
    left, op, right = match.groups()
    return left, OPERATION_LABELS[op], right


def metadata_for_problem(problem: str, domain: str) -> fab.ProblemRecord:
    canonical = canonical_problem(problem, domain)
    if domain == "fraction":
        return fab.fraction_metadata(canonical)
    return fab.decimal_metadata(canonical)


def normalize_operation(value: Any, problem: str | None = None, domain: str | None = None) -> str:
    text = clean_text(value)
    if text in OPERATION_LABELS:
        return OPERATION_LABELS[text]
    lower = text.lower()
    if lower in OPERATION_LABELS:
        return OPERATION_LABELS[lower]
    if problem and domain:
        return metadata_for_problem(problem, domain).operation
    return text


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.map(clean_text).str.lower()
    return text.isin({"1", "1.0", "true", "t", "yes", "y"})


def read_csv_auto(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", **kwargs)


def require_columns(frame: pd.DataFrame, path: Path | str, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def normalize_output_frame(
    raw: pd.DataFrame,
    *,
    domain: str,
    source_key: str,
    source_family: str,
    source_model: str,
    source_path: Path,
    problem_col: str = "problem",
    response_col: str = "model_response",
    answer_col: str = "parsed_answer",
    correct_col: str = "is_correct",
    sample_col: str = "sample_idx",
    source_row_col: str = "",
) -> SourceFrame:
    require_columns(raw, source_path, [problem_col, correct_col])
    out = pd.DataFrame(index=raw.index)
    out["source_key"] = source_key
    out["source_family"] = source_family
    out["source_model"] = source_model
    out["domain"] = domain
    out["source_path"] = str(source_path)
    out["source_row"] = raw[source_row_col].map(clean_text) if source_row_col and source_row_col in raw.columns else raw.index.astype(str)
    out["sample_idx"] = raw[sample_col] if sample_col in raw.columns else raw.index
    out["problem_raw"] = raw[problem_col].map(clean_text)
    out["problem"] = out["problem_raw"].map(lambda value: canonical_problem(value, domain))
    metadata = [metadata_for_problem(problem, domain) for problem in out["problem"]]
    out["operation"] = [
        normalize_operation(raw.iloc[idx].get("operation", ""), record.problem, domain) or record.operation
        for idx, record in enumerate(metadata)
    ]
    if domain == "fraction":
        denom_fallback = pd.Series([record.operands for record in metadata], index=raw.index)
        denom_source = raw["denom_type"] if "denom_type" in raw.columns else denom_fallback
        out["denom_type"] = denom_source.map(clean_text).replace("", pd.NA).fillna(denom_fallback)
        out["operands"] = out["denom_type"]
    else:
        operand_fallback = pd.Series([record.operands for record in metadata], index=raw.index)
        operand_source = raw["operands"] if "operands" in raw.columns else operand_fallback
        out["operands"] = operand_source.map(clean_text).replace("", pd.NA).fillna(operand_fallback)
        out["denom_type"] = ""
        op1 = []
        op2 = []
        for problem in out["problem"]:
            left, _, right = decimal_problem_parts(problem)
            op1.append(left)
            op2.append(right)
        out["op1"] = op1
        out["op2"] = op2
    if "op1" not in out.columns:
        out["op1"] = ""
    if "op2" not in out.columns:
        out["op2"] = ""
    correct_fallback = pd.Series([record.correct_answer for record in metadata], index=raw.index)
    correct_answer_source = raw["correct_answer"] if "correct_answer" in raw.columns else correct_fallback
    out["correct_answer"] = correct_answer_source.map(clean_text).replace("", pd.NA).fillna(correct_fallback)
    out["parsed_answer"] = raw[answer_col].map(clean_text) if answer_col and answer_col in raw.columns else ""
    out["model_response"] = raw[response_col].map(clean_text) if response_col and response_col in raw.columns else ""
    out["is_correct"] = coerce_bool_series(raw[correct_col])
    return SourceFrame(source_key, source_family, source_model, domain, source_path, out)


def load_frontier_sources(outputs_dir: Path, domains: Sequence[str]) -> list[SourceFrame]:
    sources: list[SourceFrame] = []
    for domain in domains:
        domain_dir = outputs_dir / domain
        if not domain_dir.exists():
            continue
        for path in sorted(domain_dir.glob("*.csv")):
            raw = read_csv_auto(path)
            if raw.empty:
                continue
            model = clean_text(raw.get("model", pd.Series([path.stem])).iloc[0])
            source_family = "centaur" if path.stem == CENTAUR_SOURCE_KEY or "centaur" in model.lower() else "frontier"
            sources.append(
                normalize_output_frame(
                    raw,
                    domain=domain,
                    source_key=path.stem,
                    source_family=source_family,
                    source_model=model or path.stem,
                    source_path=path,
                    problem_col="problem",
                    response_col="model_response",
                    answer_col="parsed_answer",
                    correct_col="is_correct",
                    sample_col="sample_idx",
                )
            )
    return sources


def load_centaur_sources(
    fraction_csv: Path,
    decimal_csv: Path,
    domains: Sequence[str],
) -> list[SourceFrame]:
    sources: list[SourceFrame] = []
    for domain, path in (("fraction", fraction_csv), ("decimal", decimal_csv)):
        if domain not in domains or not path.exists():
            continue
        raw = read_csv_auto(path)
        if raw.empty:
            continue
        model = clean_text(raw.get("model", pd.Series(["Centaur-70B"])).iloc[0]) or "Centaur-70B"
        sources.append(
            normalize_output_frame(
                raw,
                domain=domain,
                source_key=CENTAUR_SOURCE_KEY,
                source_family="centaur",
                source_model=model,
                source_path=path,
                problem_col="problem",
                response_col="model_response",
                answer_col="parsed_answer",
                correct_col="is_correct",
                sample_col="sample_idx",
                source_row_col="row_index",
            )
        )
    return sources


def dedupe_sources(sources: Sequence[SourceFrame]) -> list[SourceFrame]:
    """Keep the first source for each domain/source_key pair."""
    deduped: list[SourceFrame] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        key = (source.domain, source.source_key)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def load_uma_sources(path: Path, domains: Sequence[str]) -> list[SourceFrame]:
    if not path.exists():
        return []
    raw_all = read_csv_auto(path)
    require_columns(raw_all, path, {"domain", "prob", "answer", "is_correct"})
    sources: list[SourceFrame] = []
    for domain in domains:
        raw = raw_all[raw_all["domain"].map(clean_text) == domain].reset_index(drop=True)
        if raw.empty:
            continue
        raw = raw.rename(
            columns={
                "prob": "problem",
                "answer": "parsed_answer",
            }
        )
        raw["model_response"] = ""
        raw["sample_idx"] = raw.get("seed", raw.index)
        sources.append(
            normalize_output_frame(
                raw,
                domain=domain,
                source_key="uma_cognitive_model",
                source_family="uma",
                source_model="UMA cognitive model",
                source_path=path,
                problem_col="problem",
                response_col="model_response",
                answer_col="parsed_answer",
                correct_col="is_correct",
                sample_col="sample_idx",
            )
        )
    return sources


def load_all_sources(
    *,
    domains: Sequence[str],
    frontier_outputs_dir: Path = DEFAULT_FRONTIER_OUTPUTS_DIR,
    centaur_fraction_csv: Path = DEFAULT_CENTAUR_FRACTION_CSV,
    centaur_decimal_csv: Path = DEFAULT_CENTAUR_DECIMAL_CSV,
    uma_csv: Path = DEFAULT_UMA_CSV,
    include_frontier: bool = True,
    include_centaur: bool = True,
    include_uma: bool = True,
) -> list[SourceFrame]:
    sources: list[SourceFrame] = []
    if include_frontier:
        sources.extend(load_frontier_sources(frontier_outputs_dir, domains))
    if include_centaur:
        sources.extend(load_centaur_sources(centaur_fraction_csv, centaur_decimal_csv, domains))
    if include_uma:
        sources.extend(load_uma_sources(uma_csv, domains))
    return dedupe_sources(sources)


def load_sp2013_human_accuracy(path: Path = DEFAULT_FRACTION_HUMAN_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    human = read_csv_auto(path)
    require_columns(human, path, {"prob", "acc", "operands", "strat"})
    human = human.copy()
    human["problem"] = human["prob"].map(lambda value: canonical_problem(value, "fraction"))
    human["operation"] = human.get("operation", human.get("oper", "")).map(lambda value: normalize_operation(value, None, None))
    missing_operation = human["operation"].eq("")
    if missing_operation.any():
        human.loc[missing_operation, "operation"] = human.loc[missing_operation, "problem"].map(
            lambda value: metadata_for_problem(value, "fraction").operation
        )
    human["denom_type"] = human["operands"].map(clean_text)
    human["acc"] = pd.to_numeric(human["acc"], errors="coerce")
    human = human.dropna(subset=["acc"])
    per_problem = (
        human.groupby(["problem"], as_index=False)
        .agg(human_acc=("acc", "mean"), human_rows=("acc", "size"))
        .sort_values("problem")
    )
    per_cell = (
        human.groupby(["operation", "denom_type"], as_index=False)
        .agg(human_acc=("acc", "mean"), human_rows=("acc", "size"))
        .sort_values(["operation", "denom_type"])
    )
    return per_problem, per_cell


def load_sp2013_strategy_family_distribution(path: Path = DEFAULT_FRACTION_HUMAN_CSV) -> pd.DataFrame:
    human = read_csv_auto(path)
    require_columns(human, path, {"strat"})
    human = human.copy()
    human["strategy_family"] = human["strat"].map(map_sp2013_strat_to_family)
    human = human[human["strategy_family"].notna()]
    counts = human["strategy_family"].value_counts().to_dict()
    total = sum(counts.values())
    families = ["AS", "M", "D", "OTHER"]
    return pd.DataFrame(
        {
            "strategy_family": families,
            "reference_count": [int(counts.get(family, 0)) for family in families],
            "reference_prop": [float(counts.get(family, 0) / total) if total else 0.0 for family in families],
        }
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
