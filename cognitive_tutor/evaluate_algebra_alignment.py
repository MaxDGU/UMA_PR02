#!/usr/bin/env python3
"""Evaluate algebra trace alignment between human and synthetic sources.

The report is designed around four diagnostics:

1. Outcome alignment: accuracy by equation type.
2. Error-process alignment: where incorrect traces stop.
3. Trace-shape alignment: step-count distributions.
4. Answer-type alignment: form of the incorrect response.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Iterable, TextIO

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from cognitive_tutor.translate_algebra_uma_to_nlp import (  # noqa: E402
    clean_equation_text,
    has_variable_alone,
    infer_variable,
    short_transition_label,
)


DEFAULT_HUMAN_CSV = Path(
    "cognitive_tutor/data/processed/algebra_2006_2007/"
    "singlevar_equation_human_uma_master.csv"
)
DEFAULT_SYNTHETIC_RAW_CSV = Path(
    "results/UMA_replication/algebra_uma_k100_v3_ct_flow/"
    "algebra_uma_synth_matched_master_10k_k100_v3_ct_flow.csv"
)
DEFAULT_SYNTHETIC_TRANSLATED_CSV = Path(
    "cognitive_tutor/data/processed/algebra_2006_2007/"
    "translated_synthetic_algebra_v3_human_mixed/"
    "singlevar_algebra_synth_v3_human_mixed_train.csv.gz"
)
DEFAULT_OUT_DIR = Path("cognitive_tutor/analysis/algebra_alignment_v3_human_mixed")

SOURCE_HUMAN = "human"
SOURCE_SYNTHETIC_RAW = "synthetic_raw"
SOURCE_SYNTHETIC_TRANSLATED = "synthetic_translated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN_CSV)
    parser.add_argument("--synthetic-raw-csv", type=Path, default=DEFAULT_SYNTHETIC_RAW_CSV)
    parser.add_argument("--synthetic-translated-csv", type=Path, default=DEFAULT_SYNTHETIC_TRANSLATED_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-human-rows", type=int, default=None)
    parser.add_argument("--max-synthetic-raw-rows", type=int, default=None)
    parser.add_argument("--max-synthetic-translated-rows", type=int, default=None)
    return parser.parse_args()


def open_text(path: Path, mode: str = "rt") -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def safe_text(value: object) -> str:
    return str(value or "").strip()


def safe_int(value: object, default: int = 0) -> int:
    text = safe_text(value)
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def iter_rows(path: Path, max_rows: int | None = None) -> Iterable[dict[str, str]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            yield row
            if max_rows is not None and idx >= max_rows:
                break


def parse_trace_steps(row: dict[str, str]) -> list[str]:
    raw = safe_text(row.get("trace_steps_json"))
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
        steps = [
            safe_text(item.get("state", ""))
            for item in parsed
            if isinstance(item, dict) and safe_text(item.get("state", ""))
        ]
        if steps:
            return steps
    response = safe_text(row.get("response_nl"))
    if response:
        steps = parse_steps_from_response(response)
        if steps:
            return steps
    answer = safe_text(row.get("answer") or row.get("resp"))
    return [answer] if answer else []


def parse_steps_from_response(response: str) -> list[str]:
    steps: list[str] = []
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("### answer:"):
            continue
        if line.startswith("Step "):
            _, _, after = line.partition(":")
            steps.append(clean_surface_step(after))
            continue
        if line.startswith("Current equation:") or line.startswith("Next equation:"):
            _, _, after = line.partition(":")
            steps.append(clean_surface_step(after))
            continue
        if "->" in line:
            _, _, after = line.partition("->")
            steps.append(clean_surface_step(after))
            continue
        if "=" in line and not line.lower().startswith(("i ", "so ", "that ", "my ", "answer")):
            steps.append(clean_surface_step(line))
    return [step for step in steps if step]


def clean_surface_step(text: str) -> str:
    out = safe_text(text)
    out = re.sub(r"^(Start with|Then rewrite it as|The equation is)\s+", "", out, flags=re.I)
    return out.strip().rstrip(".")


def is_correct(row: dict[str, str]) -> bool:
    return safe_int(row.get("correct", row.get("is_correct", ""))) == 1


def equation_type(row: dict[str, str]) -> str:
    return safe_text(row.get("equation_type") or row.get("operation") or "unknown") or "unknown"


def variable_for(row: dict[str, str], steps: list[str]) -> str:
    return safe_text(row.get("variable")) or infer_variable(" ".join(steps + [safe_text(row.get("prob"))]))


def stuck_stage(row: dict[str, str]) -> str:
    steps = parse_trace_steps(row)
    if not steps:
        return "missing_trace"
    if len(steps) == 1:
        return "stuck_at_original_or_first_state"
    label = short_transition_label(steps[-2], steps[-1], {"variable": variable_for(row, steps)})
    return "after_" + label.replace(" ", "_")


def answer_text(row: dict[str, str]) -> str:
    return safe_text(row.get("answer") or row.get("resp") or row.get("first_wrong_step"))


def answer_type(row: dict[str, str]) -> str:
    answer = answer_text(row)
    variable = safe_text(row.get("variable")) or infer_variable(answer + " " + safe_text(row.get("prob")))
    cleaned = clean_equation_text(answer)
    if not answer:
        return "empty"
    if re.fullmatch(rf"{re.escape(variable)}=-?.+", cleaned) and variable:
        return "variable_assignment"
    if has_variable_alone(answer, variable):
        return "variable_isolated_equation"
    if cleaned.count("=") == 1:
        return "equation_state"
    if parse_fraction(answer) is not None:
        return "numeric_scalar"
    if re.search(r"row\d|r\d+c\d+|cell", cleaned, flags=re.I):
        return "ui_token"
    if variable and variable in cleaned:
        return "algebra_expression"
    if re.search(r"[+\-*/^]", cleaned):
        return "numeric_expression"
    return "other"


def parse_fraction(text: str) -> Fraction | None:
    value = safe_text(text)
    if not value:
        return None
    if "=" in value:
        value = value.split("=", 1)[-1].strip()
    try:
        return Fraction(value)
    except ValueError:
        return None


def response_style(row: dict[str, str]) -> str:
    text = safe_text(row.get("response_nl")).lower()
    if "not sure" in text:
        return "uncertain"
    if "i think" in text or "so i answer" in text:
        return "wrong_guess_narrative"
    if "### answer:" in text:
        return "answer_marked"
    return "unknown"


class AlignmentAccumulator:
    def __init__(self) -> None:
        self.rows_by_source: Counter[str] = Counter()
        self.accuracy: Counter[tuple[str, str, str]] = Counter()
        self.equation_type_counts: Counter[tuple[str, str]] = Counter()
        self.stuck: Counter[tuple[str, str, str]] = Counter()
        self.step_count: Counter[tuple[str, str, str, int]] = Counter()
        self.answer_types: Counter[tuple[str, str, str]] = Counter()
        self.response_styles: Counter[tuple[str, str, str]] = Counter()
        self.examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    def add(self, source: str, row: dict[str, str]) -> None:
        etype = equation_type(row)
        correct_label = "correct" if is_correct(row) else "incorrect"
        steps = parse_trace_steps(row)
        self.rows_by_source[source] += 1
        self.equation_type_counts[(source, etype)] += 1
        self.accuracy[(source, etype, correct_label)] += 1
        self.step_count[(source, etype, correct_label, len(steps))] += 1
        if not is_correct(row):
            self.stuck[(source, etype, stuck_stage(row))] += 1
            self.answer_types[(source, etype, answer_type(row))] += 1
            self.response_styles[(source, etype, response_style(row))] += 1
        if len(self.examples[source]) < 5:
            self.examples[source].append(
                {
                    "prob": safe_text(row.get("prob")),
                    "equation_type": etype,
                    "correct": str(int(is_correct(row))),
                    "answer": answer_text(row),
                    "steps": json.dumps(steps, ensure_ascii=False),
                    "response_nl": safe_text(row.get("response_nl")),
                }
            )


def counter_rows(counter: Counter[tuple], names: list[str]) -> list[dict[str, object]]:
    totals: Counter[tuple] = Counter()
    for key, count in counter.items():
        totals[key[:-1]] += count
    rows: list[dict[str, object]] = []
    for key, count in sorted(counter.items()):
        parent = key[:-1]
        denom = totals[parent]
        row = {name: value for name, value in zip(names, key)}
        row["count"] = count
        row["rate"] = count / denom if denom else 0.0
        rows.append(row)
    return rows


def accuracy_rows(acc: Counter[tuple[str, str, str]]) -> list[dict[str, object]]:
    totals: Counter[tuple[str, str]] = Counter()
    correct_counts: Counter[tuple[str, str]] = Counter()
    for (source, etype, label), count in acc.items():
        totals[(source, etype)] += count
        if label == "correct":
            correct_counts[(source, etype)] += count
    return [
        {
            "source": source,
            "equation_type": etype,
            "rows": total,
            "accuracy": correct_counts[(source, etype)] / total if total else 0.0,
        }
        for (source, etype), total in sorted(totals.items())
    ]


def distribution_from_rows(rows: list[dict[str, object]], source: str, group_cols: list[str], cat_col: str) -> dict[tuple, dict[str, float]]:
    out: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["source"] != source:
            continue
        group = tuple(row[col] for col in group_cols)
        out[group][str(row[cat_col])] = float(row["rate"])
    return out


def tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    cats = set(p) | set(q)
    return 0.5 * sum(abs(p.get(cat, 0.0) - q.get(cat, 0.0)) for cat in cats)


def comparison_rows(
    dist_rows: list[dict[str, object]],
    source_a: str,
    source_b: str,
    group_cols: list[str],
    cat_col: str,
    metric: str,
) -> list[dict[str, object]]:
    a = distribution_from_rows(dist_rows, source_a, group_cols, cat_col)
    b = distribution_from_rows(dist_rows, source_b, group_cols, cat_col)
    rows = []
    for group in sorted(set(a) | set(b)):
        row = {col: value for col, value in zip(group_cols, group)}
        row["metric"] = metric
        row["source_a"] = source_a
        row["source_b"] = source_b
        row["total_variation"] = tv_distance(a.get(group, {}), b.get(group, {}))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_alignment(
    human_csv: Path = DEFAULT_HUMAN_CSV,
    synthetic_raw_csv: Path = DEFAULT_SYNTHETIC_RAW_CSV,
    synthetic_translated_csv: Path = DEFAULT_SYNTHETIC_TRANSLATED_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_human_rows: int | None = None,
    max_synthetic_raw_rows: int | None = None,
    max_synthetic_translated_rows: int | None = None,
) -> dict[str, object]:
    acc = AlignmentAccumulator()
    for row in iter_rows(human_csv, max_human_rows):
        acc.add(SOURCE_HUMAN, row)
    for row in iter_rows(synthetic_raw_csv, max_synthetic_raw_rows):
        acc.add(SOURCE_SYNTHETIC_RAW, row)
    for row in iter_rows(synthetic_translated_csv, max_synthetic_translated_rows):
        acc.add(SOURCE_SYNTHETIC_TRANSLATED, row)

    out_dir.mkdir(parents=True, exist_ok=True)
    accuracy = accuracy_rows(acc.accuracy)
    stuck = counter_rows(acc.stuck, ["source", "equation_type", "stuck_stage"])
    step_count = counter_rows(acc.step_count, ["source", "equation_type", "correct", "num_steps"])
    answer_types = counter_rows(acc.answer_types, ["source", "equation_type", "answer_type"])
    response_styles = counter_rows(acc.response_styles, ["source", "equation_type", "response_style"])
    comparisons = (
        comparison_rows(stuck, SOURCE_HUMAN, SOURCE_SYNTHETIC_RAW, ["equation_type"], "stuck_stage", "stuck_stage_tv")
        + comparison_rows(step_count, SOURCE_HUMAN, SOURCE_SYNTHETIC_RAW, ["equation_type", "correct"], "num_steps", "step_count_tv")
        + comparison_rows(answer_types, SOURCE_HUMAN, SOURCE_SYNTHETIC_TRANSLATED, ["equation_type"], "answer_type", "answer_type_tv")
    )

    write_csv(out_dir / "accuracy_by_equation_type.csv", accuracy)
    write_csv(out_dir / "stuck_stage_distribution.csv", stuck)
    write_csv(out_dir / "step_count_distribution.csv", step_count)
    write_csv(out_dir / "answer_type_distribution.csv", answer_types)
    write_csv(out_dir / "response_style_distribution.csv", response_styles)
    write_csv(out_dir / "distribution_tv_summary.csv", comparisons)
    write_csv(
        out_dir / "examples.csv",
        [dict(source=source, **example) for source, examples in acc.examples.items() for example in examples],
    )

    summary = {
        "inputs": {
            SOURCE_HUMAN: str(human_csv),
            SOURCE_SYNTHETIC_RAW: str(synthetic_raw_csv),
            SOURCE_SYNTHETIC_TRANSLATED: str(synthetic_translated_csv),
        },
        "rows_by_source": dict(acc.rows_by_source),
        "outputs": {
            "accuracy_by_equation_type": str(out_dir / "accuracy_by_equation_type.csv"),
            "stuck_stage_distribution": str(out_dir / "stuck_stage_distribution.csv"),
            "step_count_distribution": str(out_dir / "step_count_distribution.csv"),
            "answer_type_distribution": str(out_dir / "answer_type_distribution.csv"),
            "response_style_distribution": str(out_dir / "response_style_distribution.csv"),
            "distribution_tv_summary": str(out_dir / "distribution_tv_summary.csv"),
            "examples": str(out_dir / "examples.csv"),
        },
        "notes": [
            "Outcome alignment compares accuracy by algebra equation type.",
            "Error-process alignment compares incorrect-trace stuck stages by equation type.",
            "Trace-shape alignment compares visible step-count distributions.",
            "Answer-type alignment compares incorrect answer forms; synthetic_translated is used here because human_mixed can emit numeric wrong guesses.",
        ],
    }
    write_json(out_dir / "alignment_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_alignment(
        human_csv=args.human_csv,
        synthetic_raw_csv=args.synthetic_raw_csv,
        synthetic_translated_csv=args.synthetic_translated_csv,
        out_dir=args.out_dir,
        max_human_rows=args.max_human_rows,
        max_synthetic_raw_rows=args.max_synthetic_raw_rows,
        max_synthetic_translated_rows=args.max_synthetic_translated_rows,
    )
    print(f"out_dir={args.out_dir}", flush=True)
    print(f"rows_by_source={summary['rows_by_source']}", flush=True)
    print(f"summary_json={args.out_dir / 'alignment_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
