import csv
import gzip
import json
from pathlib import Path

from cognitive_tutor.evaluate_algebra_alignment import (
    answer_type,
    run_alignment,
    stuck_stage,
)


RAW_COLUMNS = [
    "subjid",
    "prob",
    "equation_type",
    "variable",
    "answer",
    "correct",
    "trace_steps_json",
    "response_nl",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def row(prob: str, answer: str, correct: str, steps: list[str], response: str = "") -> dict[str, str]:
    return {
        "subjid": "0",
        "prob": prob,
        "equation_type": "linear_two_step",
        "variable": "x",
        "answer": answer,
        "correct": correct,
        "trace_steps_json": json.dumps([{"state": step} for step in steps]),
        "response_nl": response or f"{steps[-1]}\n### answer: {answer}",
    }


def test_alignment_categories() -> None:
    stuck = row("5*(x+7)=-345", "5*x+35=-345", "0", ["5*(x+7)=-345", "5*x+35=-345"])
    numeric = row("2x=6", "4", "0", ["2x=6", "x=3"])
    assign = row("2x=6", "x=4", "0", ["2x=6", "x=4"])
    assert stuck_stage(stuck) == "after_expand_or_simplify"
    assert answer_type(stuck) == "equation_state"
    assert answer_type(numeric) == "numeric_scalar"
    assert answer_type(assign) == "variable_assignment"


def test_run_alignment_writes_report(tmp_path: Path) -> None:
    human = tmp_path / "human.csv"
    raw = tmp_path / "raw.csv"
    translated = tmp_path / "translated.csv.gz"
    rows = [
        row("2x=6", "3", "1", ["2x=6", "x=3"]),
        row("5*(x+7)=-345", "5*x+35=-345", "0", ["5*(x+7)=-345", "5*x+35=-345"]),
    ]
    write_csv(human, rows)
    write_csv(raw, rows)
    write_csv(
        translated,
        [
            row("2x=6", "3", "1", ["2x=6", "x=3"], "I get x=3.\n### answer: 3"),
            row(
                "5*(x+7)=-345",
                "-77",
                "0",
                ["5*(x+7)=-345", "5*x+35=-345"],
                "I think x=-77.\n### answer: -77",
            ),
        ],
    )

    out_dir = tmp_path / "out"
    summary = run_alignment(human, raw, translated, out_dir)
    assert summary["rows_by_source"]["human"] == 2
    assert (out_dir / "accuracy_by_equation_type.csv").exists()
    assert (out_dir / "distribution_tv_summary.csv").exists()
    assert (out_dir / "alignment_summary.json").exists()
