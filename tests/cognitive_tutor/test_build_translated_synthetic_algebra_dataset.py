import csv
import gzip
import json
from pathlib import Path

from cognitive_tutor.build_translated_synthetic_algebra_dataset import (
    OUTPUT_COLUMNS,
    build_dataset,
)


INPUT_COLUMNS = [
    "subjid",
    "prob",
    "equation_type",
    "variable",
    "answer",
    "correct",
    "g",
    "d",
    "c",
    "rt_mu",
    "ice",
    "instruction_nl",
    "response_nl",
]


def write_input(path: Path, rows: list[dict[str, str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_split(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_row(subjid: str, prob: str, correct: str, answer: str) -> dict[str, str]:
    return {
        "subjid": subjid,
        "prob": prob,
        "equation_type": "linear_two_step",
        "variable": "x",
        "answer": answer,
        "correct": correct,
        "g": "0.01",
        "d": "0.1",
        "c": "5",
        "rt_mu": "3",
        "ice": "0",
        "instruction_nl": f"Solve this algebra equation: {prob}.",
        "response_nl": f"I get {answer}.\n### answer: {answer}",
    }


def test_build_translated_synthetic_dataset_splits_by_problem(tmp_path: Path) -> None:
    input_csv = tmp_path / "translated.csv.gz"
    rows = [
        make_row("0", "x+3=7", "1", "4"),
        make_row("1", "x+3=7", "0", "5"),
        make_row("0", "2x=6", "1", "3"),
        make_row("1", "2x=6", "0", "4"),
        make_row("0", "y/8=9", "1", "72"),
        make_row("1", "y/8=9", "0", "71"),
    ]
    write_input(input_csv, rows)

    out_dir = tmp_path / "out"
    audit = build_dataset(
        input_csv=input_csv,
        out_dir=out_dir,
        prefix="toy",
        val_frac=0.34,
        test_frac=0.33,
        seed=3,
    )

    all_rows = []
    problem_to_split = {}
    for split in ["train", "val", "test"]:
        split_rows = read_split(out_dir / f"toy_{split}.csv.gz")
        all_rows.extend(split_rows)
        for row in split_rows:
            assert list(row.keys()) == OUTPUT_COLUMNS
            assert row["split"] == split
            assert row["source_uid"]
            assert row["instruction_nl"]
            assert "### answer:" in row["response_nl"]
            previous = problem_to_split.setdefault(row["prob"], split)
            assert previous == split

    assert len(all_rows) == len(rows)
    assert set(problem_to_split) == {"x+3=7", "2x=6", "y/8=9"}
    assert audit["global"]["rows"] == 6
    assert audit["global"]["unique_probs"] == 3
    assert audit["global"]["response_missing_answer_marker"] == 0

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["primary_train_csv"].endswith("toy_train.csv.gz")
    assert manifest["prompt_col"] == "instruction_nl"
    assert manifest["response_col"] == "response_nl"
