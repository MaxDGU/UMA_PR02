import csv
import json
from pathlib import Path

import pandas as pd

from cognitive_tutor.build_trace_firstshot_dataset import (
    OUTPUT_COLUMNS,
    build_trace_firstshot_outputs,
    derive_final_answer,
    is_math_like_step,
)


HEADER = [
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


def write_tutor_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="latin-1", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_trace_firstshot_builder_groups_sorts_and_labels_attempts(tmp_path: Path) -> None:
    dataset = "toy_algebra"
    input_root = tmp_path / "extracted"
    out_dir = tmp_path / "processed"
    split_path = input_root / dataset / f"{dataset}_master.txt"
    write_tutor_split(
        split_path,
        [
            {
                "Row": "20",
                "Anon Student Id": "s1",
                "Problem Hierarchy": "Unit A",
                "Problem Name": "P-math",
                "Problem View": "1",
                "Step Name": "x = 2",
                "Step Start Time": "2007-01-01 09:02:00.0",
                "First Transaction Time": "2007-01-01 09:02:05.0",
                "Correct First Attempt": "1",
                "Incorrects": "0",
                "Hints": "0",
            },
            {
                "Row": "10",
                "Anon Student Id": "s1",
                "Problem Hierarchy": "Unit A",
                "Problem Name": "P-math",
                "Problem View": "1",
                "Step Name": "x+1 = 3",
                "Step Start Time": "2007-01-01 09:00:00.0",
                "First Transaction Time": "2007-01-01 09:00:05.0",
                "Correct First Attempt": "1",
                "Incorrects": "0",
                "Hints": "0",
            },
            {
                "Row": "30",
                "Anon Student Id": "s2",
                "Problem Hierarchy": "Unit A",
                "Problem Name": "P-opaque",
                "Problem View": "1",
                "Step Name": "R1C1",
                "Step Start Time": "2007-01-01 10:00:00.0",
                "First Transaction Time": "2007-01-01 10:00:03.0",
                "Correct First Attempt": "0",
                "Incorrects": "2",
                "Hints": "1",
            },
            {
                "Row": "31",
                "Anon Student Id": "s2",
                "Problem Hierarchy": "Unit A",
                "Problem Name": "P-opaque",
                "Problem View": "1",
                "Step Name": "R1C2",
                "Step Start Time": "2007-01-01 10:01:00.0",
                "First Transaction Time": "2007-01-01 10:01:03.0",
                "Correct First Attempt": "1",
                "Incorrects": "0",
                "Hints": "0",
            },
            {
                "Row": "50",
                "Anon Student Id": "s3",
                "Problem Hierarchy": "Unit B",
                "Problem Name": "P-row-fallback",
                "Problem View": "2",
                "Step Name": "2*3",
                "Step Start Time": "",
                "First Transaction Time": "",
                "Correct First Attempt": "1",
                "Incorrects": "0",
                "Hints": "0",
            },
            {
                "Row": "40",
                "Anon Student Id": "s3",
                "Problem Hierarchy": "Unit B",
                "Problem Name": "P-row-fallback",
                "Problem View": "2",
                "Step Name": "1+1",
                "Step Start Time": "",
                "First Transaction Time": "",
                "Correct First Attempt": "1",
                "Incorrects": "0",
                "Hints": "0",
            },
        ],
    )

    audit = build_trace_firstshot_outputs(
        dataset=dataset,
        splits=["master"],
        input_root=input_root,
        out_dir=out_dir,
    )

    complete = pd.read_csv(out_dir / "trace_firstshot_master.csv", dtype=str).fillna("")
    math_only = pd.read_csv(out_dir / "trace_firstshot_math_only_master.csv", dtype=str).fillna("")

    assert complete.columns.tolist() == OUTPUT_COLUMNS
    assert len(complete) == 3
    assert len(math_only) == 2
    assert audit["splits"]["master"]["complete"]["attempts"] == 3
    assert audit["splits"]["master"]["math_only"]["attempts"] == 2

    first = complete.loc[complete["problem_name"] == "P-math"].iloc[0]
    assert json.loads(first["trace"]) == ["x+1 = 3", "x = 2"]
    assert json.loads(first["source_rows"]) == [10, 20]
    assert first["resp"] == "2"
    assert first["target_kind"] == "derived_final_answer"
    assert first["acc"] == "1"
    assert first["final_observed_step"] == "x = 2"
    assert first["derived_final_answer"] == "2"
    assert first["first_wrong_step_index"] == "0"
    assert first["strategy"] == "Step 1: x+1 = 3\nStep 2: x = 2"
    assert first["instruction_nl"] == "Solve this algebra problem: P-math."
    assert first["response_nl"] == "Step 1: x+1 = 3\nStep 2: x = 2\n### answer: 2"

    opaque = complete.loc[complete["problem_name"] == "P-opaque"].iloc[0]
    assert json.loads(opaque["trace"]) == ["R1C1"]
    assert opaque["resp"] == "R1C1"
    assert opaque["target_kind"] == "stuck_step"
    assert opaque["acc"] == "0"
    assert opaque["num_steps"] == "1"
    assert opaque["num_original_steps"] == "2"
    assert opaque["final_observed_step"] == "R1C2"
    assert opaque["first_wrong_step_index"] == "1"
    assert opaque["first_wrong_step"] == "R1C1"
    assert opaque["first_wrong_incorrects"] == "2"
    assert opaque["first_wrong_hints"] == "1"
    assert opaque["strategy"] == "Step 1: R1C1"
    assert opaque["response_nl"] == "Step 1: R1C1\n### answer: R1C1"

    fallback = complete.loc[complete["problem_name"] == "P-row-fallback"].iloc[0]
    assert json.loads(fallback["trace"]) == ["1+1", "2*3"]
    assert json.loads(fallback["source_rows"]) == [40, 50]
    assert fallback["resp"] == "6"
    assert fallback["derived_final_answer"] == "6"
    assert fallback["acc"] == "1"


def test_math_like_step_predicate_is_conservative() -> None:
    assert is_math_like_step("x+1 = 3")
    assert is_math_like_step("(-60+sqrt(3600))/-32 = x")
    assert is_math_like_step("2*3")
    assert not is_math_like_step("R1C1")
    assert not is_math_like_step("xScale")
    assert not is_math_like_step("ValidEquations")


def test_derive_final_answer_is_conservative() -> None:
    assert derive_final_answer("5+12") == "17"
    assert derive_final_answer("1*14") == "14"
    assert derive_final_answer("x = 2") == "2"
    assert derive_final_answer("9 = y/8") == "72"
    assert derive_final_answer("R1C1") == ""
    assert derive_final_answer("(-60?sqrt(3600))/-32 = x") == ""
