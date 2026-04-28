import csv
import json
from pathlib import Path

import pandas as pd

from cognitive_tutor.build_single_variable_algebra_uma_dataset import (
    HUMAN_UMA_COLUMNS,
    build_single_variable_algebra_uma_outputs,
    parse_kc_labels,
)
from cognitive_tutor.build_trace_firstshot_dataset import OUTPUT_COLUMNS


RAW_HEADER = [
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
    "KC(Default)",
]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_raw_tutor(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="latin-1", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=RAW_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def processed_row(**overrides: object) -> dict[str, object]:
    row = {column: "" for column in OUTPUT_COLUMNS}
    row.update(
        {
            "subjid": "s1",
            "prob": "EG21 3 = y+3",
            "problem_hierarchy": "Unit A",
            "problem_name": "EG21 3 = y+3",
            "problem_view": "1",
            "resp": "0",
            "target_kind": "derived_final_answer",
            "acc": "1",
            "num_steps": "2",
            "num_original_steps": "2",
            "final_observed_step": "0 = y",
            "derived_final_answer": "0",
            "first_wrong_step_index": "0",
            "source_rows": json.dumps([1, 2]),
            "trace": json.dumps(["3 = y+3", "0 = y"]),
            "strategy": "Step 1: 3 = y+3\nStep 2: 0 = y",
            "instruction_nl": "Solve this algebra problem: EG21 3 = y+3.",
            "response_nl": "Step 1: 3 = y+3\nStep 2: 0 = y\n### answer: 0",
        }
    )
    row.update(overrides)
    return row


def test_parse_kc_labels() -> None:
    labels = parse_kc_labels(
        "[SkillRule: Remove constant; x+a=b, positive]~~"
        "[SkillRule: Isolate positive; x+a=b, positive]"
    )
    assert labels == ["Remove constant", "Isolate positive"]


def test_build_single_variable_human_uma_outputs_filters_and_formats(tmp_path: Path) -> None:
    dataset = "toy_algebra"
    processed_dir = tmp_path / "processed"
    input_root = tmp_path / "extracted"
    out_dir = tmp_path / "out"
    write_csv(
        processed_dir / "trace_firstshot_math_only_master.csv",
        [
            processed_row(),
            processed_row(
                subjid="s2",
                prob="EG12C x = 23*30",
                problem_name="EG12C x = 23*30",
                resp="690",
                trace=json.dumps(["x = 23*30"]),
                source_rows=json.dumps([3]),
            ),
            processed_row(
                subjid="s3",
                prob="EG25 9 = y/8",
                problem_name="EG25 9 = y/8",
                resp="9 = y/8",
                target_kind="stuck_step",
                acc="0",
                final_observed_step="y = 72",
                derived_final_answer="",
                first_wrong_step_index="1",
                first_wrong_step="9 = y/8",
                source_rows=json.dumps([4]),
                trace=json.dumps(["9 = y/8"]),
                strategy="Step 1: 9 = y/8",
                response_nl="Step 1: 9 = y/8\n### answer: 9 = y/8",
            ),
        ],
        OUTPUT_COLUMNS,
    )
    write_raw_tutor(
        input_root / dataset / f"{dataset}_master.txt",
        [
            {
                "Row": 1,
                "Anon Student Id": "s1",
                "Problem Hierarchy": "Unit A",
                "Problem Name": "EG21 3 = y+3",
                "Problem View": 1,
                "Step Name": "3 = y+3",
                "Correct First Attempt": 1,
                "Incorrects": 0,
                "Hints": 0,
                "KC(Default)": "[SkillRule: Remove constant; x+a=b, positive]",
            },
            {
                "Row": 2,
                "Anon Student Id": "s1",
                "Problem Hierarchy": "Unit A",
                "Problem Name": "EG21 3 = y+3",
                "Problem View": 1,
                "Step Name": "0 = y",
                "Correct First Attempt": 1,
                "Incorrects": 0,
                "Hints": 0,
                "KC(Default)": "[SkillRule: Isolate positive; x+a=b, positive]",
            },
            {
                "Row": 4,
                "Anon Student Id": "s3",
                "Problem Hierarchy": "Unit A",
                "Problem Name": "EG25 9 = y/8",
                "Problem View": 1,
                "Step Name": "9 = y/8",
                "Correct First Attempt": 0,
                "Incorrects": 2,
                "Hints": 1,
                "KC(Default)": "[SkillRule: Remove coefficient; ax=b, divide]",
            },
        ],
    )

    audit = build_single_variable_algebra_uma_outputs(
        dataset=dataset,
        splits=["master"],
        input_root=input_root,
        processed_dir=processed_dir,
        out_dir=out_dir,
    )

    output = pd.read_csv(out_dir / "singlevar_equation_human_uma_master.csv", dtype=str).fillna("")
    problem_set = pd.read_csv(out_dir / "singlevar_equation_problem_set_master.csv", dtype=str).fillna("")

    assert output.columns.tolist() == HUMAN_UMA_COLUMNS
    assert len(output) == 2
    assert len(problem_set) == 2
    assert audit["splits"]["master"]["summary"]["retained_rows"] == 2

    correct = output.loc[output["subjid"] == "s1"].iloc[0]
    assert correct["prob"] == "3 = y+3"
    assert correct["solution"] == "0"
    assert correct["answer"] == "0"
    assert correct["correct"] == "1"
    assert correct["raw_kc_labels"] == "Remove constant | Isolate positive"

    wrong = output.loc[output["subjid"] == "s3"].iloc[0]
    assert wrong["prob"] == "9 = y/8"
    assert wrong["solution"] == "72"
    assert wrong["answer"] == "9 = y/8"
    assert wrong["correct"] == "0"
    assert wrong["first_wrong_step"] == "9 = y/8"
    steps = json.loads(wrong["trace_steps_json"])
    assert steps[0]["kc_labels"] == ["Remove coefficient"]
    assert steps[0]["correct_first_attempt"] == 0
