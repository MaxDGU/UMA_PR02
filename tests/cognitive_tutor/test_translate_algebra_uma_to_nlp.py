import csv
import json
from pathlib import Path

import pandas as pd

from cognitive_tutor.translate_algebra_uma_to_nlp import (
    CONNECTED_TEMPLATES,
    HUMAN_TEMPLATES,
    OUTPUT_COLUMNS,
    RICH_TEMPLATES,
    TEMPLATED_OUTPUT_COLUMNS,
    translate_file,
)


def test_translate_algebra_uma_hides_strategy_goals_exec(tmp_path: Path) -> None:
    input_csv = tmp_path / "uma.csv"
    output_csv = tmp_path / "nlp.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
                "strategy",
                "goals",
                "exec",
                "trace_steps_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subjid": "0",
                "prob": "x+3=7",
                "equation_type": "linear_two_step",
                "variable": "x",
                "answer": "x+3=7",
                "correct": "0",
                "g": "0.01",
                "d": "0.1",
                "c": "5.0",
                "rt_mu": "3",
                "ice": "0",
                "strategy": "human-calibrated firstshot",
                "goals": "parse equation",
                "exec": "human-calibrated stuck sample",
                "trace_steps_json": json.dumps([{"step_index": 1, "state": "x+3=7"}]),
            }
        )

    count = translate_file(input_csv, output_csv)
    out = pd.read_csv(output_csv, dtype=str).fillna("")
    assert count == 1
    assert out.columns.tolist() == OUTPUT_COLUMNS
    assert "strategy" not in out.columns
    assert "goals" not in out.columns
    assert "exec" not in out.columns
    assert out.loc[0, "instruction_nl"] == "Solve this algebra equation: x+3=7."
    assert out.loc[0, "response_nl"] == "Step 1: x+3=7\n### answer: x+3=7"


def test_translate_algebra_uma_connected_templates(tmp_path: Path) -> None:
    input_csv = tmp_path / "uma.csv"
    output_csv = tmp_path / "templated.csv"
    fieldnames = [
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
        "strategy",
        "goals",
        "exec",
        "trace_steps_json",
    ]
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "subjid": "0",
                "prob": "12y-16=-652",
                "equation_type": "linear_two_step",
                "variable": "y",
                "answer": "-53",
                "correct": "1",
                "g": "0.01",
                "d": "0.1",
                "c": "5.0",
                "rt_mu": "3",
                "ice": "0",
                "strategy": "isolate variable",
                "goals": "hidden",
                "exec": "hidden",
                "trace_steps_json": json.dumps(
                    [
                        {"step_index": 1, "state": "12y-16=-652"},
                        {"step_index": 2, "state": "12*y=-636"},
                        {"step_index": 3, "state": "y=-53"},
                    ]
                ),
            }
        )
        writer.writerow(
            {
                "subjid": "1",
                "prob": "5*(x+7)=-345",
                "equation_type": "linear_two_step",
                "variable": "x",
                "answer": "5*x+35=-345",
                "correct": "0",
                "g": "0.01",
                "d": "0.1",
                "c": "5.0",
                "rt_mu": "3",
                "ice": "0",
                "strategy": "hidden",
                "goals": "hidden",
                "exec": "hidden",
                "trace_steps_json": json.dumps(
                    [
                        {"step_index": 1, "state": "5*(x+7)=-345"},
                        {"step_index": 2, "state": "5*x+35=-345"},
                    ]
                ),
            }
        )

    count = translate_file(input_csv, output_csv, expand_templates=CONNECTED_TEMPLATES)
    out = pd.read_csv(output_csv, dtype=str).fillna("")

    assert count == 2 * len(CONNECTED_TEMPLATES)
    assert out.columns.tolist() == TEMPLATED_OUTPUT_COLUMNS
    assert set(out["template_id"]) == set(CONNECTED_TEMPLATES)
    assert "strategy" not in out.columns
    assert "goals" not in out.columns
    assert "exec" not in out.columns

    correct_step_bridge = out[(out["subjid"] == "0") & (out["template_id"] == "step_bridge")].iloc[0]
    assert "Step 2: Then rewrite it as 12*y=-636." in correct_step_bridge["response_nl"]
    assert "So the final answer is -53." in correct_step_bridge["response_nl"]

    wrong_step_bridge = out[(out["subjid"] == "1") & (out["template_id"] == "step_bridge")].iloc[0]
    assert "I stop at 5*x+35=-345." in wrong_step_bridge["response_nl"]
    assert "### answer: 5*x+35=-345" in wrong_step_bridge["response_nl"]


def test_translate_algebra_uma_rich_templates_describe_visible_transitions(tmp_path: Path) -> None:
    input_csv = tmp_path / "uma.csv"
    output_csv = tmp_path / "rich.csv"
    fieldnames = [
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
        "strategy",
        "goals",
        "exec",
        "trace_steps_json",
    ]
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "subjid": "0",
                "prob": "y/42-43=-277/6",
                "equation_type": "linear_with_denominator",
                "variable": "y",
                "answer": "-133",
                "correct": "1",
                "g": "0.01",
                "d": "0.1",
                "c": "5.0",
                "rt_mu": "3",
                "ice": "0",
                "strategy": "hidden",
                "goals": "hidden",
                "exec": "hidden",
                "trace_steps_json": json.dumps(
                    [
                        {"step_index": 1, "state": "y/42-43=-277/6"},
                        {"step_index": 2, "state": "y-1806=-1939"},
                        {"step_index": 3, "state": "y=-133"},
                    ]
                ),
            }
        )
        writer.writerow(
            {
                "subjid": "1",
                "prob": "5*(x+7)=-345",
                "equation_type": "linear_two_step",
                "variable": "x",
                "answer": "5*x+35=-345",
                "correct": "0",
                "g": "0.01",
                "d": "0.1",
                "c": "5.0",
                "rt_mu": "3",
                "ice": "0",
                "strategy": "hidden",
                "goals": "hidden",
                "exec": "hidden",
                "trace_steps_json": json.dumps(
                    [
                        {"step_index": 1, "state": "5*(x+7)=-345"},
                        {"step_index": 2, "state": "5*x+35=-345"},
                    ]
                ),
            }
        )

    count = translate_file(input_csv, output_csv, expand_templates=RICH_TEMPLATES)
    out = pd.read_csv(output_csv, dtype=str).fillna("")

    assert count == 2 * len(RICH_TEMPLATES)
    assert out.columns.tolist() == TEMPLATED_OUTPUT_COLUMNS
    assert set(out["template_id"]) == set(RICH_TEMPLATES)
    assert "strategy" not in out.columns
    assert "goals" not in out.columns
    assert "exec" not in out.columns

    correct_rich = out[(out["subjid"] == "0") & (out["template_id"] == "worked_solution_rich")].iloc[0]
    assert "Clear the fraction or denominator" in correct_rich["response_nl"]
    assert "Isolate the variable" in correct_rich["response_nl"]
    assert "The equation is solved" in correct_rich["response_nl"]

    wrong_rich = out[(out["subjid"] == "1") & (out["template_id"] == "first_attempt_rich")].iloc[0]
    assert "Expand or simplify" in wrong_rich["response_nl"]
    assert "The work stops at" in wrong_rich["response_nl"]
    assert "first attempt" not in wrong_rich["response_nl"].lower()
    assert "first-shot" not in wrong_rich["response_nl"].lower()


def test_translate_algebra_uma_human_templates_include_uncertainty_and_wrong_guesses(tmp_path: Path) -> None:
    input_csv = tmp_path / "uma.csv"
    output_csv = tmp_path / "human.csv"
    fieldnames = [
        "subjid",
        "prob",
        "equation_type",
        "variable",
        "solution",
        "answer",
        "correct",
        "g",
        "d",
        "c",
        "rt_mu",
        "ice",
        "strategy",
        "goals",
        "exec",
        "trace_steps_json",
    ]
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "subjid": "1",
                "prob": "5*(x+7)=-345",
                "equation_type": "linear_two_step",
                "variable": "x",
                "solution": "-76",
                "answer": "5*x+35=-345",
                "correct": "0",
                "g": "0.01",
                "d": "0.1",
                "c": "5.0",
                "rt_mu": "3",
                "ice": "0",
                "strategy": "hidden",
                "goals": "hidden",
                "exec": "hidden",
                "trace_steps_json": json.dumps(
                    [
                        {"step_index": 1, "state": "5*(x+7)=-345"},
                        {"step_index": 2, "state": "5*x+35=-345"},
                    ]
                ),
            }
        )

    count = translate_file(input_csv, output_csv, expand_templates=HUMAN_TEMPLATES)
    out = pd.read_csv(output_csv, dtype=str).fillna("")

    assert count == len(HUMAN_TEMPLATES)
    assert set(out["template_id"]) == set(HUMAN_TEMPLATES)
    assert "strategy" not in out.columns
    assert "goals" not in out.columns
    assert "exec" not in out.columns

    uncertain = out[out["template_id"] == "human_uncertain"].iloc[0]
    assert "not sure" in uncertain["response_nl"]
    assert "first attempt" not in uncertain["response_nl"].lower()

    wrong_guess = out[out["template_id"] == "human_wrong_guess"].iloc[0]
    assert wrong_guess["answer"] != "5*x+35=-345"
    assert f"### answer: {wrong_guess['answer']}" in wrong_guess["response_nl"]
    assert "I think" in wrong_guess["response_nl"]
