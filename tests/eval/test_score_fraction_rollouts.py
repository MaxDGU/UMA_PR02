from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from eval.score_fraction_rollouts import (
    DEFAULT_HUMAN,
    load_aaai_parser,
    no_string_inference,
    problem_cell,
    score_rollouts,
)

ROOT = Path(__file__).resolve().parents[2]
PAPER_ROLLOUTS = ROOT / "eval/outputs/fraction_4b/qwen3_4b_distill_humanft_v1.csv"


def test_problem_cell_uses_denominators() -> None:
    assert problem_cell("3/5 + 1/5") == "Add ED"
    assert problem_cell("2/3:3/5") == "Div UD"


def test_stored_answers_reproduce_paper_table() -> None:
    with no_string_inference():
        result = score_rollouts(
            pd.read_csv(PAPER_ROLLOUTS), pd.read_csv(DEFAULT_HUMAN), load_aaai_parser(), answers="stored"
        )
    assert result["mae_pp"] == pytest.approx(5.238, abs=1e-3)
    assert result["categorical_nll"] == pytest.approx(2.291, abs=1e-3)


def test_reparsed_paper_rollouts_differ_only_by_truncation() -> None:
    with no_string_inference():
        result = score_rollouts(pd.read_csv(PAPER_ROLLOUTS), pd.read_csv(DEFAULT_HUMAN), load_aaai_parser())
    assert result["human_answers_scored"] == 1867
    assert result["mae_pp"] == pytest.approx(5.286, abs=1e-3)
    assert result["categorical_nll"] == pytest.approx(2.296, abs=1e-3)
