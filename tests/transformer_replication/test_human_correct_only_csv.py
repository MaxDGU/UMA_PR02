from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "results" / "transformer_replication"))

from build_human_correct_only_csv import build_correct_only_outputs  # type: ignore  # noqa: E402


def test_build_correct_only_outputs_filters_numeric_final_answers(tmp_path: Path) -> None:
    rows = [
        {
            "subjid": "1",
            "prob": "1/2+1/2",
            "resp": "1",
            "strategy": "integer answer",
            "instruction_nl": "Solve this fraction problem: 1/2+1/2=?",
            "response_nl": "I added the halves.\n### answer: 1",
        },
        {
            "subjid": "1",
            "prob": "1/2+1/2",
            "resp": "2/2",
            "strategy": "equivalent fraction",
            "instruction_nl": "Solve this fraction problem: 1/2+1/2=?",
            "response_nl": "I got two halves.\n### answer: 2/2",
        },
        {
            "subjid": "1",
            "prob": "1/2+1/4",
            "resp": "3/4",
            "strategy": "exact fraction",
            "instruction_nl": "Solve this fraction problem: 1/2+1/4=?",
            "response_nl": "I used fourths.\n### answer: 3/4",
        },
        {
            "subjid": "1",
            "prob": "1/2:1/4",
            "resp": "2",
            "strategy": "division",
            "instruction_nl": "Solve this fraction problem: 1/2:1/4=?",
            "response_nl": "I divided.\n### answer: 2",
        },
        {
            "subjid": "1",
            "prob": "1/2*1/2",
            "resp": "1/3",
            "strategy": "wrong answer",
            "instruction_nl": "Solve this fraction problem: 1/2*1/2=?",
            "response_nl": "I multiplied.\n### answer: 1/3",
        },
        {
            "subjid": "1",
            "prob": "not a problem",
            "resp": "1",
            "strategy": "unparsable",
            "instruction_nl": "Solve this fraction problem: not a problem=?",
            "response_nl": "I guessed.\n### answer: 1",
        },
    ]
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    pd.DataFrame(rows[:3]).to_csv(val_csv, index=False)

    audit = build_correct_only_outputs(train_csv=train_csv, val_csv=val_csv, out_dir=tmp_path / "out")

    train_out = pd.read_csv(tmp_path / "out" / "data_train_nlp_correct.csv")
    val_out = pd.read_csv(tmp_path / "out" / "data_val_nlp_correct.csv")

    assert train_out["resp"].tolist() == ["1", "2/2", "3/4", "2"]
    assert val_out["resp"].tolist() == ["1", "2/2", "3/4"]
    assert audit["splits"]["train"]["kept_rows"] == 4
    assert audit["splits"]["train"]["dropped_rows"] == 2
    assert audit["splits"]["val"]["kept_rows"] == 3
    assert audit["splits"]["train"]["by_operation"]["?"]["dropped_rows"] == 1
