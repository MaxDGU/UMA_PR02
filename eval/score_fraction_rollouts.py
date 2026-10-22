"""Default scorer for SP2013 fraction rollouts: profile MAE and categorical answer NLL.

Use this for every fraction number that is compared with the paper. It reports
the two metrics in the paper's main results:

* MAE: mean over the 8 operation x {ED, UD} cells of |model acc - child acc|,
  with child accuracy from Siegler & Pyke (2013).
* NLL: the paper's *categorical answer NLL* (Appendix "Behavioral NLL"): each of
  the 1,867 parseable child answers is scored under the rollout answer
  distribution, symmetric Dirichlet smoothing with total mass one.

This NLL is NOT the `val_nll` written by `human_ft/finetune_humandata.py`. That
one is the teacher-forced per-token NLL of the 96 held-out Siegler (2011)
think-aloud responses; it measures text fit and is on a different scale.

Answers are re-parsed from `model_response` with the AAAI parser
(`extract_answer` in `results/finetuning/trajectory_at_T10/eval_qwen3_chained_sp2013.py`),
ignoring any stored `parsed_answer` / `is_correct` columns, so files produced by
different generation scripts are scored identically. On the archived paper
rollouts (`eval/outputs/fraction_4b/qwen3_4b_distill_humanft_v1.csv`) this gives
MAE 5.29 and NLL 2.296; the paper's 5.24 / 2.291 used the labels stored by the
original pre-fix parser.

Example:
    python eval/score_fraction_rollouts.py path/to/rollouts/*.csv --out-csv scores.csv
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
from fractions import Fraction
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.prepare_table2_uma_matched256 import (
    canonical_problem,
    categorical_answer_nll,
    normalize_fraction_human_answer,
    normalize_fraction_model_answer,
)

AAAI_PARSER = ROOT / "results/finetuning/trajectory_at_T10/eval_qwen3_chained_sp2013.py"
DEFAULT_HUMAN = ROOT / "eval/data/siegler_fraction_human.csv"
OPERATIONS = {"+": "Add", "-": "Sub", "*": "Mul", ":": "Div"}
CELLS = [f"{op} {den}" for op in OPERATIONS.values() for den in ("ED", "UD")]
PROBLEM_RE = re.compile(r"^(\d+)/(\d+)([+\-*:])(\d+)/(\d+)$")


def no_string_inference():
    # Pandas 3 string inference turns the normalizers' None into NaN, which
    # categorical_answer_nll would then count as an answer category.
    try:
        return pd.option_context("future.infer_string", False)
    except (KeyError, pd.errors.OptionError):
        return nullcontext()


def load_aaai_parser():
    spec = importlib.util.spec_from_file_location("aaai_fraction_parser", AAAI_PARSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def problem_cell(problem: str) -> str:
    match = PROBLEM_RE.match(canonical_problem(problem))
    if match is None:
        raise ValueError(f"Unparseable SP2013 problem: {problem!r}")
    _, den_a, op, _, den_b = match.groups()
    return f"{OPERATIONS[op]} {'ED' if den_a == den_b else 'UD'}"


def human_profile(human: pd.DataFrame) -> pd.Series:
    cells = human["prob"].map(problem_cell)
    profile = human.groupby(cells)["acc"].mean().reindex(CELLS) * 100
    if profile.isna().any():
        raise ValueError("Human data is missing SP2013 cells")
    return profile


def score_rollouts(
    rollouts: pd.DataFrame,
    human: pd.DataFrame,
    parser,
    *,
    problem_column: str = "problem",
    response_column: str = "model_response",
) -> dict:
    model = pd.DataFrame({"problem": rollouts[problem_column].map(canonical_problem)})
    predictions = rollouts[response_column].fillna("").astype(str).map(parser.extract_answer)
    truth = model["problem"].map(parser.gt_answer)
    model["correct"] = [pred is not None and pred == gt for pred, gt in zip(predictions, truth)]
    model["answer"] = [None if pred is None else str(pred) for pred in predictions]
    model["cell"] = model["problem"].map(problem_cell)

    reference = human_profile(human)
    accuracy = model.groupby("cell")["correct"].mean().reindex(CELLS) * 100
    if accuracy.isna().any():
        raise ValueError("Rollouts are missing SP2013 cells")
    nll, _, scored = categorical_answer_nll(
        human,
        model,
        human_problem_column="prob",
        model_problem_column="problem",
        human_answer_column="resp_val",
        model_answer_column="answer",
        normalize_human=normalize_fraction_human_answer,
        normalize_model=normalize_fraction_model_answer,
    )
    per_problem = model.groupby("problem").size()
    result = {
        "mae_pp": float((accuracy - reference).abs().mean()),
        "categorical_nll": float(nll),
        "accuracy_pct": float(model["correct"].mean() * 100),
        "human_answers_scored": int(scored),
        "samples_per_problem_min": int(per_problem.min()),
        "samples_per_problem_max": int(per_problem.max()),
    }
    result.update({f"acc_{cell.replace(' ', '_')}": float(accuracy[cell]) for cell in CELLS})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", nargs="+", type=Path, help="Rollout CSV(s): 16 SP2013 problems x N samples.")
    parser.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--problem-column", default="problem")
    parser.add_argument("--response-column", default="model_response")
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    aaai = load_aaai_parser()
    human = pd.read_csv(args.human_csv)
    rows = []
    for path in args.rollouts:
        frame = pd.read_csv(path)
        with no_string_inference():
            result = score_rollouts(
                frame, human, aaai,
                problem_column=args.problem_column, response_column=args.response_column,
            )
        rows.append({"file": str(path), **result})
        if result["samples_per_problem_min"] != 256 or result["samples_per_problem_max"] != 256:
            print(f"warning: {path} does not have 256 samples per problem (paper protocol)", file=sys.stderr)

    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(table[["file", "mae_pp", "categorical_nll", "accuracy_pct", "human_answers_scored"]].round(3).to_string(index=False))
    if args.out_csv is not None:
        table.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
