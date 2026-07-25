#!/usr/bin/env python3
"""Build the fraction/decimal strategy-alignment table for Section 4.2.

The table reports mean per-problem TVD from the human strategy distribution.
Fractions use the four-family comparison produced by
``plot_tvd_baselines.py``. Decimals use the four-way BSS cue labels
(AS, M, BOTH, O).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plot_tvd_baselines import (
    OURS_LABEL,
    PREDICTION_FILES,
    build_statistics,
    find_repo_root,
    load_human_distribution,
    load_prediction_distribution,
    load_uma_distribution,
)


SOURCE_ORDER = [
    "UMA",
    "Centaur-70B",
    "Direct human FT",
    OURS_LABEL,
]
DECIMAL_FAMILIES = ["AS", "M", "BOTH", "O"]
DECIMAL_PREDICTION_FILES = {
    "Centaur-70B": "decimal_centaur_70b_predictions.csv.gz",
    "Direct human FT": "decimal_direct_human_ft_predictions.csv.gz",
    OURS_LABEL: "decimal_ours_humanft_predictions.csv.gz",
}


def normalize_problem(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(" ", "", regex=False).str.strip()


def decimal_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    distribution = pd.crosstab(
        frame["problem"],
        frame["family"],
        normalize="index",
    )
    return (
        distribution.reindex(columns=DECIMAL_FAMILIES, fill_value=0.0)
        .sort_index()
        .astype(float)
    )


def load_decimal_human_distribution(repo_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        repo_root
        / "eval/outputs/neurips_reproduction/raw/decimal/decimal_human_responses.csv",
        low_memory=False,
    )
    addition = frame["strat_add"].fillna(0).gt(0)
    multiplication = frame["strat_mul"].fillna(0).gt(0)
    frame["family"] = np.select(
        [addition & multiplication, addition, multiplication],
        ["BOTH", "AS", "M"],
        default="O",
    )
    frame["problem"] = normalize_problem(frame["prob"])
    return decimal_distribution(frame)


def load_decimal_prediction_distribution(
    repo_root: Path,
    source: str,
) -> pd.DataFrame:
    path = (
        repo_root
        / "eval/outputs/strategy_tvd"
        / DECIMAL_PREDICTION_FILES[source]
    )
    frame = pd.read_csv(path, low_memory=False)
    required = {"problem", "pred_code", "parse_success"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    valid = frame["parse_success"].fillna(False).astype(bool)
    if not valid.all():
        print(f"excluding {int((~valid).sum())} classifier parse failure(s) from {path}")
        frame = frame.loc[valid].copy()

    codes = frame["pred_code"].astype(str).str.upper()
    unknown = sorted(set(codes).difference(DECIMAL_FAMILIES))
    if unknown:
        raise ValueError(f"{path} contains unknown decimal strategy codes: {unknown}")
    frame["family"] = codes
    frame["problem"] = normalize_problem(frame["problem"])
    return decimal_distribution(frame)


def decimal_tvd_by_problem(
    model: pd.DataFrame,
    human: pd.DataFrame,
) -> pd.Series:
    common = human.index.intersection(model.index)
    if set(common) != set(human.index):
        missing = sorted(set(human.index).difference(common))
        raise ValueError(f"Decimal source is missing human problems: {missing}")
    return (
        0.5
        * (
            model.loc[common, DECIMAL_FAMILIES]
            - human.loc[common, DECIMAL_FAMILIES]
        )
        .abs()
        .sum(axis=1)
    )


def build_fraction_summary(repo_root: Path) -> pd.DataFrame:
    distributions = {
        "Human": load_human_distribution(repo_root),
        "UMA": load_uma_distribution(repo_root),
    }
    for source in PREDICTION_FILES:
        distributions[source] = load_prediction_distribution(repo_root, source)
    summary, _, _ = build_statistics(distributions)
    return summary.set_index("source")


def build_decimal_summary(repo_root: Path) -> pd.DataFrame:
    paper_root = repo_root / "writing/emnlp202026-humanlike-math-reasoning"
    problem_metrics = pd.read_csv(
        paper_root
        / "tables/decimal_procedural_alignment/problem_level_alignment_metrics.csv"
    )
    uma_values = problem_metrics.loc[
        problem_metrics["source_key"].eq("uma_simulator"),
        "tvd",
    ].to_numpy()
    rows = [
        {
            "source": "UMA",
            "problem_macro_tvd": float(uma_values.mean()),
            "n_problems": len(uma_values),
        }
    ]

    human = load_decimal_human_distribution(repo_root)
    for source in DECIMAL_PREDICTION_FILES:
        model = load_decimal_prediction_distribution(repo_root, source)
        values = decimal_tvd_by_problem(model, human).to_numpy()
        rows.append(
            {
                "source": source,
                "problem_macro_tvd": float(values.mean()),
                "n_problems": len(values),
            }
        )
    return pd.DataFrame(rows).set_index("source")


def write_outputs(
    fraction: pd.DataFrame,
    decimal: pd.DataFrame,
    table_dir: Path,
) -> tuple[Path, Path]:
    rows = []
    for source in SOURCE_ORDER:
        frac = fraction.loc[source]
        dec = decimal.loc[source]
        rows.append(
            {
                "source": source,
                "fraction_tvd": frac["problem_macro_tvd"],
                "fraction_n_problems": int(frac["n_problems"]),
                "decimal_tvd": dec["problem_macro_tvd"],
                "decimal_n_problems": int(dec["n_problems"]),
            }
        )
    combined = pd.DataFrame(rows)

    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "strategy_distribution_tvd.csv"
    tex_path = table_dir / "strategy_distribution_tvd.tex"
    combined.to_csv(csv_path, index=False)

    language_models = SOURCE_ORDER[1:]
    best_fraction = min(
        language_models,
        key=lambda source: float(fraction.loc[source, "problem_macro_tvd"]),
    )
    best_decimal = min(
        language_models,
        key=lambda source: float(decimal.loc[source, "problem_macro_tvd"]),
    )
    tex_rows = []
    for row in rows:
        frac_tvd = f"{row['fraction_tvd']:.3f}"
        if row["source"] == best_fraction:
            frac_tvd = rf"\textbf{{{frac_tvd}}}"
        decimal_tvd = f"{row['decimal_tvd']:.3f}"
        if row["source"] == best_decimal:
            decimal_tvd = rf"\textbf{{{decimal_tvd}}}"
        display_source = (
            r"\SYSNAME{} (ours)"
            if row["source"] == OURS_LABEL
            else row["source"]
        )
        tex_rows.append(f"{display_source} & {frac_tvd} & {decimal_tvd} \\\\")

    tex = "\n".join(
        [
            r"\begin{tabular}{@{}lcc@{}}",
            r"\toprule",
            r"Source & Fractions & Decimals \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    tex_path.write_text(tex)
    return csv_path, tex_path


def main() -> None:
    repo_root = find_repo_root(Path(__file__))
    table_dir = (
        repo_root / "writing/emnlp202026-humanlike-math-reasoning/tables"
    )
    fraction = build_fraction_summary(repo_root)
    decimal = build_decimal_summary(repo_root)
    csv_path, tex_path = write_outputs(fraction, decimal, table_dir)
    print(pd.read_csv(csv_path).to_string(index=False))
    print(f"\nwrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
