#!/usr/bin/env python3
"""Build the fraction strategy-alignment figure used in Section 4.2.

The comparison set, ordering, labels, and source colors match Figure 3:
Human, UMA, Centaur-70B, Direct human FT, and CPT-LLM (ours).

Panel (a) reports mean per-problem total variation distance (TVD) from the
human strategy distribution. Panel (b) shows the full strategy mixture in
each operation-by-denominator cell. Human strategies are the reference, UMA
strategies come from native simulator flags, and the three language-model
sources use frozen classifier labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILIES = ["AS", "M", "D", "OTHER"]
FRACTION_CELLS = [
    ("Add", "ED"),
    ("Sub", "ED"),
    ("Mul", "ED"),
    ("Div", "ED"),
    ("Add", "UD"),
    ("Sub", "UD"),
    ("Mul", "UD"),
    ("Div", "UD"),
]
SYSNAME = "CPT-LLM"
OURS_LABEL = f"{SYSNAME} (ours)"
SERIES_ORDER = [
    "Human",
    "UMA",
    "Centaur-70B",
    "Direct human FT",
    OURS_LABEL,
]
MODEL_ORDER = [
    "UMA",
    "Centaur-70B",
    "Direct human FT",
    OURS_LABEL,
]
PREDICTION_FILES = {
    "Centaur-70B": "fraction_centaur_70b_predictions.csv.gz",
    "Direct human FT": "fraction_direct_human_ft_predictions.csv.gz",
    OURS_LABEL: "fraction_ours_humanft_predictions.csv.gz",
}
CANONICAL_TO_FAMILY = {
    "KDON": "AS",
    "CDON": "AS",
    "ONOD": "M",
    "ICDM": "D",
    "CROP": "D",
    "OTHER": "OTHER",
}
HUMAN_TO_FAMILY = {
    "OpNumKeepDen": "AS",
    "IndepComp": "M",
    "InvertOper": "D",
    "Other/None": "OTHER",
}
UMA_FAMILY_COLUMNS = {
    "AS": ["KDON_AS", "KDON_OG", "CDON_AS", "CDON_OG"],
    "M": ["ONOD_M", "ONOD_OG"],
    "D": ["CROP_M", "ICDM_D", "ICDM_OG"],
}


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "eval" / "outputs").is_dir() and (
            candidate / "writing" / "emnlp202026-humanlike-math-reasoning"
        ).is_dir():
            return candidate
    raise RuntimeError("Could not locate the UMA_PR02 repository root.")


def normalize_problem(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.replace("÷", ":", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )


def distribution_by_problem(frame: pd.DataFrame) -> pd.DataFrame:
    distribution = pd.crosstab(
        frame["problem"],
        frame["family"],
        normalize="index",
    )
    return distribution.reindex(columns=FAMILIES, fill_value=0.0).sort_index()


def load_human_distribution(repo_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        repo_root / "eval/data/siegler_fraction_human.csv",
        low_memory=False,
    )
    frame["problem"] = normalize_problem(frame["prob"])
    frame["family"] = frame["strat"].map(HUMAN_TO_FAMILY).fillna("OTHER")
    return distribution_by_problem(frame)


def load_uma_distribution(repo_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        repo_root / "eval/outputs/uma_reference/verification_full_results.csv.gz",
        low_memory=False,
    )
    missing = sorted(
        {
            column
            for columns in UMA_FAMILY_COLUMNS.values()
            for column in columns
            if column not in frame.columns
        }
    )
    if missing:
        raise ValueError(f"UMA reference is missing strategy columns: {missing}")

    active = pd.DataFrame(
        {
            family: frame[columns].gt(0).any(axis=1)
            for family, columns in UMA_FAMILY_COLUMNS.items()
        }
    )
    active_count = active.sum(axis=1)
    if not active_count.eq(1).all():
        counts = active_count.value_counts().sort_index().to_dict()
        raise ValueError(f"Expected one UMA strategy family per row; found {counts}")

    frame["problem"] = normalize_problem(frame["prob"])
    frame["family"] = active.idxmax(axis=1)
    return distribution_by_problem(frame)


def load_prediction_distribution(repo_root: Path, source: str) -> pd.DataFrame:
    path = repo_root / "eval/outputs/strategy_tvd" / PREDICTION_FILES[source]
    if not path.exists():
        raise FileNotFoundError(f"Missing classified traces for {source}: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"problem", "pred_code", "parse_success"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if not frame["parse_success"].fillna(False).astype(bool).all():
        failures = int((~frame["parse_success"].fillna(False).astype(bool)).sum())
        raise ValueError(f"{path} contains {failures} classifier parse failures")
    unknown = sorted(
        set(frame["pred_code"].dropna().astype(str)).difference(CANONICAL_TO_FAMILY)
    )
    if unknown:
        raise ValueError(f"{path} contains unknown strategy codes: {unknown}")

    frame["problem"] = normalize_problem(frame["problem"])
    frame["family"] = frame["pred_code"].map(CANONICAL_TO_FAMILY)
    counts = frame.groupby("problem").size()
    if not counts.eq(100).all():
        raise ValueError(
            f"{path} must contain 100 traces/problem; found {counts.to_dict()}"
        )
    return distribution_by_problem(frame)


def load_problem_cells(repo_root: Path) -> pd.Series:
    frame = pd.read_csv(
        repo_root / "eval/data/siegler_fraction_human.csv",
        low_memory=False,
    )
    frame["problem"] = normalize_problem(frame["prob"])
    frame["operation_key"] = (
        frame["operation"].astype(str).str.strip().str[:3].str.capitalize()
    )
    frame["denominator_key"] = frame["operands"].astype(str).str.upper()
    problem_cells = (
        frame[["problem", "operation_key", "denominator_key"]]
        .drop_duplicates()
        .set_index("problem")
    )
    if problem_cells.index.duplicated().any():
        raise ValueError("A fraction problem was assigned to more than one cell")
    cells = pd.Series(
        list(
            zip(
                problem_cells["operation_key"],
                problem_cells["denominator_key"],
            )
        ),
        index=problem_cells.index,
        name="cell",
    )
    unknown = sorted(set(cells).difference(FRACTION_CELLS))
    if unknown:
        raise ValueError(f"Found unexpected fraction cells: {unknown}")
    return cells


def distribution_by_cell(
    distribution: pd.DataFrame,
    problem_cells: pd.Series,
) -> pd.DataFrame:
    missing = sorted(set(distribution.index).difference(problem_cells.index))
    if missing:
        raise ValueError(f"Missing cell assignments for fraction problems: {missing}")
    with_cells = distribution.loc[:, FAMILIES].copy()
    with_cells["cell"] = problem_cells.reindex(with_cells.index)
    return with_cells.groupby("cell")[FAMILIES].mean().reindex(FRACTION_CELLS)


def per_problem_tvd(model: pd.DataFrame, human: pd.DataFrame) -> pd.Series:
    common = human.index.intersection(model.index)
    if len(common) != len(human):
        missing = sorted(set(human.index).difference(common))
        raise ValueError(f"Model distribution is missing human problems: {missing}")
    return 0.5 * (model.loc[common, FAMILIES] - human.loc[common, FAMILIES]).abs().sum(axis=1)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 12_345,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(boot_means, 0.025)),
        float(np.quantile(boot_means, 0.975)),
    )


def build_statistics(
    distributions: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    human = distributions["Human"]
    per_problem_rows = []
    summary_rows = []
    for source in SERIES_ORDER:
        tvd = (
            pd.Series(0.0, index=human.index)
            if source == "Human"
            else per_problem_tvd(distributions[source], human)
        )
        for problem, value in tvd.items():
            per_problem_rows.append(
                {"source": source, "problem": problem, "tvd": float(value)}
            )
        point, ci_low, ci_high = bootstrap_mean_ci(tvd.to_numpy())
        summary_rows.append(
            {
                "source": source,
                "problem_macro_tvd": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_problems": len(tvd),
            }
        )

    per_problem = pd.DataFrame(per_problem_rows)
    summary = pd.DataFrame(summary_rows)
    ours = (
        per_problem[per_problem["source"] == OURS_LABEL]
        .set_index("problem")["tvd"]
        .sort_index()
    )
    paired_rows = []
    for source in ["UMA", "Centaur-70B", "Direct human FT"]:
        baseline = (
            per_problem[per_problem["source"] == source]
            .set_index("problem")["tvd"]
            .sort_index()
        )
        difference = (baseline - ours).to_numpy()
        point, ci_low, ci_high = bootstrap_mean_ci(difference)
        paired_rows.append(
            {
                "comparison": f"{source} minus {OURS_LABEL}",
                "tvd_improvement": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_problems": len(difference),
            }
        )
    return summary, per_problem, pd.DataFrame(paired_rows)


def plot_figure(
    distributions: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    problem_cells: pd.Series,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    source_colors = dict(
        zip(
            SERIES_ORDER,
            plt.colormaps["viridis"](np.linspace(0.08, 0.88, len(SERIES_ORDER))),
            strict=True,
        )
    )
    family_colors = {
        "AS": "#4C78A8",
        "M": "#F58518",
        "D": "#54A24B",
        "OTHER": "#B8B8B8",
    }
    family_labels = {
        "AS": "Operate numerator / keep denominator",
        "M": "Operate components independently",
        "D": "Invert / cross-operate",
        "OTHER": "Other / none",
    }

    fig = plt.figure(figsize=(13.6, 5.15), dpi=300)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 3.25],
        left=0.065,
        right=0.995,
        bottom=0.14,
        top=0.96,
        wspace=0.20,
    )

    ax_tvd = fig.add_subplot(outer[0, 0])
    ordered = summary.set_index("source").loc[MODEL_ORDER]
    means = ordered["problem_macro_tvd"].to_numpy()
    x = np.arange(len(MODEL_ORDER))
    bars = ax_tvd.bar(
        x,
        means,
        width=0.70,
        color=[source_colors[source] for source in MODEL_ORDER],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    for bar, source, value in zip(bars, MODEL_ORDER, means, strict=True):
        ax_tvd.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.009,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold" if source == OURS_LABEL else "normal",
        )
    short_labels = ["UMA", "Centaur-\n70B", "Direct\nhuman FT", f"{SYSNAME}\n(ours)"]
    ax_tvd.set_xticks(x, short_labels)
    ax_tvd.get_xticklabels()[-1].set_fontweight("bold")
    ax_tvd.set_ylabel("TVD to human (lower is better)", fontsize=13)
    ax_tvd.set_title(
        "(a) Fraction strategy alignment",
        fontsize=15,
        fontweight="semibold",
    )
    ax_tvd.tick_params(axis="x", labelsize=11.5)
    ax_tvd.tick_params(axis="y", labelsize=12)
    ax_tvd.set_ylim(0, 0.37)
    ax_tvd.set_yticks(np.arange(0, 0.36, 0.1))
    ax_tvd.grid(axis="y", color="#dedede", linewidth=0.55, alpha=0.9, zorder=0)
    ax_tvd.spines["top"].set_visible(False)
    ax_tvd.spines["right"].set_visible(False)

    right = outer[0, 1].subgridspec(
        3,
        4,
        height_ratios=[0.22, 1.0, 1.0],
        hspace=0.42,
        wspace=0.13,
    )
    header = fig.add_subplot(right[0, :])
    header.axis("off")
    header.set_title(
        "(b) Strategy distributions by problem type",
        fontweight="semibold",
        pad=3,
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=family_colors[family])
        for family in FAMILIES
    ]
    header.legend(
        legend_handles,
        [family_labels[family] for family in FAMILIES],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=4,
        frameon=False,
        fontsize=8.1,
        columnspacing=0.9,
        handlelength=1.25,
    )

    cell_distributions = {
        source: distribution_by_cell(distributions[source], problem_cells)
        for source in SERIES_ORDER
    }
    mixture_y = np.arange(len(SERIES_ORDER))[::-1]
    for cell_index, cell in enumerate(FRACTION_CELLS):
        row = 1 + cell_index // 4
        col = cell_index % 4
        ax = fig.add_subplot(right[row, col])
        left = np.zeros(len(SERIES_ORDER))
        mixtures = np.vstack(
            [
                cell_distributions[source].loc[[cell], FAMILIES].iloc[0].to_numpy()
                for source in SERIES_ORDER
            ]
        )
        for family_index, family in enumerate(FAMILIES):
            values = 100.0 * mixtures[:, family_index]
            ax.barh(
                mixture_y,
                values,
                left=left,
                height=0.70,
                color=family_colors[family],
                edgecolor="white",
                linewidth=0.35,
            )
            left += values

        ax.set_title(f"{cell[0]}–{cell[1]}", fontsize=10.5, pad=3)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        if row == 2:
            ax.set_xticklabels(["0", "50", "100"])
        else:
            ax.set_xticklabels([])
        if col == 0:
            compact_sources = ["Human", "UMA", "Centaur", "Direct FT", SYSNAME]
            ax.set_yticks(mixture_y, compact_sources)
            ax.get_yticklabels()[-1].set_fontweight("bold")
        else:
            ax.set_yticks(mixture_y, [])
        ax.grid(axis="x", color="#e3e3e3", linewidth=0.45, alpha=0.8, zorder=0)
        ax.tick_params(axis="both", length=2.5, pad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        0.695,
        0.035,
        "Strategy share (%)",
        ha="center",
        va="center",
        fontsize=11,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / "fraction_strategy_alignment"
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def build_outputs(
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    table_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path]:
    repo_root = find_repo_root(Path(__file__)) if repo_root is None else Path(repo_root)
    paper_root = repo_root / "writing/emnlp202026-humanlike-math-reasoning"
    output_dir = paper_root / "figures" if output_dir is None else Path(output_dir)
    table_dir = paper_root / "tables" if table_dir is None else Path(table_dir)

    distributions = {
        "Human": load_human_distribution(repo_root),
        "UMA": load_uma_distribution(repo_root),
    }
    for source in PREDICTION_FILES:
        distributions[source] = load_prediction_distribution(repo_root, source)
    human_problems = set(distributions["Human"].index)
    for source, distribution in distributions.items():
        if set(distribution.index) != human_problems:
            raise ValueError(
                f"{source} problem set differs from Human: "
                f"{sorted(set(distribution.index).symmetric_difference(human_problems))}"
            )

    problem_cells = load_problem_cells(repo_root)
    summary, per_problem, paired = build_statistics(distributions)
    table_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_dir / "fraction_strategy_alignment_summary.csv", index=False)
    per_problem.to_csv(
        table_dir / "fraction_strategy_alignment_per_problem.csv",
        index=False,
    )
    paired.to_csv(
        table_dir / "fraction_strategy_alignment_paired_bootstrap.csv",
        index=False,
    )
    png_path, pdf_path = plot_figure(
        distributions,
        summary,
        problem_cells,
        output_dir,
    )
    return summary, per_problem, paired, png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--table-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, _, paired, png_path, pdf_path = build_outputs(
        args.repo_root,
        args.output_dir,
        args.table_dir,
    )
    print(summary.to_string(index=False))
    print(f"\nPaired problem-bootstrap improvements (positive favors {SYSNAME}):")
    print(paired.to_string(index=False))
    print(f"\nwrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
