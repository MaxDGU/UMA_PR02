#!/usr/bin/env python3
"""Build the merged fraction/decimal per-cell accuracy figure.

The figure combines the former fraction and decimal per-cell figures and adds
the best persona-prompted frontier run, Centaur-70B, and a
direct-human-fine-tuning ablation.  The prompted run is selected once per
domain by whole-profile MAE (not separately by cell).  "Direct human FT" means
Qwen3-4B-Base fine-tuned on human responses without cognitive-model
distillation.  MAE is the equally weighted mean absolute gap between model and
human accuracy over the eight fraction cells or six decimal cells.

Outputs
-------
writing/emnlp202026-humanlike-math-reasoning/figures/
    fraction_decimal_per_cell_merged.png
    fraction_decimal_per_cell_merged.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRACTION_CELLS = [
    ("Add", "ED"),
    ("Add", "UD"),
    ("Sub", "ED"),
    ("Sub", "UD"),
    ("Mul", "ED"),
    ("Mul", "UD"),
    ("Div", "ED"),
    ("Div", "UD"),
]
DECIMAL_CELLS = [
    ("Add", "EDD"),
    ("Add", "UDD"),
    ("Add", "D-W"),
    ("Mul", "EDD"),
    ("Mul", "UDD"),
    ("Mul", "D-W"),
]
SYSNAME = "CPT-LLM"
OURS_LABEL = f"{SYSNAME} (ours)"
SERIES_ORDER = [
    "Human",
    "UMA",
    "Best persona prompt",
    "Centaur-70B",
    "Direct human FT",
    OURS_LABEL,
]


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "eval" / "outputs").is_dir() and (
            candidate / "writing" / "emnlp202026-humanlike-math-reasoning"
        ).is_dir():
            return candidate
    raise RuntimeError("Could not locate the UMA_PR02 repository root.")


def normalize_operation(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str[:3].str.capitalize()


def fraction_profile(path: Path, value_col: str) -> pd.Series:
    frame = pd.read_csv(path, low_memory=False)
    denom_col = "denom_type" if "denom_type" in frame.columns else "operands"
    frame["operation_key"] = normalize_operation(frame["operation"])
    frame["cell_key"] = list(
        zip(frame["operation_key"], frame[denom_col].astype(str).str.upper())
    )
    return frame.groupby("cell_key")[value_col].mean().reindex(FRACTION_CELLS)


def decimal_profile(path: Path, value_col: str) -> pd.Series:
    frame = pd.read_csv(path, low_memory=False)
    frame["operation_key"] = normalize_operation(frame["operation"])
    frame["cell_key"] = list(
        zip(frame["operation_key"], frame["operands"].astype(str).str.upper())
    )
    return frame.groupby("cell_key")[value_col].mean().reindex(DECIMAL_CELLS)


def load_fraction_profiles(repo_root: Path) -> dict[str, pd.Series]:
    human = pd.read_csv(
        repo_root / "eval/data/siegler_fraction_human.csv", low_memory=False
    ).dropna(subset=["acc"])
    human["operation_key"] = normalize_operation(human["operation"])
    human["cell_key"] = list(
        zip(human["operation_key"], human["operands"].astype(str).str.upper())
    )
    human_profile = human.groupby("cell_key")["acc"].mean().reindex(FRACTION_CELLS)

    uma = pd.read_csv(
        repo_root / "eval/outputs/uma_reference/verification_full_results.csv.gz",
        low_memory=False,
    )
    uma["operation_key"] = normalize_operation(uma["operation"])
    uma["cell_key"] = list(
        zip(uma["operation_key"], uma["denoms"].astype(str).str.upper())
    )
    uma_profile = uma.groupby("cell_key")["acc"].mean().reindex(FRACTION_CELLS)

    outputs = repo_root / "eval/outputs"
    return {
        "Human": human_profile,
        "UMA": uma_profile,
        # Best whole-profile MAE in the extended 15-persona frontier sweep:
        # GPT-5.5 low prompted as a student who has "barely started" fractions.
        "Best persona prompt": fraction_profile(
            outputs
            / "persona_sweep/fraction/gpt_5_5_low_wide_barely_started.csv.gz",
            "is_correct",
        ),
        "Centaur-70B": fraction_profile(
            outputs / "fraction/centaur_70b.csv", "is_correct"
        ),
        # The v1 files reproduce the fraction MAEs currently reported in Table 1.
        "Direct human FT": fraction_profile(
            outputs / "fraction_4b/qwen3_4b_humanft_v1.csv", "is_correct"
        ),
        OURS_LABEL: fraction_profile(
            outputs / "fraction_4b/qwen3_4b_distill_humanft_v1.csv", "is_correct"
        ),
    }


def load_decimal_profiles(repo_root: Path) -> dict[str, pd.Series]:
    human_profile = decimal_profile(
        repo_root
        / "eval/outputs/neurips_reproduction/raw/decimal/"
        "decimal_human_responses.csv",
        "acc",
    )

    uma = pd.read_csv(
        repo_root / "eval/outputs/uma_reference/uma_bss2021_summary.csv",
        low_memory=False,
    )
    uma["operation_key"] = normalize_operation(uma["operation"])
    uma["cell_key"] = list(
        zip(uma["operation_key"], uma["operands"].astype(str).str.upper())
    )
    uma["weighted_acc"] = uma["acc"] * uma["n"]
    uma_grouped = uma.groupby("cell_key", as_index=True).agg(
        weighted_acc=("weighted_acc", "sum"),
        n=("n", "sum"),
    )
    uma_profile = (uma_grouped["weighted_acc"] / uma_grouped["n"]).reindex(
        DECIMAL_CELLS
    )

    outputs = repo_root / "eval/outputs"
    return {
        "Human": human_profile,
        "UMA": uma_profile,
        # Best whole-profile MAE in the corrected 10-persona frontier sweep:
        # Claude Sonnet 4.6 prompted with the decimal-place-counting error.
        "Best persona prompt": decimal_profile(
            outputs
            / "persona_sweep/decimal/claude_sonnet_4_6_err_forgot_kcf.csv.gz",
            "is_correct",
        ),
        "Centaur-70B": decimal_profile(
            outputs / "decimal/centaur_70b.csv", "is_correct"
        ),
        "Direct human FT": decimal_profile(
            outputs / "decimal_4b/qwen3_4b_humanft.csv", "is_correct"
        ),
        OURS_LABEL: decimal_profile(
            outputs / "decimal_4b/qwen3_4b_distill_humanft.csv", "is_correct"
        ),
    }


def seed_standard_deviation(
    repo_root: Path, config: str, cells: list[tuple[str, str]]
) -> np.ndarray:
    frame = pd.read_csv(
        repo_root / "eval/outputs/seed_variance/seed_cell_acc.csv"
    )
    frame = frame[(frame["config"] == config) & (frame["cell"] != "__overall__")]
    standard_deviation = frame.groupby("cell")["acc"].std(ddof=1)
    keys = [f"{operation} {operand_type}" for operation, operand_type in cells]
    return standard_deviation.reindex(keys).to_numpy()


def profile_mae(profile: pd.Series, human_profile: pd.Series) -> float:
    return float((profile - human_profile).abs().mean() * 100.0)


def metrics_frame(
    fraction_profiles: dict[str, pd.Series],
    decimal_profiles: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for domain, profiles in [
        ("Fraction", fraction_profiles),
        ("Decimal", decimal_profiles),
    ]:
        human = profiles["Human"]
        for series in SERIES_ORDER:
            rows.append(
                {
                    "domain": domain,
                    "series": series,
                    "mae_pp": profile_mae(profiles[series], human),
                    "overall_cell_mean_accuracy_pct": float(
                        profiles[series].mean() * 100.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_panel(
    ax: plt.Axes,
    profiles: dict[str, pd.Series],
    cells: list[tuple[str, str]],
    title: str,
    colors: list[object],
    cpt_error: np.ndarray,
) -> None:
    human = profiles["Human"]
    x = np.arange(len(cells))
    width = 0.86 / len(SERIES_ORDER)
    offsets = (
        np.arange(len(SERIES_ORDER)) - (len(SERIES_ORDER) - 1) / 2
    ) * width

    for offset, series, color in zip(offsets, SERIES_ORDER, colors, strict=True):
        values = profiles[series].to_numpy() * 100.0
        error = cpt_error if series == OURS_LABEL else None
        mae = profile_mae(profiles[series], human)
        ax.bar(
            x + offset,
            values,
            width,
            label=f"{series} ({mae:.1f})",
            color=color,
            edgecolor="white",
            linewidth=0.35,
            yerr=error,
            error_kw={
                "elinewidth": 0.7,
                "capsize": 1.5,
                "ecolor": "#333333",
            },
            zorder=2,
        )

    ax.set_title(title, fontsize=18, fontweight="semibold", pad=11)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{operation}\n{operand_type}" for operation, operand_type in cells],
        fontsize=13,
    )
    ax.tick_params(axis="y", labelsize=13)
    # Reserve the area above 100% for the domain-specific legend.  Because the
    # MAE differs by domain, a single shared legend would be ambiguous.
    ax.set_ylim(0, 145)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#dedede", linewidth=0.45, alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
        fontsize=12,
        handlelength=1.1,
        columnspacing=0.7,
        borderaxespad=0,
    )
    legend.get_texts()[SERIES_ORDER.index(OURS_LABEL)].set_fontweight(
        "bold"
    )


def build_figure(
    repo_root: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    if repo_root is None:
        repo_root = find_repo_root(Path(__file__))
    repo_root = Path(repo_root).resolve()
    if output_dir is None:
        output_dir = (
            repo_root
            / "writing/emnlp202026-humanlike-math-reasoning/figures"
        )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fraction_profiles = load_fraction_profiles(repo_root)
    decimal_profiles = load_decimal_profiles(repo_root)
    metrics = metrics_frame(fraction_profiles, decimal_profiles)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Preserve the existing colors for the five shared series so this figure
    # stays visually consistent with the strategy-alignment figure.
    shared_order = [
        "Human",
        "UMA",
        "Centaur-70B",
        "Direct human FT",
        OURS_LABEL,
    ]
    color_by_series = dict(
        zip(
            shared_order,
            plt.colormaps["viridis"](np.linspace(0.08, 0.88, 5)),
            strict=True,
        )
    )
    color_by_series["Best persona prompt"] = "#9a9a9a"
    colors = [color_by_series[series] for series in SERIES_ORDER]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.15),
        dpi=300,
        gridspec_kw={"width_ratios": [8, 6]},
    )
    plot_panel(
        axes[0],
        fraction_profiles,
        FRACTION_CELLS,
        "Fraction",
        colors,
        seed_standard_deviation(repo_root, "base_frac", FRACTION_CELLS),
    )
    plot_panel(
        axes[1],
        decimal_profiles,
        DECIMAL_CELLS,
        "Decimal",
        colors,
        seed_standard_deviation(repo_root, "base_dec", DECIMAL_CELLS),
    )
    axes[0].set_ylabel("Accuracy (%)", fontsize=15)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.14, top=0.91, wspace=0.13)

    output_stem = output_dir / "fraction_decimal_per_cell_merged"
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return metrics, png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="UMA_PR02 repository root (auto-detected by default).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory (paper figures directory by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, png_path, pdf_path = build_figure(args.repo_root, args.output_dir)
    print(metrics.to_string(index=False, formatters={"mae_pp": "{:.2f}".format}))
    print(f"\nwrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
