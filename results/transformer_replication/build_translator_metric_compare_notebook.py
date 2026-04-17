#!/usr/bin/env python3
"""
Build a before/after notebook for translator metric revisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a before/after translator-metrics notebook.")
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "results" / "transformer_replication"
    parser.add_argument(
        "--before-dir",
        type=Path,
        default=results_dir / "translator_auto_metrics",
        help="Directory containing the baseline metric exports.",
    )
    parser.add_argument(
        "--after-dir",
        type=Path,
        default=results_dir / "translator_auto_metrics_v9",
        help="Directory containing the revised-template metric exports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir / "analyze_translator_metric_compare.ipynb",
        help="Path for the generated notebook.",
    )
    parser.add_argument(
        "--figure-dir-name",
        type=str,
        default="translator_metric_compare",
        help="Subdirectory under figures/ used for exported plots.",
    )
    parser.add_argument(
        "--parameter-label",
        type=str,
        default="",
        help="Optional parameter tuple label, for example '(g, d, rt_mu, ice) = (0.01, 0.1, 5, 100)'.",
    )
    parser.add_argument(
        "--slice-description",
        type=str,
        default="",
        help="Optional plain-language description of the evaluated slice.",
    )
    parser.add_argument(
        "--use-all-rows",
        action="store_true",
        help="Document the comparison as evaluating all rows from a fixed slice rather than the stable sample rule.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    }

    cells = []
    parameter_clause = ""
    if args.parameter_label:
        parameter_clause = f" on the fixed slice {args.parameter_label}"
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                f"""
                # Translator Metrics: Before vs After Template Revision

                **Question.** Did the targeted translator revision improve automatic recoverability and coverage{parameter_clause} without making the traces substantially less natural or more repetitive?
                """
            )
        )
    )
    if args.use_all_rows:
        parameter_setup = ""
        if args.parameter_label:
            parameter_setup = (
                "The source slice is defined by\n"
                "\\[\n"
                f"D_\\theta = \\{{i : (g_i, d_i, rt_{{\\mu,i}}, ice_i) = {args.parameter_label}\\}},\n"
                "\\]\n"
            )
        slice_detail = ""
        if args.slice_description:
            slice_detail = f"{args.slice_description} "
        protocol_text = (
            f"{parameter_setup}"
            f"{slice_detail}"
            "Both scorecards evaluate **all translated rows** in that slice. "
            "No extra stable-sample subsampling is applied."
        )
    else:
        protocol_text = dedent(
            r"""
            Both scorecards use the same stable sample rule,
            \[
            \texttt{source\_row\_idx} \bmod 97 = 0,
            \]
            over the first \(1.5 \times 10^6\) translated rows.
            """
        ).strip()
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                (
                    "## Formal Setup\n\n"
                    "**Protocol.** We compare two translated datasets built from the same symbolic source and evaluated with the\n"
                    "same automatic scorecard:\n\n"
                    "- **Before**: the original clean-child template (`v8`)\n"
                    "- **After**: the revised clean-child template (`v9`)\n\n"
                    f"{protocol_text}\n\n"
                    "**Comparison principle.** For a metric \\(m\\), we report\n"
                    "\\[\n"
                    "\\Delta m = m_{\\mathrm{after}} - m_{\\mathrm{before}}.\n"
                    "\\]\n"
                    "Positive deltas are improvements for accuracy-like and recall-like metrics, while negative deltas are\n"
                    "improvements for collapse-like metrics such as collision rate or top-template concentration.\n"
                )
            )
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                from __future__ import annotations

                import json
                import sys
                from pathlib import Path

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                import seaborn as sns
                from IPython.display import Markdown, display

                sns.set_theme(style="whitegrid", context="talk")
                plt.rcParams.update({
                    "figure.dpi": 160,
                    "savefig.dpi": 300,
                    "font.size": 12,
                    "axes.titlesize": 16,
                    "axes.labelsize": 13,
                    "xtick.labelsize": 11,
                    "ytick.labelsize": 11,
                    "legend.fontsize": 11,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                })
                pd.set_option("display.max_colwidth", 180)
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                f"""
                def find_project_root(start: Path) -> Path:
                    start = start.resolve()
                    for path in [start, *start.parents]:
                        if (path / "AGENTS.md").exists():
                            return path
                    raise FileNotFoundError("Could not locate project root.")


                try:
                    PROJECT_ROOT = find_project_root(Path.cwd())
                except FileNotFoundError:
                    PROJECT_ROOT = Path("/n/fs/cogai/cs1095/UMA_PR02")

                RESULTS_DIR = PROJECT_ROOT / "results" / "transformer_replication"
                FIGURE_DIR = RESULTS_DIR / "figures" / "{args.figure_dir_name}"
                FIGURE_DIR.mkdir(parents=True, exist_ok=True)

                BEFORE_DIR = Path("{args.before_dir.as_posix()}")
                AFTER_DIR = Path("{args.after_dir.as_posix()}")
                if not BEFORE_DIR.is_absolute():
                    BEFORE_DIR = PROJECT_ROOT / BEFORE_DIR
                if not AFTER_DIR.is_absolute():
                    AFTER_DIR = PROJECT_ROOT / AFTER_DIR

                def load_json(path: Path) -> dict:
                    with path.open() as f:
                        return json.load(f)

                before_overall = load_json(BEFORE_DIR / "overall_metrics.json")
                after_overall = load_json(AFTER_DIR / "overall_metrics.json")

                goal_before = pd.read_csv(BEFORE_DIR / "goal_recoverability_by_class.csv")
                goal_after = pd.read_csv(AFTER_DIR / "goal_recoverability_by_class.csv")
                goal_components_before = pd.read_csv(BEFORE_DIR / "goal_recoverability_by_component.csv")
                goal_components_after = pd.read_csv(AFTER_DIR / "goal_recoverability_by_component.csv")
                exec_before = pd.read_csv(BEFORE_DIR / "exec_recoverability_by_class.csv")
                exec_after = pd.read_csv(AFTER_DIR / "exec_recoverability_by_class.csv")
                exec_components_before = pd.read_csv(BEFORE_DIR / "exec_recoverability_by_component.csv")
                exec_components_after = pd.read_csv(AFTER_DIR / "exec_recoverability_by_component.csv")
                strategy_before = pd.read_csv(BEFORE_DIR / "strategy_family_accuracy.csv")
                strategy_after = pd.read_csv(AFTER_DIR / "strategy_family_accuracy.csv")
                minimal_before = pd.read_csv(BEFORE_DIR / "minimal_pair_by_class.csv")
                minimal_after = pd.read_csv(AFTER_DIR / "minimal_pair_by_class.csv")
                top_templates_before = pd.read_csv(BEFORE_DIR / "top_templates.csv")
                top_templates_after = pd.read_csv(AFTER_DIR / "top_templates.csv")
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Headline Deltas
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                rows = [
                    ("strategy_family_accuracy", before_overall["strategy_recoverability"]["strategy_family_accuracy"], after_overall["strategy_recoverability"]["strategy_family_accuracy"], "higher"),
                    ("goal_mean_recall", before_overall["goal_recoverability_mean"]["mean_recall"], after_overall["goal_recoverability_mean"]["mean_recall"], "higher"),
                    ("exec_mean_recall", before_overall["exec_recoverability_mean"]["mean_recall"], after_overall["exec_recoverability_mean"]["mean_recall"], "higher"),
                    ("collision_row_rate", before_overall["collision"]["collision_row_rate"], after_overall["collision"]["collision_row_rate"], "lower"),
                    ("unique_templates", before_overall["template_diversity"]["unique_templates"], after_overall["template_diversity"]["unique_templates"], "depends"),
                    ("top_10_template_row_share", before_overall["template_diversity"]["top_10_template_row_share"], after_overall["template_diversity"]["top_10_template_row_share"], "lower"),
                    ("mean_tokens", before_overall["readability"]["mean_tokens"], after_overall["readability"]["mean_tokens"], "depends"),
                    ("flesch_kincaid_grade", before_overall["readability"]["flesch_kincaid_grade"], after_overall["readability"]["flesch_kincaid_grade"], "depends"),
                ]
                summary = pd.DataFrame(rows, columns=["metric", "before", "after", "preferred_direction"])
                summary["delta"] = summary["after"] - summary["before"]
                display(summary)
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Update on `convert_CD`.** The current template no longer treats `convert_CD` as merely implied by a
                generic common-denominator sentence. In the revised `v9` translator, the narration explicitly says that
                both fractions were changed to a named denominator whenever that denominator can be recovered from the
                problem. This means the `convert_CD` change is now about both faithfulness and recoverability, not only
                about humanlike clarity.

                ## Recoverability by Coverage Class
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                goal_compare = pd.concat(
                    [goal_before.assign(version="before"), goal_after.assign(version="after")],
                    ignore_index=True,
                )
                exec_compare = pd.concat(
                    [exec_before.assign(version="before"), exec_after.assign(version="after")],
                    ignore_index=True,
                )

                fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
                sns.barplot(data=goal_compare, x="coverage_class", y="mean_recall", hue="version", ax=axes[0], palette={"before": "#9c755f", "after": "#1b9e77"})
                axes[0].set_title("Goal Recoverability by Coverage Class")
                axes[0].set_xlabel("Coverage class")
                axes[0].set_ylabel("Mean recall")
                axes[0].tick_params(axis="x", rotation=25)
                sns.barplot(data=exec_compare, x="coverage_class", y="mean_recall", hue="version", ax=axes[1], palette={"before": "#9c755f", "after": "#1b9e77"})
                axes[1].set_title("Exec Recoverability by Coverage Class")
                axes[1].set_xlabel("Coverage class")
                axes[1].set_ylabel("Mean recall")
                axes[1].tick_params(axis="x", rotation=25)
                axes[1].legend(frameon=False, title="Version")
                fig_path = FIGURE_DIR / "recoverability_compare.png"
                fig.savefig(fig_path, bbox_inches="tight")
                plt.show()
                fig_path
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Interpretation.** This panel shows whether the new template actually turned hidden symbolic components
                into recoverable surface cues. The most important gains should appear in the classes we targeted directly:
                omitted/bundled goals and omitted exec rules. At this point, the main remaining recall losses should be
                concentrated in components we still deliberately hide or never verbalize.
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Strategy Family Recovery
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                strategy_compare = strategy_before.merge(
                    strategy_after,
                    on="strategy_family_truth",
                    suffixes=("_before", "_after"),
                )
                strategy_compare["delta"] = strategy_compare["accuracy_after"] - strategy_compare["accuracy_before"]

                fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
                strategy_long = strategy_compare.melt(
                    id_vars=["strategy_family_truth"],
                    value_vars=["accuracy_before", "accuracy_after"],
                    var_name="version",
                    value_name="accuracy",
                )
                strategy_long["version"] = strategy_long["version"].str.replace("accuracy_", "", regex=False)
                sns.barplot(data=strategy_long, x="strategy_family_truth", y="accuracy", hue="version", ax=ax, palette={"before": "#9c755f", "after": "#1b9e77"})
                ax.set_ylim(0.0, 1.0)
                ax.set_title("Strategy Family Recoverability")
                ax.set_xlabel("Strategy family")
                ax.set_ylabel("Accuracy")
                ax.legend(frameon=False, title="Version")
                fig_path = FIGURE_DIR / "strategy_compare.png"
                fig.savefig(fig_path, bbox_inches="tight")
                plt.show()
                display(strategy_compare)
                fig_path
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Residual Component Gaps
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                remaining_goal_recall = goal_components_after.loc[
                    goal_components_after["recall"] < 1.0,
                    [
                        "component",
                        "coverage_class",
                        "prevalence_rows",
                        "tp",
                        "fn",
                        "precision",
                        "recall",
                    ],
                ].sort_values(["recall", "prevalence_rows"], ascending=[True, False])

                remaining_exec_recall = exec_components_after.loc[
                    exec_components_after["recall"] < 1.0,
                    [
                        "component",
                        "coverage_class",
                        "prevalence_rows",
                        "tp",
                        "fn",
                        "precision",
                        "recall",
                    ],
                ].sort_values(["recall", "prevalence_rows"], ascending=[True, False])

                low_precision_goal = goal_components_after.loc[
                    goal_components_after["precision"].fillna(1.0) < 0.9,
                    [
                        "component",
                        "coverage_class",
                        "prevalence_rows",
                        "tp",
                        "fp",
                        "precision",
                        "recall",
                    ],
                ].sort_values(["precision", "fp"], ascending=[True, False])

                low_precision_exec = exec_components_after.loc[
                    exec_components_after["precision"].fillna(1.0) < 0.9,
                    [
                        "component",
                        "coverage_class",
                        "prevalence_rows",
                        "tp",
                        "fp",
                        "precision",
                        "recall",
                    ],
                ].sort_values(["precision", "fp"], ascending=[True, False])

                display(Markdown("**Remaining goal recall gaps.**"))
                display(remaining_goal_recall)

                display(Markdown("**Remaining exec recall gaps.**"))
                display(remaining_exec_recall)

                display(Markdown("**Current low-precision goal cues.**"))
                display(low_precision_goal)

                display(Markdown("**Current low-precision exec cues.**"))
                display(low_precision_exec)
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Interpretation.** After the latest template pass, the residual recall gaps are small and concentrated.
                If a component still shows recall below one, that usually means we are still suppressing it on purpose
                or treating it as metadata rather than narrated reasoning. The more visible remaining issue is now
                precision: some surface cues, such as explicit common-denominator or keep-the-bottom wording, fire for
                a broader family of symbolic states than the exact source label.
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Minimal-Pair and Repetition Tradeoffs
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                minimal_compare = minimal_before.merge(
                    minimal_after,
                    on="coverage_class",
                    suffixes=("_before", "_after"),
                )
                minimal_compare["delta_changed_rate"] = (
                    minimal_compare["mean_response_changed_rate_after"]
                    - minimal_compare["mean_response_changed_rate_before"]
                )

                repetition_compare = pd.DataFrame(
                    [
                        {"metric": "top_10_template_row_share", "before": before_overall["template_diversity"]["top_10_template_row_share"], "after": after_overall["template_diversity"]["top_10_template_row_share"]},
                        {"metric": "unique_templates", "before": before_overall["template_diversity"]["unique_templates"], "after": after_overall["template_diversity"]["unique_templates"]},
                        {"metric": "mean_tokens", "before": before_overall["readability"]["mean_tokens"], "after": after_overall["readability"]["mean_tokens"]},
                    ]
                )
                repetition_compare["delta"] = repetition_compare["after"] - repetition_compare["before"]

                display(minimal_compare)
                display(repetition_compare)
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                ## Interpretation

                **Question.** Did the revision move us toward a better Pareto point?

                **Reading guide.**

                - Higher recoverability is good.
                - Lower collision and lower top-template concentration are good.
                - Slightly longer traces can be acceptable if they expose more symbolic state.
                - A modest readability drop can also be acceptable if the language stays natural.

                **Decision rule.** If recoverability improves substantially for the previously hidden components and the
                repetition metrics do not get dramatically worse, the revision is a net gain.
                """
            )
        )
    )

    nb["cells"] = cells
    nbf.write(nb, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
