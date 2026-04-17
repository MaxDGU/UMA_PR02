#!/usr/bin/env python3
"""
Build a notebook for the automatic translator scorecard.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results" / "transformer_replication" / "analyze_translator_automatic_metrics.ipynb"

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

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                # Translator Automatic Metrics: Current Template Evaluation

                **Question.** We evaluate the current clean-child translator with fully automatic metrics that do not require human annotation or learned classifiers.

                **Scope.** This notebook focuses on six automatic metric families:

                1. rule-based coverage mass,
                2. symbolic collision rate,
                3. rule-based recoverability from `response_nl`,
                4. minimal-pair sensitivity,
                5. template repetition and diversity,
                6. response length, readability, and lexical variety.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                ## Formal Setup

                **Data.** Let a translated row be
                \[
                z_i = (p_i, s_i, g_i, e_i, a_i, r_i),
                \]
                where \(p_i\) is the problem, \(s_i\) the strategy code, \(g_i\) the goal-token set, \(e_i\) the
                exec-token set, \(a_i\) the answer, and \(r_i=\texttt{response\_nl}\) the narrated trace.

                **Sampling protocol.** We use the same stable sample as the earlier audit: scan the first
                \(1.5 \times 10^6\) rows of the clean-child `correct_exec` CSV and keep rows with
                \[
                \texttt{source\_row\_idx} \bmod 97 = 0.
                \]

                **Recoverability.** We define rule-based extractors \(\hat{s}(r_i)\), \(\hat{g}(r_i)\), and \(\hat{e}(r_i)\)
                from the narrated text. For strategy, we evaluate family-level recovery:
                \[
                \mathrm{Acc}_{\mathrm{family}} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}[\hat{s}(r_i)=\mathrm{family}(s_i)].
                \]
                For goal and exec tokens, we evaluate token-level precision, recall, and \(F_1\).

                **Minimal-pair sensitivity.** For a component \(c\), we search for groups of rows that keep all symbolic
                state fixed except \(c\). The response-changed rate is
                \[
                \mathrm{MPS}(c) = \frac{1}{|\mathcal{G}_c|}\sum_{g \in \mathcal{G}_c}
                \mathbf{1}[T_g^{(+)} \neq T_g^{(-)}],
                \]
                where \(T_g^{(+)}\) and \(T_g^{(-)}\) are the sets of normalized response templates for the rows with and
                without \(c\).

                **Template diversity.** We normalize numeric values to `<NUM>` and study the resulting template string
                family. This lets us separate numeric variation from phrasing variation.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                from __future__ import annotations

                import json
                import math
                import sys
                from pathlib import Path
                from types import SimpleNamespace

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
                pd.set_option("display.max_columns", 30)
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                def find_project_root(start: Path) -> Path:
                    start = start.resolve()
                    for path in [start, *start.parents]:
                        if (path / "AGENTS.md").exists():
                            return path
                    raise FileNotFoundError("Could not locate the project root from the current working directory.")


                try:
                    PROJECT_ROOT = find_project_root(Path.cwd())
                except FileNotFoundError:
                    PROJECT_ROOT = Path("/n/fs/cogai/cs1095/UMA_PR02")

                RESULTS_DIR = PROJECT_ROOT / "results" / "transformer_replication"
                FIGURE_DIR = RESULTS_DIR / "figures" / "translator_auto_metrics"
                FIGURE_DIR.mkdir(parents=True, exist_ok=True)

                if str(RESULTS_DIR) not in sys.path:
                    sys.path.insert(0, str(RESULTS_DIR))

                import analyze_translator_rule_audit as audit
                import evaluate_translator_automatic_metrics as auto_eval

                args = SimpleNamespace(
                    data_csv=RESULTS_DIR / "synth_unique_seed1_all1000_nlp_clean_child_correct_exec_mincols.csv.gz",
                    out_dir=RESULTS_DIR / "translator_auto_metrics",
                    sample_modulus=97,
                    chunksize=200_000,
                    max_input_rows=1_500_000,
                    edit_cluster_threshold=0.92,
                    top_k_templates=25,
                )

                sample_df = auto_eval.build_enriched_sample(args)
                sample_df = auto_eval.add_rule_based_predictions(sample_df)
                component_rows = audit.build_component_rows(sample_df)
                coverage_lookup = auto_eval.build_component_lookup()

                coverage_outputs = auto_eval.compute_coverage_outputs(component_rows)
                collision_outputs = audit.compute_collision_outputs(sample_df)
                strategy_recoverability = auto_eval.compute_strategy_recoverability(sample_df)
                goal_recoverability = auto_eval.compute_token_recoverability(
                    sample_df,
                    component_type="goal",
                    tokens=audit.GOAL_CODES,
                    pred_column="pred_goal_tokens",
                    truth_column="goal_tokens",
                    coverage_lookup=coverage_lookup,
                )
                exec_recoverability = auto_eval.compute_token_recoverability(
                    sample_df,
                    component_type="exec",
                    tokens=audit.EXEC_CODES,
                    pred_column="pred_exec_tokens",
                    truth_column="exec_tokens",
                    coverage_lookup=coverage_lookup,
                )
                minimal_pair_outputs = auto_eval.compute_minimal_pair_outputs(sample_df, coverage_lookup)
                template_diversity = auto_eval.compute_template_diversity(
                    sample_df,
                    threshold=args.edit_cluster_threshold,
                    top_k=args.top_k_templates,
                )
                readability = auto_eval.compute_length_readability(sample_df)

                len(sample_df), len(component_rows)
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Overview

                **Setup.** We begin with a compact scorecard for the current template, then unpack each metric family.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                overview_df = pd.DataFrame(
                    [
                        {"metric": "sample_rows", "value": len(sample_df)},
                        {"metric": "strategy_family_accuracy", "value": strategy_recoverability["summary"]["strategy_family_accuracy"]},
                        {"metric": "goal_mean_recall", "value": goal_recoverability["by_component"]["recall"].mean()},
                        {"metric": "exec_mean_recall", "value": exec_recoverability["by_component"]["recall"].mean()},
                        {"metric": "collision_row_rate", "value": collision_outputs["overall"]["collision_row_rate"]},
                        {"metric": "mean_minimal_pair_changed_rate", "value": minimal_pair_outputs["by_component"]["response_changed_rate"].mean()},
                        {"metric": "unique_templates", "value": template_diversity["summary"]["unique_templates"]},
                        {"metric": "top_10_template_row_share", "value": template_diversity["summary"]["top_10_template_row_share"]},
                        {"metric": "mean_tokens", "value": readability["mean_tokens"]},
                        {"metric": "flesch_kincaid_grade", "value": readability["flesch_kincaid_grade"]},
                    ]
                )

                coverage_component = (
                    coverage_outputs["component_counts"]
                    .pivot(index="component_type", columns="coverage_class", values="n_components")
                    .fillna(0)
                    .astype(int)
                )
                coverage_mass = (
                    coverage_outputs["mass_counts"]
                    .pivot(index="component_type", columns="coverage_class", values="sample_occurrences")
                    .fillna(0)
                    .astype(int)
                )

                display(overview_df)
                display(Markdown("**Static coverage counts.**"))
                display(coverage_component)
                display(Markdown("**Sample-weighted coverage mass.**"))
                display(coverage_mass)
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Strategy Recoverability

                **Question.** How much of the strategy signal can we recover directly from `response_nl` with simple rules?
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                confusion = strategy_recoverability["confusion"].pivot(
                    index="strategy_family_truth",
                    columns="pred_strategy_family",
                    values="rows",
                ).fillna(0)

                fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
                sns.heatmap(confusion, annot=True, fmt=".0f", cmap="Blues", linewidths=0.5, linecolor="white", ax=axes[0])
                axes[0].set_title("Strategy Family Confusion")
                axes[0].set_xlabel("Predicted family")
                axes[0].set_ylabel("True family")

                by_family = strategy_recoverability["by_family"].sort_values("rows", ascending=False)
                axes[1].bar(by_family["strategy_family_truth"], by_family["accuracy"], color="#4c78a8")
                axes[1].set_ylim(0.0, 1.0)
                axes[1].set_title("Strategy Family Recoverability")
                axes[1].set_xlabel("True strategy family")
                axes[1].set_ylabel("Accuracy")
                fig_path = FIGURE_DIR / "strategy_family_recoverability.png"
                fig.savefig(fig_path, bbox_inches="tight")
                plt.show()

                display(by_family)
                fig_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Setup.** The confusion matrix evaluates rule-based family recovery from the narrated text alone. We do
                not use the symbolic columns for prediction.

                **Interpretation.** `CDON`, `KDON`, `ONOD`, and `CROP` are often recoverable because the template gives
                them distinctive surface forms. `ICDM` is much harder because many `ICDM_OG` rows fall back to generic
                “I worked with ... and got ...” narration rather than a unique strategy cue.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Goal and Exec Recoverability

                **Question.** Which symbolic components can a simple rule-based parser recover from `response_nl`, and how
                does that depend on the earlier coverage classes?
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                goal_class = goal_recoverability["by_class"].copy()
                goal_class["component_group"] = "goal"
                exec_class = exec_recoverability["by_class"].copy()
                exec_class["component_group"] = "exec"
                recoverability_class = pd.concat([goal_class, exec_class], ignore_index=True)

                fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
                for ax, metric, title in [
                    (axes[0], "mean_recall", "Mean Token Recall by Coverage Class"),
                    (axes[1], "mean_precision", "Mean Token Precision by Coverage Class"),
                ]:
                    sns.barplot(
                        data=recoverability_class,
                        x="coverage_class",
                        y=metric,
                        hue="component_group",
                        ax=ax,
                        palette={"goal": "#f2b701", "exec": "#d95f02"},
                    )
                    ax.set_ylim(0.0, 1.05)
                    ax.set_xlabel("Coverage class")
                    ax.set_ylabel(metric.replace("_", " "))
                    ax.set_title(title)
                    ax.tick_params(axis="x", rotation=25)
                axes[1].legend(title="Component group", frameon=False)
                fig_path = FIGURE_DIR / "recoverability_by_class.png"
                fig.savefig(fig_path, bbox_inches="tight")
                plt.show()

                display(Markdown("**Goal recoverability by component.**"))
                display(goal_recoverability["by_component"])
                display(Markdown("**Exec recoverability by component.**"))
                display(exec_recoverability["by_component"])
                fig_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Setup.** For goals and exec rules, the parser checks whether the narrated text contains rule-specific
                phrases such as “I checked if it could be simplified” or “I changed the bottom numbers and kept the top
                numbers the same.”

                **Interpretation.** Explicit components tend to have high recall, while omitted and ignored components
                collapse to zero recall by construction. Bundled components sit in the middle: the text often hints at
                them, but not always at the right symbolic granularity. One useful warning sign is that some components
                labeled explicit in the template audit are still not fully recoverable as exact tokens, which means the
                surface phrasing can expose a behavior without uniquely identifying the original UMA tokenization.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Minimal-Pair Sensitivity

                **Question.** When one symbolic component changes and the rest of the symbolic state stays fixed, does the
                narration change?
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                mp_class = minimal_pair_outputs["by_class"].copy()

                fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
                sns.barplot(data=mp_class, x="coverage_class", y="mean_response_changed_rate", color="#1b9e77", ax=axes[0])
                axes[0].set_ylim(0.0, 1.05)
                axes[0].set_title("Minimal-Pair Response-Changed Rate")
                axes[0].set_xlabel("Coverage class")
                axes[0].set_ylabel("Changed rate")
                axes[0].tick_params(axis="x", rotation=25)

                sns.barplot(data=mp_class, x="coverage_class", y="mean_response_collapsed_rate", color="#d95f02", ax=axes[1])
                axes[1].set_ylim(0.0, 1.05)
                axes[1].set_title("Minimal-Pair Collapsed Rate")
                axes[1].set_xlabel("Coverage class")
                axes[1].set_ylabel("Collapsed rate")
                axes[1].tick_params(axis="x", rotation=25)
                fig_path = FIGURE_DIR / "minimal_pair_sensitivity.png"
                fig.savefig(fig_path, bbox_inches="tight")
                plt.show()

                display(mp_class)
                display(Markdown("**Minimal-pair component table.**"))
                display(minimal_pair_outputs["by_component"].head(25))
                fig_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                **Setup.** A qualifying minimal pair keeps the problem, answer, and all symbolic state fixed except one
                component. The changed-rate asks whether the normalized response template changes across that toggle.

                **Interpretation.** High changed-rate means the component has a visible causal footprint in the prose.
                High collapsed-rate means the same response template survives even after the component changes. This metric
                is strict and sample-limited, so some components have few or no qualifying pairs. Still, the results are
                useful for showing that many strategy suffixes and hidden tokens do not reliably trigger a wording change.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Template Repetition and Diversity

                **Question.** How repetitive is the current translation template once we factor out the numeric values?
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                top_templates = template_diversity["top_templates"].head(15).copy().iloc[::-1]
                clusters = template_diversity["clusters"].head(15).copy().iloc[::-1]

                fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
                axes[0].barh(top_templates["response_template"], top_templates["share_rows"], color="#4c78a8")
                axes[0].set_title("Most Frequent Normalized Templates")
                axes[0].set_xlabel("Row share")
                axes[0].set_ylabel("")

                axes[1].barh(clusters["representative"], clusters["share_rows"], color="#8da0cb")
                axes[1].set_title("Largest Edit-Distance Template Families")
                axes[1].set_xlabel("Row share")
                axes[1].set_ylabel("")
                fig_path = FIGURE_DIR / "template_diversity.png"
                fig.savefig(fig_path, bbox_inches="tight")
                plt.show()

                display(pd.DataFrame([template_diversity["summary"]]))
                fig_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Setup.** We replace numeric values with `<NUM>` and treat the remaining text as a normalized template.
                The left panel counts exact normalized templates; the right panel groups similar templates by greedy
                edit-distance clustering.

                **Interpretation.** A high top-template share means a small number of phrasing patterns dominate the
                dataset. That is useful for consistency, but it also means the model sees highly repetitive linguistic
                supervision. The cluster view is slightly coarser: it measures repetition even when the exact wording
                differs by a short clause such as an added simplification sentence.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Length, Readability, and Lexical Variety

                **Question.** How simple and repetitive is the current surface text?
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                response_lengths = pd.Series([len(tokens) for tokens in sample_df["response_tokens_normalized"]], name="tokens")

                fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
                sns.histplot(response_lengths, bins=24, color="#59a14f", ax=axes[0])
                axes[0].set_title("Response Length Distribution")
                axes[0].set_xlabel("Tokens per response")
                axes[0].set_ylabel("Rows")

                readability_plot = pd.DataFrame(
                    [
                        {"metric": "mean_tokens", "value": readability["mean_tokens"]},
                        {"metric": "mean_sentences", "value": readability["mean_sentences"]},
                        {"metric": "flesch_kincaid_grade", "value": readability["flesch_kincaid_grade"]},
                        {"metric": "flesch_reading_ease / 100", "value": readability["flesch_reading_ease"] / 100.0},
                        {"metric": "type_token_ratio_normalized", "value": readability["type_token_ratio_normalized"]},
                        {"metric": "distinct_2_normalized", "value": readability["distinct_2_normalized"]},
                    ]
                )
                axes[1].bar(readability_plot["metric"], readability_plot["value"], color="#e15759")
                axes[1].set_title("Length and Variety Summary")
                axes[1].set_xlabel("")
                axes[1].set_ylabel("Value")
                axes[1].tick_params(axis="x", rotation=30)
                fig_path = FIGURE_DIR / "readability_summary.png"
                fig.savefig(fig_path, bbox_inches="tight")
                plt.show()

                display(pd.DataFrame([readability]))
                fig_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Setup.** These metrics operate on the narrated response body after removing the `### answer:` suffix.
                Numeric values are normalized for lexical-variety statistics so that number identities do not dominate the
                vocabulary counts.

                **Interpretation.** The current template is short and easy to read, but the lexical-variety metrics show
                heavy repetition. That combination is consistent with a template that is easy for a learner to imitate
                locally yet still too narrow to capture the richer symbolic distinctions present in UMA.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                ## Interpretation

                **Finding.** The current template has a mixed profile. It exposes enough signal to recover coarse
                strategy families and some explicit goal/exec rules, but it still collapses substantial symbolic state.

                **Evidence.** The strongest automatic warning signs are:

                - weak `ICDM` family recoverability because many rows fall back to generic narration,
                - zero-recall omitted and ignored components by design,
                - sparse and often collapsed minimal pairs,
                - high concentration of row mass in a small number of normalized templates.

                **Implication.** Improving the translation template should aim for a better Pareto point: more
                recoverable symbolic content without turning the traces into literal token dumps. The current automatic
                scorecard gives us a concrete way to test whether future template revisions move in that direction.
                """
            )
        )
    )

    nb["cells"] = cells
    nbf.write(nb, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
