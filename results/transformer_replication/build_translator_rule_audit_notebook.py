#!/usr/bin/env python3
"""
Build a notebook for the rule-based translator fidelity audit.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results" / "transformer_replication" / "analyze_translator_rule_audit.ipynb"

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
                # Translator Rule Audit: Rule-Based Fidelity Check

                **Question.** Does the clean-child translator preserve UMA strategy, goal, and execution-rule information in `response_nl`, or does the template collapse substantial symbolic state before training?

                **Method.** This notebook avoids learned classifiers. We use rule-based template inspection, a stable sampled audit of the translated corpus, stratified summaries, and representative manual examples.

                **Finding preview.** The current `response_nl` template preserves some coarse strategy-family cues, but it omits or bundles many goal and exec components. Several distinct symbolic states also collapse to the same translated text.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                ## Formal Setup

                **Setup.** Let one translated example be
                \[
                z_i = (p_i, s_i, g_i, e_i, a_i),
                \]
                where \(p_i\) is the fraction problem, \(s_i\) is the UMA strategy code, \(g_i\) is the set of goal tokens,
                \(e_i\) is the set of exec-rule tokens, and \(a_i\) is the produced answer. The clean-child translator maps
                \[
                T(z_i) = r_i,
                \]
                where \(r_i = \texttt{response\_nl}\) is the narrated reasoning string shown to the language model.

                **Objective.** We study whether \(r_i\) retains the symbolic state in \(s_i, g_i, e_i\). For each component
                \(c \in \{s_i\} \cup g_i \cup e_i\), we assign a rule-based coverage class
                \[
                \kappa(c) \in \{\text{explicit},\ \text{conditional},\ \text{bundled},\ \text{omitted},\ \text{ignored}\}.
                \]

                **Hidden-state burden.** For row \(i\), define
                \[
                h_i = \sum_{c \in \Gamma_i} \mathbf{1}\{\kappa(c) \in \{\text{bundled},\text{omitted},\text{ignored}\}\},
                \]
                where \(\Gamma_i\) is the set of symbolic components attached to row \(i\). Large \(h_i\) means the symbolic
                trace contains information that the narrated text does not state directly.

                **Collision test.** We flag a symbolic collision when two distinct symbolic states map to the same
                \((p_i, a_i, r_i)\):
                \[
                \chi_i = \mathbf{1}\left[\exists j \neq i :
                (p_j, a_j, r_j) = (p_i, a_i, r_i)
                \land
                (s_j, g_j, e_j) \neq (s_i, g_i, e_i)\right].
                \]
                If \(\chi_i = 1\), the translated text is not sufficient to recover the source symbolic state.

                **Protocol.** We reuse the stable audit sample defined by `source_row_idx % 97 == 0` over the first
                \(1.5 \times 10^6\) translated rows from the clean-child `correct_exec` corpus.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                from __future__ import annotations

                import importlib.util
                import math
                import sys
                from dataclasses import asdict
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
                pd.set_option("display.max_columns", 20)
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
                DATA_CSV = RESULTS_DIR / "synth_unique_seed1_all1000_nlp_clean_child_correct_exec_mincols.csv.gz"
                AUDIT_SCRIPT = RESULTS_DIR / "analyze_translator_rule_audit.py"
                FIGURE_DIR = RESULTS_DIR / "figures" / "translator_rule_audit"
                FIGURE_DIR.mkdir(parents=True, exist_ok=True)

                spec = importlib.util.spec_from_file_location("translator_rule_audit_module", AUDIT_SCRIPT)
                audit = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = audit
                spec.loader.exec_module(audit)

                SAMPLE_MODULUS = 97
                MAX_INPUT_ROWS = 1_500_000
                CHUNKSIZE = 200_000

                sample_df = audit.load_stable_sample(
                    DATA_CSV,
                    sample_modulus=SAMPLE_MODULUS,
                    chunksize=CHUNKSIZE,
                    max_input_rows=MAX_INPUT_ROWS,
                )
                component_rows = audit.build_component_rows(sample_df)
                component_summary = audit.compute_component_summary(component_rows)
                collision_outputs = audit.compute_collision_outputs(sample_df)
                coverage_static = pd.DataFrame([asdict(spec) for spec in audit.build_coverage_specs()])

                sample_df.shape, component_rows.shape
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Audit Overview

                **Setup.** We first summarize the audit sample, the static template coverage map, and the observed
                collision rate in the sampled corpus.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                overall = collision_outputs["overall"]

                overview_df = pd.DataFrame(
                    [
                        {"metric": "sample_rows", "value": overall["sample_rows"]},
                        {"metric": "component_rows", "value": int(len(component_rows))},
                        {"metric": "unique_prob_answer_response_groups", "value": overall["unique_prob_answer_response_groups"]},
                        {"metric": "collision_groups", "value": overall["collision_groups"]},
                        {"metric": "collision_group_rate", "value": overall["collision_group_rate"]},
                        {"metric": "collision_row_rate", "value": overall["collision_row_rate"]},
                        {"metric": "mean_unique_states_per_group", "value": overall["mean_unique_states_per_group"]},
                        {"metric": "max_unique_states_per_group", "value": overall["max_unique_states_per_group"]},
                    ]
                )

                coverage_counts = (
                    coverage_static.groupby(["component_type", "coverage_class"], sort=False)
                    .size()
                    .rename("n_components")
                    .reset_index()
                )
                sample_mass = (
                    component_summary.groupby(["component_type", "coverage_class"], sort=False)["sample_count"]
                    .sum()
                    .rename("sample_occurrences")
                    .reset_index()
                )

                display(Markdown("**Overview metrics.**"))
                display(overview_df)
                display(Markdown("**Static template coverage counts.**"))
                display(coverage_counts.pivot(index="component_type", columns="coverage_class", values="n_components").fillna(0).astype(int))
                display(Markdown("**Sample occurrence mass by coverage class.**"))
                display(sample_mass.pivot(index="component_type", columns="coverage_class", values="sample_occurrences").fillna(0).astype(int))
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                **How the 15,476 rows are selected.** This sample is deterministic, not random. The loader scans only
                the first \(1.5 \times 10^6\) rows of the clean-child `correct_exec` CSV, then keeps a row if and only if
                its original `source_row_idx` satisfies
                \[
                \texttt{source\_row\_idx} \bmod 97 = 0.
                \]

                **Important detail.** `source_row_idx` is inherited from the upstream source file and is not renumbered
                after filtering. So this is **not** “every 97th row of the filtered CSV.” It is “every surviving row whose
                original source index is divisible by 97.”

                **Why the count is 15,476.** If the filtered dataset contained every original row, the count would be
                close to \(1{,}500{,}000 / 97 \approx 15{,}464\). The actual count is slightly different because the
                filtered corpus drops many rows before this audit runs, so the surviving rows with `source_row_idx`
                divisible by 97 are not perfectly uniform in the first \(1.5 \times 10^6\) filtered rows.

                **Implication.** The benefit of this rule is stability: rerunning the audit on the same translated CSV
                gives the same sample, which makes before/after translator comparisons easier.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                COVERAGE_ORDER = [
                    "explicit_template",
                    "conditional_template",
                    "bundled_not_explicit",
                    "omitted",
                    "omitted_or_generic",
                    "intentionally_ignored",
                ]
                COVERAGE_LABEL = {
                    "explicit_template": "Explicit",
                    "conditional_template": "Conditional",
                    "bundled_not_explicit": "Bundled",
                    "omitted": "Omitted",
                    "omitted_or_generic": "Omitted",
                    "intentionally_ignored": "Ignored",
                }
                COVERAGE_COLOR = {
                    "Explicit": "#1b9e77",
                    "Conditional": "#7570b3",
                    "Bundled": "#e6ab02",
                    "Omitted": "#d95f02",
                    "Ignored": "#666666",
                }

                static_plot = (
                    coverage_counts.assign(coverage_label=lambda df: df["coverage_class"].map(COVERAGE_LABEL))
                    .groupby(["component_type", "coverage_label"], as_index=False)["n_components"]
                    .sum()
                )
                sample_plot = (
                    sample_mass.assign(coverage_label=lambda df: df["coverage_class"].map(COVERAGE_LABEL))
                    .groupby(["component_type", "coverage_label"], as_index=False)["sample_occurrences"]
                    .sum()
                )

                static_wide = (
                    static_plot.pivot(index="component_type", columns="coverage_label", values="n_components")
                    .fillna(0)
                    [["Explicit", "Conditional", "Bundled", "Omitted", "Ignored"]]
                )
                sample_wide = (
                    sample_plot.pivot(index="component_type", columns="coverage_label", values="sample_occurrences")
                    .fillna(0)
                    [["Explicit", "Conditional", "Bundled", "Omitted", "Ignored"]]
                )

                fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
                for ax, data, title, xlabel in [
                    (axes[0], static_wide, "Template Coverage by Component Type", "Number of symbolic components"),
                    (axes[1], sample_wide, "Sample Mass by Coverage Class", "Sample occurrences"),
                ]:
                    left = np.zeros(len(data))
                    for label in data.columns:
                        ax.barh(
                            data.index,
                            data[label].values,
                            left=left,
                            color=COVERAGE_COLOR[label],
                            label=label,
                            height=0.65,
                        )
                        left = left + data[label].values
                    ax.set_title(title)
                    ax.set_xlabel(xlabel)
                axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Coverage")
                fig_path = FIGURE_DIR / "coverage_overview.png"
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
                r"""
                **Setup.** In these plots, a **symbolic component** means one labeled UMA element attached to a row:
                one strategy code, one goal token, or one exec-rule token. For example, a row might contribute one
                strategy component such as `CDON_OG`, several goal components such as `convert_CD`, `operate_nums`,
                `check_simplify`, and several exec components such as `sub_fact` or `div_calculator`.

                **Coverage labels.**

                - **Explicit** means the component is stated directly in `response_nl`.
                - **Conditional** means the component is only stated in some branches of the template.
                - **Bundled** means the component is partly reflected in the prose, but only through a coarser sentence.
                - **Omitted** means the component is present in the symbolic trace but not verbalized in `response_nl`.
                - **Ignored** means the translator intentionally suppresses that component.

                **Interpretation.** The left panel counts how many distinct symbolic components fall into each coverage
                class for `strategy`, `goal`, and `exec`. The right panel weights the same classes by how often those
                components occur in the sampled dataset. The important pattern is that strategy cues are often explicit,
                while goals and exec rules carry much more bundled, omitted, or ignored mass. So even before model
                fitting, the text target is already discarding a substantial part of the UMA trace.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Which UMA Components Are Hidden?

                **Method.** We rank goals and exec rules by sampled occurrence count, but keep the rule-based coverage
                class attached to each component. This separates rare omissions from common omissions.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                goal_hidden = (
                    component_summary[
                        (component_summary["component_type"] == "goal")
                        & (component_summary["coverage_class"].isin(["bundled_not_explicit", "omitted", "intentionally_ignored"]))
                    ]
                    .sort_values("sample_count", ascending=False)
                    .head(10)
                    .copy()
                )
                exec_hidden = (
                    component_summary[
                        (component_summary["component_type"] == "exec")
                        & (component_summary["coverage_class"].isin(["omitted", "intentionally_ignored"]))
                    ]
                    .sort_values("sample_count", ascending=False)
                    .head(10)
                    .copy()
                )

                goal_hidden["coverage_label"] = goal_hidden["coverage_class"].map(COVERAGE_LABEL)
                exec_hidden["coverage_label"] = exec_hidden["coverage_class"].map(COVERAGE_LABEL)

                fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)
                for ax, df, title in [
                    (axes[0], goal_hidden.iloc[::-1], "Most Common Hidden Goals"),
                    (axes[1], exec_hidden.iloc[::-1], "Most Common Hidden Exec Rules"),
                ]:
                    ax.barh(
                        df["component"],
                        df["sample_count"],
                        color=[COVERAGE_COLOR[label] for label in df["coverage_label"]],
                        height=0.7,
                    )
                    ax.set_title(title)
                    ax.set_xlabel("Sample occurrences")
                fig_path = FIGURE_DIR / "hidden_components.png"
                fig.savefig(fig_path, bbox_inches="tight")
                plt.show()

                display(Markdown("**Goal components with the largest hidden mass.**"))
                display(goal_hidden[["component", "coverage_class", "sample_count", "coverage_rationale"]])
                display(Markdown("**Exec components with the largest hidden mass.**"))
                display(exec_hidden[["component", "coverage_class", "sample_count", "coverage_rationale"]])

                fig_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Setup.** These bars rank individual goal and exec tokens by how often they appear in the sample while
                still being hidden from the translated text. The x-axis is the number of sampled row occurrences, not the
                number of unique problems.

                **Interpretation.** Large bars indicate components that are systematically unavailable to a model trained
                only on `response_nl`. For example, if `get_GCD`, `sub_fact`, or `div_calculator` appear frequently here,
                then the symbolic traces contain those distinctions but the text target does not expose them clearly.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Stratified Hidden-State Burden

                **Method.** We aggregate symbolic components back to the row level and compute hidden-state burden
                \((h_i)\) by operation, denominator type, correctness, and strategy family.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                row_coverage = (
                    component_rows.assign(
                        explicit=lambda df: df["coverage_class"].eq("explicit_template").astype(int),
                        conditional=lambda df: df["coverage_class"].eq("conditional_template").astype(int),
                        bundled=lambda df: df["coverage_class"].eq("bundled_not_explicit").astype(int),
                        omitted=lambda df: df["coverage_class"].isin(["omitted", "omitted_or_generic"]).astype(int),
                        ignored=lambda df: df["coverage_class"].eq("intentionally_ignored").astype(int),
                    )
                    .groupby(
                        ["source_uid", "op_label", "denom_type", "is_correct", "strategy_family"],
                        as_index=False,
                    )
                    .agg(
                        total_components=("component", "size"),
                        explicit_n=("explicit", "sum"),
                        conditional_n=("conditional", "sum"),
                        bundled_n=("bundled", "sum"),
                        omitted_n=("omitted", "sum"),
                        ignored_n=("ignored", "sum"),
                    )
                )
                row_coverage["hidden_n"] = row_coverage["bundled_n"] + row_coverage["omitted_n"] + row_coverage["ignored_n"]
                row_coverage["hidden_share"] = row_coverage["hidden_n"] / row_coverage["total_components"]

                op_order = ["add", "sub", "mul", "div"]
                denom_order = ["ED", "UD"]

                heatmap_df = (
                    row_coverage.groupby(["is_correct", "denom_type", "op_label"], as_index=False)["hidden_share"]
                    .mean()
                )

                fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
                for ax, is_correct, title in [
                    (axes[0], False, "Mean Hidden Share | Incorrect Rows"),
                    (axes[1], True, "Mean Hidden Share | Correct Rows"),
                ]:
                    matrix = (
                        heatmap_df[heatmap_df["is_correct"] == is_correct]
                        .pivot(index="denom_type", columns="op_label", values="hidden_share")
                        .reindex(index=denom_order, columns=op_order)
                    )
                    sns.heatmap(
                        matrix,
                        ax=ax,
                        annot=True,
                        fmt=".2f",
                        cmap="YlOrRd",
                        vmin=0.0,
                        vmax=float(heatmap_df["hidden_share"].max()),
                        cbar=is_correct,
                        linewidths=0.5,
                        linecolor="white",
                    )
                    ax.set_title(title)
                    ax.set_xlabel("Operation")
                    ax.set_ylabel("Denominator type")
                fig_path = FIGURE_DIR / "hidden_share_heatmap.png"
                fig.savefig(fig_path, bbox_inches="tight")
                plt.show()

                stratified_hidden = (
                    row_coverage.groupby(["op_label", "denom_type", "is_correct", "strategy_family"], as_index=False)
                    .agg(
                        rows=("source_uid", "size"),
                        mean_hidden_n=("hidden_n", "mean"),
                        mean_hidden_share=("hidden_share", "mean"),
                        mean_explicit_n=("explicit_n", "mean"),
                        mean_conditional_n=("conditional_n", "mean"),
                    )
                    .sort_values(["mean_hidden_share", "rows"], ascending=[False, False])
                    .reset_index(drop=True)
                )

                display(Markdown("**Top strata by mean hidden-state share.**"))
                display(stratified_hidden.head(20))

                fig_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                **Setup.** For each sampled row \(i\), the hidden-share statistic is
                \[
                \frac{h_i}{|\Gamma_i|},
                \]
                where \(h_i\) is the number of bundled, omitted, or ignored components and \(|\Gamma_i|\) is the total
                number of symbolic components on that row. The heatmaps average this quantity within each `ED/UD × op`
                cell, separately for correct and incorrect rows.

                **Interpretation.** A darker cell means a larger fraction of the source symbolic state is hidden from
                the narrated text. If a stratum has a high hidden share, then weak NLP fit there is consistent with a
                representational bottleneck: the model is asked to imitate text that does not actually verbalize much of
                the underlying strategy/goals/exec structure.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Collision Analysis

                **Question.** Even if the translator is locally faithful, does it collapse distinct symbolic traces onto
                the same narrated text?
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                collision_strata = collision_outputs["stratified"].copy()
                collision_strata["stratum"] = (
                    collision_strata["denom_type"]
                    + " | "
                    + collision_strata["op_label"]
                    + " | "
                    + collision_strata["is_correct"].map({True: "correct", False: "incorrect"})
                    + " | "
                    + collision_strata["strategy_family"]
                )

                collision_plot = (
                    collision_strata[collision_strata["rows"] >= 20]
                    .sort_values(["collision_row_rate", "rows"], ascending=[False, False])
                    .head(12)
                    .iloc[::-1]
                )

                fig, ax = plt.subplots(figsize=(12.5, 6.5), constrained_layout=True)
                ax.barh(collision_plot["stratum"], collision_plot["collision_row_rate"], color="#4c78a8", height=0.7)
                ax.set_title("Highest Collision-Rate Strata")
                ax.set_xlabel("Collision row rate")
                ax.set_ylabel("")
                fig_path = FIGURE_DIR / "collision_strata.png"
                fig.savefig(fig_path, bbox_inches="tight")
                plt.show()

                display(Markdown("**Representative symbolic collisions.**"))
                display(
                    collision_outputs["top_collisions"][
                        ["prob", "answer", "n_rows", "n_unique_states", "strategies", "goals", "execs", "op_label", "denom_type", "response_nl"]
                    ].head(15)
                )

                fig_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                **Setup.** A collision means that two or more different symbolic states \((strategy, goals, exec)\)
                produce the same `(prob, answer, response_nl)` tuple. Each bar here is one stratum defined by denominator
                type, operation, correctness, and strategy family; the x-axis is the fraction of sampled rows in that
                stratum that belong to a collision group.

                **Interpretation.** High collision rate means the translation is not one-to-one even when the problem and
                final answer are fixed. In those strata, the narrated text is insufficient to recover the original UMA
                state, because distinct traces collapse to the same prose.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Manual Strategy Samples

                **Method.** We sample representative rows for each strategy code and inspect whether the strategy is
                actually visible in the narrated text.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                strategy_examples = (
                    component_rows[component_rows["component_type"] == "strategy"][
                        [
                            "strategy",
                            "coverage_class",
                            "coverage_rationale",
                            "prob",
                            "answer",
                            "goals",
                            "exec",
                            "response_nl",
                            "op_label",
                            "denom_type",
                            "is_correct",
                            "source_row_idx",
                        ]
                    ]
                    .drop_duplicates(subset=["strategy", "prob", "answer", "response_nl"])
                    .sort_values(["strategy", "source_row_idx"])
                    .groupby("strategy", dropna=False)
                    .head(2)
                    .reset_index(drop=True)
                )

                display(strategy_examples)

                display(Markdown("**ICDM non-division rows often collapse to generic narration.**"))
                icdm_generic = (
                    sample_df[(sample_df["strategy"] == "ICDM_OG") & (sample_df["op_label"] != "div")]
                    .sort_values("source_row_idx")
                    .head(8)[["prob", "answer", "strategy", "goals", "exec", "response_nl", "op_label", "denom_type", "is_correct"]]
                )
                display(icdm_generic)
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Manual Goal and Exec Samples

                **Finding.** The examples below show components that are common in the symbolic trace but weakly realized,
                bundled into coarse arithmetic language, or omitted entirely.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                goal_focus = [
                    "operate_nums",
                    "pass_den",
                    "convert_CD_LCM",
                    "get_LCM",
                    "convert_fra_to_den",
                    "get_GCD",
                ]
                goal_examples = (
                    component_rows[
                        (component_rows["component_type"] == "goal")
                        & (component_rows["component"].isin(goal_focus))
                    ][
                        [
                            "component",
                            "coverage_class",
                            "coverage_rationale",
                            "prob",
                            "answer",
                            "strategy",
                            "goals",
                            "exec",
                            "response_nl",
                            "op_label",
                            "denom_type",
                            "is_correct",
                            "source_row_idx",
                        ]
                    ]
                    .sort_values(["component", "source_row_idx"])
                    .groupby("component", dropna=False)
                    .head(2)
                    .reset_index(drop=True)
                )

                exec_focus = [
                    "sub_fact",
                    "div_calculator",
                    "div_to_mul_denied",
                    "acc_skip",
                    "acc_extra",
                ]
                exec_examples = (
                    component_rows[
                        (component_rows["component_type"] == "exec")
                        & (component_rows["component"].isin(exec_focus))
                    ][
                        [
                            "component",
                            "coverage_class",
                            "coverage_rationale",
                            "prob",
                            "answer",
                            "strategy",
                            "goals",
                            "exec",
                            "response_nl",
                            "op_label",
                            "denom_type",
                            "is_correct",
                            "source_row_idx",
                        ]
                    ]
                    .sort_values(["component", "source_row_idx"])
                    .groupby("component", dropna=False)
                    .head(2)
                    .reset_index(drop=True)
                )

                display(Markdown("**Goal-side examples.**"))
                display(goal_examples)
                display(Markdown("**Exec-side examples.**"))
                display(exec_examples)
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                ## Interpretation

                **Finding.** The template makes some strategy families easy to recover: `CDON`, `KDON`, `ONOD`, `CROP`,
                and division-specific `ICDM` cues surface directly in text. But even there, the narration often stops at
                the family level rather than preserving the full symbolic state.

                **Evidence.** Many goals and exec rules remain hidden. Common examples include `get_GCD`,
                `convert_CD_LCM`, `get_LCM`, `convert_fra_to_den`, `sub_fact`, `div_calculator`, and
                `div_to_mul_denied`. These are present in the source trace but are not stated in `response_nl`.

                **Implication.** If training only consumes `instruction_nl -> response_nl`, then the learner cannot
                recover symbolic information that never appears in the target text. Poor NLP fit can therefore reflect
                representational collapse in the translation layer, not only optimization failure.

                **Next step.** If we want the NLP target to be faithful to UMA traces, we should either enrich
                `response_nl` to verbalize the hidden components or change the training target to expose structured
                strategy, goals, and exec channels directly.
                """
            )
        )
    )

    nb["cells"] = cells
    nbf.write(nb, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
