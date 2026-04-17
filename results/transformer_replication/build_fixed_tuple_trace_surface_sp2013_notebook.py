#!/usr/bin/env python3
"""
Build and execute the fixed-tuple trace-surface SP2013 comparison notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    tr_dir = root / "results" / "transformer_replication"
    parser = argparse.ArgumentParser(description="Build and execute the fixed-tuple SP2013 comparison notebook.")
    parser.add_argument("--default_rollouts", type=Path, required=True)
    parser.add_argument("--surface_rollouts", type=Path, required=True)
    parser.add_argument("--uma_target_csv", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=tr_dir / "analyze_fixed_tuple_trace_surface_sp2013.ipynb",
    )
    parser.add_argument(
        "--figure_dir_name",
        type=str,
        default="fixed_tuple_trace_surface_sp2013",
    )
    parser.add_argument("--default_label", type=str, default="Humanlike Trace")
    parser.add_argument("--surface_label", type=str, default="Surface All Steps")
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
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                f"""
                # Fixed-Tuple Trace-Surface Comparison on SP2013

                **Question.** For the top SP2013-matching UMA model `subjid=472`, does training on more explicit translated traces improve the match between LLM answer distributions and same-tuple UMA answer distributions on held-out SP2013?

                **Method.** We compare two models trained on the same raw synthetic traces for `(g,d,rt_\\mu,ice)=(0.05,0.7,5,50)`:

                - **{args.default_label}**: default humanlike translation
                - **{args.surface_label}**: translation with hidden/internal steps surfaced

                Both models are evaluated with `1000` sampled rollouts per SP2013 problem, and we compare them to `1000` same-tuple UMA rollouts per problem.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                ## Formal Setup

                **Setup.** Let \(p \in \mathcal{P}\) index the 16 held-out SP2013 fraction problems, and let
                \[
                q_{\mathrm{UMA}}(a \mid p)
                \]
                denote the empirical answer distribution from `1000` same-tuple UMA rollouts for problem \(p\).
                For each trained language model variant \(m \in \{\text{humanlike}, \text{surface}\}\), let
                \[
                q_m(a \mid p)
                \]
                denote the empirical answer distribution from `1000` sampled generations for the same problem under the fixed student prefix.

                **Primary metric.** We measure total variation distance problem by problem:
                \[
                \mathrm{TV}(q_m, q_{\mathrm{UMA}} \mid p)
                = \frac{1}{2}\sum_a \left|q_m(a \mid p) - q_{\mathrm{UMA}}(a \mid p)\right|.
                \]
                We report the mean over all SP2013 problems,
                \[
                \overline{\mathrm{TV}}_m = \frac{1}{|\mathcal{P}|}\sum_{p \in \mathcal{P}} \mathrm{TV}(q_m, q_{\mathrm{UMA}} \mid p).
                \]

                **Secondary metrics.** We also report TV by operation, TV by ED/UD cell, modal-answer agreement with UMA, parseable coverage, and mathematically true-answer accuracy for context.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                from __future__ import annotations

                import math
                from fractions import Fraction
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
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 11,
                    "legend.fontsize": 11,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                })
                pd.set_option("display.max_columns", 24)
                pd.set_option("display.max_colwidth", 200)
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

                DEFAULT_ROLLOUTS = Path("{args.default_rollouts.as_posix()}")
                SURFACE_ROLLOUTS = Path("{args.surface_rollouts.as_posix()}")
                UMA_TARGET_CSV = Path("{args.uma_target_csv.as_posix()}")
                if not DEFAULT_ROLLOUTS.is_absolute():
                    DEFAULT_ROLLOUTS = PROJECT_ROOT / DEFAULT_ROLLOUTS
                if not SURFACE_ROLLOUTS.is_absolute():
                    SURFACE_ROLLOUTS = PROJECT_ROOT / SURFACE_ROLLOUTS
                if not UMA_TARGET_CSV.is_absolute():
                    UMA_TARGET_CSV = PROJECT_ROOT / UMA_TARGET_CSV

                DEFAULT_LABEL = "{args.default_label}"
                SURFACE_LABEL = "{args.surface_label}"

                default_df = pd.read_csv(DEFAULT_ROLLOUTS)
                surface_df = pd.read_csv(SURFACE_ROLLOUTS)
                uma_df = pd.read_csv(UMA_TARGET_CSV)

                len(default_df), len(surface_df), len(uma_df)
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                def parse_problem(prob: str):
                    text = str(prob).replace(" ", "").replace("÷", ":").replace("−", "-")
                    for op in ["+", "-", "*", ":"]:
                        if op in text:
                            left, right = text.split(op, 1)
                            return left, op, right
                    return "", "?", ""


                def fraction_den(token: str):
                    try:
                        frac = Fraction(token)
                    except Exception:
                        return None
                    return int(frac.denominator)


                def classify_problem(prob: str):
                    left, op, right = parse_problem(prob)
                    left_den = fraction_den(left)
                    right_den = fraction_den(right)
                    edud = "ED" if left_den == right_den else "UD"
                    op_label = "/" if op == ":" else op
                    return op_label, edud


                def build_distribution(series: pd.Series) -> dict[str, float]:
                    counts = series.fillna("?").astype(str).value_counts(dropna=False)
                    total = float(counts.sum())
                    return {str(key): float(value / total) for key, value in counts.items()}


                def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
                    support = set(p) | set(q)
                    return 0.5 * sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in support)


                def modal_answer(dist: dict[str, float]) -> str:
                    if not dist:
                        return "?"
                    items = sorted(dist.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
                    return str(items[0][0])


                def per_problem_metrics(run_df: pd.DataFrame, target_df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
                    records = []
                    target_rows = []
                    pred_rows = []
                    grouped_target = target_df.groupby("prob")
                    grouped_run = run_df.groupby("prob")
                    for prob, target_group in grouped_target:
                        pred_group = grouped_run.get_group(prob)
                        target_dist = build_distribution(target_group["ans"].astype(str))
                        pred_dist = build_distribution(pred_group["pred_answer_label"].astype(str))
                        op_label, edud = classify_problem(prob)
                        tv = total_variation(pred_dist, target_dist)
                        target_rows.append({"variant": label, "prob": prob, "distribution": target_dist})
                        pred_rows.append({"variant": label, "prob": prob, "distribution": pred_dist})
                        records.append(
                            {
                                "variant": label,
                                "prob": prob,
                                "op": op_label,
                                "edud": edud,
                                "tv_distance": float(tv),
                                "modal_answer": modal_answer(pred_dist),
                                "modal_target": modal_answer(target_dist),
                                "modal_agree": int(modal_answer(pred_dist) == modal_answer(target_dist)),
                                "parseable_coverage": float(pred_group["pred_answer_parseable"].mean()),
                                "true_acc_all": float(pred_group["is_correct_true"].mean()),
                            }
                        )
                    return pd.DataFrame(records), {
                        "target_distribution_rows": pd.DataFrame(target_rows),
                        "pred_distribution_rows": pd.DataFrame(pred_rows),
                    }


                default_problem_df, default_dist = per_problem_metrics(default_df, uma_df, DEFAULT_LABEL)
                surface_problem_df, surface_dist = per_problem_metrics(surface_df, uma_df, SURFACE_LABEL)
                problem_df = pd.concat([default_problem_df, surface_problem_df], ignore_index=True)
                problem_df.head()
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Summary Table

                **Finding.** We start with compact aggregate metrics so we can compare the two trace styles before drilling into problem-level structure.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                summary_rows = []
                for label, run_df in [(DEFAULT_LABEL, default_df), (SURFACE_LABEL, surface_df)]:
                    sub = problem_df[problem_df["variant"] == label]
                    summary_rows.append(
                        {
                            "variant": label,
                            "tv_mean": float(sub["tv_distance"].mean()),
                            "tv_std": float(sub["tv_distance"].std(ddof=0)),
                            "modal_agreement_mean": float(sub["modal_agree"].mean()),
                            "parseable_coverage": float(run_df["pred_answer_parseable"].mean()),
                            "true_acc_all": float(run_df["is_correct_true"].mean()),
                            "n_rollouts": int(len(run_df)),
                            "n_problems": int(sub["prob"].nunique()),
                        }
                    )

                summary_df = pd.DataFrame(summary_rows).sort_values("tv_mean")
                summary_df
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                summary_plot = summary_df.set_index("variant")[["tv_mean", "modal_agreement_mean", "true_acc_all"]]

                fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
                columns = ["tv_mean", "modal_agreement_mean", "true_acc_all"]
                titles = ["Mean TV", "Modal Agreement", "True Accuracy"]
                for ax, column, title in zip(axes, columns, titles):
                    plot_df = summary_df.copy()
                    sns.barplot(data=plot_df, x="variant", y=column, ax=ax, palette="Set2")
                    ax.set_title(title)
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", rotation=15)
                    if column == "tv_mean":
                        ax.set_ylabel("Distance")
                    else:
                        ax.set_ylabel("Rate")

                plot_path = FIGURE_DIR / "summary_bars.png"
                fig.savefig(plot_path, bbox_inches="tight")
                plt.show()
                plot_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Interpretation.** Lower TV is better because it means the LLM answer distribution is closer to same-tuple UMA on average across the 16 SP2013 problems. Higher modal agreement and higher true-answer accuracy are helpful context, but the primary target here is the distributional match to UMA.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Problem-Level TV Heatmap

                **Question.** Are the gains broad across problems, or concentrated in a few SP2013 items?
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                order = (
                    problem_df.groupby("prob", as_index=False)["tv_distance"]
                    .mean()
                    .sort_values(["tv_distance", "prob"], ascending=[False, True])["prob"]
                    .tolist()
                )
                tv_pivot = problem_df.pivot(index="variant", columns="prob", values="tv_distance")[order]

                fig, ax = plt.subplots(figsize=(18, 4.8), constrained_layout=True)
                sns.heatmap(tv_pivot, cmap="mako_r", vmin=0.0, vmax=max(0.25, float(tv_pivot.max().max())), ax=ax)
                ax.set_title("Per-Problem Total Variation Distance")
                ax.set_xlabel("SP2013 problem")
                ax.set_ylabel("")
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

                plot_path = FIGURE_DIR / "per_problem_tv_heatmap.png"
                fig.savefig(plot_path, bbox_inches="tight")
                plt.show()
                plot_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Interpretation.** Each cell is one SP2013 problem. Darker cells indicate larger total variation distance and therefore a poorer match to the UMA answer distribution for that item. This plot is useful for seeing whether one trace style helps broadly or only on a narrow subset of problems.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## TV by Operation and ED/UD Cell

                **Question.** Where do the distributional differences concentrate structurally: by arithmetic operation, or by equal versus unequal denominators?
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                cell_summary = (
                    problem_df.groupby(["variant", "edud", "op"], as_index=False)["tv_distance"]
                    .mean()
                    .sort_values(["variant", "edud", "op"])
                )
                cell_summary
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                variants = [DEFAULT_LABEL, SURFACE_LABEL]
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
                vmax = max(0.25, float(cell_summary["tv_distance"].max()))
                for ax, label in zip(axes, variants):
                    mat = (
                        cell_summary[cell_summary["variant"] == label]
                        .pivot(index="edud", columns="op", values="tv_distance")
                        .reindex(index=["ED", "UD"], columns=["+", "-", "*", "/"])
                    )
                    sns.heatmap(mat, cmap="mako_r", vmin=0.0, vmax=vmax, annot=True, fmt=".3f", ax=ax)
                    ax.set_title(label)
                    ax.set_xlabel("Operation")
                    ax.set_ylabel("")

                plot_path = FIGURE_DIR / "tv_by_cell_heatmap.png"
                fig.savefig(plot_path, bbox_inches="tight")
                plt.show()
                plot_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Interpretation.** This heatmap aggregates the 16 problems into ED/UD by operation cells. It helps separate broad pattern changes from isolated item effects, especially when one trace style changes the match mainly for UD problems or for a specific operation family.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## ED/UD Accuracy by Operation

                **Question.** How do the mathematically correct answer rates look across the standard ED/UD by operation view, in the same combined profile style we used for the LR-ablation analysis?
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                op_order = ["+", "-", "*", "/"]
                op_display = {"+": "Add", "-": "Sub", "*": "Mul", "/": "Div"}

                def attach_problem_structure(df: pd.DataFrame) -> pd.DataFrame:
                    enriched = df.copy()
                    structure = enriched["prob"].map(classify_problem)
                    enriched["op_label"] = [item[0] for item in structure]
                    enriched["edud"] = [item[1] for item in structure]
                    return enriched


                default_acc_df = attach_problem_structure(default_df)
                surface_acc_df = attach_problem_structure(surface_df)
                uma_acc_df = attach_problem_structure(uma_df)
                human_ref_df = pd.read_csv(PROJECT_ROOT / "sp2013_human.csv")

                accuracy_rows = []
                for label, run_df, acc_col in [
                    (DEFAULT_LABEL, default_acc_df, "is_correct_true"),
                    (SURFACE_LABEL, surface_acc_df, "is_correct_true"),
                    ("UMA Target", uma_acc_df, "acc"),
                ]:
                    grouped = (
                        run_df.groupby(["edud", "op_label"], as_index=False)[acc_col]
                        .mean()
                        .rename(columns={acc_col: "accuracy"})
                    )
                    grouped["variant"] = label
                    accuracy_rows.append(grouped)

                accuracy_cell_df = pd.concat(accuracy_rows, ignore_index=True)
                accuracy_cell_df["op_label"] = pd.Categorical(
                    accuracy_cell_df["op_label"], categories=op_order, ordered=True
                )
                accuracy_cell_df = accuracy_cell_df.sort_values(["variant", "edud", "op_label"]).reset_index(drop=True)

                human_cell_df = (
                    human_ref_df.groupby(["operands", "operation"], as_index=False)["acc"]
                    .mean()
                    .rename(columns={"operands": "edud", "operation": "operation_name", "acc": "accuracy"})
                )
                human_cell_df["op_label"] = human_cell_df["operation_name"].map(
                    {"add": "+", "sub": "-", "mul": "*", "div": "/"}
                )
                human_cell_df["variant"] = "Human"

                accuracy_cell_df
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                profile_grid = [
                    ("ED", "+"), ("UD", "+"),
                    ("ED", "-"), ("UD", "-"),
                    ("ED", "*"), ("UD", "*"),
                    ("ED", "/"), ("UD", "/"),
                ]
                profile_ticklabels = [
                    "add\\nED", "add\\nUD",
                    "sub\\nED", "sub\\nUD",
                    "mul\\nED", "mul\\nUD",
                    "div\\nED", "div\\nUD",
                ]

                def profile_curve(df: pd.DataFrame, label: str) -> np.ndarray:
                    lookup = (
                        df[df["variant"] == label]
                        .set_index(["edud", "op_label"])["accuracy"]
                    )
                    return np.array([100.0 * float(lookup.loc[(edud, op)]) for edud, op in profile_grid])


                human_curve = profile_curve(human_cell_df, "Human")
                uma_curve = profile_curve(accuracy_cell_df, "UMA Target")
                default_curve = profile_curve(accuracy_cell_df, DEFAULT_LABEL)
                surface_curve = profile_curve(accuracy_cell_df, SURFACE_LABEL)

                fig, ax = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
                x = np.arange(len(profile_grid))
                series = [
                    ("Human", human_curve, "#111111", "o", "-", 2.0, 5.0),
                    ("UMA", uma_curve, "#e76f51", "D", "-", 2.0, 4.8),
                    (DEFAULT_LABEL, default_curve, "#3a86ff", "s", "-", 2.2, 5.0),
                    (SURFACE_LABEL, surface_curve, "#2a9d8f", "^", "-", 2.2, 5.0),
                ]
                for label, values, color, marker, linestyle, linewidth, markersize in series:
                    ax.plot(
                        x,
                        values,
                        label=label,
                        color=color,
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        markersize=markersize,
                    )

                ax.set_title("Fixed-Tuple Trace-Surface: SP2013 True-Answer Profile by Cell")
                ax.set_xlabel("SP2013 cell")
                ax.set_ylabel("Accuracy (%)")
                ax.set_xticks(x)
                ax.set_xticklabels(profile_ticklabels)
                ax.set_ylim(0.0, 100.0)
                ax.legend(loc="upper right", frameon=True)

                plot_path = FIGURE_DIR / "accuracy_profile_combined.png"
                fig.savefig(plot_path, bbox_inches="tight")
                plt.show()
                plot_path
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                **Interpretation.** This figure matches the compact SP2013 cell-profile view from the LR-ablation notebook. The black line is the human reference profile, the orange line is the same-tuple UMA target, and the two colored lines are the final fixed-tuple LLM runs. Because this experiment only saved post-hoc SP2013 rollouts for the final checkpoint, this notebook shows the final profile rather than a per-epoch grid.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Modal Answers and Reproducibility Check

                **Setup.** We finish with the modal-answer comparison and a direct consistency check that the summary table is reproduced from the saved rollout CSVs rather than a separate hidden artifact.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                modal_table = (
                    problem_df.loc[:, ["variant", "prob", "modal_answer", "modal_target", "modal_agree"]]
                    .sort_values(["prob", "variant"])
                    .reset_index(drop=True)
                )
                modal_table.head(12)
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                reproduced = []
                for label, run_df in [(DEFAULT_LABEL, default_df), (SURFACE_LABEL, surface_df)]:
                    sub = problem_df[problem_df["variant"] == label]
                    reproduced.append(
                        {
                            "variant": label,
                            "tv_mean": float(sub["tv_distance"].mean()),
                            "modal_agreement_mean": float(sub["modal_agree"].mean()),
                            "parseable_coverage": float(run_df["pred_answer_parseable"].mean()),
                            "true_acc_all": float(run_df["is_correct_true"].mean()),
                        }
                    )
                reproduced_df = pd.DataFrame(reproduced).sort_values("variant").reset_index(drop=True)
                summary_check = (
                    summary_df.loc[:, ["variant", "tv_mean", "modal_agreement_mean", "parseable_coverage", "true_acc_all"]]
                    .sort_values("variant")
                    .reset_index(drop=True)
                )
                pd.testing.assert_frame_equal(
                    summary_check.round(12),
                    reproduced_df.round(12),
                    check_dtype=False,
                )
                Markdown("Reproduction check passed: the aggregate summary matches the saved rollout CSVs exactly.")
                """
            )
        )
    )

    nb["cells"] = cells
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

    client = NotebookClient(nb, timeout=1200, kernel_name="python3")
    client.execute()
    with out.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Wrote executed notebook -> {out}")


if __name__ == "__main__":
    main()
