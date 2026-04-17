#!/usr/bin/env python3
"""
Build and execute a notebook that analyzes the 135M panel-25 runs.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results" / "transformer_replication" / "analyze_panel25_135m_comparison.ipynb"

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

    cells = [
        md(
            r"""
            # Panel-25 Runs: 135M Default vs Surface, Pretrained vs Scratch

            **Question.** How do the four `SmolLM2-135M` panel-25 runs differ when we vary the prompt representation
            (`default` vs `surface`) and initialization (`pretrained` vs `scratch`) while holding the optimizer setup
            fixed?

            **Setup.** The notebook reads the latest on-disk runs matching
            `smol2_135m_panel25_even_g_rt5_ice50_seed1_{default,surface}_{pretrained,scratch}_e*_TIMESTAMP`
            under `results/transformer_replication/`.

            **Design.** This is a \(2 \times 2\) comparison over
            \[
            v \in \{\mathrm{default}, \mathrm{surface}\}, \qquad
            i \in \{\mathrm{pretrained}, \mathrm{scratch}\}.
            \]
            All four runs use the same optimizer family, learning rate, scheduler, batch size, and panel-25 synthetic
            training source. The factors that vary are the representation of the student state in the prompt and whether
            training starts from the pretrained checkpoint or from scratch.
            """
        ),
        md(
            r"""
            ## Formal Setup

            **Objects.** Let \(r = (v, i)\) index a run, let \(t\) denote an evaluation point along training, and let
            \(N = 64\) denote the size of the saved `id_val` slice used in the on-disk row-level evaluation files.

            We track four quantities:
            \[
            L_r^{\mathrm{train}}(t), \qquad L_r^{\mathrm{val}}(t),
            \]
            \[
            A_r^{\mathrm{resp,scored}}(t)
            = \frac{c_r^{\mathrm{resp}}(t)}{n_r^{\mathrm{parseable}}(t)},
            \qquad
            C_r(t)
            = \frac{n_r^{\mathrm{parseable}}(t)}{N},
            \]
            \[
            A_r^{\mathrm{resp,all}}(t)
            = \frac{c_r^{\mathrm{resp}}(t)}{N}
            = C_r(t)\, A_r^{\mathrm{resp,scored}}(t),
            \qquad
            A_r^{\mathrm{true}}(t)
            = \frac{c_r^{\mathrm{true}}(t)}{N}.
            \]

            Here \(c_r^{\mathrm{resp}}(t)\) counts agreement with the UMA response target, \(c_r^{\mathrm{true}}(t)\)
            counts agreement with the mathematically correct answer, and
            \(n_r^{\mathrm{parseable}}(t)\) counts how many generations produce a parseable predicted answer.

            **Panel-25 grid.** Each evaluation prompt contains a student tuple
            \((g, d, rt, ice)\). For this family, \(rt=5\) and \(ice=50\) are fixed, while \(g\) and \(d\) vary over a
            \(5 \times 5\) panel. For a given run \(r\), student cell \((g,d)\), and metric \(m \in \{\mathrm{resp}, \mathrm{true}\}\),
            the notebook reports
            \[
            A_{r,g,d}^{m}
            = \frac{1}{N_{g,d}} \sum_{p \in \mathcal{P}_{g,d}} \mathbf{1}\{\hat{y}_r(p) = y^{m}(p)\},
            \]
            where \(\mathcal{P}_{g,d}\) is the subset of the saved `id_val` problems whose prompt contains that
            student cell.

            **Interpretation.** Because \(N=64\) is small and coverage is below \(100\%\), we separate scored accuracy
            from all-example accuracy and display counts in every subgroup heatmap.
            """
        ),
        code(
            """
            from __future__ import annotations

            import json
            import re
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import Markdown, display
            from matplotlib.lines import Line2D
            from matplotlib.ticker import FuncFormatter

            plt.rcParams.update(
                {
                    "figure.dpi": 160,
                    "savefig.dpi": 300,
                    "font.size": 12,
                    "axes.titlesize": 16,
                    "axes.labelsize": 13,
                    "legend.fontsize": 10,
                    "xtick.labelsize": 11,
                    "ytick.labelsize": 11,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                }
            )

            RUN_RE = re.compile(
                r"^smol2_135m_panel25_even_g_rt5_ice50_seed1_"
                r"(?P<variant>default|surface)_"
                r"(?P<init>pretrained|scratch)_"
                r"e(?P<epochs>\\d+)_"
                r"(?P<timestamp>\\d{8}_\\d{6})$"
            )
            PROMPT_RE = re.compile(
                r"<student> g (?P<g>[0-9.]+) d (?P<d>[0-9.]+) rt (?P<rt>[0-9.]+) ice (?P<ice>[0-9.]+) </student>"
            )
            PROB_RE = re.compile(r"^(\\d+)/(\\d+)([+\\-*:])(\\d+)/(\\d+)$")

            VARIANT_ORDER = ["default", "surface"]
            INIT_ORDER = ["pretrained", "scratch"]
            DENOM_ORDER = ["ED", "UD"]
            OP_ORDER = ["add", "sub", "mul", "div"]
            G_LEVELS = [0.02, 0.04, 0.06, 0.08, 0.10]
            D_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
            SAVE_FIGURE = True

            STYLE_MAP = {
                ("default", "pretrained"): {
                    "label": "Default | Pretrained",
                    "color": "#1f77b4",
                    "marker": "o",
                    "linestyle": "-",
                },
                ("surface", "pretrained"): {
                    "label": "Surface | Pretrained",
                    "color": "#ff7f0e",
                    "marker": "s",
                    "linestyle": "-",
                },
                ("default", "scratch"): {
                    "label": "Default | Scratch",
                    "color": "#2ca02c",
                    "marker": "^",
                    "linestyle": "--",
                },
                ("surface", "scratch"): {
                    "label": "Surface | Scratch",
                    "color": "#d62728",
                    "marker": "D",
                    "linestyle": "--",
                },
            }


            def resolve_project_root() -> Path:
                candidates = [Path.cwd(), *Path.cwd().parents]
                for candidate in candidates:
                    if (candidate / "results" / "transformer_replication").exists():
                        return candidate
                raise FileNotFoundError("Could not resolve project root.")


            def load_json(path: Path):
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)


            def parse_prob(prob: str) -> tuple[str, str]:
                match = PROB_RE.match(str(prob).replace(" ", ""))
                if match is None:
                    raise ValueError(f"Could not parse probability string: {prob}")
                left_den = int(match.group(2))
                right_den = int(match.group(5))
                denom = "ED" if left_den == right_den else "UD"
                operation = {"+": "add", "-": "sub", "*": "mul", ":": "div"}[match.group(3)]
                return denom, operation


            def parse_prompt_tuple(prompt: str) -> pd.Series:
                match = PROMPT_RE.search(str(prompt))
                if match is None:
                    return pd.Series({"g": np.nan, "d": np.nan, "rt": np.nan, "ice": np.nan})
                return pd.Series(
                    {
                        "g": float(match.group("g")),
                        "d": float(match.group("d")),
                        "rt": float(match.group("rt")),
                        "ice": float(match.group("ice")),
                    }
                )


            def find_panel25_runs(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
                records = []
                for path in sorted(run_dir.glob("smol2_135m_panel25_even_g_rt5_ice50_seed1_*")):
                    if not path.is_dir():
                        continue
                    match = RUN_RE.fullmatch(path.name)
                    if match is None:
                        continue
                    records.append(
                        {
                            "variant": match.group("variant"),
                            "init": match.group("init"),
                            "epochs": int(match.group("epochs")),
                            "timestamp": match.group("timestamp"),
                            "run_dir": str(path),
                        }
                    )
                catalog = pd.DataFrame(records)
                if catalog.empty:
                    return catalog, catalog
                selected = (
                    catalog.sort_values(["variant", "init", "timestamp"])
                    .groupby(["variant", "init"], as_index=False)
                    .tail(1)
                    .sort_values(
                        ["init", "variant"],
                        key=lambda s: s.map({name: idx for idx, name in enumerate(INIT_ORDER + VARIANT_ORDER)}),
                    )
                    .reset_index(drop=True)
                )
                return catalog, selected


            def style_for(variant: str, init: str) -> dict[str, str]:
                return STYLE_MAP[(variant, init)]


            def make_label(variant: str, init: str) -> str:
                return style_for(variant, init)["label"]


            def format_epoch_progress(value, _):
                if abs(value - round(value)) < 1e-8:
                    return f"{int(round(value))}"
                return f"{value:.1f}"


            def metric_delta_table(
                frame: pd.DataFrame,
                group_col: str,
                compare_col: str,
                left_name: str,
                right_name: str,
                metrics: list[str],
            ) -> pd.DataFrame:
                left = frame.loc[frame[compare_col] == left_name, [group_col, *metrics]].set_index(group_col)
                right = frame.loc[frame[compare_col] == right_name, [group_col, *metrics]].set_index(group_col)
                merged = left.join(right, lsuffix=f"_{left_name}", rsuffix=f"_{right_name}", how="inner")
                rows = []
                for group_value, row in merged.iterrows():
                    out = {group_col: group_value}
                    for metric in metrics:
                        left_value = row[f"{metric}_{left_name}"]
                        right_value = row[f"{metric}_{right_name}"]
                        out[f"{left_name}_{metric}"] = left_value
                        out[f"{right_name}_{metric}"] = right_value
                        out[f"delta_{right_name}_minus_{left_name}_{metric}"] = right_value - left_value
                    rows.append(out)
                return pd.DataFrame(rows)


            PROJECT_ROOT = resolve_project_root()
            RUN_DIR = PROJECT_ROOT / "results" / "transformer_replication"
            FIGURE_DIR = RUN_DIR / "figures" / "panel25_135m"
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)

            catalog_df, selected_df = find_panel25_runs(RUN_DIR)
            if selected_df.empty:
                raise FileNotFoundError("No panel-25 135M run directories were found.")

            display(Markdown(f"Detected **{len(catalog_df)}** candidate panel-25 directories and selected the latest run for each condition."))
            display(selected_df[["variant", "init", "epochs", "timestamp", "run_dir"]])
            """
        ),
        code(
            """
            summary_rows = []
            history_rows = []
            profile_rows = []
            gd_rows = []
            disagreement_rows = []
            example_frames = {}

            for record in selected_df.to_dict("records"):
                variant = record["variant"]
                init = record["init"]
                label = make_label(variant, init)
                style = style_for(variant, init)
                out_dir = Path(record["run_dir"])

                best_summary = load_json(out_dir / "best_checkpoint_summary.json")
                summary_metrics = load_json(out_dir / "summary_metrics.json")
                best_metrics = load_json(out_dir / "best_id_val_metrics.json")
                train_args = load_json(out_dir / "best" / "train_args.json")
                history = load_json(out_dir / "last" / "history.json")

                updates_per_epoch = max(
                    int(row["update_in_epoch"])
                    for row in history
                    if str(row.get("eval_scope", "full")) == "full"
                )

                for row in history:
                    epoch = int(row["epoch"])
                    update_in_epoch = int(row["update_in_epoch"])
                    epoch_progress = (epoch - 1) + update_in_epoch / updates_per_epoch
                    scored_response_pct = 100 * float(row.get("id_val_acc", np.nan))
                    coverage_pct = 100 * float(row.get("id_val_coverage", np.nan))
                    response_all_pct = scored_response_pct * coverage_pct / 100 if np.isfinite(scored_response_pct) and np.isfinite(coverage_pct) else np.nan
                    true_acc_pct = 100 * float(row.get("id_val_true_acc", np.nan))
                    history_rows.append(
                        {
                            "label": label,
                            "variant": variant,
                            "init": init,
                            "epoch": epoch,
                            "eval_tag": row.get("eval_tag"),
                            "update_in_epoch": update_in_epoch,
                            "updates_done": int(row["updates_done"]),
                            "epoch_progress": epoch_progress,
                            "train_loss": float(row["train_loss"]),
                            "val_loss": float(row["val_loss"]),
                            "test_loss": float(row["test_loss"]),
                            "scored_response_acc_pct": scored_response_pct,
                            "coverage_pct": coverage_pct,
                            "all_response_acc_pct": response_all_pct,
                            "true_acc_pct": true_acc_pct,
                        }
                    )

                best_df = pd.read_csv(out_dir / "best_id_val.csv").copy()
                best_df[["g", "d", "rt", "ice"]] = best_df["prompt"].apply(parse_prompt_tuple)
                best_df[["denom", "operation"]] = best_df["prob"].apply(lambda prob: pd.Series(parse_prob(prob)))
                best_df["label"] = label
                best_df["variant"] = variant
                best_df["init"] = init
                example_frames[label] = best_df.copy()

                profile_df = (
                    best_df.groupby(["denom", "operation"], as_index=False)
                    .agg(
                        n=("prob", "size"),
                        response_correct_n=("is_correct_response", "sum"),
                        true_correct_n=("is_correct_true", "sum"),
                        response_acc=("is_correct_response", "mean"),
                        true_acc=("is_correct_true", "mean"),
                    )
                )
                for row in profile_df.to_dict("records"):
                    profile_rows.append(
                        {
                            "label": label,
                            "variant": variant,
                            "init": init,
                            "denom": row["denom"],
                            "operation": row["operation"],
                            "n": int(row["n"]),
                            "response_correct_n": int(row["response_correct_n"]),
                            "true_correct_n": int(row["true_correct_n"]),
                            "response_acc_pct": 100 * float(row["response_acc"]),
                            "true_acc_pct": 100 * float(row["true_acc"]),
                        }
                    )

                gd_df = (
                    best_df.groupby(["g", "d"], as_index=False)
                    .agg(
                        n=("prob", "size"),
                        response_correct_n=("is_correct_response", "sum"),
                        true_correct_n=("is_correct_true", "sum"),
                        response_acc=("is_correct_response", "mean"),
                        true_acc=("is_correct_true", "mean"),
                    )
                )
                for row in gd_df.to_dict("records"):
                    gd_rows.append(
                        {
                            "label": label,
                            "variant": variant,
                            "init": init,
                            "g": float(row["g"]),
                            "d": float(row["d"]),
                            "n": int(row["n"]),
                            "response_correct_n": int(row["response_correct_n"]),
                            "true_correct_n": int(row["true_correct_n"]),
                            "response_acc_pct": 100 * float(row["response_acc"]),
                            "true_acc_pct": 100 * float(row["true_acc"]),
                        }
                    )

                summary_rows.append(
                    {
                        "label": label,
                        "variant": variant,
                        "init": init,
                        "timestamp": record["timestamp"],
                        "epochs": int(record["epochs"]),
                        "run_dir": str(out_dir),
                        "data_csv": train_args["data_csv"],
                        "lr": float(train_args["lr"]),
                        "scheduler": train_args["lr_scheduler_type"],
                        "max_length": int(train_args["max_length"]),
                        "best_epoch": int(summary_metrics["best_epoch"]),
                        "best_updates_done": int(summary_metrics["best_updates_done"]),
                        "best_train_loss": float(best_summary["train_loss"]),
                        "best_val_loss": float(summary_metrics["best_val_loss"]),
                        "best_test_loss": float(best_summary["test_loss"]),
                        "scored_response_acc_pct": 100 * float(best_metrics["acc_scored"]),
                        "all_response_acc_pct": 100 * float(best_metrics["acc_all"]),
                        "coverage_pct": 100 * float(best_metrics["coverage"]),
                        "true_acc_pct": 100 * float(best_metrics["true_acc_all"]),
                        "n_total": int(best_metrics["n_total"]),
                        "n_scored": int(best_metrics["n_scored"]),
                        "response_correct_n": int(best_metrics["n_correct"]),
                        "true_correct_n": int(best_metrics["true_n_correct"]),
                    }
                )

            summary_df = pd.DataFrame(summary_rows).sort_values(
                ["init", "variant"],
                key=lambda s: s.map(
                    {
                        "pretrained": 0,
                        "scratch": 1,
                        "default": 0,
                        "surface": 1,
                    }
                ),
            ).reset_index(drop=True)
            history_df = pd.DataFrame(history_rows).sort_values(["label", "updates_done"]).reset_index(drop=True)
            profile_df = pd.DataFrame(profile_rows).sort_values(["label", "denom", "operation"]).reset_index(drop=True)
            gd_df = pd.DataFrame(gd_rows).sort_values(["label", "g", "d"]).reset_index(drop=True)

            display(
                Markdown(
                    f"### Common Training Setup\\n"
                    f"All four runs use `lr = {summary_df['lr'].iloc[0]:.0e}`, "
                    f"`scheduler = {summary_df['scheduler'].iloc[0]}`, and "
                    f"`epochs = {summary_df['epochs'].iloc[0]}`."
                )
            )

            summary_view = summary_df[
                [
                    "label",
                    "best_val_loss",
                    "best_test_loss",
                    "scored_response_acc_pct",
                    "all_response_acc_pct",
                    "true_acc_pct",
                    "coverage_pct",
                    "response_correct_n",
                    "true_correct_n",
                    "n_scored",
                    "n_total",
                ]
            ].copy()
            display(Markdown("### Best-Checkpoint Summary"))
            display(summary_view.round(3))

            metrics_for_delta = [
                "best_val_loss",
                "all_response_acc_pct",
                "true_acc_pct",
                "coverage_pct",
            ]
            surface_effect_df = metric_delta_table(
                summary_df,
                group_col="init",
                compare_col="variant",
                left_name="default",
                right_name="surface",
                metrics=metrics_for_delta,
            )
            init_effect_df = metric_delta_table(
                summary_df,
                group_col="variant",
                compare_col="init",
                left_name="pretrained",
                right_name="scratch",
                metrics=metrics_for_delta,
            )

            display(Markdown("### Surface Effect Within Each Initialization (`surface - default`)"))
            display(surface_effect_df.round(3))
            display(Markdown("### Scratch Effect Within Each Prompt Variant (`scratch - pretrained`)"))
            display(init_effect_df.round(3))

            best_loss_row = summary_df.loc[summary_df["best_val_loss"].idxmin()]
            best_resp_row = summary_df.loc[summary_df["all_response_acc_pct"].idxmax()]
            best_true_row = summary_df.loc[summary_df["true_acc_pct"].idxmax()]
            display(
                Markdown(
                    f"Lowest validation loss: **{best_loss_row['label']}** "
                    f"(`{best_loss_row['best_val_loss']:.4f}`). "
                    f"Highest all-example response accuracy: **{best_resp_row['label']}** "
                    f"(`{best_resp_row['all_response_acc_pct']:.2f}%`). "
                    f"Highest true-answer accuracy: **{best_true_row['label']}** "
                    f"(`{best_true_row['true_acc_pct']:.2f}%`)."
                )
            )

            profile_view = (
                profile_df.pivot_table(
                    index=["denom", "operation", "n"],
                    columns="label",
                    values=["response_acc_pct", "true_acc_pct"],
                )
                .sort_index()
            )
            display(Markdown("### Best-Checkpoint Accuracy by `ED/UD x operation`"))
            display(profile_view.round(2))
            """
        ),
        code(
            """
            ordered_labels = [make_label(variant, init) for init in INIT_ORDER for variant in VARIANT_ORDER]
            cell_order = [(denom, operation) for denom in DENOM_ORDER for operation in OP_ORDER]
            cell_ticklabels = [f"{denom}\\n{operation}" for denom, operation in cell_order]
            cell_counts = (
                profile_df.loc[profile_df["label"] == ordered_labels[0], ["denom", "operation", "n"]]
                .set_index(["denom", "operation"])
                .reindex(cell_order)["n"]
                .tolist()
            )

            legend_handles = []
            for init in INIT_ORDER:
                for variant in VARIANT_ORDER:
                    style = style_for(variant, init)
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=style["color"],
                            linestyle=style["linestyle"],
                            marker=style["marker"],
                            linewidth=2.1,
                            markersize=6,
                            label=style["label"],
                        )
                    )

            fig, axes = plt.subplots(2, 3, figsize=(18.8, 10.2), constrained_layout=True)
            axes = axes.ravel()
            metric_specs = [
                ("train_loss", "Train Loss", "Loss"),
                ("val_loss", "Validation Loss", "Loss"),
                ("coverage_pct", "Parseable Coverage", "Coverage (%)"),
                ("scored_response_acc_pct", "Response Accuracy on Parseable Outputs", "Accuracy (%)"),
                ("all_response_acc_pct", "Response Accuracy on All 64 Examples", "Accuracy (%)"),
                ("true_acc_pct", "True-Answer Accuracy on All 64 Examples", "Accuracy (%)"),
            ]

            for ax, (metric, title, ylabel) in zip(axes, metric_specs):
                for label in ordered_labels:
                    run_history = history_df.loc[history_df["label"] == label]
                    if run_history.empty:
                        continue
                    variant = run_history["variant"].iloc[0]
                    init = run_history["init"].iloc[0]
                    style = style_for(variant, init)
                    ax.plot(
                        run_history["epoch_progress"],
                        run_history[metric],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        linewidth=2.0,
                        markersize=5.0,
                        alpha=0.95,
                    )
                ax.set_title(title)
                ax.set_xlabel("Epoch Progress")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25, linewidth=0.8)
                ax.xaxis.set_major_formatter(FuncFormatter(format_epoch_progress))
                ax.set_xticks(sorted(history_df["epoch_progress"].unique()))
                if "Accuracy" in ylabel or "Coverage" in ylabel:
                    ax.set_ylim(0, 100)

            axes[-1].legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                title="Run",
            )

            fig.suptitle("Panel-25 135M Runs: Optimization, Coverage, and Accuracy Trajectories", fontsize=18)

            if SAVE_FIGURE:
                for suffix in ["png", "pdf"]:
                    fig.savefig(FIGURE_DIR / f"panel25_135m_training_trajectories.{suffix}", bbox_inches="tight")

            plt.show()

            fig_cells, axes_cells = plt.subplots(2, 1, figsize=(14.8, 8.8), sharex=True, constrained_layout=True)
            bar_width = 0.18
            x = np.arange(len(cell_order))

            for idx, label in enumerate(ordered_labels):
                run_profile = (
                    profile_df.loc[profile_df["label"] == label, ["denom", "operation", "response_acc_pct", "true_acc_pct"]]
                    .set_index(["denom", "operation"])
                    .reindex(cell_order)
                )
                variant = profile_df.loc[profile_df["label"] == label, "variant"].iloc[0]
                init = profile_df.loc[profile_df["label"] == label, "init"].iloc[0]
                style = style_for(variant, init)
                offset = (idx - (len(ordered_labels) - 1) / 2) * bar_width

                axes_cells[0].bar(
                    x + offset,
                    run_profile["response_acc_pct"],
                    width=bar_width,
                    color=style["color"],
                    alpha=0.92,
                    label=style["label"],
                )
                axes_cells[1].bar(
                    x + offset,
                    run_profile["true_acc_pct"],
                    width=bar_width,
                    color=style["color"],
                    alpha=0.92,
                    label=style["label"],
                )

            for ax, title in zip(
                axes_cells,
                ["Accuracy Against UMA Response Target", "Accuracy Against True Answer"],
            ):
                ax.set_title(title)
                ax.set_ylabel("Accuracy (%)")
                ax.set_ylim(0, 100)
                ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
                ax.axvline(3.5, color="#777777", linewidth=1.0, alpha=0.6)

            axes_cells[1].set_xticks(x)
            axes_cells[1].set_xticklabels(cell_ticklabels)
            axes_cells[1].set_xlabel("SP2013 cell")

            for x_pos, count in zip(x, cell_counts):
                axes_cells[1].text(
                    x_pos,
                    -11,
                    f"n={int(count)}",
                    ha="center",
                    va="top",
                    fontsize=9,
                    clip_on=False,
                )

            axes_cells[0].legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                title="Run",
            )

            fig_cells.suptitle("Panel-25 135M Runs: ED/UD x add/sub/mul/div Profile at the Best Checkpoint", fontsize=18)

            if SAVE_FIGURE:
                for suffix in ["png", "pdf"]:
                    fig_cells.savefig(FIGURE_DIR / f"panel25_135m_edud_operation_profiles.{suffix}", bbox_inches="tight")

            plt.show()
            """
        ),
        code(
            """
            def build_grid_mats(run_label: str, value_col: str, correct_col: str):
                value_mat = np.full((len(G_LEVELS), len(D_LEVELS)), np.nan)
                annot_mat = np.full((len(G_LEVELS), len(D_LEVELS)), "", dtype=object)
                run_grid = gd_df.loc[gd_df["label"] == run_label]
                for _, row in run_grid.iterrows():
                    i = G_LEVELS.index(round(float(row["g"]), 2))
                    j = D_LEVELS.index(round(float(row["d"]), 1))
                    value_mat[i, j] = float(row[value_col])
                    annot_mat[i, j] = f"{int(row[correct_col])}/{int(row['n'])}"
                return value_mat, annot_mat


            def plot_gd_heatmaps(value_col: str, correct_col: str, title: str, file_stem: str):
                fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.8), sharex=True, sharey=True, constrained_layout=True)
                cmap = plt.get_cmap("YlGnBu").copy()
                cmap.set_bad("#efefef")
                ims = []
                for row_idx, init in enumerate(INIT_ORDER):
                    for col_idx, variant in enumerate(VARIANT_ORDER):
                        ax = axes[row_idx, col_idx]
                        label = make_label(variant, init)
                        values, annots = build_grid_mats(label, value_col, correct_col)
                        masked = np.ma.masked_invalid(values)
                        im = ax.imshow(masked, origin="lower", cmap=cmap, vmin=0, vmax=100, aspect="auto")
                        ims.append(im)
                        for i in range(values.shape[0]):
                            for j in range(values.shape[1]):
                                text = annots[i, j] if annots[i, j] else "--"
                                color = "black" if np.isnan(values[i, j]) or values[i, j] < 70 else "white"
                                ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)
                        ax.set_title(label)
                        ax.set_xticks(range(len(D_LEVELS)))
                        ax.set_xticklabels([f"{d:.1f}" for d in D_LEVELS])
                        ax.set_yticks(range(len(G_LEVELS)))
                        ax.set_yticklabels([f"{g:.2f}" for g in G_LEVELS])
                        ax.set_xlabel("d")
                        ax.set_ylabel("g")
                cbar = fig.colorbar(ims[-1], ax=axes, shrink=0.92)
                cbar.set_label("Accuracy (%)")
                fig.suptitle(title, fontsize=18)
                if SAVE_FIGURE:
                    for suffix in ["png", "pdf"]:
                        fig.savefig(FIGURE_DIR / f"{file_stem}.{suffix}", bbox_inches="tight")
                plt.show()


            display(
                Markdown(
                    "### `g x d` Grid Heatmaps at the Best Checkpoint\\n"
                    "Each cell is annotated as `correct/total` on the saved 64-row `id_val` slice."
                )
            )

            plot_gd_heatmaps(
                value_col="response_acc_pct",
                correct_col="response_correct_n",
                title="Panel-25 135M Runs: Response-Target Accuracy by Student Cell",
                file_stem="panel25_135m_gd_response_heatmaps",
            )
            plot_gd_heatmaps(
                value_col="true_acc_pct",
                correct_col="true_correct_n",
                title="Panel-25 135M Runs: True-Answer Accuracy by Student Cell",
                file_stem="panel25_135m_gd_true_heatmaps",
            )

            comparison_specs = [
                ("Default vs Surface | Pretrained", make_label("default", "pretrained"), make_label("surface", "pretrained")),
                ("Default vs Surface | Scratch", make_label("default", "scratch"), make_label("surface", "scratch")),
                ("Pretrained vs Scratch | Default", make_label("default", "pretrained"), make_label("default", "scratch")),
                ("Pretrained vs Scratch | Surface", make_label("surface", "pretrained"), make_label("surface", "scratch")),
            ]

            comparison_rows = []
            merged_examples = {}
            for title, left_label, right_label in comparison_specs:
                left_df = example_frames[left_label][
                    [
                        "prob",
                        "response_target_answer",
                        "true_target_answer",
                        "pred_answer",
                        "is_correct_response",
                        "is_correct_true",
                        "generation_text",
                    ]
                ].copy()
                right_df = example_frames[right_label][
                    [
                        "prob",
                        "pred_answer",
                        "is_correct_response",
                        "is_correct_true",
                        "generation_text",
                    ]
                ].copy()
                merged = left_df.merge(right_df, on="prob", suffixes=("_left", "_right"))
                merged_examples[title] = merged
                comparison_rows.append(
                    {
                        "comparison": title,
                        "response_label_flips": int((merged["is_correct_response_left"] != merged["is_correct_response_right"]).sum()),
                        "true_label_flips": int((merged["is_correct_true_left"] != merged["is_correct_true_right"]).sum()),
                        "pred_answer_changes": int(
                            (merged["pred_answer_left"].fillna("") != merged["pred_answer_right"].fillna("")).sum()
                        ),
                    }
                )

            comparison_df = pd.DataFrame(comparison_rows)
            display(Markdown("### Example-Level Disagreement Summary"))
            display(comparison_df)

            for title, _, _ in comparison_specs:
                merged = merged_examples[title]
                flip_mask = (
                    (merged["is_correct_response_left"] != merged["is_correct_response_right"])
                    | (merged["is_correct_true_left"] != merged["is_correct_true_right"])
                )
                pred_only_mask = (
                    ~flip_mask
                    & (merged["pred_answer_left"].fillna("") != merged["pred_answer_right"].fillna(""))
                )

                if flip_mask.any():
                    display(Markdown(f"#### {title}: correctness flips"))
                    display(
                        merged.loc[
                            flip_mask,
                            [
                                "prob",
                                "response_target_answer",
                                "true_target_answer",
                                "pred_answer_left",
                                "pred_answer_right",
                                "is_correct_response_left",
                                "is_correct_response_right",
                                "is_correct_true_left",
                                "is_correct_true_right",
                            ],
                        ]
                        .head(10)
                        .reset_index(drop=True)
                    )
                elif pred_only_mask.any():
                    display(Markdown(f"#### {title}: prediction changes without metric flips"))
                    display(
                        merged.loc[
                            pred_only_mask,
                            [
                                "prob",
                                "response_target_answer",
                                "true_target_answer",
                                "pred_answer_left",
                                "pred_answer_right",
                            ],
                        ]
                        .head(10)
                        .reset_index(drop=True)
                    )
            """
        ),
        md(
            r"""
            ## Reading Guide

            **Finding.** The notebook keeps optimization and evaluation separate. Validation loss favors the `surface`
            representation, especially with pretrained initialization, while response-level accuracy on the saved
            `id_val` slice is not ranked the same way.

            **Implication.** For this panel-25 family, a lower loss does not automatically imply better agreement with
            either the UMA response target or the true answer. The disagreement tables are useful when two runs land on
            similar aggregate scores but differ on particular problems.

            **Caution.** The subgroup heatmaps use only the saved 64-example `id_val` slice. They are most useful as a
            directional diagnostic, not as a stable estimate of panel-wide performance.
            """
        ),
    ]

    nb["cells"] = cells
    with out.open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)

    client = NotebookClient(nb, timeout=1800, kernel_name="python3")
    executed = client.execute(cwd=str(root))

    with out.open("w", encoding="utf-8") as handle:
        nbf.write(executed, handle)

    print(f"Wrote executed notebook to {out}")


if __name__ == "__main__":
    main()
