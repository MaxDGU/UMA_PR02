#!/usr/bin/env python3
"""
Build and execute the 135M LR ablation notebook with epochwise SP2013 profiles.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results" / "transformer_replication" / "analyze_135m_lr_ablation_training_curves.ipynb"

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
                # 135M LR Ablation: Epochwise Training and SP2013 Accuracy Profiles

                **Question.** How do the updated pretrained `SmolLM2-135M` learning-rate ablation runs change their SP2013 performance across completed epochs, and where do those changes concentrate in the `ED/UD × add/sub/mul/div` profile?

                **Setup.** The notebook reads the current pretrained `135M` run directories under
                `results/transformer_replication/smol2_135m_*_pretrained_matchspval_iduma_*`.
                Those directories were extended by the continuation and rerun `r135_*` jobs launched on
                `2026-04-12`, so the analysis uses the latest artifacts on disk rather than a stale launch manifest.

                **Objects.** For run \(r\), completed validation epoch \(e\), denominator class
                \(d \in \{\mathrm{ED}, \mathrm{UD}\}\), and operation
                \(o \in \{\mathrm{add}, \mathrm{sub}, \mathrm{mul}, \mathrm{div}\}\), we study two accuracies:
                \[
                a^{(\mathrm{resp})}_{r,e,d,o}
                = \frac{1}{N_{d,o}}\sum_{p \in \mathcal{P}_{d,o}} \mathbf{1}\{\hat{y}_{r,e}(p)=y^{\mathrm{UMA}}(p)\},
                \]
                \[
                a^{(\mathrm{true})}_{r,e,d,o}
                = \frac{1}{N_{d,o}}\sum_{p \in \mathcal{P}_{d,o}} \mathbf{1}\{\hat{y}_{r,e}(p)=y^{\star}(p)\},
                \]
                where \(\mathcal{P}_{d,o}\) is the SP2013 subset for cell \((d,o)\), \(y^{\mathrm{UMA}}(p)\) is the UMA
                response target used for checkpoint selection, and \(y^{\star}(p)\) is the mathematically correct answer.

                For a reference profile \(b \in \{\mathrm{Human}, \mathrm{UMA}\}\), we also report the mean absolute gap
                \[
                \mathrm{MAG}_{b}(r,e)
                = \frac{1}{8}\sum_{(d,o)\in \mathcal{C}}
                \left|a^{(\mathrm{true})}_{r,e,d,o} - a_b(d,o)\right|,
                \]
                where \(\mathcal{C}\) is the 8-cell `ED/UD × add/sub/mul/div` grid. Lower MAG means the epochwise
                SP2013 profile is closer to the reference profile.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                ## Formal Setup

                **Protocol.** Every run uses the same translated synthetic corpus and the same held-out SP2013
                validation condition. The comparison only varies learning rate and scheduler. The notebook keeps two
                views separate:

                1. **Optimization view.** We plot the logged training loss from `last/history.json` against epoch
                   progress. This view can include partial in-progress epochs.
                2. **Evaluation view.** We plot per-epoch SP2013 accuracy using the saved `id_val_metrics_epoch_XX.json`
                   and `id_val_epoch_XX.csv` files. This view stops at the latest completed full validation epoch on disk.

                **Important detail.** The training-loss figure can extend beyond the cellwise profile figures whenever a
                run has logged later history checkpoints than completed `id_val_epoch_XX.csv` artifacts. The availability
                table below reports the current per-run state directly from disk.
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
                import re
                from pathlib import Path

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                from IPython.display import Markdown, display
                from matplotlib.lines import Line2D
                from matplotlib.ticker import FuncFormatter

                plt.rcParams.update({
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
                })

                RUN_NAME_RE = re.compile(
                    r"lr(?P<lr_token>[0-9]+em[0-9]+)_(?P<scheduler_token>cos|lin)_e5_20260409_104738$"
                )
                SCHEDULER_NAME = {"cos": "cosine", "lin": "linear"}
                PROFILE_GRID = [("ED", "add"), ("ED", "sub"), ("ED", "mul"), ("ED", "div"),
                                ("UD", "add"), ("UD", "sub"), ("UD", "mul"), ("UD", "div")]
                PROFILE_CURVE_GRID = [
                    ("ED", "add"), ("UD", "add"),
                    ("ED", "sub"), ("UD", "sub"),
                    ("ED", "mul"), ("UD", "mul"),
                    ("ED", "div"), ("UD", "div"),
                ]
                PROFILE_CURVE_TICKLABELS = [
                    "add\\nED", "add\\nUD",
                    "sub\\nED", "sub\\nUD",
                    "mul\\nED", "mul\\nUD",
                    "div\\nED", "div\\nUD",
                ]
                OP_ORDER = ["add", "sub", "mul", "div"]
                DENOM_ORDER = ["ED", "UD"]
                SAVE_FIGURE = True


                def resolve_project_root() -> Path:
                    candidates = [Path.cwd(), *Path.cwd().parents]
                    for candidate in candidates:
                        if (candidate / "results" / "transformer_replication").exists():
                            return candidate
                    raise FileNotFoundError("Could not resolve project root.")


                def lr_token_to_float(token: str) -> float:
                    return float(token.replace("em", "e-"))


                def format_lr_label(value: float) -> str:
                    text = f"{value:.0e}"
                    return text.replace("e-0", "e-").replace("e+0", "e+")


                def format_epoch_progress(x, _):
                    if abs(x - round(x)) < 1e-6:
                        return f"{int(round(x))}"
                    return f"{x:.1f}"


                def parse_prob(prob: str) -> tuple[str, str]:
                    match = re.match(r"^(\\d+)/(\\d+)([+\\-*:])(\\d+)/(\\d+)$", str(prob).replace(" ", ""))
                    if match is None:
                        raise ValueError(f"Could not parse SP2013 problem: {prob}")
                    left_den = int(match.group(2))
                    op_symbol = match.group(3)
                    right_den = int(match.group(5))
                    denom = "ED" if left_den == right_den else "UD"
                    operation = {"+": "add", "-": "sub", "*": "mul", ":": "div"}[op_symbol]
                    return denom, operation


                PROJECT_ROOT = resolve_project_root()
                RUN_DIR = PROJECT_ROOT / "results" / "transformer_replication"
                FIGURE_DIR = RUN_DIR / "figures"
                FIGURE_DIR.mkdir(parents=True, exist_ok=True)

                run_dirs = sorted(
                    RUN_DIR.glob(
                        "smol2_135m_synth_unique_seed1_all1000_mincols_paramsalways_pretrained_matchspval_iduma_lr*_e5_20260409_104738"
                    )
                )
                if not run_dirs:
                    raise FileNotFoundError("No pretrained 135M LR ablation run directories were found.")

                display(Markdown(f"Detected **{len(run_dirs)}** pretrained LR-ablation runs with current on-disk artifacts."))
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                human_df = pd.read_csv(PROJECT_ROOT / "sp2013_human.csv")
                uma_df = pd.read_csv(PROJECT_ROOT / "results/UMA_replication/sp2013_eval/seed_1/sp2013_seed1_all_models.csv")

                reference_rows = []
                for denom, operation in PROFILE_GRID:
                    human_value = (
                        human_df.loc[
                            (human_df["operation"] == operation) & (human_df["operands"] == denom),
                            "acc",
                        ]
                        .mean()
                        * 100
                    )
                    uma_value = (
                        uma_df.loc[
                            (uma_df["operation"] == operation) & (uma_df["denoms"] == denom),
                            "acc",
                        ]
                        .mean()
                        * 100
                    )
                    reference_rows.append(
                        {
                            "denom": denom,
                            "operation": operation,
                            "Human (%)": human_value,
                            "UMA (%)": uma_value,
                        }
                    )

                reference_df = pd.DataFrame(reference_rows)
                reference_lookup = reference_df.set_index(["denom", "operation"])
                human_curve = np.array(
                    [reference_lookup.loc[(denom, operation), "Human (%)"] for denom, operation in PROFILE_CURVE_GRID]
                )
                uma_curve = np.array(
                    [reference_lookup.loc[(denom, operation), "UMA (%)"] for denom, operation in PROFILE_CURVE_GRID]
                )
                uma_mag_human_pct = float(np.mean(np.abs(uma_curve - human_curve)))

                run_rows = []
                history_rows = []
                overall_rows = []
                profile_rows = []

                for out_dir in run_dirs:
                    match = RUN_NAME_RE.search(out_dir.name)
                    if match is None:
                        continue

                    lr_value = lr_token_to_float(match.group("lr_token"))
                    scheduler = SCHEDULER_NAME[match.group("scheduler_token")]
                    label = f"{format_lr_label(lr_value)} {scheduler}"

                    history_path = out_dir / "last" / "history.json"
                    if not history_path.exists():
                        raise FileNotFoundError(history_path)
                    with history_path.open("r", encoding="utf-8") as handle:
                        history = json.load(handle)

                    completed_profile_epochs = sorted(
                        int(path.stem.split("_")[-1])
                        for path in out_dir.glob("id_val_epoch_*.csv")
                    )
                    completed_metric_epochs = sorted(
                        int(path.stem.split("_")[-1])
                        for path in out_dir.glob("id_val_metrics_epoch_*.json")
                    )
                    updates_per_epoch = max(
                        int(row["update_in_epoch"])
                        for row in history
                        if str(row.get("eval_scope", "full")) == "full"
                    )

                    history_for_run = []
                    for row in history:
                        update_in_epoch = int(row["update_in_epoch"])
                        epoch = int(row["epoch"])
                        history_record = {
                            "label": label,
                            "lr_value": lr_value,
                            "scheduler": scheduler,
                            "epoch": epoch,
                            "eval_tag": row.get("eval_tag"),
                            "eval_scope": row.get("eval_scope", "full"),
                            "update_in_epoch": update_in_epoch,
                            "updates_done": int(row["updates_done"]),
                            "epoch_progress": (epoch - 1) + update_in_epoch / updates_per_epoch,
                            "train_loss": float(row["train_loss"]),
                            "val_loss": float(row["val_loss"]),
                        }
                        history_for_run.append(history_record)
                        history_rows.append(history_record)

                    metrics_for_run = []
                    for metrics_path in sorted(out_dir.glob("id_val_metrics_epoch_*.json")):
                        epoch = int(metrics_path.stem.split("_")[-1])
                        with metrics_path.open("r", encoding="utf-8") as handle:
                            metrics = json.load(handle)
                        metric_record = {
                            "label": label,
                            "lr_value": lr_value,
                            "scheduler": scheduler,
                            "epoch": epoch,
                            "response_acc_pct": 100 * float(metrics["acc_all"]),
                            "response_acc_scored_pct": 100 * float(metrics["acc_scored"]),
                            "response_coverage_pct": 100 * float(metrics["coverage"]),
                            "true_acc_pct": 100 * float(metrics["true_acc_all"]),
                        }
                        metrics_for_run.append(metric_record)
                        overall_rows.append(metric_record)

                    for profile_path in sorted(out_dir.glob("id_val_epoch_*.csv")):
                        epoch = int(profile_path.stem.split("_")[-1])
                        profile_df = pd.read_csv(
                            profile_path,
                            usecols=["prob", "is_correct_true", "is_correct_response"],
                        )
                        profile_df[["denom", "operation"]] = profile_df["prob"].apply(
                            lambda prob: pd.Series(parse_prob(prob))
                        )
                        grouped = (
                            profile_df.groupby(["denom", "operation"])[["is_correct_true", "is_correct_response"]]
                            .mean()
                            .mul(100)
                            .reset_index()
                        )
                        for denom, operation in PROFILE_GRID:
                            row = grouped.loc[
                                (grouped["denom"] == denom) & (grouped["operation"] == operation)
                            ]
                            if row.empty:
                                raise RuntimeError(f"Missing profile cell {(denom, operation)} for {label} epoch {epoch}.")
                            profile_rows.append(
                                {
                                    "label": label,
                                    "lr_value": lr_value,
                                    "scheduler": scheduler,
                                    "epoch": epoch,
                                    "denom": denom,
                                    "operation": operation,
                                    "true_accuracy_pct": float(row["is_correct_true"].iloc[0]),
                                    "response_accuracy_pct": float(row["is_correct_response"].iloc[0]),
                                }
                            )

                    latest_history = max(history_for_run, key=lambda row: row["epoch_progress"])
                    latest_metrics = max(metrics_for_run, key=lambda row: row["epoch"]) if metrics_for_run else None
                    best_metrics = max(metrics_for_run, key=lambda row: row["response_acc_pct"]) if metrics_for_run else None

                    run_rows.append(
                        {
                            "label": label,
                            "lr_value": lr_value,
                            "scheduler": scheduler,
                            "completed_profile_epochs": ",".join(str(epoch) for epoch in completed_profile_epochs),
                            "completed_metric_epochs": ",".join(str(epoch) for epoch in completed_metric_epochs),
                            "latest_train_epoch_progress": latest_history["epoch_progress"],
                            "latest_train_loss": latest_history["train_loss"],
                            "latest_response_acc_pct": np.nan if latest_metrics is None else latest_metrics["response_acc_pct"],
                            "latest_true_acc_pct": np.nan if latest_metrics is None else latest_metrics["true_acc_pct"],
                            "latest_response_coverage_pct": np.nan if latest_metrics is None else latest_metrics["response_coverage_pct"],
                            "best_response_acc_pct": np.nan if best_metrics is None else best_metrics["response_acc_pct"],
                        }
                    )

                runs_df = pd.DataFrame(run_rows).sort_values(["lr_value", "scheduler"], ascending=[False, True]).reset_index(drop=True)
                history_df = pd.DataFrame(history_rows).sort_values(["lr_value", "scheduler", "updates_done"], ascending=[False, True, True]).reset_index(drop=True)
                overall_df = pd.DataFrame(overall_rows).sort_values(["lr_value", "scheduler", "epoch"], ascending=[False, True, True]).reset_index(drop=True)
                profile_long_df = pd.DataFrame(profile_rows).sort_values(["lr_value", "scheduler", "epoch", "denom", "operation"], ascending=[False, True, True, True, True]).reset_index(drop=True)

                mag_rows = []
                for (label, lr_value, scheduler, epoch), group in profile_long_df.groupby(
                    ["label", "lr_value", "scheduler", "epoch"], sort=False
                ):
                    ordered_true = (
                        group.set_index(["denom", "operation"])
                        .loc[PROFILE_CURVE_GRID, "true_accuracy_pct"]
                        .to_numpy()
                    )
                    mag_rows.append(
                        {
                            "label": label,
                            "lr_value": lr_value,
                            "scheduler": scheduler,
                            "epoch": epoch,
                            "true_mag_human_pct": float(np.mean(np.abs(ordered_true - human_curve))),
                            "true_mag_uma_pct": float(np.mean(np.abs(ordered_true - uma_curve))),
                        }
                    )
                mag_df = pd.DataFrame(mag_rows).sort_values(["lr_value", "scheduler", "epoch"], ascending=[False, True, True]).reset_index(drop=True)
                overall_df = overall_df.merge(
                    mag_df,
                    on=["label", "lr_value", "scheduler", "epoch"],
                    how="left",
                )

                first_mag_df = (
                    overall_df.sort_values(["label", "epoch"])
                    .groupby("label", as_index=False)
                    .head(1)
                    .loc[:, ["label", "epoch", "true_mag_human_pct", "true_mag_uma_pct"]]
                    .rename(
                        columns={
                            "epoch": "first_epoch",
                            "true_mag_human_pct": "first_true_mag_human_pct",
                            "true_mag_uma_pct": "first_true_mag_uma_pct",
                        }
                    )
                )
                latest_mag_df = (
                    overall_df.sort_values(["label", "epoch"])
                    .groupby("label", as_index=False)
                    .tail(1)
                    .loc[:, ["label", "epoch", "true_mag_human_pct", "true_mag_uma_pct"]]
                    .rename(
                        columns={
                            "epoch": "latest_epoch",
                            "true_mag_human_pct": "latest_true_mag_human_pct",
                            "true_mag_uma_pct": "latest_true_mag_uma_pct",
                        }
                    )
                )
                mag_trend_df = (
                    runs_df.loc[:, ["label", "lr_value", "scheduler"]]
                    .merge(first_mag_df, on="label", how="left")
                    .merge(latest_mag_df, on="label", how="left")
                    .sort_values(["lr_value", "scheduler"], ascending=[False, True])
                    .reset_index(drop=True)
                )
                mag_trend_df["delta_true_mag_human_pct"] = (
                    mag_trend_df["latest_true_mag_human_pct"] - mag_trend_df["first_true_mag_human_pct"]
                )
                mag_trend_df["delta_true_mag_uma_pct"] = (
                    mag_trend_df["latest_true_mag_uma_pct"] - mag_trend_df["first_true_mag_uma_pct"]
                )
                runs_df = runs_df.merge(
                    latest_mag_df.loc[:, ["label", "latest_true_mag_human_pct", "latest_true_mag_uma_pct"]],
                    on="label",
                    how="left",
                )

                display(Markdown("### Run Availability and Latest Completed Results"))
                display(
                    runs_df[
                        [
                            "label",
                            "completed_profile_epochs",
                            "latest_train_epoch_progress",
                            "latest_train_loss",
                            "latest_response_acc_pct",
                            "latest_true_acc_pct",
                            "latest_response_coverage_pct",
                            "best_response_acc_pct",
                            "latest_true_mag_human_pct",
                            "latest_true_mag_uma_pct",
                        ]
                    ].round(3)
                )

                display(Markdown("### Epochwise MAG Summary (lower is better)"))
                display(
                    overall_df[
                        [
                            "label",
                            "epoch",
                            "response_acc_pct",
                            "true_acc_pct",
                            "true_mag_human_pct",
                            "true_mag_uma_pct",
                        ]
                    ].round(3)
                )

                display(Markdown("### MAG Change From First to Latest Completed Epoch"))
                display(
                    mag_trend_df[
                        [
                            "label",
                            "first_epoch",
                            "latest_epoch",
                            "first_true_mag_human_pct",
                            "latest_true_mag_human_pct",
                            "delta_true_mag_human_pct",
                            "first_true_mag_uma_pct",
                            "latest_true_mag_uma_pct",
                            "delta_true_mag_uma_pct",
                        ]
                    ].round(3)
                )
                display(
                    Markdown(
                        f"Mean change from first to latest completed epoch: "
                        f"Human MAG `{mag_trend_df['delta_true_mag_human_pct'].mean():+.2f}` pts, "
                        f"UMA MAG `{mag_trend_df['delta_true_mag_uma_pct'].mean():+.2f}` pts. "
                        f"Negative is improvement."
                    )
                )
                display(
                    Markdown(
                        f"Fixed UMA-vs-Human baseline: `true_mag_human_pct = {uma_mag_human_pct:.2f}` pts. "
                        f"The epochwise Human-MAG plot below overlays this quantity as a dashed reference line."
                    )
                )

                display(Markdown("### Reference SP2013 Cellwise True-Answer Accuracy (%)"))
                display(
                    reference_df.pivot(index="denom", columns="operation", values=["Human (%)", "UMA (%)"]).round(2)
                )
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                palette = plt.get_cmap("viridis")
                unique_lrs = sorted(runs_df["lr_value"].unique(), reverse=True)
                color_map = {lr: palette(i / max(len(unique_lrs) - 1, 1)) for i, lr in enumerate(unique_lrs)}
                linestyle_map = {"linear": "-", "cosine": "--"}
                marker_map = {"linear": "o", "cosine": "s"}
                curve_x = np.arange(len(PROFILE_CURVE_GRID))

                ordered_runs = runs_df[["label", "lr_value", "scheduler"]].drop_duplicates().to_dict("records")
                student_handles = [
                    Line2D(
                        [0],
                        [0],
                        color=color_map[row["lr_value"]],
                        linestyle=linestyle_map[row["scheduler"]],
                        marker=marker_map[row["scheduler"]],
                        linewidth=2.0,
                        markersize=5,
                        label=row["label"],
                    )
                    for row in ordered_runs
                ]
                uma_mag_baseline_handle = Line2D(
                    [0],
                    [0],
                    color="#E76F51",
                    linestyle=(0, (6, 3)),
                    linewidth=2.2,
                    label=f"UMA vs Human ({uma_mag_human_pct:.2f})",
                )

                fig1, axes1 = plt.subplots(1, 3, figsize=(18.5, 5.6), constrained_layout=True)

                for row in ordered_runs:
                    run_history = history_df.loc[history_df["label"] == row["label"]]
                    run_overall = overall_df.loc[overall_df["label"] == row["label"]]
                    axes1[0].plot(
                        run_history["epoch_progress"],
                        run_history["train_loss"],
                        color=color_map[row["lr_value"]],
                        linestyle=linestyle_map[row["scheduler"]],
                        marker=marker_map[row["scheduler"]],
                        linewidth=1.7,
                        markersize=4.5,
                        alpha=0.92,
                    )
                    axes1[1].plot(
                        run_overall["epoch"],
                        run_overall["response_acc_pct"],
                        color=color_map[row["lr_value"]],
                        linestyle=linestyle_map[row["scheduler"]],
                        marker=marker_map[row["scheduler"]],
                        linewidth=1.7,
                        markersize=4.5,
                        alpha=0.92,
                    )
                    axes1[2].plot(
                        run_overall["epoch"],
                        run_overall["true_acc_pct"],
                        color=color_map[row["lr_value"]],
                        linestyle=linestyle_map[row["scheduler"]],
                        marker=marker_map[row["scheduler"]],
                        linewidth=1.7,
                        markersize=4.5,
                        alpha=0.92,
                    )

                axes1[0].set_title("Logged Training Loss")
                axes1[0].set_xlabel("Epoch Progress")
                axes1[0].set_ylabel("Loss")
                axes1[0].set_yscale("log")
                axes1[0].xaxis.set_major_formatter(FuncFormatter(format_epoch_progress))
                axes1[0].grid(True, alpha=0.25, linewidth=0.8)

                axes1[1].set_title("SP2013 Accuracy vs UMA Target")
                axes1[1].set_xlabel("Completed Epoch")
                axes1[1].set_ylabel("Accuracy (%)")
                axes1[1].set_ylim(48, 61)
                axes1[1].grid(True, alpha=0.25, linewidth=0.8)

                axes1[2].set_title("SP2013 True-Answer Accuracy")
                axes1[2].set_xlabel("Completed Epoch")
                axes1[2].set_ylabel("Accuracy (%)")
                axes1[2].set_ylim(63, 69.5)
                axes1[2].grid(True, alpha=0.25, linewidth=0.8)

                for axis in axes1[1:]:
                    axis.set_xticks(sorted(overall_df["epoch"].unique()))

                axes1[2].legend(
                    handles=student_handles,
                    title="LR + scheduler",
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                )

                fig1.suptitle("135M LR Ablation: Optimization and Overall Accuracy by Epoch", fontsize=18)

                fig2, axes2 = plt.subplots(1, 2, figsize=(15.6, 5.6), constrained_layout=True, sharex=True)
                for row in ordered_runs:
                    run_mag = overall_df.loc[overall_df["label"] == row["label"]]
                    if run_mag.empty:
                        continue
                    axes2[0].plot(
                        run_mag["epoch"],
                        run_mag["true_mag_human_pct"],
                        color=color_map[row["lr_value"]],
                        linestyle=linestyle_map[row["scheduler"]],
                        marker=marker_map[row["scheduler"]],
                        linewidth=1.7,
                        markersize=4.5,
                        alpha=0.92,
                    )
                    axes2[1].plot(
                        run_mag["epoch"],
                        run_mag["true_mag_uma_pct"],
                        color=color_map[row["lr_value"]],
                        linestyle=linestyle_map[row["scheduler"]],
                        marker=marker_map[row["scheduler"]],
                        linewidth=1.7,
                        markersize=4.5,
                        alpha=0.92,
                    )

                axes2[0].axhline(
                    uma_mag_human_pct,
                    color="#E76F51",
                    linestyle=(0, (6, 3)),
                    linewidth=2.2,
                    alpha=0.95,
                )
                axes2[0].set_title("true_mag_human_pct (MAG) vs Human")
                axes2[0].set_xlabel("Completed Epoch")
                axes2[0].set_ylabel("MAG (pct points)")
                axes2[0].grid(True, alpha=0.25, linewidth=0.8)
                axes2[1].set_title("True-Profile MAG vs UMA")
                axes2[1].set_xlabel("Completed Epoch")
                axes2[1].set_ylabel("MAG (pct points)")
                axes2[1].grid(True, alpha=0.25, linewidth=0.8)

                for axis in axes2:
                    axis.set_xticks(sorted(overall_df["epoch"].unique()))

                axes2[0].legend(
                    handles=[uma_mag_baseline_handle],
                    title="Reference",
                    loc="upper right",
                    frameon=False,
                )
                axes2[1].legend(
                    handles=student_handles,
                    title="LR + scheduler",
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                )

                fig2.suptitle("135M LR Ablation: Mean Absolute Gap by Epoch", fontsize=18)


                def plot_epoch_profile_curves(title: str, file_stem: str) -> plt.Figure:
                    epoch_values = sorted(profile_long_df["epoch"].unique())
                    n_cols = 2 if len(epoch_values) > 1 else 1
                    n_rows = int(np.ceil(len(epoch_values) / n_cols))
                    fig, axes = plt.subplots(
                        n_rows,
                        n_cols,
                        figsize=(9.3 * n_cols, 5.8 * n_rows),
                        sharey=True,
                        constrained_layout=True,
                    )
                    axes = np.atleast_1d(axes).ravel()

                    for axis, epoch in zip(axes, epoch_values):
                        for row in ordered_runs:
                            run_profile = profile_long_df.loc[
                                (profile_long_df["label"] == row["label"]) & (profile_long_df["epoch"] == epoch),
                                ["denom", "operation", "true_accuracy_pct"],
                            ]
                            if run_profile.empty:
                                continue
                            run_profile = (
                                run_profile.set_index(["denom", "operation"])
                                .loc[PROFILE_CURVE_GRID, "true_accuracy_pct"]
                                .to_numpy()
                            )
                            axis.plot(
                                curve_x,
                                run_profile,
                                color=color_map[row["lr_value"]],
                                linestyle=linestyle_map[row["scheduler"]],
                                marker=marker_map[row["scheduler"]],
                                linewidth=1.5,
                                markersize=4.3,
                                alpha=0.82,
                            )

                        axis.plot(
                            curve_x,
                            human_curve,
                            color="black",
                            marker="o",
                            markersize=6,
                            linewidth=2.4,
                        )
                        axis.plot(
                            curve_x,
                            uma_curve,
                            color="#E76F51",
                            marker="D",
                            markersize=5.5,
                            linewidth=2.4,
                        )
                        axis.set_title(f"Epoch {epoch}")
                        axis.set_xticks(curve_x)
                        axis.set_xticklabels(PROFILE_CURVE_TICKLABELS)
                        axis.set_ylim(0, 105)
                        axis.grid(True, axis="y", alpha=0.25, linewidth=0.8)
                        axis.set_xlabel("SP2013 cell")
                        axis.set_ylabel("Accuracy (%)")

                    for axis in axes[len(epoch_values):]:
                        axis.set_visible(False)

                    axes[0].legend(
                        handles=[
                            Line2D([0], [0], color="black", marker="o", linewidth=2.4, markersize=6, label="Human"),
                            Line2D([0], [0], color="#E76F51", marker="D", linewidth=2.4, markersize=5.5, label="UMA"),
                        ],
                        loc="upper left",
                        frameon=False,
                    )
                    axes[min(len(epoch_values) - 1, len(axes) - 1)].legend(
                        handles=student_handles,
                        title="LR + scheduler",
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=False,
                    )
                    fig.suptitle(title, fontsize=18)

                    if SAVE_FIGURE:
                        png_path = FIGURE_DIR / f"{file_stem}.png"
                        pdf_path = FIGURE_DIR / f"{file_stem}.pdf"
                        fig.savefig(png_path, bbox_inches="tight")
                        fig.savefig(pdf_path, bbox_inches="tight")
                        print(f"Saved figure to: {png_path}")
                        print(f"Saved figure to: {pdf_path}")

                    return fig


                def plot_cell_trajectories(value_col: str, title: str, file_stem: str, include_refs: bool) -> plt.Figure:
                    fig, axes = plt.subplots(2, 4, figsize=(18.8, 8.6), sharex=True, sharey=True, constrained_layout=True)
                    axes = np.asarray(axes)

                    for row_idx, denom in enumerate(DENOM_ORDER):
                        for col_idx, operation in enumerate(OP_ORDER):
                            axis = axes[row_idx, col_idx]
                            for row in ordered_runs:
                                run_cell = profile_long_df.loc[
                                    (profile_long_df["label"] == row["label"])
                                    & (profile_long_df["denom"] == denom)
                                    & (profile_long_df["operation"] == operation)
                                ]
                                if run_cell.empty:
                                    continue
                                axis.plot(
                                    run_cell["epoch"],
                                    run_cell[value_col],
                                    color=color_map[row["lr_value"]],
                                    linestyle=linestyle_map[row["scheduler"]],
                                    marker=marker_map[row["scheduler"]],
                                    linewidth=1.6,
                                    markersize=4.5,
                                    alpha=0.9,
                                )

                            if include_refs:
                                axis.axhline(
                                    reference_lookup.loc[(denom, operation), "Human (%)"],
                                    color="black",
                                    linewidth=1.9,
                                    alpha=0.85,
                                )
                                axis.axhline(
                                    reference_lookup.loc[(denom, operation), "UMA (%)"],
                                    color="#E76F51",
                                    linewidth=1.9,
                                    linestyle=":",
                                    alpha=0.95,
                                )

                            axis.set_title(f"{denom} · {operation}")
                            axis.set_xticks(sorted(profile_long_df["epoch"].unique()))
                            axis.set_ylim(0, 105)
                            axis.grid(True, alpha=0.25, linewidth=0.8)

                            if row_idx == len(DENOM_ORDER) - 1:
                                axis.set_xlabel("Completed Epoch")
                            if col_idx == 0:
                                axis.set_ylabel("Accuracy (%)")

                    if include_refs:
                        axes[0, 0].legend(
                            handles=[
                                Line2D([0], [0], color="black", linewidth=1.9, label="Human"),
                                Line2D([0], [0], color="#E76F51", linewidth=1.9, linestyle=":", label="UMA"),
                            ],
                            loc="lower left",
                            frameon=False,
                        )

                    axes[0, -1].legend(
                        handles=student_handles,
                        title="LR + scheduler",
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.0),
                        frameon=False,
                    )
                    fig.suptitle(title, fontsize=18)

                    if SAVE_FIGURE:
                        png_path = FIGURE_DIR / f"{file_stem}.png"
                        pdf_path = FIGURE_DIR / f"{file_stem}.pdf"
                        fig.savefig(png_path, bbox_inches="tight")
                        fig.savefig(pdf_path, bbox_inches="tight")
                        print(f"Saved figure to: {png_path}")
                        print(f"Saved figure to: {pdf_path}")

                    return fig


                if SAVE_FIGURE:
                    for suffix in ["png", "pdf"]:
                        fig1.savefig(FIGURE_DIR / f"135m_lr_ablation_epochwise_summary.{suffix}", bbox_inches="tight")
                        fig2.savefig(FIGURE_DIR / f"135m_lr_ablation_epochwise_mag.{suffix}", bbox_inches="tight")

                fig3 = plot_epoch_profile_curves(
                    title="135M LR Ablation: SP2013 True-Answer Profile by Completed Epoch",
                    file_stem="135m_lr_ablation_epochwise_true_curve_profiles",
                )
                fig4 = plot_cell_trajectories(
                    value_col="true_accuracy_pct",
                    title="135M LR Ablation: Cellwise True-Answer Accuracy by Epoch",
                    file_stem="135m_lr_ablation_epochwise_true_profiles",
                    include_refs=True,
                )
                fig5 = plot_cell_trajectories(
                    value_col="response_accuracy_pct",
                    title="135M LR Ablation: Cellwise Accuracy Against UMA Target by Epoch",
                    file_stem="135m_lr_ablation_epochwise_response_profiles",
                    include_refs=False,
                )

                plt.show()
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                ## Reading Guide

                **Finding.** The notebook now offers two epochwise profile views. The new multi-panel curve figure keeps
                the original 8-cell profile style and shows one full profile per completed epoch. The 2×4 cell figure
                still shows each `ED/UD × operation` cell as a trajectory across epochs.

                **Interpretation.** The overall-accuracy figure uses `acc_all`, so unparseable responses count as
                incorrect. The true-answer figures isolate mathematical correctness from response-target agreement. The
                MAG figures summarize the same 8-cell true-answer profile as a single number relative to the Human and
                UMA reference curves; lower is better. In the `true_mag_human_pct` panel, the dashed horizontal line is
                the fixed UMA-to-Human gap, so student curves below that line are closer to the Human profile than UMA is.

                **Availability.** If a line stops at `epoch 2`, `epoch 3`, or `epoch 4`, that is the latest completed full
                `id_val_epoch_XX.csv` file currently present on disk for that run, even if the training-loss plot shows
                a partial later epoch in progress.
                """
            )
        )
    )

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
