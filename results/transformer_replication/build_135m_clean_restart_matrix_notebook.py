#!/usr/bin/env python3
"""
Build and execute the 135M clean distillation restart matrix notebook.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results" / "transformer_replication" / "analyze_135m_clean_restart_matrix.ipynb"

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
                # 135M Clean Distillation Restart Matrix: Updated Run Audit and Partial Result Analysis

                This notebook updates the clean-distillation restart matrix after the later reruns on `2026-04-12`.
                The original `2026-04-11 20:10:16` launch failed, but a later rerun produced a **partial**
                `2 \times 2` result: both `params_always` conditions completed training and SP2013 evaluation, while
                both `no_params` conditions still failed in validation setup.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                ## Formal Setup

                **Question.** We study the four-condition clean-distillation restart matrix defined by the Cartesian product
                \[
                \mathcal{D} = \{\texttt{correct\_exec},\; \texttt{correct\_exec\_and\_answer}\}
                \]
                and
                \[
                \mathcal{P} = \{\texttt{params\_always},\; \texttt{no\_params}\}.
                \]
                Each run is indexed by
                \[
                r = (d, p) \in \mathcal{D} \times \mathcal{P}.
                \]

                **Training protocol.** All reruns keep the common optimizer configuration
                \[
                \eta = 10^{-4}, \qquad s = \texttt{linear}, \qquad E = 2,
                \]
                with SP2013 evaluation matched to prompt condition:
                \[
                p = \texttt{params\_always} \Rightarrow \text{tuple-conditioned greedy SP2013 evaluation},
                \]
                \[
                p = \texttt{no\_params} \Rightarrow \text{1000-rollout sampled SP2013 distribution evaluation}.
                \]

                **Observed objects.** For each run \(r\), we audit
                \[
                A_r = \text{attempt lineage and launcher diagnostics},
                \]
                \[
                F_r = \text{filesystem artifacts written to the output directory},
                \]
                \[
                H_r = \text{epoch-level trainer history},
                \]
                \[
                M_r = \text{SP2013 evaluation metrics, if any were produced}.
                \]

                **Objective.** The notebook answers three questions.

                1. Which restart attempt first produced usable training artifacts?
                2. Which cells in the `2 \times 2` matrix completed training versus failed in validation setup?
                3. For the successful cells, how do SP2013 outcomes compare across epochs and across data conditions?
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

                import matplotlib.colors as mcolors
                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                import seaborn as sns
                from IPython.display import Markdown, display


                def resolve_project_root() -> Path:
                    candidates = [Path.cwd(), *Path.cwd().parents]
                    for candidate in candidates:
                        if (candidate / "results" / "transformer_replication").exists():
                            return candidate
                    raise FileNotFoundError("Could not resolve project root.")


                PROJECT_ROOT = resolve_project_root()
                BASE_DIR = PROJECT_ROOT / "results" / "transformer_replication"
                FIGURE_DIR = BASE_DIR / "figures"
                FIGURE_DIR.mkdir(parents=True, exist_ok=True)

                ATTEMPT_MANIFESTS = [
                    BASE_DIR / "jobs_135m_clean_distill_matrix_20260411_201016.tsv",
                    BASE_DIR / "jobs_135m_clean_distill_matrix_20260412_151559.tsv",
                    BASE_DIR / "jobs_135m_clean_distill_matrix_20260412_204603.tsv",
                    BASE_DIR / "jobs_135m_clean_distill_matrix_20260412_211956.tsv",
                ]
                LATEST_MANIFEST_PATH = ATTEMPT_MANIFESTS[-1]
                PROFILE_GRID = [
                    ("ED", "add"), ("UD", "add"),
                    ("ED", "sub"), ("UD", "sub"),
                    ("ED", "mul"), ("UD", "mul"),
                    ("ED", "div"), ("UD", "div"),
                ]
                PROFILE_TICKLABELS = [
                    "add\\nED", "add\\nUD",
                    "sub\\nED", "sub\\nUD",
                    "mul\\nED", "mul\\nUD",
                    "div\\nED", "div\\nUD",
                ]
                OUTCOME_TO_CODE = {
                    "metrics_written": 2,
                    "partial_artifacts_only": 1,
                    "failed_validation_layout": -1,
                    "failed_before_artifacts": -2,
                    "missing": -3,
                }
                OUTCOME_TO_LABEL = {
                    "metrics_written": "metrics written",
                    "partial_artifacts_only": "partial artifacts only",
                    "failed_validation_layout": "failed validation layout",
                    "failed_before_artifacts": "failed before artifacts",
                    "missing": "missing",
                }
                OUTCOME_COLORS = {
                    2: "#2A9D8F",
                    1: "#E9C46A",
                    -1: "#D62828",
                    -2: "#6D597A",
                    -3: "#ADB5BD",
                }

                plt.rcParams.update({
                    "figure.dpi": 160,
                    "savefig.dpi": 300,
                    "font.size": 12,
                    "axes.titlesize": 16,
                    "axes.labelsize": 13,
                    "xtick.labelsize": 11,
                    "ytick.labelsize": 11,
                    "legend.fontsize": 10,
                    "legend.title_fontsize": 11,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                })
                sns.set_theme(style="whitegrid", context="talk")


                def resolve_log_path(template: str, job_id: int) -> Path:
                    return Path(template.replace("%j", str(job_id)))


                def read_text(path: Path) -> str:
                    if not path.exists():
                        return ""
                    return path.read_text(encoding="utf-8", errors="ignore")


                def load_json(path: Path) -> dict:
                    if not path.exists():
                        return {}
                    with path.open("r", encoding="utf-8") as handle:
                        return json.load(handle)


                def gather_artifacts(out_dir: Path) -> dict[str, object]:
                    recursive_files = [p for p in out_dir.rglob("*") if p.is_file()] if out_dir.exists() else []
                    return {
                        "out_dir_exists": out_dir.exists(),
                        "recursive_file_count": len(recursive_files),
                        "top_level_items": len(list(out_dir.iterdir())) if out_dir.exists() else 0,
                        "best_summary_exists": (out_dir / "best_checkpoint_summary.json").exists(),
                        "summary_metrics_exists": (out_dir / "summary_metrics.json").exists(),
                        "last_history_exists": (out_dir / "last" / "history.json").exists(),
                        "best_history_exists": (out_dir / "best" / "history.json").exists(),
                        "sp2013_metrics_files": len(list(out_dir.glob("sp2013_metrics_epoch_*.json"))) if out_dir.exists() else 0,
                        "sp2013_epoch_files": len(list(out_dir.glob("sp2013_epoch_*.csv"))) if out_dir.exists() else 0,
                    }


                def simplify_failure_reason(stdout_text: str, stderr_text: str) -> str:
                    text = stderr_text if stderr_text.strip() else stdout_text
                    if not text.strip():
                        return "no failure text captured"
                    if "strict layout requires student parameter columns" in text:
                        return "strict SP2013 validation needs g/d/rt_mu/ice"
                    if "ChildFailedError" in text:
                        return "trainer failed before artifact write"
                    if "Traceback" in text:
                        lines = [line.strip() for line in text.splitlines() if line.strip()]
                        return lines[-1][:120] if lines else "python traceback"
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    return lines[-1][:120] if lines else "stderr present but unparsed"


                def infer_outcome(stdout_text: str, stderr_text: str, artifacts: dict[str, object]) -> str:
                    if artifacts["sp2013_metrics_files"] > 0:
                        return "metrics_written"
                    if "strict layout requires student parameter columns" in stderr_text:
                        return "failed_validation_layout"
                    if artifacts["recursive_file_count"] > 0:
                        return "partial_artifacts_only"
                    if stdout_text.strip() or stderr_text.strip():
                        return "failed_before_artifacts"
                    return "missing"


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
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                attempt_rows = []
                for manifest_path in ATTEMPT_MANIFESTS:
                    manifest = pd.read_csv(manifest_path, sep="\\t")
                    manifest["out_dir"] = manifest["out_dir"].map(Path)
                    artifact_rows = [gather_artifacts(path) for path in manifest["out_dir"]]
                    artifact_df = pd.DataFrame(artifact_rows)
                    attempt_rows.append(
                        {
                            "attempt_tag": manifest_path.stem.replace("jobs_135m_clean_distill_matrix_", ""),
                            "runs_submitted": len(manifest),
                            "runs_with_nonempty_dirs": int((artifact_df["recursive_file_count"] > 0).sum()),
                            "runs_with_history": int(artifact_df["last_history_exists"].sum()),
                            "runs_with_sp2013_metrics": int((artifact_df["sp2013_metrics_files"] > 0).sum()),
                            "runs_with_best_summary": int(artifact_df["best_summary_exists"].sum()),
                        }
                    )

                attempt_df = pd.DataFrame(attempt_rows)

                manifest = pd.read_csv(LATEST_MANIFEST_PATH, sep="\\t")
                manifest["out_dir"] = manifest["out_dir"].map(Path)
                manifest["data_csv"] = manifest["data_csv"].map(Path)
                manifest["log_out_resolved"] = [
                    resolve_log_path(template, job_id)
                    for template, job_id in zip(manifest["log_out"], manifest["job_id"])
                ]
                manifest["log_err_resolved"] = [
                    resolve_log_path(template, job_id)
                    for template, job_id in zip(manifest["log_err"], manifest["job_id"])
                ]
                manifest["stdout_text"] = manifest["log_out_resolved"].map(read_text)
                manifest["stderr_text"] = manifest["log_err_resolved"].map(read_text)

                artifact_rows = [gather_artifacts(path) for path in manifest["out_dir"]]
                manifest = pd.concat([manifest, pd.DataFrame(artifact_rows)], axis=1)

                manifest["outcome"] = [
                    infer_outcome(stdout, stderr, {
                        "sp2013_metrics_files": sp2013_metrics_files,
                        "recursive_file_count": recursive_file_count,
                    })
                    for stdout, stderr, sp2013_metrics_files, recursive_file_count in zip(
                        manifest["stdout_text"],
                        manifest["stderr_text"],
                        manifest["sp2013_metrics_files"],
                        manifest["recursive_file_count"],
                    )
                ]
                manifest["failure_reason"] = [
                    simplify_failure_reason(stdout, stderr)
                    for stdout, stderr in zip(manifest["stdout_text"], manifest["stderr_text"])
                ]

                summary_payloads = [load_json(path / "summary_metrics.json") for path in manifest["out_dir"]]
                best_payloads = [load_json(path / "best_checkpoint_summary.json") for path in manifest["out_dir"]]
                manifest["best_sp2013_primary"] = [payload.get("best_sp2013_primary", np.nan) for payload in summary_payloads]
                manifest["best_epoch"] = [payload.get("best_epoch", np.nan) for payload in summary_payloads]
                manifest["best_eval_tag"] = [payload.get("best_eval_tag", "") for payload in summary_payloads]
                manifest["best_val_loss"] = [payload.get("best_val_loss", np.nan) for payload in summary_payloads]
                manifest["sp2013_primary_name"] = [payload.get("best_sp2013_primary_name", "") or best.get("sp2013_primary_name", "") for payload, best in zip(summary_payloads, best_payloads)]
                manifest["latest_artifact_tag"] = LATEST_MANIFEST_PATH.stem.replace("jobs_135m_clean_distill_matrix_", "")
                manifest["condition_label"] = manifest["data_condition"] + " | " + manifest["prompt_condition"]

                display(Markdown("### Attempt Lineage"))
                display(attempt_df)

                display(Markdown("### Latest Restart Matrix Audit"))
                display(
                    manifest[
                        [
                            "job_id",
                            "data_condition",
                            "prompt_condition",
                            "sp2013_target_mode",
                            "outcome",
                            "best_sp2013_primary",
                            "best_epoch",
                            "failure_reason",
                        ]
                    ].round(4)
                )
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                dataset_records = []
                for data_csv in sorted(manifest["data_csv"].unique()):
                    subset = manifest.loc[manifest["data_csv"] == data_csv].iloc[0]
                    summary_rows = []
                    for out_dir in manifest.loc[manifest["data_csv"] == data_csv, "out_dir"]:
                        payload = load_json(out_dir / "summary_metrics.json")
                        train_rows = payload.get("train_rows")
                        if train_rows is not None:
                            summary_rows.append(int(train_rows))
                    inferred_rows = summary_rows[0] if summary_rows else np.nan
                    dataset_records.append(
                        {
                            "data_condition": subset["data_condition"],
                            "rows": inferred_rows,
                            "rows_millions": inferred_rows / 1_000_000 if pd.notna(inferred_rows) else np.nan,
                            "compressed_size_gb": data_csv.stat().st_size / (1024 ** 3),
                            "data_csv": data_csv,
                        }
                    )
                dataset_df = pd.DataFrame(dataset_records).sort_values("rows", ascending=False).reset_index(drop=True)

                outcome_matrix = manifest.pivot(index="data_condition", columns="prompt_condition", values="outcome")
                outcome_code_matrix = outcome_matrix.replace(OUTCOME_TO_CODE)
                outcome_annot = pd.DataFrame("", index=outcome_matrix.index, columns=outcome_matrix.columns)
                for _, row in manifest.iterrows():
                    cell_text = OUTCOME_TO_LABEL[row["outcome"]]
                    if pd.notna(row["best_sp2013_primary"]):
                        cell_text += f"\\nbest={float(row['best_sp2013_primary']):.3f}"
                    elif row["failure_reason"]:
                        cell_text += f"\\n{row['failure_reason']}"
                    outcome_annot.loc[row["data_condition"], row["prompt_condition"]] = cell_text

                artifact_plot_df = (
                    manifest.set_index("condition_label")[
                        [
                            "out_dir_exists",
                            "last_history_exists",
                            "sp2013_metrics_files",
                            "best_summary_exists",
                        ]
                    ]
                    .copy()
                )
                artifact_plot_df["sp2013_metrics_files"] = (artifact_plot_df["sp2013_metrics_files"] > 0).astype(int)
                artifact_plot_df = artifact_plot_df.astype(int).rename(
                    columns={
                        "out_dir_exists": "out dir",
                        "last_history_exists": "last/history.json",
                        "sp2013_metrics_files": "sp2013 metrics",
                        "best_summary_exists": "best_checkpoint_summary.json",
                    }
                )

                best_bar_df = manifest.copy()
                best_bar_df["best_sp2013_primary_pct"] = 100 * best_bar_df["best_sp2013_primary"]

                fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)

                axes[0, 0].bar(
                    dataset_df["data_condition"],
                    dataset_df["rows_millions"],
                    color=["#1D3557", "#457B9D"],
                    width=0.65,
                )
                axes[0, 0].set_title("Filtered Dataset Size by Data Condition")
                axes[0, 0].set_ylabel("Rows (millions)")
                axes[0, 0].set_xlabel("")
                for idx, row in dataset_df.iterrows():
                    axes[0, 0].text(
                        idx,
                        row["rows_millions"] + 0.15,
                        f"{row['rows_millions']:.2f}M",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                    )

                cmap = mcolors.ListedColormap([OUTCOME_COLORS[k] for k in [-3, -2, -1, 1, 2]])
                bounds = [-3.5, -2.5, -1.5, 0.0, 1.5, 2.5]
                norm = mcolors.BoundaryNorm(bounds, cmap.N)
                sns.heatmap(
                    outcome_code_matrix,
                    ax=axes[0, 1],
                    cmap=cmap,
                    norm=norm,
                    cbar=False,
                    linewidths=2,
                    linecolor="white",
                    annot=outcome_annot,
                    fmt="",
                    annot_kws={"fontsize": 10},
                )
                axes[0, 1].set_title("Latest Restart Matrix Outcome")
                axes[0, 1].set_xlabel("Prompt condition")
                axes[0, 1].set_ylabel("Data condition")

                sns.heatmap(
                    artifact_plot_df,
                    ax=axes[1, 0],
                    cmap=sns.color_palette(["#F1F3F5", "#2A9D8F"], as_cmap=True),
                    cbar=False,
                    linewidths=1.5,
                    linecolor="white",
                    annot=True,
                    fmt="d",
                    annot_kws={"fontsize": 11},
                )
                axes[1, 0].set_title("Artifact Presence by Latest Run")
                axes[1, 0].set_xlabel("Artifact")
                axes[1, 0].set_ylabel("Run condition")

                plot_df = best_bar_df.copy()
                plot_df["bar_color"] = plot_df["data_condition"].map(
                    {
                        "correct_exec": "#2A9D8F",
                        "correct_exec_and_answer": "#E76F51",
                    }
                ).fillna("#ADB5BD")
                axes[1, 1].bar(
                    plot_df["condition_label"],
                    plot_df["best_sp2013_primary_pct"].fillna(0.0),
                    color=plot_df["bar_color"],
                    width=0.65,
                    alpha=0.9,
                )
                axes[1, 1].set_title("Best SP2013 Primary by Latest Run")
                axes[1, 1].set_ylabel("Best SP2013 primary (%)")
                axes[1, 1].set_xlabel("")
                axes[1, 1].tick_params(axis="x", rotation=20)
                for idx, row in plot_df.reset_index(drop=True).iterrows():
                    if pd.notna(row["best_sp2013_primary_pct"]):
                        label = f"{row['best_sp2013_primary_pct']:.2f}"
                    else:
                        label = "fail"
                    axes[1, 1].text(idx, row["best_sp2013_primary_pct"] if pd.notna(row["best_sp2013_primary_pct"]) else 0.3, label, ha="center", va="bottom", fontsize=10)

                png_path = FIGURE_DIR / "135m_clean_restart_matrix_overview.png"
                pdf_path = FIGURE_DIR / "135m_clean_restart_matrix_overview.pdf"
                fig.savefig(png_path, bbox_inches="tight")
                fig.savefig(pdf_path, bbox_inches="tight")
                plt.show()

                print(f"Saved figure to: {png_path}")
                print(f"Saved figure to: {pdf_path}")
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                metric_records = []
                profile_records = []
                summary_records = []

                for _, row in manifest.iterrows():
                    out_dir = row["out_dir"]
                    summary = load_json(out_dir / "summary_metrics.json")
                    best_summary = load_json(out_dir / "best_checkpoint_summary.json")

                    if summary:
                        summary_records.append(
                            {
                                "data_condition": row["data_condition"],
                                "prompt_condition": row["prompt_condition"],
                                "condition_label": row["condition_label"],
                                "best_epoch": summary.get("best_epoch"),
                                "best_eval_tag": summary.get("best_eval_tag"),
                                "best_sp2013_primary_pct": 100 * float(summary.get("best_sp2013_primary")),
                                "best_sp2013_primary_name": summary.get("best_sp2013_primary_name"),
                                "best_val_loss": summary.get("best_val_loss"),
                                "train_rows": summary.get("train_rows"),
                                "val_rows": summary.get("val_rows"),
                            }
                        )

                    for metric_path in sorted(out_dir.glob("sp2013_metrics_epoch_*.json")):
                        epoch = int(metric_path.stem.split("_")[-1])
                        payload = load_json(metric_path)
                        metric_records.append(
                            {
                                "data_condition": row["data_condition"],
                                "prompt_condition": row["prompt_condition"],
                                "condition_label": row["condition_label"],
                                "epoch": epoch,
                                "primary_name": payload.get("primary_name"),
                                "primary_value_pct": 100 * float(payload.get("primary_value")),
                                "teacher_acc_all_pct": 100 * float(payload.get("teacher_acc_all")),
                                "teacher_coverage_pct": 100 * float(payload.get("teacher_coverage")),
                                "true_acc_all_pct": 100 * float(payload.get("true_acc_all")),
                            }
                        )

                        csv_path = out_dir / f"sp2013_epoch_{epoch:02d}.csv"
                        if csv_path.exists():
                            profile_df = pd.read_csv(csv_path, usecols=["prob", "is_correct_true", "is_correct_target"])
                            profile_df[["denom", "operation"]] = profile_df["prob"].apply(
                                lambda prob: pd.Series(parse_prob(prob))
                            )
                            grouped = (
                                profile_df.groupby(["denom", "operation"])[["is_correct_true", "is_correct_target"]]
                                .mean()
                                .mul(100)
                                .reset_index()
                            )
                            for denom, operation in PROFILE_GRID:
                                profile_row = grouped.loc[
                                    (grouped["denom"] == denom) & (grouped["operation"] == operation)
                                ]
                                if profile_row.empty:
                                    continue
                                profile_records.append(
                                    {
                                        "condition_label": row["condition_label"],
                                        "epoch": epoch,
                                        "denom": denom,
                                        "operation": operation,
                                        "true_accuracy_pct": float(profile_row["is_correct_true"].iloc[0]),
                                        "target_accuracy_pct": float(profile_row["is_correct_target"].iloc[0]),
                                    }
                                )

                summary_df = pd.DataFrame(summary_records)
                metric_df = pd.DataFrame(metric_records)
                profile_long_df = pd.DataFrame(profile_records)
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
                    [reference_lookup.loc[(denom, operation), "Human (%)"] for denom, operation in PROFILE_GRID]
                )
                uma_curve = np.array(
                    [reference_lookup.loc[(denom, operation), "UMA (%)"] for denom, operation in PROFILE_GRID]
                )
                uma_mag_human_pct = float(np.mean(np.abs(uma_curve - human_curve)))


                def annotate_profile_lines(axis, label_rows: list[dict[str, object]], x_anchor: float) -> None:
                    if not label_rows:
                        return

                    ordered_rows = sorted(label_rows, key=lambda row: row["y"])
                    min_gap = 4.8
                    y_positions = []
                    for row in ordered_rows:
                        y_value = float(row["y"])
                        if y_positions:
                            y_value = max(y_value, y_positions[-1] + min_gap)
                        y_positions.append(y_value)

                    upper_bound = 101.0
                    if y_positions[-1] > upper_bound:
                        shift = y_positions[-1] - upper_bound
                        y_positions = [y - shift for y in y_positions]

                    lower_bound = 4.0
                    if y_positions[0] < lower_bound:
                        shift = lower_bound - y_positions[0]
                        y_positions = [y + shift for y in y_positions]

                    for idx in range(len(y_positions) - 2, -1, -1):
                        y_positions[idx] = min(y_positions[idx], y_positions[idx + 1] - min_gap)

                    for row, y_text in zip(ordered_rows, y_positions):
                        axis.plot(
                            [len(PROFILE_GRID) - 1, x_anchor - 0.05],
                            [float(row["y"]), y_text],
                            color=row["color"],
                            linewidth=1.0,
                            alpha=0.7,
                        )
                        axis.text(
                            x_anchor,
                            y_text,
                            row["text"],
                            color=row["color"],
                            fontsize=9.5,
                            va="center",
                            ha="left",
                        )

                if summary_df.empty:
                    display(
                        Markdown(
                            "### Metric Harvest\\n"
                            "No successful SP2013 metric files were found in the latest matrix."
                        )
                    )
                else:
                    display(Markdown("### Best Available Successful Runs"))
                    display(summary_df.round(4))

                    display(Markdown("### Epoch-Level SP2013 Metrics for Successful Cells"))
                    display(metric_df.round(4))

                    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True, sharex=True)
                    palette = {
                        "correct_exec | params_always": "#2A9D8F",
                        "correct_exec_and_answer | params_always": "#E76F51",
                    }
                    for condition_label, group in metric_df.groupby("condition_label"):
                        group = group.sort_values("epoch")
                        color = palette.get(condition_label, "#457B9D")
                        axes[0].plot(
                            group["epoch"],
                            group["primary_value_pct"],
                            marker="o",
                            linewidth=2.2,
                            markersize=6,
                            color=color,
                            label=condition_label,
                        )
                        axes[1].plot(
                            group["epoch"],
                            group["true_acc_all_pct"],
                            marker="o",
                            linewidth=2.2,
                            markersize=6,
                            color=color,
                            label=condition_label,
                        )

                    axes[0].set_title("SP2013 Primary by Epoch")
                    axes[0].set_xlabel("Epoch")
                    axes[0].set_ylabel("Accuracy (%)")
                    axes[0].grid(True, alpha=0.25, linewidth=0.8)
                    axes[0].set_xticks(sorted(metric_df["epoch"].unique()))

                    axes[1].set_title("SP2013 True Accuracy by Epoch")
                    axes[1].set_xlabel("Epoch")
                    axes[1].set_ylabel("Accuracy (%)")
                    axes[1].grid(True, alpha=0.25, linewidth=0.8)
                    axes[1].set_xticks(sorted(metric_df["epoch"].unique()))
                    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Successful cell")

                    png_path = FIGURE_DIR / "135m_clean_restart_matrix_epoch_metrics.png"
                    pdf_path = FIGURE_DIR / "135m_clean_restart_matrix_epoch_metrics.pdf"
                    fig.savefig(png_path, bbox_inches="tight")
                    fig.savefig(pdf_path, bbox_inches="tight")
                    plt.show()
                    print(f"Saved figure to: {png_path}")
                    print(f"Saved figure to: {pdf_path}")

                    if not profile_long_df.empty:
                        display(
                            Markdown(
                                f"### Reference Profiles\\n"
                                f"Human and UMA are plotted against the 8-cell true-answer profile. "
                                f"UMA has fixed `MAG_h = {uma_mag_human_pct:.2f}` percentage points."
                            )
                        )

                        fig2, axes2 = plt.subplots(1, len(sorted(profile_long_df["epoch"].unique())), figsize=(10.4 * len(sorted(profile_long_df["epoch"].unique())), 5.8), constrained_layout=True, sharey=True)
                        axes2 = np.atleast_1d(axes2)
                        x = np.arange(len(PROFILE_GRID))
                        line_name_map = {
                            "correct_exec | params_always": "correct_exec",
                            "correct_exec_and_answer | params_always": "correct_exec+answer",
                        }
                        uma_color = "#E9C46A"
                        for axis, epoch in zip(axes2, sorted(profile_long_df["epoch"].unique())):
                            subset = profile_long_df.loc[profile_long_df["epoch"] == epoch]
                            label_rows = []
                            for condition_label, group in subset.groupby("condition_label"):
                                ordered = (
                                    group.set_index(["denom", "operation"])
                                    .loc[PROFILE_GRID, "true_accuracy_pct"]
                                    .to_numpy()
                                )
                                mag_h = float(np.mean(np.abs(ordered - human_curve)))
                                color = palette.get(condition_label, "#457B9D")
                                axis.plot(
                                    x,
                                    ordered,
                                    marker="o",
                                    linewidth=2.0,
                                    markersize=5,
                                    color=color,
                                )
                                label_rows.append(
                                    {
                                        "y": float(ordered[-1]),
                                        "color": color,
                                        "text": f"{line_name_map.get(condition_label, condition_label)} | MAG_h={mag_h:.1f}",
                                    }
                                )

                            axis.plot(
                                x,
                                human_curve,
                                color="black",
                                marker="o",
                                linewidth=2.3,
                                markersize=5.2,
                            )
                            axis.plot(
                                x,
                                uma_curve,
                                color=uma_color,
                                marker="D",
                                linewidth=2.1,
                                linestyle="--",
                                markersize=4.8,
                            )
                            label_rows.extend(
                                [
                                    {
                                        "y": float(human_curve[-1]),
                                        "color": "black",
                                        "text": "Human | MAG_h=0.0",
                                    },
                                    {
                                        "y": float(uma_curve[-1]),
                                        "color": uma_color,
                                        "text": f"UMA | MAG_h={uma_mag_human_pct:.1f}",
                                    },
                                ]
                            )
                            annotate_profile_lines(axis, label_rows, x_anchor=len(PROFILE_GRID) - 0.25 + 1.0)
                            axis.set_title(f"Epoch {epoch}")
                            axis.set_xticks(x)
                            axis.set_xticklabels(PROFILE_TICKLABELS)
                            axis.set_ylim(0, 105)
                            axis.set_xlim(-0.4, len(PROFILE_GRID) + 1.45)
                            axis.set_xlabel("SP2013 cell")
                            axis.set_ylabel("True-answer accuracy (%)")
                            axis.grid(True, axis="y", alpha=0.25, linewidth=0.8)

                        fig2.suptitle("Successful Params-Always Cells: SP2013 True-Answer Profile by Epoch", fontsize=17)

                        png_path = FIGURE_DIR / "135m_clean_restart_matrix_profile_curves.png"
                        pdf_path = FIGURE_DIR / "135m_clean_restart_matrix_profile_curves.pdf"
                        fig2.savefig(png_path, bbox_inches="tight")
                        fig2.savefig(pdf_path, bbox_inches="tight")
                        plt.show()
                        print(f"Saved figure to: {png_path}")
                        print(f"Saved figure to: {pdf_path}")
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                r"""
                ## Reading Guide

                **Finding.** The original `20260411_201016` matrix should no longer be read as the final result. Later
                reruns changed the answer materially.

                **Finding.** The latest rerun `20260412_211956` produced a **partial success**. Both
                `params_always` cells completed two training epochs and wrote SP2013 metrics. Both `no_params` cells
                still failed during strict SP2013-only validation because the validation rows did not contain the
                required student parameter columns `g/d/rt_mu/ice`.

                **Finding.** Among the successful cells, `correct_exec + params_always` achieved the stronger
                tuple-conditioned SP2013 primary score (`49.69%` best), while
                `correct_exec_and_answer + params_always` remained lower on the same target (`41.42%` best) despite
                much higher mathematically true-answer accuracy.

                **Interpretation.** The epochwise profile figure now overlays Human and UMA true-answer references.
                Each line label reports
                \[
                \mathrm{MAG}_h = \frac{1}{8}\sum_{(d,o)\in\mathcal{C}}
                \left|a^{(\mathrm{true})}_{r,e,d,o} - a_{\mathrm{Human}}(d,o)\right|,
                \]
                so lower `MAG_h` means the student profile is closer to the Human SP2013 profile.

                **Implication.** The clean filtered-data experiments were rerun, and the current evidence is not “the
                matrix never trained.” The more precise statement is: the `params_always` half of the matrix trained,
                while the `no_params` half is still blocked by validation-layout assumptions.
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
