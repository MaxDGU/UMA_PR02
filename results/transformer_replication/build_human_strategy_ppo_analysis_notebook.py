#!/usr/bin/env python3
"""
Build the human strategy PPO analysis notebook with old-vs-patched comparisons.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results" / "transformer_replication" / "analyze_human_strategy_ppo_results.ipynb"

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
                # Human Strategy PPO: Old vs Guarded Comparison

                **Question.** How did the guarded rerun change behavior relative to the original human strategy-matching PPO grid, and where are the gains concentrated across equal-denominator and unequal-denominator fraction problems?

                **Setup.** We compare two grids:

                - **Unguarded grid** with run timestamp `20260411_083827`
                - **Guarded grid** with run timestamp `20260411_174305`

                For each run $r$, denominator class $d \in \{\mathrm{ED}, \mathrm{UD}\}$, and operation
                $o \in \{\mathrm{add}, \mathrm{sub}, \mathrm{mul}, \mathrm{div}\}$, we study the out-of-distribution
                accuracy

                \[
                a_{r,d,o} = \frac{1}{N_{d,o}} \sum_{p \in \mathcal{P}_{d,o}} \mathrm{Acc}(r,p),
                \]

                where $\mathcal{P}_{d,o}$ is the set of unique `sp2013_human.csv` problems with denominator group
                $d$ and operation $o$, and $\mathrm{Acc}(r,p)$ is the rollout-level answer accuracy on problem $p$.

                **Primary comparison.** The notebook focuses on the `ED/UD x add/sub/mul/div` accuracy distribution.
                We also keep a compact collapse diagnostic because the guarded patch was designed to prevent empty-output
                reward hacking.

                **Important note.** The unguarded grid has no completed `360m + KL` or `360m + Wasserstein` runs,
                because those jobs were canceled before the guarded rerun. The notebook marks those comparisons as missing.
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
                import statistics
                from pathlib import Path

                import numpy as np
                import pandas as pd
                import matplotlib.pyplot as plt
                import matplotlib as mpl
                from IPython.display import display, Markdown

                plt.rcParams.update({
                    "figure.dpi": 160,
                    "savefig.dpi": 160,
                    "font.size": 11,
                    "axes.titlesize": 13,
                    "axes.labelsize": 12,
                    "legend.fontsize": 10,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                })

                MODELS = ["135m", "360m", "1p7b"]
                LOSSES = ["tv", "kl", "wasserstein"]
                LOSS_TITLE = {"tv": "TV", "kl": "KL", "wasserstein": "Wasserstein"}
                OP_ORDER = ["add", "sub", "mul", "div"]
                DENOM_ORDER = ["ED", "UD"]
                CONDITION_ORDER = [(d, o) for d in DENOM_ORDER for o in OP_ORDER]
                GRID_SPECS = [
                    {"tag": "unguarded", "label": "Unguarded", "run_ts": "20260411_083827"},
                    {"tag": "guarded", "label": "Guarded", "run_ts": "20260411_174305"},
                ]
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
                    raise FileNotFoundError("Could not locate project root from the current working directory.")


                try:
                    PROJECT_ROOT = find_project_root(Path.cwd())
                except FileNotFoundError:
                    PROJECT_ROOT = Path("/n/fs/cogai/cs1095/UMA_PR02")

                RESULTS_DIR = PROJECT_ROOT / "results" / "transformer_replication"
                SP2013_PATH = PROJECT_ROOT / "sp2013_human.csv"


                def safe_float(value):
                    if value is None:
                        return np.nan
                    try:
                        out = float(value)
                    except (TypeError, ValueError):
                        return np.nan
                    if math.isnan(out):
                        return np.nan
                    return out


                def op_name_from_value(value: str) -> str:
                    mapping = {
                        "+": "add",
                        "-": "sub",
                        "*": "mul",
                        ":": "div",
                        "add": "add",
                        "sub": "sub",
                        "mul": "mul",
                        "div": "div",
                    }
                    return mapping.get(str(value).strip(), str(value).strip())


                def denom_group_from_value(value: str) -> str:
                    mapping = {
                        "same-denom": "ED",
                        "diff-denom": "UD",
                        "ED": "ED",
                        "UD": "UD",
                    }
                    return mapping.get(str(value).strip(), str(value).strip())


                def load_json(path: Path) -> dict:
                    with path.open() as f:
                        return json.load(f)


                sp2013_df = pd.read_csv(SP2013_PATH)
                sp2013_meta = (
                    sp2013_df[["prob", "denom", "operation"]]
                    .copy()
                    .assign(
                        prob=lambda df: df["prob"].astype(str).str.strip(),
                        denom_group=lambda df: df["denom"].map(denom_group_from_value),
                        op_name=lambda df: df["operation"].map(op_name_from_value),
                    )
                    .drop_duplicates(subset=["prob"])
                    .sort_values("prob")
                    .reset_index(drop=True)
                )

                sp2013_condition_counts = (
                    sp2013_meta.groupby(["denom_group", "op_name"], sort=True)
                    .size()
                    .rename("problem_count")
                    .reset_index()
                )


                def load_run_record(grid: dict, model: str, loss: str) -> dict:
                    run_dir = RESULTS_DIR / f"human_strategy_ppo_{model}_{loss}_{grid['run_ts']}"
                    run_summary_path = run_dir / "run_summary.json"
                    eval_summary_path = run_dir / "final_eval" / "eval_summary.json"
                    id_rollouts_path = run_dir / "final_eval" / "id_rollouts.csv"

                    record = {
                        "grid_tag": grid["tag"],
                        "grid_label": grid["label"],
                        "run_ts": grid["run_ts"],
                        "model": model,
                        "loss": loss,
                        "run_label": f"{model} | {LOSS_TITLE[loss]}",
                        "run_dir": str(run_dir),
                        "has_run_dir": run_dir.exists(),
                        "has_run_summary": run_summary_path.exists(),
                        "has_final_eval": eval_summary_path.exists(),
                    }

                    if run_summary_path.exists():
                        run_summary = load_json(run_summary_path)
                        best = run_summary.get("best_id_summary", {})
                        final_id = run_summary.get("final_id_summary", {})
                        final_ood = run_summary.get("final_ood_summary", {})
                        record.update(
                            {
                                "best_update": run_summary.get("best_update"),
                                "best_id_accuracy_all": safe_float(best.get("accuracy_all")),
                                "best_id_valid_rollout_fraction": safe_float(best.get("valid_rollout_fraction", best.get("parseable_coverage"))),
                                "best_id_selection_metric": safe_float(best.get("selection_metric", best.get("mean_" + ("wasserstein" if loss == "wasserstein" else loss) + "_distance"))),
                                "final_id_accuracy_all": safe_float(final_id.get("accuracy_all")),
                                "final_id_valid_rollout_fraction": safe_float(final_id.get("valid_rollout_fraction", final_id.get("parseable_coverage"))),
                                "final_id_selection_metric": safe_float(final_id.get("selection_metric", final_id.get("mean_" + ("wasserstein" if loss == "wasserstein" else loss) + "_distance"))),
                                "final_ood_accuracy_all": safe_float(final_ood.get("accuracy_all")),
                            }
                        )
                    else:
                        record.update(
                            {
                                "best_update": np.nan,
                                "best_id_accuracy_all": np.nan,
                                "best_id_valid_rollout_fraction": np.nan,
                                "best_id_selection_metric": np.nan,
                                "final_id_accuracy_all": np.nan,
                                "final_id_valid_rollout_fraction": np.nan,
                                "final_id_selection_metric": np.nan,
                                "final_ood_accuracy_all": np.nan,
                            }
                        )

                    if id_rollouts_path.exists():
                        id_rollouts = pd.read_csv(id_rollouts_path, usecols=["response_text"])
                        lengths = id_rollouts["response_text"].fillna("").str.len()
                        record["final_id_mean_response_len"] = float(lengths.mean())
                        record["final_id_frac_empty"] = float((lengths == 0).mean())
                    else:
                        record["final_id_mean_response_len"] = np.nan
                        record["final_id_frac_empty"] = np.nan

                    return record


                def load_ood_per_problem(grid: dict, model: str, loss: str) -> pd.DataFrame:
                    path = RESULTS_DIR / f"human_strategy_ppo_{model}_{loss}_{grid['run_ts']}" / "final_eval" / "ood_per_problem.csv"
                    if not path.exists():
                        return pd.DataFrame()
                    df = pd.read_csv(path)
                    df["prob"] = df["prob"].astype(str).str.strip()
                    df["op_name"] = df["op"].map(op_name_from_value)
                    df = df.merge(sp2013_meta[["prob", "denom_group", "op_name"]], on="prob", how="left", suffixes=("", "_meta"))
                    df["op_name"] = df["op_name_meta"].fillna(df["op_name"])
                    df["grid_tag"] = grid["tag"]
                    df["grid_label"] = grid["label"]
                    df["run_ts"] = grid["run_ts"]
                    df["model"] = model
                    df["loss"] = loss
                    df["run_label"] = f"{model} | {LOSS_TITLE[loss]}"
                    df["condition"] = df["denom_group"].astype(str) + " | " + df["op_name"].astype(str)
                    return df.drop(columns=["op_name_meta"])


                run_records = []
                ood_problem_frames = []
                for grid in GRID_SPECS:
                    for model in MODELS:
                        for loss in LOSSES:
                            run_records.append(load_run_record(grid, model, loss))
                            ood_df = load_ood_per_problem(grid, model, loss)
                            if not ood_df.empty:
                                ood_problem_frames.append(ood_df)

                runs_df = pd.DataFrame(run_records)
                ood_problem_df = pd.concat(ood_problem_frames, ignore_index=True) if ood_problem_frames else pd.DataFrame()

                if not ood_problem_df.empty:
                    condition_df = (
                        ood_problem_df.groupby(["grid_tag", "grid_label", "model", "loss", "run_label", "denom_group", "op_name"], as_index=False)
                        .agg(
                            accuracy_all=("accuracy_all", "mean"),
                            accuracy_parseable=("accuracy_parseable", "mean"),
                            parseable_coverage=("parseable_coverage", "mean"),
                            problem_count=("prob", "nunique"),
                        )
                    )
                else:
                    condition_df = pd.DataFrame()

                runs_df["model"] = pd.Categorical(runs_df["model"], categories=MODELS, ordered=True)
                runs_df["loss"] = pd.Categorical(runs_df["loss"], categories=LOSSES, ordered=True)
                runs_df = runs_df.sort_values(["grid_tag", "model", "loss"]).reset_index(drop=True)

                display(Markdown(f"Detected **{len(runs_df)}** run slots across the two grids."))
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## SP2013 OOD Structure

                The comparison target is the deduplicated `sp2013_human.csv` problem set. The next table confirms how
                those 16 problems split across equal-denominator and unequal-denominator conditions.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                display(sp2013_condition_counts.sort_values(["denom_group", "op_name"]).reset_index(drop=True))
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Run Snapshot

                This table keeps the high-level status concise: final OOD accuracy, final ID accuracy, guarded-validity
                fraction, and a simple empty-response diagnostic based on the saved final ID rollouts.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                snapshot_cols = [
                    "grid_label",
                    "model",
                    "loss",
                    "has_final_eval",
                    "best_update",
                    "final_ood_accuracy_all",
                    "final_id_accuracy_all",
                    "final_id_valid_rollout_fraction",
                    "final_id_frac_empty",
                    "final_id_mean_response_len",
                ]
                snapshot_df = runs_df[snapshot_cols].copy()
                for col in ["final_ood_accuracy_all", "final_id_accuracy_all", "final_id_valid_rollout_fraction", "final_id_frac_empty", "final_id_mean_response_len"]:
                    snapshot_df[col] = pd.to_numeric(snapshot_df[col], errors="coerce")
                snapshot_df = snapshot_df.round(4)
                display(snapshot_df)
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Overall OOD Accuracy: Unguarded vs Guarded

                The bars compare final OOD answer accuracy for matched runs. This is the coarsest summary; the next
                sections unpack where the gains land within `ED/UD x add/sub/mul/div`.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 4.8), constrained_layout=True)
                width = 0.34
                for ax, model in zip(axes, MODELS):
                    sub = runs_df[runs_df["model"] == model]
                    x = np.arange(len(LOSSES))
                    old_vals = (
                        sub[sub["grid_tag"] == "unguarded"]
                        .set_index("loss")
                        .reindex(LOSSES)["final_ood_accuracy_all"]
                        .astype(float)
                        .to_numpy()
                    )
                    new_vals = (
                        sub[sub["grid_tag"] == "guarded"]
                        .set_index("loss")
                        .reindex(LOSSES)["final_ood_accuracy_all"]
                        .astype(float)
                        .to_numpy()
                    )

                    ax.bar(x - width / 2, old_vals, width=width, color="#bab0ac", label="Unguarded")
                    ax.bar(x + width / 2, new_vals, width=width, color="#4c78a8", label="Guarded")

                    for xpos, val in zip(x - width / 2, old_vals):
                        label = "NA" if np.isnan(val) else f"{val:.3f}"
                        ax.text(xpos, 0.002 if np.isnan(val) else val + 0.002, label, ha="center", va="bottom", fontsize=9)
                    for xpos, val in zip(x + width / 2, new_vals):
                        label = "NA" if np.isnan(val) else f"{val:.3f}"
                        ax.text(xpos, 0.002 if np.isnan(val) else val + 0.002, label, ha="center", va="bottom", fontsize=9)

                    ax.set_title(model)
                    ax.set_xticks(x)
                    ax.set_xticklabels([LOSS_TITLE[loss] for loss in LOSSES])
                    ax.set_ylabel("Final OOD accuracy")
                    ax.set_ylim(0.0, max(0.16, np.nanmax(new_vals) + 0.04))
                    ax.grid(axis="y", alpha=0.25)

                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
                plt.show()
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Patched Grid: OOD Accuracy over `ED/UD x add/sub/mul/div`

                Each heatmap summarizes a final patched run. Rows are denominator class and columns are operation.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                def condition_matrix(df: pd.DataFrame) -> pd.DataFrame:
                    matrix = df.pivot(index="denom_group", columns="op_name", values="accuracy_all")
                    return matrix.reindex(index=DENOM_ORDER, columns=OP_ORDER)


                patched_condition = condition_df[condition_df["grid_tag"] == "guarded"].copy()
                vmax = float(patched_condition["accuracy_all"].max()) if not patched_condition.empty else 1.0

                fig, axes = plt.subplots(len(MODELS), len(LOSSES), figsize=(14.5, 9.0), constrained_layout=True)
                for row_idx, model in enumerate(MODELS):
                    for col_idx, loss in enumerate(LOSSES):
                        ax = axes[row_idx, col_idx]
                        sub = patched_condition[(patched_condition["model"] == model) & (patched_condition["loss"] == loss)]
                        matrix = condition_matrix(sub)
                        im = ax.imshow(matrix.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect="auto")
                        ax.set_title(f"{model} | {LOSS_TITLE[loss]}")
                        ax.set_xticks(np.arange(len(OP_ORDER)))
                        ax.set_xticklabels(OP_ORDER)
                        ax.set_yticks(np.arange(len(DENOM_ORDER)))
                        ax.set_yticklabels(DENOM_ORDER)
                        for i in range(len(DENOM_ORDER)):
                            for j in range(len(OP_ORDER)):
                                value = matrix.iloc[i, j]
                                text = "NA" if pd.isna(value) else f"{value:.3f}"
                                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)

                cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
                cbar.set_label("OOD accuracy")
                plt.show()
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Guarded Minus Unguarded: Condition-Level OOD Accuracy Delta

                A positive cell means the guarded rerun improved OOD accuracy for that specific condition. Missing
                subplots correspond to runs that never finished in the unguarded grid.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                old_condition = condition_df[condition_df["grid_tag"] == "unguarded"].copy()
                merged_delta = patched_condition.merge(
                    old_condition,
                    on=["model", "loss", "denom_group", "op_name"],
                    how="left",
                    suffixes=("_guarded", "_unguarded"),
                )
                merged_delta["accuracy_delta"] = merged_delta["accuracy_all_guarded"] - merged_delta["accuracy_all_unguarded"]

                delta_vals = merged_delta["accuracy_delta"].dropna()
                delta_lim = float(max(abs(delta_vals.min()) if not delta_vals.empty else 0.0, abs(delta_vals.max()) if not delta_vals.empty else 0.0, 0.02))

                fig, axes = plt.subplots(len(MODELS), len(LOSSES), figsize=(14.5, 9.0), constrained_layout=True)
                for row_idx, model in enumerate(MODELS):
                    for col_idx, loss in enumerate(LOSSES):
                        ax = axes[row_idx, col_idx]
                        sub = merged_delta[(merged_delta["model"] == model) & (merged_delta["loss"] == loss)]
                        if sub["accuracy_all_unguarded"].notna().sum() == 0:
                            ax.axis("off")
                            ax.text(0.5, 0.5, "Old run unavailable", ha="center", va="center", fontsize=11)
                            ax.set_title(f"{model} | {LOSS_TITLE[loss]}")
                            continue
                        matrix = (
                            sub.pivot(index="denom_group", columns="op_name", values="accuracy_delta")
                            .reindex(index=DENOM_ORDER, columns=OP_ORDER)
                        )
                        im = ax.imshow(matrix.to_numpy(dtype=float), cmap="RdBu_r", vmin=-delta_lim, vmax=delta_lim, aspect="auto")
                        ax.set_title(f"{model} | {LOSS_TITLE[loss]}")
                        ax.set_xticks(np.arange(len(OP_ORDER)))
                        ax.set_xticklabels(OP_ORDER)
                        ax.set_yticks(np.arange(len(DENOM_ORDER)))
                        ax.set_yticklabels(DENOM_ORDER)
                        for i in range(len(DENOM_ORDER)):
                            for j in range(len(OP_ORDER)):
                                value = matrix.iloc[i, j]
                                text = "NA" if pd.isna(value) else f"{value:+.3f}"
                                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)

                cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
                cbar.set_label("Guarded - Unguarded OOD accuracy")
                plt.show()
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Best Patched Run: Condition Table

                The next table gives the main condition-level comparison for the strongest patched run, using the same
                `ED/UD x add/sub/mul/div` structure as the heatmaps.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                patched_best = runs_df[runs_df["grid_tag"] == "guarded"].sort_values("final_ood_accuracy_all", ascending=False).iloc[0]
                best_model = str(patched_best["model"])
                best_loss = str(patched_best["loss"])

                best_cmp = merged_delta[(merged_delta["model"] == best_model) & (merged_delta["loss"] == best_loss)].copy()
                best_cmp["condition"] = best_cmp["denom_group"] + " | " + best_cmp["op_name"]
                best_table = best_cmp[["condition", "accuracy_all_unguarded", "accuracy_all_guarded", "accuracy_delta"]].copy()
                best_table = best_table.rename(
                    columns={
                        "accuracy_all_unguarded": "unguarded_acc",
                        "accuracy_all_guarded": "guarded_acc",
                        "accuracy_delta": "delta",
                    }
                )
                best_table[["unguarded_acc", "guarded_acc", "delta"]] = best_table[["unguarded_acc", "guarded_acc", "delta"]].round(4)
                display(Markdown(f"**Best patched run:** `{best_model} + {LOSS_TITLE[best_loss]}`"))
                display(best_table.sort_values("condition").reset_index(drop=True))
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Collapse Diagnostic

                The guarded patch was designed to stop empty-output reward hacking. The table below compares the final
                empty-response fraction and mean final-ID rollout length between the two grids.
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                collapse_df = runs_df[
                    [
                        "grid_label",
                        "model",
                        "loss",
                        "final_id_frac_empty",
                        "final_id_mean_response_len",
                        "final_id_valid_rollout_fraction",
                    ]
                ].copy()
                collapse_df[["final_id_frac_empty", "final_id_mean_response_len", "final_id_valid_rollout_fraction"]] = (
                    collapse_df[["final_id_frac_empty", "final_id_mean_response_len", "final_id_valid_rollout_fraction"]].round(4)
                )
                display(collapse_df)
                """
            )
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Notes

                **Finding.** The main question is no longer whether the guard stops empty generations; it does. The
                remaining question is which loss family turns that stability into real OOD arithmetic accuracy.

                **Implication.** The `ED/UD x add/sub/mul/div` heatmaps are now the most informative lens for follow-up
                ablations, because they show whether gains are broad or concentrated in a few operation families.
                """
            )
        )
    )

    nb["cells"] = cells
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        nbf.write(nb, f)
    print(out)


if __name__ == "__main__":
    main()
