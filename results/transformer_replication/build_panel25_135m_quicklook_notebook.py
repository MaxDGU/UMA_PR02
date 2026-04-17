#!/usr/bin/env python3
"""
Build a lightweight panel-25 comparison notebook.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results" / "transformer_replication" / "analyze_panel25_135m_quicklook.ipynb"

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

    nb["cells"] = [
        md(
            r"""
            # Panel-25 135M Quicklook

            **Question.** How do the four `SmolLM2-135M` panel-25 runs compare at a high level when we vary
            representation (`default` vs `surface`) and initialization (`pretrained` vs `scratch`)?

            **Setup.** We load the latest run in each condition from
            `results/transformer_replication/smol2_135m_panel25_even_g_rt5_ice50_seed1_*`.
            Let \(r\) index a run. We report:
            \[
            L_r^{\mathrm{val}}, \qquad
            A_r^{\mathrm{resp,all}} = \frac{c_r^{\mathrm{resp}}}{N}, \qquad
            A_r^{\mathrm{true}} = \frac{c_r^{\mathrm{true}}}{N}, \qquad
            C_r = \frac{n_r^{\mathrm{parseable}}}{N},
            \]
            where \(N=64\) is the saved `id_val` slice size, \(c_r^{\mathrm{resp}}\) counts agreement with the UMA
            response target, \(c_r^{\mathrm{true}}\) counts agreement with the mathematically correct answer, and
            \(n_r^{\mathrm{parseable}}\) counts parseable predictions.

            **Objective.** This notebook is intentionally small and easy to open. It gives a compact summary first,
            then one lightweight subgroup table over `ED/UD × operation`.
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

            plt.rcParams.update(
                {
                    "figure.dpi": 150,
                    "savefig.dpi": 250,
                    "font.size": 12,
                    "axes.titlesize": 15,
                    "axes.labelsize": 12,
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
            PROB_RE = re.compile(r"^(\\d+)/(\\d+)([+\\-*:])(\\d+)/(\\d+)$")

            STYLE = {
                ("default", "pretrained"): {"label": "Default | Pretrained", "color": "#1f77b4"},
                ("surface", "pretrained"): {"label": "Surface | Pretrained", "color": "#ff7f0e"},
                ("default", "scratch"): {"label": "Default | Scratch", "color": "#2ca02c"},
                ("surface", "scratch"): {"label": "Surface | Scratch", "color": "#d62728"},
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
                    raise ValueError(f"Could not parse problem: {prob}")
                left_den = int(match.group(2))
                right_den = int(match.group(5))
                denom = "ED" if left_den == right_den else "UD"
                operation = {"+": "add", "-": "sub", "*": "mul", ":": "div"}[match.group(3)]
                return denom, operation


            PROJECT_ROOT = resolve_project_root()
            RUN_DIR = PROJECT_ROOT / "results" / "transformer_replication"

            candidates = []
            for path in sorted(RUN_DIR.glob("smol2_135m_panel25_even_g_rt5_ice50_seed1_*")):
                if not path.is_dir():
                    continue
                match = RUN_RE.fullmatch(path.name)
                if match is None:
                    continue
                candidates.append(
                    {
                        "variant": match.group("variant"),
                        "init": match.group("init"),
                        "epochs": int(match.group("epochs")),
                        "timestamp": match.group("timestamp"),
                        "run_dir": path,
                    }
                )

            catalog_df = pd.DataFrame(candidates)
            if catalog_df.empty:
                raise FileNotFoundError("No 135M panel-25 run directories found.")

            selected_df = (
                catalog_df.sort_values(["variant", "init", "timestamp"])
                .groupby(["variant", "init"], as_index=False)
                .tail(1)
                .reset_index(drop=True)
            )

            summary_rows = []
            profile_rows = []

            for record in selected_df.to_dict("records"):
                variant = record["variant"]
                init = record["init"]
                label = STYLE[(variant, init)]["label"]
                out_dir = Path(record["run_dir"])

                best_summary = load_json(out_dir / "best_checkpoint_summary.json")
                summary_metrics = load_json(out_dir / "summary_metrics.json")
                best_metrics = load_json(out_dir / "best_id_val_metrics.json")
                train_args = load_json(out_dir / "best" / "train_args.json")
                best_df = pd.read_csv(out_dir / "best_id_val.csv")
                best_df[["denom", "operation"]] = best_df["prob"].apply(lambda prob: pd.Series(parse_prob(prob)))

                summary_rows.append(
                    {
                        "label": label,
                        "variant": variant,
                        "init": init,
                        "timestamp": record["timestamp"],
                        "run_dir": str(out_dir),
                        "data_csv": Path(train_args["data_csv"]).name,
                        "best_val_loss": float(summary_metrics["best_val_loss"]),
                        "best_test_loss": float(best_summary["test_loss"]),
                        "response_acc_scored_pct": 100 * float(best_metrics["acc_scored"]),
                        "response_acc_all_pct": 100 * float(best_metrics["acc_all"]),
                        "true_acc_pct": 100 * float(best_metrics["true_acc_all"]),
                        "coverage_pct": 100 * float(best_metrics["coverage"]),
                        "response_correct_n": int(best_metrics["n_correct"]),
                        "true_correct_n": int(best_metrics["true_n_correct"]),
                        "n_scored": int(best_metrics["n_scored"]),
                        "n_total": int(best_metrics["n_total"]),
                    }
                )

                grouped = (
                    best_df.groupby(["denom", "operation"], as_index=False)[["is_correct_response", "is_correct_true"]]
                    .mean()
                )
                grouped["label"] = label
                grouped["variant"] = variant
                grouped["init"] = init
                grouped["response_acc_pct"] = 100 * grouped["is_correct_response"]
                grouped["true_acc_pct"] = 100 * grouped["is_correct_true"]
                profile_rows.append(grouped)

            summary_df = pd.DataFrame(summary_rows).sort_values(["init", "variant"]).reset_index(drop=True)
            profile_df = pd.concat(profile_rows, ignore_index=True)

            display(Markdown("### Selected Runs"))
            display(selected_df[["variant", "init", "epochs", "timestamp", "run_dir"]])

            display(Markdown("### High-Level Summary"))
            display(
                summary_df[
                    [
                        "label",
                        "best_val_loss",
                        "best_test_loss",
                        "response_acc_scored_pct",
                        "response_acc_all_pct",
                        "true_acc_pct",
                        "coverage_pct",
                        "response_correct_n",
                        "true_correct_n",
                        "n_scored",
                        "n_total",
                    ]
                ].round(3)
            )
            """
        ),
        code(
            """
            order = [STYLE[(variant, init)]["label"] for init in ["pretrained", "scratch"] for variant in ["default", "surface"]]
            plot_df = summary_df.set_index("label").loc[order].reset_index()
            colors = [STYLE[(row["variant"], row["init"])]["color"] for _, row in plot_df.iterrows()]

            fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6), constrained_layout=True)
            metrics = [
                ("best_val_loss", "Best Val Loss", "Loss"),
                ("response_acc_all_pct", "Response Accuracy", "Accuracy (%)"),
                ("true_acc_pct", "True-Answer Accuracy", "Accuracy (%)"),
                ("coverage_pct", "Parseable Coverage", "Coverage (%)"),
            ]

            for ax, (metric, title, ylabel) in zip(axes, metrics):
                ax.bar(plot_df["label"], plot_df[metric], color=colors, alpha=0.9)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
                ax.tick_params(axis="x", rotation=28)
                if "Accuracy" in ylabel or "Coverage" in ylabel:
                    ax.set_ylim(0, 100)

            plt.show()

            display(Markdown("### Best-Checkpoint Accuracy by `ED/UD x operation`"))
            display(
                profile_df.pivot_table(
                    index=["denom", "operation"],
                    columns="label",
                    values=["response_acc_pct", "true_acc_pct"],
                ).round(2)
            )

            surface_effect = (
                summary_df.pivot(index="init", columns="variant", values=["best_val_loss", "response_acc_all_pct", "true_acc_pct", "coverage_pct"])
            )
            scratch_effect = (
                summary_df.pivot(index="variant", columns="init", values=["best_val_loss", "response_acc_all_pct", "true_acc_pct", "coverage_pct"])
            )

            display(Markdown("### Surface Minus Default"))
            display(
                pd.DataFrame(
                    {
                        "delta_val_loss": surface_effect["best_val_loss"]["surface"] - surface_effect["best_val_loss"]["default"],
                        "delta_response_acc_all_pct": surface_effect["response_acc_all_pct"]["surface"] - surface_effect["response_acc_all_pct"]["default"],
                        "delta_true_acc_pct": surface_effect["true_acc_pct"]["surface"] - surface_effect["true_acc_pct"]["default"],
                        "delta_coverage_pct": surface_effect["coverage_pct"]["surface"] - surface_effect["coverage_pct"]["default"],
                    }
                ).round(3)
            )

            display(Markdown("### Scratch Minus Pretrained"))
            display(
                pd.DataFrame(
                    {
                        "delta_val_loss": scratch_effect["best_val_loss"]["scratch"] - scratch_effect["best_val_loss"]["pretrained"],
                        "delta_response_acc_all_pct": scratch_effect["response_acc_all_pct"]["scratch"] - scratch_effect["response_acc_all_pct"]["pretrained"],
                        "delta_true_acc_pct": scratch_effect["true_acc_pct"]["scratch"] - scratch_effect["true_acc_pct"]["pretrained"],
                        "delta_coverage_pct": scratch_effect["coverage_pct"]["scratch"] - scratch_effect["coverage_pct"]["pretrained"],
                    }
                ).round(3)
            )
            """
        ),
    ]

    with out.open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)

    print(f"Wrote lightweight notebook to {out}")


if __name__ == "__main__":
    main()
