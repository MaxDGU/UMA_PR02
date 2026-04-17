#!/usr/bin/env python3
"""
Summarize strategy-classifier scaling runs.

Reads run directories produced by train_strategy_classifier.py and reports:
- effective train size
- validation/test metrics
- a simple log-linear fit of metric vs log10(train_size)
- the smallest train size that reaches a chosen fraction of the best test metric
- an optional publication-ready scaling plot
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize strategy-classifier scaling runs.")
    parser.add_argument(
        "--run_glob",
        type=str,
        required=True,
        help="Glob for run directories, e.g. 'results/transformer_replication/strategy_scale_*'",
    )
    parser.add_argument(
        "--metric",
        choices=["macro_f1", "accuracy", "balanced_accuracy"],
        default="macro_f1",
        help="Test metric to summarize and fit.",
    )
    parser.add_argument(
        "--target_frac_of_best",
        type=float,
        default=0.95,
        help="Pick the smallest train size reaching this fraction of the best test metric.",
    )
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument(
        "--plot_path",
        type=str,
        default="",
        help="Optional path for a scaling plot (png/pdf).",
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        default="Strategy Classifier Scaling",
        help="Title for the optional scaling plot.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(run_glob: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in sorted(glob.glob(run_glob)):
        run_dir = Path(raw).resolve()
        train_args_path = run_dir / "train_args.json"
        label_space_path = run_dir / "label_space.json"
        test_metrics_path = run_dir / "test_metrics.json"
        history_path = run_dir / "history.json"
        if not (train_args_path.exists() and label_space_path.exists() and test_metrics_path.exists() and history_path.exists()):
            continue

        train_args = load_json(train_args_path)
        label_space = load_json(label_space_path)
        test_metrics = load_json(test_metrics_path)
        history = load_json(history_path)
        best_epoch = history.get("best_epoch")
        hist_rows = history.get("history", [])
        best_row = None
        for row in hist_rows:
            if row.get("epoch") == best_epoch:
                best_row = row
                break

        rows.append(
            {
                "run_dir": str(run_dir),
                "effective_train_examples": int(label_space["effective_train_examples"]),
                "effective_validation_examples": int(label_space["effective_validation_examples"]),
                "effective_test_examples": int(label_space["effective_test_examples"]),
                "epochs": int(train_args["epochs"]),
                "batch_size": int(train_args["batch_size"]),
                "text_mode": str(train_args["text_mode"]),
                "split_mode": str(train_args["split_mode"]),
                "drop_answer_line": bool(train_args["drop_answer_line"]),
                "train_subsample_count": train_args.get("train_subsample_count"),
                "best_epoch": best_epoch,
                "val_macro_f1": None if best_row is None else float(best_row["macro_f1"]),
                "val_accuracy": None if best_row is None else float(best_row["accuracy"]),
                "val_balanced_accuracy": None if best_row is None else float(best_row["balanced_accuracy"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            }
        )
    rows.sort(key=lambda x: x["effective_train_examples"])
    return rows


def fit_log_linear(xs: List[int], ys: List[float]) -> Dict[str, float]:
    if len(xs) < 2:
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}
    logx = np.log10(np.asarray(xs, dtype=np.float64))
    y = np.asarray(ys, dtype=np.float64)
    slope, intercept = np.polyfit(logx, y, deg=1)
    pred = intercept + slope * logx
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {"intercept": float(intercept), "slope": float(slope), "r2": r2}


def format_train_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        value = n / 1_000
        return f"{value:.0f}k" if float(value).is_integer() else f"{value:.1f}k"
    return str(n)


def write_plot(
    rows: List[Dict[str, Any]],
    metric: str,
    target_metric: float,
    recommended_train_examples: int,
    plot_path: str,
    plot_title: str,
) -> None:
    import matplotlib.pyplot as plt

    xs = [int(row["effective_train_examples"]) for row in rows]
    macro_f1 = [float(row["test_macro_f1"]) for row in rows]
    accuracy = [float(row["test_accuracy"]) for row in rows]
    balanced_accuracy = [float(row["test_balanced_accuracy"]) for row in rows]

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=220, constrained_layout=True)

    series = [
        ("Macro-F1", macro_f1, "#1f77b4"),
        ("Accuracy", accuracy, "#2ca02c"),
        ("Balanced Acc.", balanced_accuracy, "#d62728"),
    ]
    for label, ys, color in series:
        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=6.5, color=color, label=label)

    ax.axhline(target_metric, color="#4d4d4d", linestyle="--", linewidth=1.3, label=f"{metric} target")
    ax.axvline(
        recommended_train_examples,
        color="#4d4d4d",
        linestyle=":",
        linewidth=1.5,
        label=f"recommended = {format_train_size(recommended_train_examples)}",
    )

    ax.set_xscale("log", base=10)
    ax.set_xlabel("Effective Train Examples")
    ax.set_ylabel("Held-out Test Metric")
    ax.set_title(plot_title)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, which="major", axis="both", alpha=0.25, linewidth=0.8)
    ax.set_xticks(xs, [format_train_size(x) for x in xs])
    ax.legend(loc="lower right", frameon=True)

    out_path = Path(plot_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = collect_runs(args.run_glob)
    if not rows:
        raise SystemExit(f"No complete runs matched: {args.run_glob}")

    metric_key = f"test_{args.metric}"
    xs = [int(row["effective_train_examples"]) for row in rows]
    ys = [float(row[metric_key]) for row in rows]
    best_metric = max(ys)
    target = float(args.target_frac_of_best) * best_metric

    recommended_row = None
    for row in rows:
        if float(row[metric_key]) >= target:
            recommended_row = row
            break
    if recommended_row is None:
        recommended_row = rows[-1]

    fit = fit_log_linear(xs, ys)

    summary = {
        "metric": args.metric,
        "n_runs": len(rows),
        "best_metric": best_metric,
        "target_frac_of_best": float(args.target_frac_of_best),
        "target_metric": target,
        "recommended_train_examples": int(recommended_row["effective_train_examples"]),
        "recommended_run_dir": str(recommended_row["run_dir"]),
        "log_linear_fit": fit,
        "rows": rows,
    }

    if args.out_csv:
        out_csv = Path(args.out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if args.out_json:
        out_json = Path(args.out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    else:
        print(json.dumps(summary, indent=2))

    if args.plot_path:
        write_plot(
            rows=rows,
            metric=args.metric,
            target_metric=target,
            recommended_train_examples=int(recommended_row["effective_train_examples"]),
            plot_path=args.plot_path,
            plot_title=args.plot_title,
        )


if __name__ == "__main__":
    main()
