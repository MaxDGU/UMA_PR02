#!/usr/bin/env python3
"""Importance-weight distilled-model traces using training-human answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACES = (
    ROOT / "data" / "human_ft" / "decimal_rerun" / "distill_generated_rollouts.csv.gz"
)
DEFAULT_HUMAN_TRAIN = ROOT / "data" / "human_ft" / "decimal_rerun" / "decimal_train_nlp.csv"
DEFAULT_OUTPUT = (
    ROOT / "data" / "human_ft" / "decimal_rerun" / "distill_selftrain_weighted_nlp.csv.gz"
)
UNPARSEABLE = "__UNPARSEABLE__"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_decimal_answer(value: object, precision: int = 4) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"} or "?" in text:
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            number = Decimal(numerator) / Decimal(denominator)
        else:
            number = Decimal(text)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None
    if not number.is_finite():
        return None
    quantum = Decimal(1).scaleb(-precision)
    number = number.quantize(quantum)
    normalized = format(number, "f").rstrip("0").rstrip(".")
    return normalized if normalized not in {"", "-0"} else "0"


def _distribution(values: pd.Series) -> dict[str, float]:
    counts = values.value_counts(dropna=False)
    total = int(counts.sum())
    if total == 0:
        return {}
    return {str(answer): float(count / total) for answer, count in counts.items()}


def prepare_distilled_rollouts(
    rollouts: pd.DataFrame,
    human: pd.DataFrame,
    *,
    expected_samples_per_problem: int = 1000,
) -> pd.DataFrame:
    """Validate generated rollouts and render them as completion-SFT examples."""
    required_rollout = {"problem", "parsed_answer", "model_response", "sample_idx"}
    required_human = {"prob", "resp"}
    if missing := sorted(required_rollout.difference(rollouts.columns)):
        raise ValueError(f"Distilled rollout CSV missing columns: {missing}")
    if missing := sorted(required_human.difference(human.columns)):
        raise ValueError(f"Human CSV missing columns: {missing}")

    human_problems = sorted(human["prob"].astype(str).unique().tolist())
    rollout_problems = sorted(rollouts["problem"].astype(str).unique().tolist())
    if len(human_problems) != 12 or rollout_problems != human_problems:
        raise ValueError("Human and distilled-rollout inputs must cover the same 12 problems")

    counts = rollouts.assign(problem=rollouts["problem"].astype(str)).groupby("problem").size()
    if not counts.eq(expected_samples_per_problem).all():
        raise ValueError(
            f"Expected {expected_samples_per_problem} distilled rollouts per problem; "
            f"observed {counts.to_dict()}"
        )
    if rollouts.assign(problem=rollouts["problem"].astype(str)).duplicated(
        ["problem", "sample_idx"]
    ).any():
        raise ValueError("Distilled rollouts contain duplicate (problem, sample_idx) rows")

    prepared = rollouts.copy()
    prepared["prob"] = prepared["problem"].astype(str)
    responses = prepared["model_response"].fillna("").astype(str)
    prepared["trainable_trace"] = ~responses.str.strip().eq("")
    prepared["instruction_nl"] = prepared["prob"].map(
        lambda problem: f"Solve this decimal problem: {problem}=?"
    )
    prepared["response_nl"] = responses
    normalized = prepared["parsed_answer"].map(normalize_decimal_answer)
    prepared["normalized_answer"] = normalized.fillna(UNPARSEABLE)
    prepared["answer"] = prepared["normalized_answer"]
    return prepared


def add_importance_weights(
    traces: pd.DataFrame,
    human: pd.DataFrame,
    *,
    smoothing_mass: float,
    max_weight: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add clipped p_human(answer|problem)/p_distill(answer|problem) weights."""
    if not 0 < smoothing_mass < 1:
        raise ValueError("smoothing_mass must be between zero and one")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    required_trace = {"prob", "normalized_answer", "instruction_nl", "response_nl"}
    required_human = {"prob", "resp"}
    if missing := sorted(required_trace.difference(traces.columns)):
        raise ValueError(f"Prepared trace frame missing columns: {missing}")
    if missing := sorted(required_human.difference(human.columns)):
        raise ValueError(f"Human CSV missing columns: {missing}")

    weighted = traces.copy().reset_index(drop=True)
    problems = sorted(human["prob"].astype(str).unique().tolist())
    if len(problems) != 12 or set(weighted["prob"].astype(str).unique()) != set(problems):
        raise ValueError("Human and distilled-trace inputs must cover the same 12 problems")

    raw_weights = np.ones(len(weighted), dtype=np.float64)
    diagnostics: dict[str, Any] = {"problems": {}}
    for problem in problems:
        human_problem = human[human["prob"].astype(str).eq(problem)]
        trace_mask = weighted["prob"].astype(str).eq(problem)
        trace_problem = weighted.loc[trace_mask]

        human_answers = human_problem["resp"].map(normalize_decimal_answer).fillna(UNPARSEABLE)
        human_dist = _distribution(human_answers)
        distill_dist = _distribution(trace_problem["normalized_answer"].astype(str))
        support = sorted(set(human_dist) | set(distill_dist))
        if not support:
            raise ValueError(f"No human or distilled answers for {problem}")

        uniform = 1.0 / len(support)
        answer_weights = {
            answer: min(
                max_weight,
                (
                    ((1 - smoothing_mass) * human_dist.get(answer, 0.0))
                    + smoothing_mass * uniform
                )
                / (
                    ((1 - smoothing_mass) * distill_dist.get(answer, 0.0))
                    + smoothing_mass * uniform
                ),
            )
            for answer in support
        }
        positions = np.flatnonzero(trace_mask.to_numpy())
        problem_weights = np.array(
            [answer_weights[str(answer)] for answer in trace_problem["normalized_answer"]],
            dtype=np.float64,
        )
        raw_weights[positions] = problem_weights
        human_covered_mass = sum(
            probability for answer, probability in human_dist.items() if answer in distill_dist
        )
        diagnostics["problems"][problem] = {
            "human_responses": int(len(human_problem)),
            "unparseable_human_responses": int(human_answers.eq(UNPARSEABLE).sum()),
            "generated_distilled_traces": int(len(trace_problem)),
            "retained_distilled_traces": int(trace_problem["trainable_trace"].sum()),
            "empty_distilled_traces": int((~trace_problem["trainable_trace"]).sum()),
            "human_answers": int(len(human_dist)),
            "distilled_answers": int(len(distill_dist)),
            "unparseable_distilled_traces": int(
                trace_problem["normalized_answer"].astype(str).eq(UNPARSEABLE).sum()
            ),
            "human_mass_covered_by_distilled_model": float(human_covered_mass),
            "raw_weight_min": float(problem_weights.min()),
            "raw_weight_mean": float(problem_weights.mean()),
            "raw_weight_max": float(problem_weights.max()),
        }

    retained_mask = weighted["trainable_trace"].to_numpy(dtype=bool)
    if not retained_mask.any():
        raise ValueError("All distilled traces have empty model responses")
    weighted = weighted.loc[retained_mask].reset_index(drop=True)
    retained_raw_weights = raw_weights[retained_mask]
    mean_weight = float(retained_raw_weights.mean())
    if not math.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError(f"Invalid raw importance-weight mean: {mean_weight}")
    weighted["importance_weight"] = retained_raw_weights / mean_weight
    diagnostics["overall"] = {
        "rows": int(len(weighted)),
        "generated_rows": int(len(traces)),
        "empty_rows_excluded_from_sft": int((~retained_mask).sum()),
        "raw_weight_mean": mean_weight,
        "normalized_weight_min": float(weighted["importance_weight"].min()),
        "normalized_weight_mean": float(weighted["importance_weight"].mean()),
        "normalized_weight_max": float(weighted["importance_weight"].max()),
        "mean_human_mass_covered_by_distilled_model": float(
            np.mean(
                [
                    values["human_mass_covered_by_distilled_model"]
                    for values in diagnostics["problems"].values()
                ]
            )
        ),
    }
    return weighted, diagnostics


def build_weighted_bundle(
    traces_path: Path,
    human_train_path: Path,
    output_path: Path,
    *,
    expected_samples_per_problem: int = 1000,
    smoothing_mass: float = 0.05,
    max_weight: float = 5.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    paths = [traces_path.resolve(), human_train_path.resolve()]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    rollouts = pd.read_csv(paths[0], dtype={"parsed_answer": "string"})
    human = pd.read_csv(paths[1], dtype={"resp": "string"})
    prepared = prepare_distilled_rollouts(
        rollouts,
        human,
        expected_samples_per_problem=expected_samples_per_problem,
    )
    weighted, diagnostics = add_importance_weights(
        prepared,
        human,
        smoothing_mass=smoothing_mass,
        max_weight=max_weight,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    weighted.to_csv(output_path, index=False, compression="infer")
    manifest = {
        "protocol": "decimal_distilled_selftraining_importance_weights_v1",
        "inputs": {
            "distilled_rollouts": str(paths[0]),
            "distilled_rollouts_sha256": sha256_file(paths[0]),
            "human_train": str(paths[1]),
            "human_train_sha256": sha256_file(paths[1]),
        },
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "expected_samples_per_problem": int(expected_samples_per_problem),
        "smoothing_mass": float(smoothing_mass),
        "max_weight": float(max_weight),
        "weight_diagnostics": diagnostics,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_weighted_bundle(
    output_path: Path,
    traces_path: Path,
    human_train_path: Path,
    *,
    expected_samples_per_problem: int = 1000,
) -> dict[str, Any]:
    """Validate provenance and numerical invariants of a weighted trace bundle."""
    output_path = output_path.resolve()
    traces_path = traces_path.resolve()
    human_train_path = human_train_path.resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    errors: list[str] = []
    for path in (output_path, traces_path, human_train_path, manifest_path):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing artifact: {path}")
    if errors:
        return {"valid": False, "output": str(output_path), "errors": errors}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        weighted = pd.read_csv(output_path)
    except (json.JSONDecodeError, OSError, pd.errors.ParserError) as exc:
        return {
            "valid": False,
            "output": str(output_path),
            "errors": [f"unreadable artifact: {exc}"],
        }

    required = {
        "prob",
        "instruction_nl",
        "response_nl",
        "normalized_answer",
        "importance_weight",
    }
    if missing := sorted(required.difference(weighted.columns)):
        errors.append(f"missing columns: {missing}")
    else:
        weights = pd.to_numeric(weighted["importance_weight"], errors="coerce")
        if not np.isfinite(weights).all() or bool((weights <= 0).any()):
            errors.append("importance weights must be finite and positive")
        elif not math.isclose(float(weights.mean()), 1.0, rel_tol=0, abs_tol=1e-10):
            errors.append("importance weights do not have mean one")
        if weighted["response_nl"].fillna("").astype(str).str.strip().eq("").any():
            errors.append("weighted bundle contains an empty training response")
        if weighted["prob"].astype(str).nunique() != 12:
            errors.append("weighted bundle does not cover 12 problems")

    inputs = manifest.get("inputs", {})
    diagnostics = manifest.get("weight_diagnostics", {})
    overall = diagnostics.get("overall", {})
    if manifest.get("protocol") != "decimal_distilled_selftraining_importance_weights_v1":
        errors.append("protocol mismatch")
    if inputs.get("distilled_rollouts_sha256") != sha256_file(traces_path):
        errors.append("distilled-rollout input SHA-256 mismatch")
    if inputs.get("human_train_sha256") != sha256_file(human_train_path):
        errors.append("human-train input SHA-256 mismatch")
    if manifest.get("output_sha256") != sha256_file(output_path):
        errors.append("output SHA-256 mismatch")
    if int(manifest.get("expected_samples_per_problem", -1)) != expected_samples_per_problem:
        errors.append("expected samples-per-problem mismatch")
    if int(overall.get("generated_rows", -1)) != 12 * expected_samples_per_problem:
        errors.append("generated-row count mismatch")
    if int(overall.get("rows", -1)) != len(weighted):
        errors.append("retained-row count mismatch")
    problem_diagnostics = diagnostics.get("problems", {})
    if len(problem_diagnostics) != 12 or any(
        int(values.get("generated_distilled_traces", -1)) != expected_samples_per_problem
        for values in problem_diagnostics.values()
    ):
        errors.append("per-problem generation diagnostics mismatch")
    return {
        "valid": not errors,
        "output": str(output_path),
        "manifest": str(manifest_path),
        "rows": int(len(weighted)),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--human-train", type=Path, default=DEFAULT_HUMAN_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-samples-per-problem", type=int, default=1000)
    parser.add_argument("--smoothing-mass", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        result = validate_weighted_bundle(
            args.output,
            args.traces,
            args.human_train,
            expected_samples_per_problem=args.expected_samples_per_problem,
        )
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
        return
    manifest = build_weighted_bundle(
        args.traces,
        args.human_train,
        args.output,
        expected_samples_per_problem=args.expected_samples_per_problem,
        smoothing_mass=args.smoothing_mass,
        max_weight=args.max_weight,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["weight_diagnostics"]["overall"], indent=2))


if __name__ == "__main__":
    main()
