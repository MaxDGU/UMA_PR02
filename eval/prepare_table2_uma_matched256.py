#!/usr/bin/env python3
"""Build and score the matched-256 UMA baseline for Table 2.

UMA already has one saved response from each of 996 trained learner profiles
for every SP2013 fraction and BSS2021 decimal problem. To match the language
model rollout budget and the five-seed Qwen evaluation, this script selects
five independently sampled cohorts of 256 complete profiles. Within a seed,
the same cohort is used in both domains.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_ft.build_decimal_importance_weights import normalize_decimal_answer


DEFAULT_UMA = (
    ROOT
    / "results"
    / "procedural_alignment"
    / "uma_cognitive_model_sp2013_bss2021_20260516"
    / "uma_cognitive_model_sp2013_bss2021_trials.csv.gz"
)
DEFAULT_FRACTION_HUMAN = ROOT / "eval" / "data" / "siegler_fraction_human.csv"
DEFAULT_DECIMAL_HUMAN = (
    ROOT / "data" / "human_ft" / "decimal_rerun" / "decimal_test_nlp.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "eval" / "outputs" / "table2_matched256" / "uma"
DEFAULT_COHORT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_SAMPLES_PER_PROBLEM = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_problem(value: object) -> str:
    return str(value).strip().replace("÷", ":").replace(" ", "")


def normalize_fraction_human_answer(value: object) -> str | None:
    """Match the fraction-human normalization used for the paper table."""
    if pd.isna(value):
        return None
    try:
        return str(Fraction(str(float(value))).limit_denominator(10**6))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def normalize_fraction_model_answer(value: object) -> str | None:
    """Normalize numeric UMA/model fraction answers; reject invalid ratios."""
    if pd.isna(value):
        return None
    text = (
        str(value)
        .strip()
        .replace("−", "-")
        .replace("–", "-")
        .replace("÷", "/")
        .replace(" ", "")
    )
    try:
        return str(Fraction(text))
    except (TypeError, ValueError, ZeroDivisionError):
        try:
            return str(Fraction(str(float(text))).limit_denominator(10**6))
        except (TypeError, ValueError, ZeroDivisionError):
            return None


def categorical_answer_nll(
    human: pd.DataFrame,
    model: pd.DataFrame,
    *,
    human_problem_column: str,
    model_problem_column: str,
    human_answer_column: str,
    model_answer_column: str,
    normalize_human: Callable[[object], str | None],
    normalize_model: Callable[[object], str | None],
) -> tuple[float, pd.DataFrame, int]:
    """Categorical answer NLL with total symmetric Dirichlet mass one."""
    human_problems = human[human_problem_column].map(canonical_problem)
    model_problems = model[model_problem_column].map(canonical_problem)
    if set(human_problems.unique()) != set(model_problems.unique()):
        raise ValueError("Human and UMA problem sets differ")

    rows: list[dict[str, Any]] = []
    total_nll = 0.0
    total_human = 0
    for problem in sorted(human_problems.unique()):
        human_answers = [
            answer
            for answer in human.loc[
                human_problems.eq(problem), human_answer_column
            ].map(normalize_human)
            if answer is not None
        ]
        human_counts = Counter(human_answers)
        human_support = set(human_counts)
        categories = sorted(human_support) + ["Other", "Unparseable"]

        model_subset = model.loc[
            model_problems.eq(problem), model_answer_column
        ]
        model_counts: Counter[str] = Counter()
        for raw_answer in model_subset:
            answer = normalize_model(raw_answer)
            category = (
                "Unparseable"
                if answer is None
                else answer
                if answer in human_support
                else "Other"
            )
            model_counts[category] += 1

        prior_per_category = 1.0 / len(categories)
        denominator = len(model_subset) + 1.0
        for category, human_count in human_counts.items():
            probability = (
                model_counts.get(category, 0) + prior_per_category
            ) / denominator
            contribution = -int(human_count) * math.log(probability)
            total_nll += contribution
            total_human += int(human_count)
            rows.append(
                {
                    "problem": problem,
                    "category": category,
                    "human_count": int(human_count),
                    "model_count": int(model_counts.get(category, 0)),
                    "model_probability_smoothed": float(probability),
                    "nll_sum": float(contribution),
                }
            )

    if total_human == 0:
        raise ValueError("No parseable human answers")
    return total_nll / total_human, pd.DataFrame(rows), total_human


def load_problem_metadata(domain: str) -> pd.DataFrame:
    path = ROOT / "eval" / "data" / f"{domain}_problems.csv"
    metadata = pd.read_csv(path, dtype=str)
    metadata["problem"] = metadata["prob"].map(canonical_problem)
    if domain == "fraction":
        metadata = metadata.rename(columns={"denom_type": "operands"})
    return metadata[["problem", "operation", "operands", "correct_answer"]]


def complete_profile_ids(uma: pd.DataFrame) -> list[object]:
    expected = {
        domain: set(load_problem_metadata(domain)["problem"])
        for domain in ("fraction", "decimal")
    }
    complete: list[object] = []
    for subjid, profile in uma.groupby("subjid", sort=True):
        valid = True
        for domain, expected_problems in expected.items():
            domain_rows = profile[profile["domain"].eq(domain)]
            problems = domain_rows["canonical_problem"]
            if set(problems) != expected_problems or problems.duplicated().any():
                valid = False
                break
        if valid:
            complete.append(subjid)
    return complete


def select_profile_ids(
    complete_ids: list[object], *, samples_per_problem: int, cohort_seed: int
) -> list[object]:
    if len(complete_ids) < samples_per_problem:
        raise ValueError(
            f"Need {samples_per_problem} complete UMA profiles; found {len(complete_ids)}"
        )
    rng = np.random.default_rng(cohort_seed)
    selected = rng.choice(
        np.asarray(complete_ids), size=samples_per_problem, replace=False
    )
    return sorted(selected.tolist())


def build_rollout(
    uma: pd.DataFrame, domain: str, selected_ids: list[object]
) -> pd.DataFrame:
    selected = uma[
        uma["domain"].eq(domain) & uma["subjid"].isin(selected_ids)
    ].copy()
    selected["problem"] = selected["canonical_problem"]
    selected = selected.drop(
        columns=["operation", "operands", "correct_answer"], errors="ignore"
    )
    selected = selected.merge(
        load_problem_metadata(domain),
        on="problem",
        how="left",
        validate="many_to_one",
    )
    if selected[["operation", "operands", "correct_answer"]].isna().any().any():
        raise ValueError(f"Missing {domain} problem metadata after merge")

    selected = selected.sort_values(["problem", "subjid"]).reset_index(drop=True)
    selected["sample_idx"] = selected.groupby("problem", sort=False).cumcount()
    counts = selected.groupby("problem").size()
    if len(counts) != len(load_problem_metadata(domain)) or not counts.eq(
        len(selected_ids)
    ).all():
        raise ValueError(f"Incomplete matched UMA {domain} grid: {counts.to_dict()}")

    selected = selected.rename(
        columns={"subjid": "uma_subjid", "answer": "parsed_answer"}
    )
    selected["model_key"] = "uma"
    selected["model"] = "UMA"
    selected["training_seed"] = -1
    selected["model_response"] = ""
    selected["is_correct"] = selected["is_correct"].astype(bool)
    columns = [
        "problem",
        "operation",
        "operands",
        "correct_answer",
        "model_key",
        "model",
        "training_seed",
        "sample_idx",
        "uma_subjid",
        "parsed_answer",
        "is_correct",
        "model_response",
        "g",
        "d",
        "rt_mu",
        "ice",
    ]
    return selected[columns]


def score_domain(
    domain: str,
    rollout: pd.DataFrame,
    fraction_human_path: Path,
    decimal_human_path: Path,
    cohort_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if domain == "fraction":
        human = pd.read_csv(fraction_human_path, low_memory=False)
        nll, detail, human_responses = categorical_answer_nll(
            human,
            rollout,
            human_problem_column="prob",
            model_problem_column="problem",
            human_answer_column="resp_val",
            model_answer_column="parsed_answer",
            normalize_human=normalize_fraction_human_answer,
            normalize_model=normalize_fraction_model_answer,
        )
    else:
        human = pd.read_csv(decimal_human_path, low_memory=False)
        nll, detail, human_responses = categorical_answer_nll(
            human,
            rollout,
            human_problem_column="prob",
            model_problem_column="problem",
            human_answer_column="resp",
            model_answer_column="parsed_answer",
            normalize_human=normalize_decimal_answer,
            normalize_model=normalize_decimal_answer,
        )
    detail.insert(0, "cohort_seed", cohort_seed)
    detail.insert(0, "domain", domain)
    return (
        {
            "domain": domain,
            "model_key": "uma",
            "model_label": "UMA",
            "cohort_seed": cohort_seed,
            "problems": int(rollout["problem"].nunique()),
            "samples_per_problem": int(rollout.groupby("problem").size().iloc[0]),
            "model_rollouts": int(len(rollout)),
            "human_responses": int(human_responses),
            "answer_nll": float(nll),
        },
        detail,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uma", type=Path, default=DEFAULT_UMA)
    parser.add_argument("--fraction-human", type=Path, default=DEFAULT_FRACTION_HUMAN)
    parser.add_argument("--decimal-human", type=Path, default=DEFAULT_DECIMAL_HUMAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--samples-per-problem", type=int, default=DEFAULT_SAMPLES_PER_PROBLEM
    )
    parser.add_argument(
        "--cohort-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_COHORT_SEEDS),
        help="Independent UMA cohort seeds; default: 42 43 44 45 46.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.uma, args.fraction_human, args.decimal_human):
        if not path.is_file():
            raise FileNotFoundError(path)

    uma = pd.read_csv(args.uma, low_memory=False)
    required = {
        "domain",
        "subjid",
        "prob",
        "answer",
        "is_correct",
        "g",
        "d",
        "rt_mu",
        "ice",
    }
    if missing := sorted(required.difference(uma.columns)):
        raise ValueError(f"UMA source missing columns: {missing}")
    uma["canonical_problem"] = uma["prob"].map(canonical_problem)

    complete_ids = complete_profile_ids(uma)
    cohort_seeds = list(dict.fromkeys(args.cohort_seeds))
    if len(cohort_seeds) != len(args.cohort_seeds):
        raise ValueError("Cohort seeds must be unique")
    if not cohort_seeds:
        raise ValueError("At least one cohort seed is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    output_paths: dict[str, Path] = {}
    for cohort_seed in cohort_seeds:
        selected_ids = select_profile_ids(
            complete_ids,
            samples_per_problem=args.samples_per_problem,
            cohort_seed=cohort_seed,
        )
        cohort_dir = args.output_dir / "cohorts" / f"seed_{cohort_seed}"
        cohort_dir.mkdir(parents=True, exist_ok=True)
        selected_profiles_path = cohort_dir / "selected_profiles.csv"
        pd.DataFrame({"uma_subjid": selected_ids}).to_csv(
            selected_profiles_path, index=False
        )
        output_paths[f"seed_{cohort_seed}_selected_profiles"] = (
            selected_profiles_path
        )
        for domain in ("fraction", "decimal"):
            rollout = build_rollout(uma, domain, selected_ids)
            path = cohort_dir / f"uma_{domain}_matched256.csv"
            rollout.to_csv(path, index=False)
            output_paths[f"seed_{cohort_seed}_{domain}"] = path
            summary, detail = score_domain(
                domain,
                rollout,
                args.fraction_human,
                args.decimal_human,
                cohort_seed,
            )
            summaries.append(summary)
            details.append(detail)

    per_run = pd.DataFrame(summaries).sort_values(["domain", "cohort_seed"])
    aggregate_rows: list[dict[str, Any]] = []
    for domain, group in per_run.groupby("domain", sort=True):
        values = group["answer_nll"].astype(float)
        aggregate_rows.append(
            {
                "domain": domain,
                "model_key": "uma",
                "model_label": "UMA",
                "cohorts": int(len(values)),
                "cohort_seeds": ",".join(
                    str(seed) for seed in group["cohort_seed"]
                ),
                "samples_per_problem_per_cohort": args.samples_per_problem,
                "answer_nll_mean": float(values.mean()),
                "answer_nll_sd": (
                    float(values.std(ddof=1)) if len(values) > 1 else float("nan")
                ),
                "answer_nll_se": (
                    float(values.std(ddof=1) / math.sqrt(len(values)))
                    if len(values) > 1
                    else float("nan")
                ),
                "answer_nll_min": float(values.min()),
                "answer_nll_max": float(values.max()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)

    per_run_path = args.output_dir / "per_run.csv"
    summary_path = args.output_dir / "summary.csv"
    detail_path = args.output_dir / "per_answer_nll.csv"
    per_run.to_csv(per_run_path, index=False)
    aggregate.to_csv(summary_path, index=False)
    pd.concat(details, ignore_index=True).to_csv(detail_path, index=False)
    output_paths.update(
        {
            "per_run": per_run_path,
            "summary": summary_path,
            "per_answer_nll": detail_path,
        }
    )

    manifest = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "table2_uma_five_shared_profile_cohorts_matched256_v2",
        "source_uma": str(args.uma.resolve()),
        "source_uma_sha256": sha256_file(args.uma),
        "fraction_human": str(args.fraction_human.resolve()),
        "fraction_human_sha256": sha256_file(args.fraction_human),
        "decimal_human": str(args.decimal_human.resolve()),
        "decimal_human_sha256": sha256_file(args.decimal_human),
        "complete_source_profiles": len(complete_ids),
        "selected_profiles_per_cohort": args.samples_per_problem,
        "cohort_seeds": cohort_seeds,
        "profile_sampling": (
            "independent simple-random samples without replacement; "
            "the same sampled profiles are used in both domains within each seed"
        ),
        "categorical_nll": {
            "support": "distinct parseable human answers plus Other and Unparseable",
            "smoothing": "symmetric Dirichlet with total prior mass one",
            "aggregation": "response-averaged",
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in output_paths.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(per_run.to_string(index=False))
    print("\nAggregate across cohort seeds:")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
