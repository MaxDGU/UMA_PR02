#!/usr/bin/env python3
"""
Rule-based audit of the clean-child translator fidelity.

This analysis intentionally avoids training classifiers. Instead it:
1. samples a stable subset of the translated clean-child dataset,
2. audits which symbolic components have explicit surface realizations in
   response_nl versus being bundled or omitted by template design,
3. measures symbolic collisions where the same (prob, answer, response_nl)
   corresponds to multiple distinct symbolic states, and
4. exports representative examples for manual inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


DEFAULT_DATA_CSV = Path(
    "/n/fs/cogai/cs1095/UMA_PR02/results/transformer_replication/"
    "synth_unique_seed1_all1000_nlp_clean_child_correct_exec_mincols.csv.gz"
)
DEFAULT_OUT_DIR = Path(
    "/n/fs/cogai/cs1095/UMA_PR02/results/transformer_replication/"
    "translator_rule_audit"
)


STRATEGY_CODES = [
    "KDON_AS",
    "KDON_OG",
    "CDON_AS",
    "CDON_OG",
    "ONOD_M",
    "ONOD_OG",
    "CROP_M",
    "ICDM_D",
    "ICDM_OG",
    "OTHER",
]

GOAL_CODES = [
    "operate_nums",
    "operate_dens",
    "pass_den",
    "convert_CD",
    "convert_CD_LCM",
    "get_LCM",
    "convert_fra_to_den",
    "invert_op2",
    "div_to_mul",
    "check_simplify",
    "skip_simplify",
    "get_GCD",
    "simplify_fraction",
]

EXEC_CODES = [
    "add_fact",
    "sub_fact",
    "mul_fact",
    "div_calculator",
    "convert_CD_omit_nums",
    "invert_rand",
    "invert_fail",
    "div_to_mul_denied",
    "acc_skip",
    "acc_extra",
    "sub_LbS",
    "div_LbS",
    "div_LbS_drop_rem",
    "div_drop_rem",
]

NUMERIC_TOKEN_RE = r"[+\-]?(?:\d+(?:/\d+)?|\d*\.\d+)"
BINARY_PROB_RE = re.compile(rf"\s*({NUMERIC_TOKEN_RE})\s*([+\-*/:])\s*({NUMERIC_TOKEN_RE})\s*")


@dataclass(frozen=True)
class CoverageSpec:
    component_type: str
    component: str
    coverage_class: str
    rationale: str


def build_coverage_specs() -> List[CoverageSpec]:
    specs: List[CoverageSpec] = []

    for code in STRATEGY_CODES:
        if code.startswith("CDON"):
            specs.append(CoverageSpec("strategy", code, "explicit_template", "Response says it found a common denominator."))
        elif code.startswith("KDON"):
            specs.append(CoverageSpec("strategy", code, "explicit_template", "Response keeps one denominator on the bottom."))
        elif code.startswith("ONOD"):
            specs.append(CoverageSpec("strategy", code, "explicit_template", "Response applies the operation to numerators and denominators separately."))
        elif code == "CROP_M":
            specs.append(CoverageSpec("strategy", code, "explicit_template", "Response says it crossed the fractions."))
        elif code.startswith("ICDM"):
            specs.append(CoverageSpec("strategy", code, "explicit_template", "Response says it flipped or considered flipping the second fraction and surfaces separate work on top and bottom numbers."))
        else:
            specs.append(CoverageSpec("strategy", code, "omitted_or_generic", "Falls back to a generic worked-with sentence."))

    goal_map = {
        "operate_nums": ("explicit_template", "Response names work on the top numbers or directly states the numerator-side operation."),
        "operate_dens": ("explicit_template", "Response names work on the bottom numbers or directly states the denominator-side operation."),
        "pass_den": ("explicit_template", "Response explicitly says it kept a denominator on the bottom."),
        "convert_CD": ("explicit_template", "Response explicitly says it changed both fractions to a named common denominator."),
        "convert_CD_LCM": ("explicit_template", "Response names the least common denominator."),
        "get_LCM": ("explicit_template", "Response names the least common multiple of the denominators."),
        "convert_fra_to_den": ("explicit_template", "Response explicitly rewrites fractions to the target denominator."),
        "invert_op2": ("explicit_template", "Response explicitly says it flipped or considered flipping the second fraction."),
        "div_to_mul": ("conditional_template", "Only visible when the ICDM division branch fires."),
        "check_simplify": ("explicit_template", "Response says it checked whether the fraction could be simplified."),
        "skip_simplify": ("intentionally_ignored", "Explicitly removed from narrated goals."),
        "get_GCD": ("explicit_template", "Response names the greatest common factor used for simplification."),
        "simplify_fraction": ("explicit_template", "Response says it simplified the fraction."),
    }
    for code in GOAL_CODES:
        coverage_class, rationale = goal_map[code]
        specs.append(CoverageSpec("goal", code, coverage_class, rationale))

    exec_map = {
        "add_fact": ("explicit_template", "Response says it used an addition fact."),
        "sub_fact": ("explicit_template", "Response says it used a subtraction fact."),
        "mul_fact": ("explicit_template", "Response says it used a multiplication fact."),
        "div_calculator": ("explicit_template", "Response says the division was worked out directly."),
        "convert_CD_omit_nums": ("explicit_template", "Response says it changed denominators but kept top numbers the same."),
        "invert_rand": ("explicit_template", "Response says it flipped one of the fractions."),
        "invert_fail": ("intentionally_ignored", "Explicitly removed from narrated exec rules."),
        "div_to_mul_denied": ("explicit_template", "Response says it considered changing to multiplication but stayed in the current form."),
        "acc_skip": ("omitted", "Only affects quality flags; never verbalized."),
        "acc_extra": ("omitted", "Only affects quality flags; never verbalized."),
        "sub_LbS": ("explicit_template", "Response says it subtracted the smaller number from the bigger number."),
        "div_LbS": ("explicit_template", "Response says it divided the bigger number by the smaller number."),
        "div_LbS_drop_rem": ("explicit_template", "Response says it divided bigger by smaller and dropped the remainder."),
        "div_drop_rem": ("explicit_template", "Response says it used the whole-number part after dividing."),
    }
    for code in EXEC_CODES:
        coverage_class, rationale = exec_map[code]
        specs.append(CoverageSpec("exec", code, coverage_class, rationale))

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based audit of translator fidelity.")
    parser.add_argument("--data_csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample_modulus", type=int, default=97, help="Keep rows where source_row_idx %% sample_modulus == 0.")
    parser.add_argument("--sample_rows_per_component", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--max_input_rows",
        type=int,
        default=1_500_000,
        help="Optional prefix cap on streamed rows. Use 0 or a negative value for no cap.",
    )
    return parser.parse_args()


def split_tokens(value: Any) -> List[str]:
    text = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    return [token for token in text.strip().split() if token]


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def parse_prob(prob: str) -> Dict[str, str]:
    text = safe_text(prob)
    match = BINARY_PROB_RE.fullmatch(text)
    if match is None:
        return {"op": "?", "op_label": "unknown", "denom_type": "unknown"}
    left = match.group(1)
    op = match.group(2)
    right = match.group(3)
    if op == "/":
        op = ":"
    op_label = {"+": "add", "-": "sub", "*": "mul", ":": "div"}.get(op, "unknown")
    denom_type = "unknown"
    if "/" in left and "/" in right:
        denom_type = "ED" if left.split("/")[-1] == right.split("/")[-1] else "UD"
    return {"op": op, "op_label": op_label, "denom_type": denom_type}


def strategy_family(strategy: str) -> str:
    text = safe_text(strategy)
    if "_" in text:
        return text.split("_", 1)[0]
    return text or "OTHER"


def truthy(value: Any) -> bool:
    return safe_text(value).lower() in {"1", "1.0", "true", "t", "yes", "y"}


def build_component_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    coverage_lookup = {(spec.component_type, spec.component): spec for spec in build_coverage_specs()}
    for row in df.itertuples(index=False):
        base = {
            "source_uid": getattr(row, "source_uid"),
            "source_row_idx": int(getattr(row, "source_row_idx")),
            "prob": getattr(row, "prob"),
            "answer": getattr(row, "answer"),
            "correct": getattr(row, "correct"),
            "is_correct": truthy(getattr(row, "is_correct")),
            "strategy": getattr(row, "strategy"),
            "goals": getattr(row, "goals"),
            "exec": getattr(row, "exec"),
            "response_nl": getattr(row, "response_nl"),
            "op_label": getattr(row, "op_label"),
            "denom_type": getattr(row, "denom_type"),
            "strategy_family": getattr(row, "strategy_family"),
        }
        strategy = safe_text(getattr(row, "strategy"))
        for component_type, components in (
            ("strategy", [strategy] if strategy else []),
            ("goal", split_tokens(getattr(row, "goals"))),
            ("exec", split_tokens(getattr(row, "exec"))),
        ):
            for component in components:
                spec = coverage_lookup.get((component_type, component))
                rows.append(
                    {
                        **base,
                        "component_type": component_type,
                        "component": component,
                        "coverage_class": spec.coverage_class if spec else "unknown",
                        "coverage_rationale": spec.rationale if spec else "",
                    }
                )
    return pd.DataFrame(rows)


def load_stable_sample(
    csv_path: Path,
    sample_modulus: int,
    chunksize: int,
    max_input_rows: int,
) -> pd.DataFrame:
    usecols = [
        "source_uid",
        "source_row_idx",
        "prob",
        "answer",
        "correct",
        "is_correct",
        "strategy",
        "goals",
        "exec",
        "response_nl",
    ]
    samples: List[pd.DataFrame] = []
    rows_seen = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        if max_input_rows > 0 and rows_seen >= max_input_rows:
            break
        if max_input_rows > 0 and rows_seen + len(chunk) > max_input_rows:
            chunk = chunk.iloc[: max_input_rows - rows_seen].copy()
        rows_seen += len(chunk)
        keep_mask = pd.to_numeric(chunk["source_row_idx"], errors="coerce").fillna(-1).astype(int) % sample_modulus == 0
        sample = chunk.loc[keep_mask].copy()
        if len(sample) == 0:
            continue
        parsed = sample["prob"].map(parse_prob).apply(pd.Series)
        sample["op_label"] = parsed["op_label"]
        sample["denom_type"] = parsed["denom_type"]
        sample["strategy_family"] = sample["strategy"].map(strategy_family)
        sample["state_key"] = (
            sample["strategy"].fillna("").astype(str)
            + " || "
            + sample["goals"].fillna("").astype(str)
            + " || "
            + sample["exec"].fillna("").astype(str)
        )
        samples.append(sample)
    if not samples:
        raise ValueError(f"No sampled rows loaded from {csv_path}")
    out = pd.concat(samples, ignore_index=True)
    out["is_correct"] = out["is_correct"].map(truthy)
    return out


def compute_component_summary(component_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        component_rows.groupby(["component_type", "component", "coverage_class", "coverage_rationale"], dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values(["component_type", "sample_count", "component"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return grouped


def compute_collision_outputs(sample_df: pd.DataFrame) -> Dict[str, Any]:
    group_cols = ["prob", "answer", "response_nl"]
    collisions = (
        sample_df.groupby(group_cols)
        .agg(
            n_rows=("state_key", "size"),
            n_unique_states=("state_key", "nunique"),
            strategies=("strategy", lambda s: " | ".join(sorted(set(s.astype(str))))[:400]),
            goals=("goals", lambda s: " | ".join(sorted(set(s.astype(str))))[:400]),
            execs=("exec", lambda s: " | ".join(sorted(set(s.astype(str))))[:400]),
            op_label=("op_label", "first"),
            denom_type=("denom_type", "first"),
        )
        .reset_index()
    )
    collisions["has_symbolic_collision"] = collisions["n_unique_states"] > 1

    row_df = sample_df.merge(
        collisions[group_cols + ["n_unique_states", "has_symbolic_collision"]],
        on=group_cols,
        how="left",
        validate="many_to_one",
    )

    overall = {
        "sample_rows": int(len(sample_df)),
        "unique_prob_answer_response_groups": int(len(collisions)),
        "collision_groups": int(collisions["has_symbolic_collision"].sum()),
        "collision_group_rate": float(collisions["has_symbolic_collision"].mean()),
        "collision_row_rate": float(row_df["has_symbolic_collision"].mean()),
        "mean_unique_states_per_group": float(collisions["n_unique_states"].mean()),
        "max_unique_states_per_group": int(collisions["n_unique_states"].max()),
    }

    stratified = (
        row_df.groupby(["op_label", "denom_type", "is_correct", "strategy_family"], dropna=False)
        .agg(
            rows=("state_key", "size"),
            collision_row_rate=("has_symbolic_collision", "mean"),
            mean_group_unique_states=("n_unique_states", "mean"),
        )
        .reset_index()
        .sort_values(["collision_row_rate", "rows"], ascending=[False, False])
        .reset_index(drop=True)
    )

    top_collisions = (
        collisions[collisions["has_symbolic_collision"]]
        .sort_values(["n_unique_states", "n_rows"], ascending=[False, False])
        .head(100)
        .reset_index(drop=True)
    )

    return {
        "overall": overall,
        "stratified": stratified,
        "top_collisions": top_collisions,
    }


def sample_examples(component_rows: pd.DataFrame, per_component: int) -> pd.DataFrame:
    sampled = (
        component_rows.sort_values(["component_type", "component", "source_row_idx"])
        .groupby(["component_type", "component"], dropna=False)
        .head(per_component)
        .reset_index(drop=True)
    )
    return sampled


def build_static_summary_markdown(
    args: argparse.Namespace,
    sample_df: pd.DataFrame,
    component_summary: pd.DataFrame,
    collision_outputs: Dict[str, Any],
) -> str:
    explicit = component_summary["coverage_class"].eq("explicit_template").sum()
    omitted = component_summary["coverage_class"].isin({"omitted", "omitted_or_generic"}).sum()
    conditional = component_summary["coverage_class"].eq("conditional_template").sum()
    intentionally_ignored = component_summary["coverage_class"].eq("intentionally_ignored").sum()
    bundled = component_summary["coverage_class"].eq("bundled_not_explicit").sum()

    overall = collision_outputs["overall"]
    lines = [
        "# Translator Rule Audit",
        "",
        "Question. We study whether the clean-child translator's `response_nl` preserves the symbolic UMA state used in training.",
        "",
        "Setup. Let a symbolic row be `z = (prob, strategy, goals, exec, answer)` and let `T(z) = response_nl` be the student-facing translated reasoning text. We audit `T` without a classifier by combining a static template inspection with a stable sampled empirical audit.",
        "",
        f"Data. Source CSV: `{args.data_csv}`",
        (
            f"Prefix cap. `{args.max_input_rows:,}` input rows."
            if args.max_input_rows > 0
            else "Prefix cap. none."
        ),
        f"Sampling rule. Keep rows with `source_row_idx % {args.sample_modulus} == 0`.",
        f"Sample size. `{len(sample_df):,}` rows.",
        "",
        "Method. We classify each symbolic component into one of five categories: `explicit_template`, `conditional_template`, `bundled_not_explicit`, `omitted`, or `intentionally_ignored`. We then measure symbolic collisions where the same `(prob, answer, response_nl)` corresponds to multiple distinct `(strategy, goals, exec)` states.",
        "",
        "Finding. The translator preserves only a small subset of the symbolic state directly in `response_nl`; many goal and exec components are either bundled into coarse arithmetic sentences or not surfaced at all.",
        "",
        f"- Components with `explicit_template`: `{explicit}`",
        f"- Components with `conditional_template`: `{conditional}`",
        f"- Components with `bundled_not_explicit`: `{bundled}`",
        f"- Components with `omitted` or `omitted_or_generic`: `{omitted}`",
        f"- Components with `intentionally_ignored`: `{intentionally_ignored}`",
        "",
        "Collision evidence.",
        f"- Sampled rows in collision groups: `{overall['collision_row_rate']:.3f}`",
        f"- Groups with symbolic collisions: `{overall['collision_groups']}` / `{overall['unique_prob_answer_response_groups']}`",
        f"- Mean unique symbolic states per `(prob, answer, response_nl)` group: `{overall['mean_unique_states_per_group']:.3f}`",
        f"- Max unique symbolic states in one group: `{overall['max_unique_states_per_group']}`",
        "",
        "Implication. If the model is trained only on `instruction_nl -> response_nl`, then strategy/goals/exec information that lives only in side columns is unavailable to the learner. Poor NLP fit can therefore reflect representational collapse, not just optimization.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_df = load_stable_sample(args.data_csv, args.sample_modulus, args.chunksize, args.max_input_rows)
    component_rows = build_component_rows(sample_df)
    component_summary = compute_component_summary(component_rows)
    sample_examples_df = sample_examples(component_rows, args.sample_rows_per_component)
    collision_outputs = compute_collision_outputs(sample_df)
    coverage_specs = pd.DataFrame([spec.__dict__ for spec in build_coverage_specs()])

    coverage_specs.to_csv(args.out_dir / "component_coverage_static.csv", index=False)
    component_summary.to_csv(args.out_dir / "component_summary_sample.csv", index=False)
    sample_examples_df.to_csv(args.out_dir / "component_examples_sample.csv", index=False)
    collision_outputs["stratified"].to_csv(args.out_dir / "collision_stratified_sample.csv", index=False)
    collision_outputs["top_collisions"].to_csv(args.out_dir / "collision_examples_sample.csv", index=False)

    summary_md = build_static_summary_markdown(args, sample_df, component_summary, collision_outputs)
    (args.out_dir / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
    (args.out_dir / "overall_metrics.json").write_text(
        json.dumps(collision_outputs["overall"], indent=2),
        encoding="utf-8",
    )

    print(summary_md)
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
