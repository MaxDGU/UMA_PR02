#!/usr/bin/env python3
"""
Automatic scorecard for the clean-child translator.

This extends the earlier rule-based audit with programmatic metrics for:
1. coverage mass,
2. symbolic collisions,
3. rule-based recoverability from response_nl,
4. minimal-pair sensitivity,
5. template repetition/diversity,
6. length, readability, and lexical variety.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

import analyze_translator_rule_audit as audit


DEFAULT_DATA_CSV = audit.DEFAULT_DATA_CSV
DEFAULT_OUT_DIR = Path(
    "/n/fs/cogai/cs1095/UMA_PR02/results/transformer_replication/"
    "translator_auto_metrics"
)

WORD_RE = re.compile(r"[A-Za-z<>']+")
NUM_RE = re.compile(audit.NUMERIC_TOKEN_RE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate automatic translator metrics.")
    parser.add_argument("--data_csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample_modulus", type=int, default=97)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max_input_rows", type=int, default=1_500_000)
    parser.add_argument(
        "--edit_cluster_threshold",
        type=float,
        default=0.92,
        help="Greedy normalized-string similarity threshold for template-family clustering.",
    )
    parser.add_argument(
        "--top_k_templates",
        type=int,
        default=25,
        help="Number of top templates / clusters to retain in exported tables.",
    )
    return parser.parse_args()


def strip_answer_suffix(text: Any) -> str:
    raw = audit.safe_text(text)
    if "### answer:" in raw:
        raw = raw.split("### answer:", 1)[0]
    return raw.strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_numbers(text: str) -> str:
    return NUM_RE.sub("<NUM>", text)


def normalize_template(text: Any) -> str:
    body = strip_answer_suffix(text).lower()
    body = normalize_numbers(body)
    body = normalize_whitespace(body)
    return body


def normalize_response_body(text: Any) -> str:
    return normalize_whitespace(strip_answer_suffix(text))


def split_tokens(value: Any) -> Tuple[str, ...]:
    return tuple(sorted(audit.split_tokens(value)))


def tokenize_text(text: str, *, normalize_numbers_flag: bool = False) -> List[str]:
    cleaned = text.lower()
    if normalize_numbers_flag:
        cleaned = normalize_numbers(cleaned)
    tokens = WORD_RE.findall(cleaned)
    return [token for token in tokens if token]


def count_sentences(text: str) -> int:
    chunks = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    return max(1, len(chunks))


def count_syllables(word: str) -> int:
    token = re.sub(r"[^a-z]", "", word.lower())
    if not token:
        return 0
    groups = re.findall(r"[aeiouy]+", token)
    count = len(groups)
    if token.endswith("e") and len(groups) > 1:
        count -= 1
    return max(1, count)


def build_enriched_sample(args: argparse.Namespace) -> pd.DataFrame:
    sample_df = audit.load_stable_sample(
        args.data_csv,
        sample_modulus=args.sample_modulus,
        chunksize=args.chunksize,
        max_input_rows=args.max_input_rows,
    ).copy()
    sample_df["response_body"] = sample_df["response_nl"].map(normalize_response_body)
    sample_df["response_template"] = sample_df["response_nl"].map(normalize_template)
    sample_df["goal_tokens"] = sample_df["goals"].map(split_tokens)
    sample_df["exec_tokens"] = sample_df["exec"].map(split_tokens)
    sample_df["strategy_token"] = sample_df["strategy"].map(audit.safe_text)
    sample_df["strategy_family_truth"] = sample_df["strategy_token"].map(audit.strategy_family)
    sample_df["response_tokens"] = sample_df["response_body"].map(lambda text: tokenize_text(text, normalize_numbers_flag=False))
    sample_df["response_tokens_normalized"] = sample_df["response_body"].map(lambda text: tokenize_text(text, normalize_numbers_flag=True))
    return sample_df


def build_component_lookup() -> Dict[Tuple[str, str], audit.CoverageSpec]:
    return {(spec.component_type, spec.component): spec for spec in audit.build_coverage_specs()}


def infer_strategy_family(response_body: str) -> str:
    text = response_body.lower()
    if "changed it to multiplication" in text or "changing it to multiplication" in text:
        return "ICDM"
    if "i found a common denominator" in text:
        return "CDON"
    if "least common denominator" in text or "least common multiple of the denominators" in text:
        return "CDON"
    if "i crossed them" in text:
        return "CROP"
    if "kept" in text and "on the bottom" in text:
        return "KDON"
    if re.search(r"i took .* and got .*?, and .* and got .*", text):
        return "ONOD"
    if "i applied" in text and "top numbers and bottom numbers separately" in text:
        return "ONOD"
    if "i worked with" in text:
        return "OTHER"
    return "UNKNOWN"


def infer_goal_tokens(response_body: str) -> Set[str]:
    text = response_body.lower()
    predicted: Set[str] = set()
    if "i changed both fractions to use denominator" in text:
        predicted.add("convert_CD")
    if "least common denominator" in text:
        predicted.add("convert_CD_LCM")
    if "least common multiple of the denominators" in text:
        predicted.update({"convert_CD_LCM", "get_LCM"})
    if re.search(r"kept [^.!?]{1,40} on the bottom", text):
        predicted.add("pass_den")
    if re.search(r"i took .*? (plus|minus|times|divided by) .*? and got", text):
        predicted.add("operate_nums")
    if re.search(r"i did .*? (plus|minus|times|divided by) .*? (to get|and got)", text):
        predicted.add("operate_nums")
    if re.search(r"i combined the top numbers with (plus|minus|times|divided by)", text):
        predicted.add("operate_nums")
    if "i combined the top numbers and kept" in text:
        predicted.add("operate_nums")
    if "worked on the top numbers" in text:
        predicted.add("operate_nums")
    if "worked on the bottom numbers" in text:
        predicted.add("operate_dens")
    if re.search(r", and .*? (plus|minus|times|divided by) .*? and got", text) and "kept" not in text:
        predicted.add("operate_dens")
    if re.search(r", and .*? (plus|minus|times|divided by) .*? (to get|and got)", text):
        predicted.add("operate_dens")
    if "i applied" in text and "top numbers and bottom numbers separately" in text:
        predicted.update({"operate_nums", "operate_dens"})
    if "i worked on the top numbers and on the bottom numbers separately" in text:
        predicted.update({"operate_nums", "operate_dens"})
    if "i crossed them" in text:
        predicted.update({"operate_nums", "operate_dens"})
    if re.search(r"i rewrote .*? as .*? and .*? as .*?", text) or "i tried to change both fractions to use denominator" in text:
        predicted.add("convert_fra_to_den")
    if re.search(r"i flipped [^.!?]+ and changed it to multiplication", text):
        predicted.add("div_to_mul")
    if "i flipped the second fraction" in text:
        predicted.add("invert_op2")
    if "i thought about flipping the second fraction and changing it to multiplication" in text:
        predicted.add("invert_op2")
    if "i checked if it could be simplified" in text:
        predicted.add("check_simplify")
    if "i checked for a greatest common factor" in text or "i checked for a greatest common divisor" in text:
        predicted.update({"check_simplify", "get_GCD"})
    if "greatest common factor" in text or "greatest common divisor" in text:
        predicted.add("get_GCD")
    if "then i simplified it" in text:
        predicted.add("simplify_fraction")
    if "used the greatest common factor to simplify it" in text or "used the greatest common divisor to simplify it" in text:
        predicted.update({"simplify_fraction", "get_GCD"})
    if "checked for a greatest common factor and then used it to simplify it" in text or "checked for a greatest common divisor and then used it to simplify it" in text:
        predicted.update({"check_simplify", "simplify_fraction", "get_GCD"})
    if "i decided not to simplify the fraction any further" in text:
        predicted.add("skip_simplify")
    return predicted


def infer_exec_tokens(response_body: str) -> Set[str]:
    text = response_body.lower()
    predicted: Set[str] = set()
    if "i used an addition fact to get that result" in text:
        predicted.add("add_fact")
    if "i used a subtraction fact to get that result" in text:
        predicted.add("sub_fact")
    if "i used a multiplication fact to get that result" in text:
        predicted.add("mul_fact")
    if "i changed the bottom numbers and kept the top numbers the same" in text:
        predicted.add("convert_CD_omit_nums")
    if "i flipped one of the fractions" in text:
        predicted.add("invert_rand")
    if "i tried to flip a fraction, but i did not change it" in text:
        predicted.add("invert_fail")
    if "changing it to multiplication, but i kept it in this form" in text:
        predicted.add("div_to_mul_denied")
    if "i worked out that division directly" in text:
        predicted.add("div_calculator")
    if "i subtracted the smaller number from the bigger number" in text:
        predicted.add("sub_LbS")
    has_div_combo = "i divided the bigger number by the smaller number and kept only the whole-number part" in text
    has_div_larger = re.search(r"i divided the bigger number by the smaller number(?:[.!?]|$)", text) is not None
    has_drop_rem = re.search(r"i used the whole-number part after dividing(?:[.!?]|$)", text) is not None
    if has_div_combo:
        predicted.add("div_LbS_drop_rem")
    if has_div_larger:
        predicted.add("div_LbS")
    if has_drop_rem:
        predicted.add("div_drop_rem")
    if "i moved on without updating the running answer" in text:
        predicted.add("acc_skip")
    if "i made an extra running-answer update" in text:
        predicted.add("acc_extra")
    return predicted


def add_rule_based_predictions(sample_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_df.copy()
    df["pred_strategy_family"] = df["response_body"].map(infer_strategy_family)
    df["pred_goal_tokens"] = df["response_body"].map(infer_goal_tokens)
    df["pred_exec_tokens"] = df["response_body"].map(infer_exec_tokens)
    return df


def compute_coverage_outputs(component_rows: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    by_component = (
        component_rows.groupby(["component_type", "coverage_class"], dropna=False)
        .size()
        .rename("n_components")
        .reset_index()
    )
    by_mass = (
        component_rows.groupby(["component_type", "coverage_class"], dropna=False)
        .size()
        .rename("sample_occurrences")
        .reset_index()
    )
    return {
        "component_counts": by_component,
        "mass_counts": by_mass,
    }


def compute_strategy_recoverability(sample_df: pd.DataFrame) -> Dict[str, Any]:
    family_df = (
        sample_df.groupby(["strategy_family_truth", "pred_strategy_family"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    family_acc = float((sample_df["strategy_family_truth"] == sample_df["pred_strategy_family"]).mean())
    by_family = (
        sample_df.assign(match=sample_df["strategy_family_truth"] == sample_df["pred_strategy_family"])
        .groupby("strategy_family_truth", dropna=False)
        .agg(rows=("strategy_family_truth", "size"), accuracy=("match", "mean"))
        .reset_index()
        .sort_values("rows", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "summary": {
            "strategy_family_accuracy": family_acc,
            "strategy_family_classes": int(sample_df["strategy_family_truth"].nunique()),
        },
        "confusion": family_df,
        "by_family": by_family,
    }


def compute_token_recoverability(
    sample_df: pd.DataFrame,
    component_type: str,
    tokens: Sequence[str],
    pred_column: str,
    truth_column: str,
    coverage_lookup: Dict[Tuple[str, str], audit.CoverageSpec],
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    truth_sets = sample_df[truth_column]
    pred_sets = sample_df[pred_column]
    for token in tokens:
        tp = fp = fn = 0
        for truth, pred in zip(truth_sets, pred_sets):
            truth_has = token in truth
            pred_has = token in pred
            if truth_has and pred_has:
                tp += 1
            elif pred_has and not truth_has:
                fp += 1
            elif truth_has and not pred_has:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else math.nan
        recall = tp / (tp + fn) if (tp + fn) else math.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) and not math.isnan(precision) and not math.isnan(recall) else math.nan
        prevalence = sum(token in truth for truth in truth_sets)
        spec = coverage_lookup[(component_type, token)]
        records.append(
            {
                "component_type": component_type,
                "component": token,
                "coverage_class": spec.coverage_class,
                "coverage_rationale": spec.rationale,
                "prevalence_rows": int(prevalence),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    by_component = pd.DataFrame(records).sort_values(["coverage_class", "prevalence_rows", "component"], ascending=[True, False, True]).reset_index(drop=True)
    by_class = (
        by_component.groupby("coverage_class", dropna=False)
        .agg(
            components=("component", "size"),
            prevalence_rows=("prevalence_rows", "sum"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
        )
        .reset_index()
        .sort_values("prevalence_rows", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "by_component": by_component,
        "by_class": by_class,
    }


def tuple_key(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(value for value in values if value))


def compute_minimal_pair_for_component(
    sample_df: pd.DataFrame,
    component_type: str,
    component: str,
    coverage_lookup: Dict[Tuple[str, str], audit.CoverageSpec],
) -> Dict[str, Any]:
    groups: Dict[Tuple[Any, ...], Dict[bool, Set[str]]] = defaultdict(lambda: {True: set(), False: set()})

    for row in sample_df.itertuples(index=False):
        prob = getattr(row, "prob")
        answer = getattr(row, "answer")
        strategy = getattr(row, "strategy_token")
        goals = set(getattr(row, "goal_tokens"))
        execs = set(getattr(row, "exec_tokens"))
        response_template = getattr(row, "response_template")

        if component_type == "strategy":
            presence = strategy == component
            key = (prob, answer, tuple_key(goals), tuple_key(execs))
        elif component_type == "goal":
            presence = component in goals
            key = (prob, answer, strategy, tuple_key(goals - {component}), tuple_key(execs))
        elif component_type == "exec":
            presence = component in execs
            key = (prob, answer, strategy, tuple_key(goals), tuple_key(execs - {component}))
        else:
            raise ValueError(f"Unsupported component_type {component_type}")

        groups[key][presence].add(response_template)

    qualifying = 0
    changed = 0
    collapsed = 0
    changed_examples: List[Dict[str, Any]] = []
    collapsed_examples: List[Dict[str, Any]] = []
    for key, bucket in groups.items():
        if not bucket[True] or not bucket[False]:
            continue
        qualifying += 1
        present_templates = bucket[True]
        absent_templates = bucket[False]
        if present_templates != absent_templates:
            changed += 1
            if len(changed_examples) < 3:
                changed_examples.append(
                    {
                        "component_type": component_type,
                        "component": component,
                        "group_key": repr(key),
                        "present_templates": " || ".join(sorted(present_templates))[:500],
                        "absent_templates": " || ".join(sorted(absent_templates))[:500],
                    }
                )
        if present_templates.intersection(absent_templates):
            collapsed += 1
            if len(collapsed_examples) < 3:
                collapsed_examples.append(
                    {
                        "component_type": component_type,
                        "component": component,
                        "group_key": repr(key),
                        "shared_templates": " || ".join(sorted(present_templates.intersection(absent_templates)))[:500],
                    }
                )

    spec = coverage_lookup[(component_type, component)]
    return {
        "component_type": component_type,
        "component": component,
        "coverage_class": spec.coverage_class,
        "coverage_rationale": spec.rationale,
        "qualifying_groups": qualifying,
        "response_changed_rate": (changed / qualifying) if qualifying else math.nan,
        "response_collapsed_rate": (collapsed / qualifying) if qualifying else math.nan,
        "changed_examples": changed_examples,
        "collapsed_examples": collapsed_examples,
    }


def compute_minimal_pair_outputs(sample_df: pd.DataFrame, coverage_lookup: Dict[Tuple[str, str], audit.CoverageSpec]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []
    for component in audit.STRATEGY_CODES:
        record = compute_minimal_pair_for_component(sample_df, "strategy", component, coverage_lookup)
        rows.append({k: v for k, v in record.items() if k not in {"changed_examples", "collapsed_examples"}})
        example_rows.extend(record["changed_examples"])
        example_rows.extend(record["collapsed_examples"])
    for component in audit.GOAL_CODES:
        record = compute_minimal_pair_for_component(sample_df, "goal", component, coverage_lookup)
        rows.append({k: v for k, v in record.items() if k not in {"changed_examples", "collapsed_examples"}})
        example_rows.extend(record["changed_examples"])
        example_rows.extend(record["collapsed_examples"])
    for component in audit.EXEC_CODES:
        record = compute_minimal_pair_for_component(sample_df, "exec", component, coverage_lookup)
        rows.append({k: v for k, v in record.items() if k not in {"changed_examples", "collapsed_examples"}})
        example_rows.extend(record["changed_examples"])
        example_rows.extend(record["collapsed_examples"])

    by_component = pd.DataFrame(rows).sort_values(
        ["qualifying_groups", "response_changed_rate", "component"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    by_class = (
        by_component.groupby("coverage_class", dropna=False)
        .agg(
            components=("component", "size"),
            components_with_pairs=("qualifying_groups", lambda s: int((s > 0).sum())),
            total_qualifying_groups=("qualifying_groups", "sum"),
            mean_response_changed_rate=("response_changed_rate", "mean"),
            mean_response_collapsed_rate=("response_collapsed_rate", "mean"),
        )
        .reset_index()
        .sort_values("total_qualifying_groups", ascending=False)
        .reset_index(drop=True)
    )
    examples_df = pd.DataFrame(example_rows)
    return {
        "by_component": by_component,
        "by_class": by_class,
        "examples": examples_df,
    }


def compute_template_diversity(sample_df: pd.DataFrame, threshold: float, top_k: int) -> Dict[str, Any]:
    exact_counts = Counter(sample_df["response_body"])
    skeleton_counts = Counter(sample_df["response_template"])
    total_rows = len(sample_df)

    token_stream = [token for tokens in sample_df["response_tokens_normalized"] for token in tokens]
    bigrams = list(zip(token_stream, token_stream[1:])) if len(token_stream) >= 2 else []
    distinct_1 = len(set(token_stream)) / len(token_stream) if token_stream else math.nan
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else math.nan

    sorted_templates = skeleton_counts.most_common()
    clusters: List[Dict[str, Any]] = []
    for template, count in sorted_templates:
        assigned = False
        for cluster in clusters:
            score = difflib.SequenceMatcher(None, template, cluster["representative"]).ratio()
            if score >= threshold:
                cluster["row_count"] += count
                cluster["unique_templates"] += 1
                cluster["members"].append((template, count, score))
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    "representative": template,
                    "row_count": count,
                    "unique_templates": 1,
                    "members": [(template, count, 1.0)],
                }
            )

    cluster_rows = [
        {
            "cluster_id": idx,
            "representative": cluster["representative"],
            "row_count": cluster["row_count"],
            "unique_templates": cluster["unique_templates"],
            "share_rows": cluster["row_count"] / total_rows if total_rows else math.nan,
        }
        for idx, cluster in enumerate(sorted(clusters, key=lambda c: (-c["row_count"], c["representative"])), start=1)
    ]
    cluster_df = pd.DataFrame(cluster_rows)

    top_templates = pd.DataFrame(
        [
            {"response_template": template, "rows": count, "share_rows": count / total_rows if total_rows else math.nan}
            for template, count in sorted_templates[:top_k]
        ]
    )
    summary = {
        "sample_rows": int(total_rows),
        "unique_exact_responses": int(len(exact_counts)),
        "unique_exact_response_rate": float(len(exact_counts) / total_rows) if total_rows else math.nan,
        "unique_templates": int(len(skeleton_counts)),
        "unique_template_rate": float(len(skeleton_counts) / total_rows) if total_rows else math.nan,
        "mean_rows_per_template": float(total_rows / len(skeleton_counts)) if skeleton_counts else math.nan,
        "max_rows_single_template": int(max(skeleton_counts.values())) if skeleton_counts else 0,
        "top_10_template_row_share": float(sum(count for _, count in sorted_templates[:10]) / total_rows) if total_rows else math.nan,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "edit_cluster_threshold": threshold,
        "edit_distance_clusters": int(len(cluster_df)),
        "cluster_compression_ratio": float(len(cluster_df) / len(skeleton_counts)) if skeleton_counts else math.nan,
    }
    return {
        "summary": summary,
        "top_templates": top_templates,
        "clusters": cluster_df.head(top_k).copy(),
    }


def compute_length_readability(sample_df: pd.DataFrame) -> Dict[str, Any]:
    response_bodies = sample_df["response_body"].tolist()
    token_lists = sample_df["response_tokens_normalized"].tolist()
    token_counts = [len(tokens) for tokens in token_lists]
    sentence_counts = [count_sentences(text) for text in response_bodies]
    char_counts = [len(text) for text in response_bodies]
    total_words = sum(token_counts)
    total_sentences = sum(sentence_counts)
    total_chars = sum(char_counts)
    total_syllables = sum(count_syllables(token) for tokens in token_lists for token in tokens)
    vocab = {token for tokens in token_lists for token in tokens}
    bigrams = [bg for tokens in token_lists for bg in zip(tokens, tokens[1:])]

    avg_words_per_sentence = (total_words / total_sentences) if total_sentences else math.nan
    avg_syllables_per_word = (total_syllables / total_words) if total_words else math.nan
    flesch_reading_ease = (
        206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        if total_words and total_sentences
        else math.nan
    )
    flesch_kincaid_grade = (
        0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
        if total_words and total_sentences
        else math.nan
    )

    summary = {
        "sample_rows": int(len(sample_df)),
        "mean_tokens": float(sum(token_counts) / len(token_counts)) if token_counts else math.nan,
        "median_tokens": float(pd.Series(token_counts).median()) if token_counts else math.nan,
        "p90_tokens": float(pd.Series(token_counts).quantile(0.90)) if token_counts else math.nan,
        "mean_sentences": float(sum(sentence_counts) / len(sentence_counts)) if sentence_counts else math.nan,
        "mean_characters": float(total_chars / len(char_counts)) if char_counts else math.nan,
        "vocabulary_size_normalized": int(len(vocab)),
        "type_token_ratio_normalized": float(len(vocab) / total_words) if total_words else math.nan,
        "distinct_1_normalized": float(len(set(token for tokens in token_lists for token in tokens)) / total_words) if total_words else math.nan,
        "distinct_2_normalized": float(len(set(bigrams)) / len(bigrams)) if bigrams else math.nan,
        "avg_words_per_sentence": avg_words_per_sentence,
        "avg_syllables_per_word": avg_syllables_per_word,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
    }
    return summary


def build_summary_markdown(
    args: argparse.Namespace,
    sample_df: pd.DataFrame,
    strategy_recovery: Dict[str, Any],
    goal_recovery: Dict[str, Any],
    exec_recovery: Dict[str, Any],
    minimal_pairs: Dict[str, Any],
    template_diversity: Dict[str, Any],
    readability: Dict[str, Any],
    collision: Dict[str, Any],
) -> str:
    lines = [
        "# Translator Automatic Metrics",
        "",
        "Question. We evaluate the current clean-child translator with fully automatic metrics over a stable sampled audit set.",
        "",
        f"Data. Source CSV: `{args.data_csv}`",
        f"Sampling. First `{args.max_input_rows:,}` rows, keep rows with `source_row_idx % {args.sample_modulus} == 0`.",
        f"Sample size. `{len(sample_df):,}` rows.",
        "",
        "Metrics.",
        "- coverage mass from the existing rule-based audit,",
        "- symbolic collision rate,",
        "- rule-based recoverability from `response_nl`,",
        "- minimal-pair sensitivity,",
        "- template repetition and diversity,",
        "- length, readability, and lexical variety.",
        "",
        "Headline results.",
        f"- Strategy-family recoverability accuracy: `{strategy_recovery['summary']['strategy_family_accuracy']:.3f}`",
        f"- Goal recoverability mean recall across tokens: `{goal_recovery['by_component']['recall'].mean():.3f}`",
        f"- Exec recoverability mean recall across tokens: `{exec_recovery['by_component']['recall'].mean():.3f}`",
        f"- Collision row rate: `{collision['overall']['collision_row_rate']:.3f}`",
        f"- Mean minimal-pair response-changed rate: `{minimal_pairs['by_component']['response_changed_rate'].mean():.3f}`",
        f"- Unique normalized templates: `{template_diversity['summary']['unique_templates']}`",
        f"- Top-10 template row share: `{template_diversity['summary']['top_10_template_row_share']:.3f}`",
        f"- Mean response length (tokens): `{readability['mean_tokens']:.1f}`",
        f"- Flesch-Kincaid grade: `{readability['flesch_kincaid_grade']:.2f}`",
        "",
        "Interpretation. The current template exposes coarse strategy-family information and a subset of goals/exec rules, but the automatic metrics still show substantial collapse through omitted components, collisions, and repeated template families.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_df = build_enriched_sample(args)
    sample_df = add_rule_based_predictions(sample_df)
    component_rows = audit.build_component_rows(sample_df)
    coverage_lookup = build_component_lookup()

    coverage_outputs = compute_coverage_outputs(component_rows)
    collision_outputs = audit.compute_collision_outputs(sample_df)
    strategy_recoverability = compute_strategy_recoverability(sample_df)
    goal_recoverability = compute_token_recoverability(
        sample_df,
        component_type="goal",
        tokens=audit.GOAL_CODES,
        pred_column="pred_goal_tokens",
        truth_column="goal_tokens",
        coverage_lookup=coverage_lookup,
    )
    exec_recoverability = compute_token_recoverability(
        sample_df,
        component_type="exec",
        tokens=audit.EXEC_CODES,
        pred_column="pred_exec_tokens",
        truth_column="exec_tokens",
        coverage_lookup=coverage_lookup,
    )
    minimal_pair_outputs = compute_minimal_pair_outputs(sample_df, coverage_lookup)
    template_diversity = compute_template_diversity(sample_df, args.edit_cluster_threshold, args.top_k_templates)
    readability = compute_length_readability(sample_df)

    coverage_outputs["component_counts"].to_csv(args.out_dir / "coverage_component_counts.csv", index=False)
    coverage_outputs["mass_counts"].to_csv(args.out_dir / "coverage_mass_counts.csv", index=False)
    pd.DataFrame([collision_outputs["overall"]]).to_csv(args.out_dir / "collision_overall.csv", index=False)
    collision_outputs["stratified"].to_csv(args.out_dir / "collision_stratified.csv", index=False)
    collision_outputs["top_collisions"].to_csv(args.out_dir / "collision_examples.csv", index=False)
    strategy_recoverability["confusion"].to_csv(args.out_dir / "strategy_family_confusion.csv", index=False)
    strategy_recoverability["by_family"].to_csv(args.out_dir / "strategy_family_accuracy.csv", index=False)
    goal_recoverability["by_component"].to_csv(args.out_dir / "goal_recoverability_by_component.csv", index=False)
    goal_recoverability["by_class"].to_csv(args.out_dir / "goal_recoverability_by_class.csv", index=False)
    exec_recoverability["by_component"].to_csv(args.out_dir / "exec_recoverability_by_component.csv", index=False)
    exec_recoverability["by_class"].to_csv(args.out_dir / "exec_recoverability_by_class.csv", index=False)
    minimal_pair_outputs["by_component"].to_csv(args.out_dir / "minimal_pair_by_component.csv", index=False)
    minimal_pair_outputs["by_class"].to_csv(args.out_dir / "minimal_pair_by_class.csv", index=False)
    minimal_pair_outputs["examples"].to_csv(args.out_dir / "minimal_pair_examples.csv", index=False)
    template_diversity["top_templates"].to_csv(args.out_dir / "top_templates.csv", index=False)
    template_diversity["clusters"].to_csv(args.out_dir / "template_clusters.csv", index=False)

    overall_json = {
        "strategy_recoverability": strategy_recoverability["summary"],
        "goal_recoverability_mean": {
            "mean_precision": float(goal_recoverability["by_component"]["precision"].mean()),
            "mean_recall": float(goal_recoverability["by_component"]["recall"].mean()),
            "mean_f1": float(goal_recoverability["by_component"]["f1"].mean()),
        },
        "exec_recoverability_mean": {
            "mean_precision": float(exec_recoverability["by_component"]["precision"].mean()),
            "mean_recall": float(exec_recoverability["by_component"]["recall"].mean()),
            "mean_f1": float(exec_recoverability["by_component"]["f1"].mean()),
        },
        "collision": collision_outputs["overall"],
        "template_diversity": template_diversity["summary"],
        "readability": readability,
    }
    (args.out_dir / "overall_metrics.json").write_text(json.dumps(overall_json, indent=2), encoding="utf-8")

    summary_md = build_summary_markdown(
        args=args,
        sample_df=sample_df,
        strategy_recovery=strategy_recoverability,
        goal_recovery=goal_recoverability,
        exec_recovery=exec_recoverability,
        minimal_pairs=minimal_pair_outputs,
        template_diversity=template_diversity,
        readability=readability,
        collision=collision_outputs,
    )
    (args.out_dir / "SUMMARY.md").write_text(summary_md, encoding="utf-8")

    print(summary_md)
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
