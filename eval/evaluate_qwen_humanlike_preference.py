#!/usr/bin/env python3
"""Run Qwen-vs-baseline humanlike preference evaluations.

This script is intentionally self-contained so the Qwen decimal and fraction
preference results can be reproduced from the shared ``eval`` folder. It can
write request files without API calls, or call Gemini directly with
``GEMINI_API_KEY``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROMPT_VERSION = "qwen_humanlike_preference_v1"
DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_API_VERSION = "v1alpha"

FRACTION_CONTEXT = (
    "Target population: sixth- and eighth-grade students, roughly ages 10-14, "
    "solving written fraction arithmetic problems."
)
DECIMAL_CONTEXT = (
    "Target population: sixth- and eighth-grade students, roughly ages 10-14, "
    "solving written decimal arithmetic problems."
)
DECIMAL_FORMAT_NOTE = (
    "Some responses may be compact transcriptions of written work, while others "
    "may be prose explanations. Do not prefer a response just because it is "
    "longer, more polished, or written in complete sentences."
)

PREFERENCE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "preferred": {"type": "string", "enum": ["A", "B", "TIE"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reason": {"type": "string"},
    },
    "required": ["preferred", "confidence", "reason"],
}


@dataclass(frozen=True)
class EntityFrame:
    entity: str
    domain: str
    path: Path
    frame: pd.DataFrame
    sample_with_replacement: bool = False
    required: bool = True


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def normalize_problem(value: Any) -> str:
    text = re.sub(r"\s+", "", clean_text(value))
    return text.replace("÷", ":").replace("×", "*")


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", clean_text(text)).strip("_").lower() or "item"


def stable_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", **kwargs)


def resolve_existing(candidates: Sequence[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def original_tree_path(*parts: str) -> Path:
    return ROOT_DIR.parent / "UMA_PR02" / Path(*parts)


def source_id_from_columns(row: Mapping[str, Any], fallback_row: int, columns: Sequence[str]) -> str:
    values = [clean_text(row.get(column, "")) for column in columns]
    values = [value for value in values if value]
    return ":".join(values) if values else f"row:{fallback_row}"


def normalize_model_csv(path: Path, entity: str, domain: str) -> pd.DataFrame:
    frame = read_csv_flexible(path, dtype=str, keep_default_na=False)
    required = {"problem", "model_response"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    rows = []
    for source_row, row in frame.reset_index(drop=True).iterrows():
        rows.append(
            {
                "entity": entity,
                "domain": domain,
                "source_path": str(path),
                "source_row": int(source_row),
                "source_id": source_id_from_columns(row, source_row, ["sample_idx", "request_id"]),
                "prob": normalize_problem(row["problem"]),
                "answer": clean_text(row.get("parsed_answer", "")),
                "is_correct": clean_text(row.get("is_correct", "")),
                "response_text": clean_text(row["model_response"]),
            }
        )
    return pd.DataFrame(rows).query("prob != '' and response_text != ''").reset_index(drop=True)


def operation_symbol(row: Mapping[str, Any]) -> str:
    operation = clean_text(row.get("operation", "")).lower()
    problem = clean_text(row.get("prob", ""))
    if operation.startswith("add") or "+" in problem:
        return "+"
    if operation.startswith("sub") or "-" in problem:
        return "-"
    if operation.startswith("mul") or "*" in problem or "x" in problem.lower():
        return "x"
    if operation.startswith("div") or ":" in problem:
        return ":"
    return ""


def render_decimal_human(row: Mapping[str, Any]) -> str:
    operand_1 = clean_text(row.get("work_operand_1", ""))
    operand_2 = clean_text(row.get("work_operand_2", ""))
    work_answer = clean_text(row.get("work_answer", ""))
    response = clean_text(row.get("resp", ""))
    symbol = operation_symbol(row)
    lines: list[str] = []
    if operand_1:
        lines.append(operand_1)
    if operand_2:
        lines.append(f"{symbol} {operand_2}" if symbol else operand_2)
    if work_answer:
        lines.append(work_answer)
    if response:
        lines.append(f"Answer: {response}")
    return "\n".join(lines)


def normalize_decimal_human(path: Path) -> pd.DataFrame:
    frame = read_csv_flexible(path, dtype=str, keep_default_na=False)
    required = {"prob", "resp"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    rows = []
    for source_row, row in frame.reset_index(drop=True).iterrows():
        response_text = render_decimal_human(row)
        rows.append(
            {
                "entity": "human",
                "domain": "decimal",
                "source_path": str(path),
                "source_row": int(source_row),
                "source_id": source_id_from_columns(row, source_row, ["subjid", "idx"]),
                "prob": normalize_problem(row["prob"]),
                "answer": clean_text(row.get("resp", "")),
                "is_correct": clean_text(row.get("acc", "")),
                "response_text": response_text,
            }
        )
    return pd.DataFrame(rows).query("prob != '' and response_text != ''").reset_index(drop=True)


def normalize_fraction_human(path: Path) -> pd.DataFrame:
    frame = read_csv_flexible(path, dtype=str, keep_default_na=False)
    required = {"prob", "resp"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    rows = []
    for source_row, row in frame.reset_index(drop=True).iterrows():
        strategy = clean_text(row.get("strat", ""))
        answer = clean_text(row.get("resp", ""))
        response_text = "\n".join(part for part in [strategy, f"Answer: {answer}" if answer else ""] if part)
        rows.append(
            {
                "entity": "human",
                "domain": "fraction",
                "source_path": str(path),
                "source_row": int(source_row),
                "source_id": source_id_from_columns(row, source_row, ["subjid", "seq_idx"]),
                "prob": normalize_problem(row["prob"]),
                "answer": answer,
                "is_correct": clean_text(row.get("acc", "")),
                "response_text": response_text,
            }
        )
    return pd.DataFrame(rows).query("prob != '' and response_text != ''").reset_index(drop=True)


def render_decimal_uma(row: Mapping[str, Any]) -> str:
    work = clean_text(row.get("work", ""))
    if work:
        answer = clean_text(row.get("answer", ""))
        return f"{work}\nAnswer: {answer}" if answer else work
    pieces = [
        clean_text(row.get("strategy", "")),
        clean_text(row.get("goals", "")),
        clean_text(row.get("exec", "")),
    ]
    text = "\n".join(piece for piece in pieces if piece)
    answer = clean_text(row.get("answer", ""))
    return f"{text}\nAnswer: {answer}".strip() if answer else text


def normalize_decimal_uma(path: Path) -> pd.DataFrame:
    frame = read_csv_flexible(path, dtype=str, keep_default_na=False)
    required = {"prob", "answer"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    rows = []
    for source_row, row in frame.reset_index(drop=True).iterrows():
        rows.append(
            {
                "entity": "uma",
                "domain": "decimal",
                "source_path": str(path),
                "source_row": int(source_row),
                "source_id": source_id_from_columns(row, source_row, ["subjid", "sample_idx", "seed"]),
                "prob": normalize_problem(row["prob"]),
                "answer": clean_text(row.get("answer", "")),
                "is_correct": clean_text(row.get("is_correct", "")),
                "response_text": render_decimal_uma(row),
            }
        )
    return pd.DataFrame(rows).query("prob != '' and response_text != ''").reset_index(drop=True)


def normalize_fraction_uma(path: Path, target_probs: set[str]) -> pd.DataFrame:
    if path.is_dir():
        parts = sorted(path.glob("part-*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet shards found in {path}")
        frames = []
        for part in parts:
            shard = pd.read_parquet(part, columns=["source_uid", "prob", "response_nl", "answer", "is_correct"])
            shard = shard.assign(prob=shard["prob"].map(normalize_problem))
            frames.append(shard[shard["prob"].isin(target_probs)])
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = read_csv_flexible(path, dtype=str, keep_default_na=False)
        frame = frame.assign(prob=frame["prob"].map(normalize_problem))
        frame = frame[frame["prob"].isin(target_probs)].reset_index(drop=True)

    rows = []
    for source_row, row in frame.reset_index(drop=True).iterrows():
        response_text = clean_text(row.get("response_nl", ""))
        if not response_text:
            response_text = render_decimal_uma(row)
        rows.append(
            {
                "entity": "uma",
                "domain": "fraction",
                "source_path": str(path),
                "source_row": int(source_row),
                "source_id": source_id_from_columns(row, source_row, ["source_uid", "subjid"]),
                "prob": normalize_problem(row["prob"]),
                "answer": clean_text(row.get("answer", "")),
                "is_correct": clean_text(row.get("is_correct", "")),
                "response_text": response_text,
            }
        )
    return pd.DataFrame(rows).query("prob != '' and response_text != ''").reset_index(drop=True)


def counts_by_problem(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=int)
    return frame.groupby("prob").size()


def inventory_row(
    *,
    domain: str,
    entity: str,
    path: Path | None,
    frame: pd.DataFrame | None,
    status: str,
    reason: str = "",
) -> dict[str, Any]:
    counts = counts_by_problem(frame) if frame is not None else pd.Series(dtype=int)
    return {
        "domain": domain,
        "entity": entity,
        "source_path": str(path) if path else "",
        "status": status,
        "reason": reason,
        "row_count": int(len(frame)) if frame is not None else 0,
        "problem_count": int(len(counts)),
        "min_rows_per_problem": int(counts.min()) if not counts.empty else 0,
        "max_rows_per_problem": int(counts.max()) if not counts.empty else 0,
    }


def add_source(
    sources: dict[str, EntityFrame],
    inventory: list[dict[str, Any]],
    *,
    entity: str,
    domain: str,
    path: Path | None,
    loader: Any,
    samples_per_problem: int,
    sample_with_replacement: bool = False,
    required: bool = True,
) -> None:
    if path is None or not path.exists():
        status = "missing_required" if required else "missing_optional"
        inventory.append(inventory_row(domain=domain, entity=entity, path=path, frame=None, status=status))
        if required:
            raise FileNotFoundError(f"Missing required {domain} source for {entity}: {path}")
        return
    frame = loader(path)
    counts = counts_by_problem(frame)
    if counts.empty:
        inventory.append(
            inventory_row(domain=domain, entity=entity, path=path, frame=frame, status="empty", reason="no usable rows")
        )
        if required:
            raise ValueError(f"Required {domain} source for {entity} has no usable rows: {path}")
        return
    low = counts[counts < samples_per_problem]
    if not sample_with_replacement and not low.empty:
        status = "skipped_low_count" if not required else "low_count_required"
        reason = f"needs {samples_per_problem} rows/problem without replacement; low counts: {low.to_dict()}"
        inventory.append(inventory_row(domain=domain, entity=entity, path=path, frame=frame, status=status, reason=reason))
        if required:
            raise ValueError(f"{entity} is under-sampled for {domain}: {reason}")
        return
    inventory.append(inventory_row(domain=domain, entity=entity, path=path, frame=frame, status="included"))
    sources[entity] = EntityFrame(
        entity=entity,
        domain=domain,
        path=path,
        frame=frame,
        sample_with_replacement=sample_with_replacement,
        required=required,
    )


def load_decimal_sources(args: argparse.Namespace) -> tuple[str, dict[str, EntityFrame], pd.DataFrame]:
    samples_per_problem = args.samples_per_problem
    sources: dict[str, EntityFrame] = {}
    inventory: list[dict[str, Any]] = []
    domain = "decimal"
    qwen_entity = args.decimal_qwen_entity
    qwen = Path(args.decimal_qwen_csv) if args.decimal_qwen_csv else (
        ROOT_DIR / "eval" / "outputs" / "decimal" / "qwen3_4b_distill_humanft.csv"
    )
    add_source(
        sources,
        inventory,
        entity=qwen_entity,
        domain=domain,
        path=qwen,
        loader=lambda path: normalize_model_csv(path, qwen_entity, domain),
        samples_per_problem=samples_per_problem,
        required=True,
    )
    human_path = resolve_existing(
        [
            Path(args.decimal_human_csv) if args.decimal_human_csv else Path("__missing__"),
            ROOT_DIR / "fractionGPT" / "decimals" / "bss2021_human_responses.csv",
            original_tree_path("fractionGPT", "decimals", "bss2021_human_responses.csv"),
        ]
    )
    add_source(
        sources,
        inventory,
        entity="human",
        domain=domain,
        path=human_path,
        loader=normalize_decimal_human,
        samples_per_problem=samples_per_problem,
        sample_with_replacement=True,
        required=True,
    )
    uma_path = resolve_existing(
        [
            Path(args.decimal_uma_csv) if args.decimal_uma_csv else Path("__missing__"),
            ROOT_DIR
            / "results"
            / "UMA_replication"
            / "decimal_bss2021_human_problems_k100_x100_traces"
            / "uma_traces_decimal_human_problems.csv.gz",
            original_tree_path(
                "results",
                "UMA_replication",
                "decimal_bss2021_human_problems_k100_x100_traces",
                "uma_traces_decimal_human_problems.csv.gz",
            ),
        ]
    )
    add_source(
        sources,
        inventory,
        entity="uma",
        domain=domain,
        path=uma_path,
        loader=normalize_decimal_uma,
        samples_per_problem=samples_per_problem,
        required=True,
    )
    for csv_path in sorted((ROOT_DIR / "eval" / "outputs" / "decimal").glob("*.csv")):
        entity = csv_path.stem
        if entity == "qwen3_4b_distill_humanft":
            continue
        add_source(
            sources,
            inventory,
            entity=entity,
            domain=domain,
            path=csv_path,
            loader=lambda path, entity=entity: normalize_model_csv(path, entity, domain),
            samples_per_problem=samples_per_problem,
            required=True,
        )
    return qwen_entity, sources, pd.DataFrame(inventory)


def load_fraction_sources(args: argparse.Namespace) -> tuple[str, dict[str, EntityFrame], pd.DataFrame]:
    samples_per_problem = args.samples_per_problem
    sources: dict[str, EntityFrame] = {}
    inventory: list[dict[str, Any]] = []
    domain = "fraction"
    qwen_entity = args.fraction_qwen_entity
    qwen = Path(args.fraction_qwen_csv) if args.fraction_qwen_csv else (
        ROOT_DIR / "eval" / "outputs" / "fraction" / "qwen3_8b_distill_humanft.csv"
    )
    add_source(
        sources,
        inventory,
        entity=qwen_entity,
        domain=domain,
        path=qwen,
        loader=lambda path: normalize_model_csv(path, qwen_entity, domain),
        samples_per_problem=samples_per_problem,
        required=True,
    )
    target_probs = set(sources[qwen_entity].frame["prob"].unique())
    add_source(
        sources,
        inventory,
        entity="human",
        domain=domain,
        path=ROOT_DIR / "eval" / "data" / "siegler_fraction_human.csv",
        loader=normalize_fraction_human,
        samples_per_problem=samples_per_problem,
        sample_with_replacement=True,
        required=True,
    )
    add_source(
        sources,
        inventory,
        entity="uma",
        domain=domain,
        path=ROOT_DIR / "data" / "distillation" / "uma_fraction_distillation_1000",
        loader=lambda path: normalize_fraction_uma(path, target_probs),
        samples_per_problem=samples_per_problem,
        required=True,
    )
    centaur = ROOT_DIR / "eval" / "outputs" / "fraction" / "centaur_70b.csv"
    add_source(
        sources,
        inventory,
        entity="centaur_70b",
        domain=domain,
        path=centaur,
        loader=lambda path: normalize_model_csv(path, "centaur_70b", domain),
        samples_per_problem=samples_per_problem,
        required=True,
    )
    for csv_path in sorted((ROOT_DIR / "eval" / "outputs" / "fraction").glob("*.csv")):
        entity = csv_path.stem
        if entity in {qwen_entity, "centaur_70b"}:
            continue
        add_source(
            sources,
            inventory,
            entity=entity,
            domain=domain,
            path=csv_path,
            loader=lambda path, entity=entity: normalize_model_csv(path, entity, domain),
            samples_per_problem=samples_per_problem,
            required=False,
        )
    return qwen_entity, sources, pd.DataFrame(inventory)


def parse_baseline_filter(value: str) -> set[str]:
    return {clean_text(item) for item in value.split(",") if clean_text(item)}


def apply_baseline_filter(
    *,
    qwen_entity: str,
    sources: dict[str, EntityFrame],
    inventory: pd.DataFrame,
    only_baselines: str,
) -> tuple[dict[str, EntityFrame], pd.DataFrame]:
    requested = parse_baseline_filter(only_baselines)
    if not requested:
        return sources, inventory

    available = set(sources) - {qwen_entity}
    missing = sorted(requested - available)
    if missing:
        statuses = {}
        if not inventory.empty and {"entity", "status"}.issubset(inventory.columns):
            statuses = dict(zip(inventory["entity"], inventory["status"]))
        raise ValueError(
            "Requested baselines are not available after source loading: "
            f"{missing}. Source statuses: {statuses}"
        )

    keep = requested | {qwen_entity}
    filtered_sources = {entity: source for entity, source in sources.items() if entity in keep}
    filtered_inventory = inventory.copy()
    if not filtered_inventory.empty and "entity" in filtered_inventory.columns:
        selected = filtered_inventory["entity"].isin(keep)
        included_not_selected = (~selected) & (filtered_inventory["status"] == "included")
        filtered_inventory.loc[included_not_selected, "status"] = "skipped_by_filter"
        filtered_inventory.loc[included_not_selected, "reason"] = (
            "excluded by --only-baselines"
        )
    return filtered_sources, filtered_inventory


def sample_entity(entity_frame: EntityFrame, samples_per_problem: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    selected_indices: list[int] = []
    for _, group in entity_frame.frame.groupby("prob", sort=True):
        indices = group.index.tolist()
        replace = entity_frame.sample_with_replacement or len(indices) < samples_per_problem
        if replace:
            selected_indices.extend(rng.choice(indices) for _ in range(samples_per_problem))
        else:
            selected_indices.extend(rng.sample(indices, samples_per_problem))
    sampled = entity_frame.frame.loc[selected_indices].copy().reset_index(drop=True)
    sampled["entity_sample_slot"] = sampled.groupby(["entity", "prob"]).cumcount()
    sampled["entity_sample_id"] = sampled.apply(
        lambda row: f"{row['entity']}:{safe_slug(row['prob'])}:{int(row['entity_sample_slot']):03d}",
        axis=1,
    )
    return sampled


def build_entity_samples(sources: Mapping[str, EntityFrame], samples_per_problem: int, seed: int) -> pd.DataFrame:
    frames = []
    for idx, entity in enumerate(sorted(sources)):
        frames.append(sample_entity(sources[entity], samples_per_problem, seed + 17 * (idx + 1)))
    return pd.concat(frames, ignore_index=True).sort_values(["entity", "prob", "entity_sample_slot"]).reset_index(drop=True)


def side_payload(row: Mapping[str, Any], side: str) -> dict[str, Any]:
    prefix = f"{side.lower()}_"
    return {
        f"{prefix}entity": clean_text(row["entity"]),
        f"{prefix}entity_sample_id": clean_text(row["entity_sample_id"]),
        f"{prefix}source_id": clean_text(row["source_id"]),
        f"{prefix}source_path": clean_text(row["source_path"]),
        f"{prefix}source_row": int(row["source_row"]),
        f"{prefix}answer": clean_text(row["answer"]),
        f"response_{side.lower()}": clean_text(row["response_text"]),
    }


def build_pairings(
    *,
    qwen_entity: str,
    entity_samples: pd.DataFrame,
    samples_per_problem: int,
    seed: int,
    max_pairs_per_comparison: int = 0,
) -> pd.DataFrame:
    if samples_per_problem % 2 != 0:
        raise ValueError("samples_per_problem must be even for balanced A/B side assignment.")
    rng = random.Random(seed + 101)
    by_entity_problem = {
        (entity, prob): group.sort_values("entity_sample_slot").reset_index(drop=True)
        for (entity, prob), group in entity_samples.groupby(["entity", "prob"], sort=True)
    }
    problems = sorted(entity_samples["prob"].unique())
    baselines = sorted(entity for entity in entity_samples["entity"].unique() if entity != qwen_entity)
    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        pair_count = 0
        preference_set = f"{qwen_entity}_vs_{baseline}"
        for problem_index, prob in enumerate(problems):
            qwen_rows = by_entity_problem.get((qwen_entity, prob))
            baseline_rows = by_entity_problem.get((baseline, prob))
            if qwen_rows is None or baseline_rows is None:
                continue
            qwen_order = list(range(samples_per_problem))
            baseline_order = list(range(samples_per_problem))
            qwen_on_a_flags = [True] * (samples_per_problem // 2) + [False] * (samples_per_problem // 2)
            rng.shuffle(qwen_order)
            rng.shuffle(baseline_order)
            rng.shuffle(qwen_on_a_flags)
            for pair_index, (q_idx, b_idx, qwen_on_a) in enumerate(
                zip(qwen_order, baseline_order, qwen_on_a_flags)
            ):
                qwen_row = qwen_rows.iloc[q_idx].to_dict()
                baseline_row = baseline_rows.iloc[b_idx].to_dict()
                if qwen_on_a:
                    side_a = side_payload(qwen_row, "a")
                    side_b = side_payload(baseline_row, "b")
                    qwen_side = "A"
                    baseline_side = "B"
                else:
                    side_a = side_payload(baseline_row, "a")
                    side_b = side_payload(qwen_row, "b")
                    qwen_side = "B"
                    baseline_side = "A"
                rows.append(
                    {
                        "request_id": f"{preference_set}:problem:{problem_index:03d}:pair:{pair_index:03d}",
                        "preference_set": preference_set,
                        "prob": prob,
                        "problem_index": problem_index,
                        "pair_index": pair_index,
                        "qwen_entity": qwen_entity,
                        "baseline_entity": baseline,
                        "qwen_side": qwen_side,
                        "baseline_side": baseline_side,
                        **side_a,
                        **side_b,
                    }
                )
                pair_count += 1
                if max_pairs_per_comparison > 0 and pair_count >= max_pairs_per_comparison:
                    break
            if max_pairs_per_comparison > 0 and pair_count >= max_pairs_per_comparison:
                break
    return pd.DataFrame(rows)


def build_system_prompt(domain: str) -> str:
    context = DECIMAL_CONTEXT if domain == "decimal" else FRACTION_CONTEXT
    lines = [
        f"Compare two written solutions to the same {domain} arithmetic problem.",
        "Your task is to decide which solution seems more human-like for the target student population.",
        "",
        context,
        "",
        "For each pair, choose the response that is more human-like for this population.",
        "Focus on whether the work sounds like a student's written reasoning: wording, procedure choice, step order, amount of explanation, and kinds of mistakes.",
        "Do not choose solely because one answer is mathematically correct. A wrong answer can be more human-like than a correct answer.",
    ]
    if domain == "decimal":
        lines.append(DECIMAL_FORMAT_NOTE)
    lines.extend(
        [
            "Do not assume either response is human-written or model-generated.",
            "Return JSON only.",
        ]
    )
    return "\n".join(lines)


def build_user_prompt(row: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"Problem: {row['prob']}",
            "",
            "Response A:",
            clean_text(row["response_a"]),
            "",
            "Response B:",
            clean_text(row["response_b"]),
            "",
            "Which response is more human-like for the target student population?",
            "",
            "Return exactly one JSON object with this schema:",
            '{"preferred": "A|B|TIE", "confidence": "low|medium|high", "reason": "one short sentence"}',
        ]
    )


def make_request_records(pairings: pd.DataFrame, args: argparse.Namespace, domain: str) -> list[dict[str, Any]]:
    system_prompt = build_system_prompt(domain)
    records = []
    for row in pairings.to_dict("records"):
        user_prompt = build_user_prompt(row)
        key_payload = {
            "prompt_version": PROMPT_VERSION,
            "judge_model": args.judge_model,
            "request_id": row["request_id"],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
        records.append(
            {
                **row,
                "prompt_version": PROMPT_VERSION,
                "judge_model": args.judge_model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "request_key": stable_hash(key_payload),
            }
        )
    return records


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_gemini_client(args: argparse.Namespace) -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError("Gemini backend requires google-genai in the active environment.") from exc
    api_key = os.environ.get(args.gemini_api_key_env)
    if not api_key:
        raise EnvironmentError(f"{args.gemini_api_key_env} is not set.")
    return genai.Client(api_key=api_key, http_options={"api_version": args.gemini_api_version})


def gemini_config(args: argparse.Namespace, system_prompt: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_tokens,
        "response_mime_type": "application/json",
        "response_schema": PREFERENCE_RESPONSE_SCHEMA,
    }
    if args.thinking_budget >= 0:
        config["thinking_config"] = {"thinking_budget": args.thinking_budget}
    return config


def response_to_payload(response: Any) -> Any:
    if isinstance(response, Mapping):
        return dict(response)
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_json_dict"):
        return response.to_json_dict()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return {"repr": repr(response)}


def text_from_response(response: Any) -> str:
    if not response:
        return ""
    text = getattr(response, "text", None)
    if text:
        return clean_text(text)
    if isinstance(response, Mapping):
        text = response.get("text")
        if text:
            return clean_text(text)
        candidates = response.get("candidates") or []
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(clean_text(part.get("text", "")) for part in parts)
    return ""


def gemini_usage(response_json: Mapping[str, Any] | None) -> dict[str, int]:
    if not response_json:
        return {}
    usage = response_json.get("usage_metadata") or response_json.get("usageMetadata") or response_json.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_token_count") or usage.get("promptTokenCount") or usage.get("prompt_tokens") or 0),
        "completion_tokens": int(
            usage.get("candidates_token_count") or usage.get("candidatesTokenCount") or usage.get("completion_tokens") or 0
        ),
        "total_tokens": int(usage.get("total_token_count") or usage.get("totalTokenCount") or usage.get("total_tokens") or 0),
    }


def run_gemini_request(
    record: Mapping[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
    client: Any,
) -> dict[str, Any]:
    cache_path = out_dir / "cache" / f"{record['request_key']}.json"
    if cache_path.exists() and not args.force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    response_json: dict[str, Any] | None = None
    response_text = ""
    request_error = ""
    payload = {
        "model": args.judge_model,
        "contents": str(record["user_prompt"]),
        "config": gemini_config(args, str(record["system_prompt"])),
    }
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.models.generate_content(**payload)
            response_json = response_to_payload(response)
            response_text = text_from_response(response)
            request_error = ""
            break
        except Exception as exc:  # noqa: BLE001
            request_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                time.sleep(args.retry_base_seconds * (2 ** (attempt - 1)))
    cached = {
        "request": dict(record),
        "payload": payload,
        "response_json": response_json,
        "response_text": response_text,
        "request_error": request_error,
        "usage": gemini_usage(response_json),
    }
    cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
    return cached


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = clean_text(text)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def normalize_preferred(value: Any) -> str:
    text = clean_text(value).upper()
    if text in {"A", "RESPONSE A", "STUDENT A"}:
        return "A"
    if text in {"B", "RESPONSE B", "STUDENT B"}:
        return "B"
    if text in {"TIE", "BOTH", "NEITHER", "UNCLEAR", "EQUAL"}:
        return "TIE"
    return ""


def prediction_row(record: Mapping[str, Any], cached: Mapping[str, Any]) -> dict[str, Any]:
    parsed = parse_json_object(clean_text(cached.get("response_text", "")))
    preferred = normalize_preferred(parsed.get("preferred"))
    preferred_entity = ""
    if preferred == "A":
        preferred_entity = clean_text(record["a_entity"])
    elif preferred == "B":
        preferred_entity = clean_text(record["b_entity"])
    elif preferred == "TIE":
        preferred_entity = "TIE"
    usage = cached.get("usage") or {}
    return {
        "request_id": record["request_id"],
        "preference_set": record["preference_set"],
        "prob": record["prob"],
        "qwen_entity": record["qwen_entity"],
        "baseline_entity": record["baseline_entity"],
        "qwen_side": record["qwen_side"],
        "preferred": preferred,
        "preferred_entity": preferred_entity,
        "parse_success": bool(preferred),
        "confidence": clean_text(parsed.get("confidence")),
        "reason": clean_text(parsed.get("reason")),
        "raw_response_text": clean_text(cached.get("response_text", "")),
        "request_error": clean_text(cached.get("request_error", "")),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "request_key": record["request_key"],
    }


def summarize_predictions(predictions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return {"row_count": 0}, pd.DataFrame(), pd.DataFrame()
    pred = predictions.copy()
    pred["qwen_preferred"] = pred["preferred_entity"] == pred["qwen_entity"]
    pred["baseline_preferred"] = pred["preferred_entity"] == pred["baseline_entity"]
    pred["tie"] = pred["preferred_entity"] == "TIE"
    summary = (
        pred.groupby(["qwen_entity", "baseline_entity", "preference_set"], as_index=False)
        .agg(
            rows=("request_id", "size"),
            parse_success_rate=("parse_success", "mean"),
            qwen_preference_rate=("qwen_preferred", "mean"),
            baseline_preference_rate=("baseline_preferred", "mean"),
            tie_rate=("tie", "mean"),
            qwen_on_a_rate=("qwen_side", lambda x: float((x == "A").mean())),
        )
        .sort_values(["qwen_entity", "baseline_entity"])
    )
    by_problem = (
        pred.groupby(["qwen_entity", "baseline_entity", "prob"], as_index=False)
        .agg(
            rows=("request_id", "size"),
            parse_success_rate=("parse_success", "mean"),
            qwen_preference_rate=("qwen_preferred", "mean"),
            baseline_preference_rate=("baseline_preferred", "mean"),
            tie_rate=("tie", "mean"),
        )
        .sort_values(["qwen_entity", "baseline_entity", "prob"])
    )
    metrics = {
        "row_count": int(len(pred)),
        "parse_success_rate": float(pred["parse_success"].mean()),
        "preference_set_count": int(pred["preference_set"].nunique()),
        "problem_count": int(pred["prob"].nunique()),
        "total_prompt_tokens": int(pd.to_numeric(pred["prompt_tokens"], errors="coerce").fillna(0).sum()),
        "total_completion_tokens": int(pd.to_numeric(pred["completion_tokens"], errors="coerce").fillna(0).sum()),
        "total_tokens": int(pd.to_numeric(pred["total_tokens"], errors="coerce").fillna(0).sum()),
    }
    return metrics, summary, by_problem


def run_plan(domain: str, args: argparse.Namespace, out_dir: Path) -> int:
    if domain == "decimal":
        qwen_entity, sources, inventory = load_decimal_sources(args)
    elif domain == "fraction":
        qwen_entity, sources, inventory = load_fraction_sources(args)
    else:
        raise ValueError(f"Unsupported domain: {domain}")
    sources, inventory = apply_baseline_filter(
        qwen_entity=qwen_entity,
        sources=sources,
        inventory=inventory,
        only_baselines=args.only_baselines,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)
    inventory.to_csv(out_dir / "source_inventory.csv", index=False)

    entity_samples = build_entity_samples(sources, args.samples_per_problem, args.seed)
    pairings = build_pairings(
        qwen_entity=qwen_entity,
        entity_samples=entity_samples,
        samples_per_problem=args.samples_per_problem,
        seed=args.seed,
        max_pairs_per_comparison=args.max_pairs_per_comparison,
    )
    if pairings.empty:
        raise ValueError(f"No pairings generated for {domain}.")
    records = make_request_records(pairings, args, domain)

    entity_samples.to_csv(out_dir / "entity_samples.csv", index=False)
    pairings.to_csv(out_dir / "pairings.csv", index=False)
    write_jsonl(out_dir / "requests.jsonl", records)
    manifest = {
        "domain": domain,
        "prompt_version": PROMPT_VERSION,
        "backend": args.backend,
        "judge_model": args.judge_model,
        "seed": args.seed,
        "samples_per_problem": args.samples_per_problem,
        "max_pairs_per_comparison": args.max_pairs_per_comparison,
        "only_baselines": sorted(parse_baseline_filter(args.only_baselines)),
        "row_count": len(records),
        "qwen_entity": qwen_entity,
        "included_entities": sorted(sources),
        "skipped_sources": inventory[inventory["status"] != "included"].to_dict("records"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.backend == "requests":
        metrics = {
            "backend": args.backend,
            "judge_model": args.judge_model,
            "row_count": len(records),
            "preference_set_count": int(pairings["preference_set"].nunique()),
            "problem_count": int(pairings["prob"].nunique()),
            "prompt_version": PROMPT_VERSION,
        }
        (out_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {len(records)} {domain} preference requests to {out_dir / 'requests.jsonl'}", flush=True)
        return 0

    client = make_gemini_client(args)
    raw_rows = []
    pred_rows = []
    for idx, record in enumerate(records, start=1):
        cached = run_gemini_request(record, args, out_dir, client)
        raw_rows.append(cached)
        pred_rows.append(prediction_row(record, cached))
        if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == len(records)):
            print(f"{domain}: completed {idx}/{len(records)} Gemini judgments", flush=True)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_jsonl(out_dir / "responses_raw.jsonl", raw_rows)
    predictions = pd.DataFrame(pred_rows)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    metrics, preference_summary, preference_by_problem = summarize_predictions(predictions)
    metrics.update(
        {
            "backend": args.backend,
            "judge_model": args.judge_model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "prompt_version": PROMPT_VERSION,
        }
    )
    preference_summary.to_csv(out_dir / "preference_summary.csv", index=False)
    preference_by_problem.to_csv(out_dir / "preference_by_problem.csv", index=False)
    (out_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Completed {domain} Gemini judgments: rows={metrics['row_count']} parse_success={metrics['parse_success_rate']:.3f}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=["decimal", "fraction", "both"], default="both")
    parser.add_argument("--backend", choices=["requests", "gemini_sync"], default="requests")
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR / "results" / "humanlike_preference" / "qwen_baselines")
    parser.add_argument("--samples-per-problem", type=int, default=100)
    parser.add_argument("--max-pairs-per-comparison", type=int, default=0)
    parser.add_argument(
        "--only-baselines",
        default="",
        help="Comma-separated baseline entity names to compare against Qwen.",
    )
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--gemini-api-version", default=DEFAULT_GEMINI_API_VERSION)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--decimal-qwen-csv",
        default="",
        help="Optional replacement CSV for the decimal Qwen entity.",
    )
    parser.add_argument(
        "--decimal-qwen-entity",
        default="qwen3_4b_distill_humanft",
        help="Entity name to use for the decimal Qwen source.",
    )
    parser.add_argument("--decimal-human-csv", default="")
    parser.add_argument("--decimal-uma-csv", default="")
    parser.add_argument(
        "--fraction-qwen-csv",
        default="",
        help="Optional replacement CSV for the fraction Qwen entity.",
    )
    parser.add_argument(
        "--fraction-qwen-entity",
        default="qwen3_8b_distill_humanft",
        help="Entity name to use for the fraction Qwen source.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    domains = ["decimal", "fraction"] if args.plan == "both" else [args.plan]
    for domain in domains:
        out_dir = args.out_dir / domain if args.plan == "both" else args.out_dir
        run_plan(domain, args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
