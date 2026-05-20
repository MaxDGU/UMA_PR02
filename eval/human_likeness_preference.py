#!/usr/bin/env python3
"""Build and optionally run human-likeness preference judgments.

The default comparison is panel25 distilled fraction traces versus the repaired
Max-prompt frontier traces on the eight human-FT fraction problems. The prompt
asks a judge model which response is more likely to have been written by a
student from the target child population, while explicitly not treating
mathematical correctness as the only criterion.
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
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import quote

import pandas as pd
import requests as http_requests


EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_EVAL_DIR = (
    ROOT_DIR
    / "results"
    / "procedural_alignment"
    / "human_ft_fraction_eval_20260428_200211"
)
DEFAULT_FRONTIER_DIR = (
    ROOT_DIR
    / "results"
    / "procedural_alignment"
    / "frontier_human_ft_max_prompt"
)
DEFAULT_DISTILLED_CSV = (
    DEFAULT_EVAL_DIR
    / "panel25_distilled"
    / "panel25_distilled_human_ft_rollouts.csv.gz"
)
DEFAULT_GEMINI_CSV = (
    DEFAULT_FRONTIER_DIR
    / "gemini_2p5_flash_2000_repaired"
    / "parsed_samples.csv"
)
DEFAULT_GPT_CSV = (
    DEFAULT_FRONTIER_DIR
    / "gpt_4p1_mini_2000_repaired"
    / "parsed_samples.csv"
)
DEFAULT_OUT_DIR = (
    EVAL_DIR
    / "results"
    / "human_likeness_preference"
    / "distilled_vs_frontier_v1"
)
DEFAULT_SANDBOX_ENDPOINT = "https://api-ai-sandbox.princeton.edu/"
DEFAULT_PORTKEY_BASE_URL = "https://api.portkey.ai/v1"
DEFAULT_SANDBOX_API_VERSION = "2025-03-01-preview"

PROMPT_VERSION = "human_likeness_preference_v5"
DEFAULT_DEMOGRAPHIC_CONTEXT = (
    "Target population: sixth- and eighth-grade students, roughly ages 10-14, "
    "solving written fraction arithmetic problems."
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    trace_col: str
    answer_col: str
    id_col: str
    problem_col: str = "prob"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug or "source"


def stable_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def parse_source_spec(text: str, default_trace_col: str, default_answer_col: str, default_id_col: str) -> SourceSpec:
    """Parse name/path source specs.

    Format:
      name=gemini,path=results/...csv,trace_col=model_response,answer_col=parsed_answer,id_col=request_id

    Only name and path are required when defaults are supplied by the caller.
    """

    fields: dict[str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Source spec item must be key=value: {item!r}")
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()
    missing = [key for key in ("name", "path") if key not in fields]
    if missing:
        raise ValueError(f"Source spec missing required keys {missing}: {text!r}")
    return SourceSpec(
        name=fields["name"],
        path=Path(fields["path"]),
        trace_col=fields.get("trace_col", default_trace_col),
        answer_col=fields.get("answer_col", default_answer_col),
        id_col=fields.get("id_col", default_id_col),
        problem_col=fields.get("problem_col", "prob"),
    )


def load_source(spec: SourceSpec) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(f"Source CSV not found for {spec.name}: {spec.path}")
    frame = read_csv_flexible(spec.path).copy()
    required = [spec.problem_col, spec.trace_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{spec.name} source missing required columns {missing}: {spec.path}")
    if spec.answer_col and spec.answer_col not in frame.columns:
        frame[spec.answer_col] = ""
    if spec.id_col and spec.id_col not in frame.columns:
        frame[spec.id_col] = ""

    out = pd.DataFrame(
        {
            "source_name": spec.name,
            "source_path": str(spec.path),
            "source_row": range(len(frame)),
            "source_id": frame[spec.id_col].map(clean_text) if spec.id_col else "",
            "prob": frame[spec.problem_col].map(clean_text),
            "answer": frame[spec.answer_col].map(clean_text) if spec.answer_col else "",
            "trace": frame[spec.trace_col].map(clean_text),
        }
    )
    out = out[(out["prob"] != "") & (out["trace"] != "")].reset_index(drop=True)
    return out


def sample_group(frame: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    replace = len(frame) < n
    indices = frame.index.tolist()
    if replace:
        selected = [rng.choice(indices) for _ in range(n)]
    else:
        selected = rng.sample(indices, n)
    return frame.loc[selected].reset_index(drop=True)


def build_system_prompt(demographic_context: str) -> str:
    return "\n".join(
        [
            "Compare two written solutions to the same fraction arithmetic problem.",
            "Your task is to decide which solution seems more human-like for the target student population.",
            "",
            demographic_context,
            "",
            "For each pair, choose the response that is more human-like for this population.",
            "Focus on whether the work sounds like a student's written reasoning: wording, procedure choice, step order, amount of explanation, and kinds of mistakes.",
            "Do not choose solely because one answer is mathematically correct. A wrong answer can be more human-like than a correct answer.",
            "Do not assume either response is human-written or model-generated.",
            "Return JSON only.",
        ]
    )


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


def source_row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_name": clean_text(row["source_name"]),
        "source_path": clean_text(row["source_path"]),
        "source_row": int(row["source_row"]),
        "source_id": clean_text(row["source_id"]),
        "answer": clean_text(row["answer"]),
        "trace": clean_text(row["trace"]),
    }


def make_comparison_rows(
    baseline: pd.DataFrame,
    comparisons: Mapping[str, pd.DataFrame],
    samples_per_problem: int,
    seed: int,
    max_pairs_per_comparison: int = 0,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    baseline_name = clean_text(baseline["source_name"].iloc[0]) if not baseline.empty else "baseline"
    baseline_by_problem = {prob: group.copy() for prob, group in baseline.groupby("prob", sort=True)}

    for comparison_name, comparison in comparisons.items():
        comparison_by_problem = {prob: group.copy() for prob, group in comparison.groupby("prob", sort=True)}
        shared_probs = sorted(set(baseline_by_problem) & set(comparison_by_problem))
        pair_count = 0
        for problem_index, prob in enumerate(shared_probs):
            left_rows = sample_group(baseline_by_problem[prob], samples_per_problem, rng)
            right_rows = sample_group(comparison_by_problem[prob], samples_per_problem, rng)
            for sample_idx in range(samples_per_problem):
                baseline_payload = source_row_payload(left_rows.iloc[sample_idx])
                comparison_payload = source_row_payload(right_rows.iloc[sample_idx])
                baseline_on_a = rng.random() < 0.5
                if baseline_on_a:
                    source_a = baseline_payload
                    source_b = comparison_payload
                    label_a = baseline_name
                    label_b = comparison_name
                else:
                    source_a = comparison_payload
                    source_b = baseline_payload
                    label_a = comparison_name
                    label_b = baseline_name
                request_id = (
                    f"{safe_slug(baseline_name)}_vs_{safe_slug(comparison_name)}:"
                    f"problem:{problem_index:03d}:sample:{sample_idx:03d}"
                )
                rows.append(
                    {
                        "request_id": request_id,
                        "comparison_name": comparison_name,
                        "baseline_name": baseline_name,
                        "prob": prob,
                        "problem_index": problem_index,
                        "sample_idx": sample_idx,
                        "source_a": label_a,
                        "source_b": label_b,
                        "response_a": source_a["trace"],
                        "response_b": source_b["trace"],
                        "answer_a": source_a["answer"],
                        "answer_b": source_b["answer"],
                        "source_a_row": source_a["source_row"],
                        "source_b_row": source_b["source_row"],
                        "source_a_id": source_a["source_id"],
                        "source_b_id": source_b["source_id"],
                        "baseline_side": "A" if baseline_on_a else "B",
                        "comparison_side": "B" if baseline_on_a else "A",
                    }
                )
                pair_count += 1
                if max_pairs_per_comparison > 0 and pair_count >= max_pairs_per_comparison:
                    break
            if max_pairs_per_comparison > 0 and pair_count >= max_pairs_per_comparison:
                break
    return pd.DataFrame(rows)


def make_request_records(comparisons: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    system_prompt = build_system_prompt(args.demographic_context)
    records = []
    for row in comparisons.to_dict("records"):
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


def request_payload(record: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.judge_model,
        "messages": [
            {"role": "system", "content": record["system_prompt"]},
            {"role": "user", "content": record["user_prompt"]},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }


def sandbox_chat_url(args: argparse.Namespace) -> str:
    if args.api_style == "portkey":
        return f"{args.portkey_base_url.rstrip('/')}/chat/completions"
    endpoint = args.sandbox_endpoint.rstrip("/")
    model = quote(args.judge_model, safe="")
    return f"{endpoint}/openai/deployments/{model}/chat/completions?api-version={args.api_version}"


def extract_response_text(response_json: Mapping[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(clean_text(item.get("text", item)) if isinstance(item, Mapping) else clean_text(item) for item in content)
    return clean_text(content)


def call_sandbox(payload: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"{args.api_key_env} is not set.")
    if args.api_style == "portkey":
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    else:
        headers = {"api-key": api_key, "Content-Type": "application/json"}
    last_error = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            response = http_requests.post(
                sandbox_chat_url(args),
                headers=headers,
                json=payload,
                timeout=args.request_timeout_seconds,
            )
            if response.status_code < 400:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < args.max_retries:
            time.sleep(args.retry_base_seconds * (2 ** (attempt - 1)))
    raise RuntimeError(last_error)


def run_request(record: Mapping[str, Any], args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    cache_path = out_dir / "cache" / f"{record['request_key']}.json"
    if cache_path.exists() and not args.force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    payload = request_payload(record, args)
    response_json: Optional[dict[str, Any]] = None
    response_text = ""
    request_error = ""
    try:
        response_json = call_sandbox(payload, args)
        response_text = extract_response_text(response_json)
    except Exception as exc:  # noqa: BLE001
        request_error = f"{type(exc).__name__}: {exc}"

    cached = {
        "request": dict(record),
        "payload": payload,
        "response_json": response_json,
        "response_text": response_text,
        "request_error": request_error,
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
    response_text = clean_text(cached.get("response_text", ""))
    parsed = parse_json_object(response_text)
    preferred = normalize_preferred(parsed.get("preferred"))
    preferred_source = ""
    if preferred == "A":
        preferred_source = clean_text(record["source_a"])
    elif preferred == "B":
        preferred_source = clean_text(record["source_b"])
    elif preferred == "TIE":
        preferred_source = "TIE"
    usage = (cached.get("response_json") or {}).get("usage") or {}
    return {
        "request_id": record["request_id"],
        "comparison_name": record["comparison_name"],
        "baseline_name": record["baseline_name"],
        "prob": record["prob"],
        "sample_idx": record["sample_idx"],
        "source_a": record["source_a"],
        "source_b": record["source_b"],
        "baseline_side": record["baseline_side"],
        "preferred": preferred,
        "preferred_source": preferred_source,
        "parse_success": bool(preferred),
        "confidence": clean_text(parsed.get("confidence")),
        "reason": clean_text(parsed.get("reason")),
        "raw_response_text": response_text,
        "request_error": clean_text(cached.get("request_error", "")),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "request_key": record["request_key"],
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_predictions(predictions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return {"row_count": 0}, pd.DataFrame(), pd.DataFrame()
    pred = predictions.copy()
    pred["baseline_preferred"] = pred["preferred_source"] == pred["baseline_name"]
    pred["comparison_preferred"] = pred["preferred_source"] == pred["comparison_name"]
    pred["tie"] = pred["preferred_source"] == "TIE"
    summary = (
        pred.groupby(["baseline_name", "comparison_name"], as_index=False)
        .agg(
            rows=("request_id", "size"),
            parse_success_rate=("parse_success", "mean"),
            baseline_preference_rate=("baseline_preferred", "mean"),
            comparison_preference_rate=("comparison_preferred", "mean"),
            tie_rate=("tie", "mean"),
            baseline_on_a_rate=("baseline_side", lambda x: float((x == "A").mean())),
        )
        .sort_values(["baseline_name", "comparison_name"])
    )
    by_problem = (
        pred.groupby(["baseline_name", "comparison_name", "prob"], as_index=False)
        .agg(
            rows=("request_id", "size"),
            parse_success_rate=("parse_success", "mean"),
            baseline_preference_rate=("baseline_preferred", "mean"),
            comparison_preference_rate=("comparison_preferred", "mean"),
            tie_rate=("tie", "mean"),
        )
        .sort_values(["comparison_name", "prob"])
    )
    metrics = {
        "row_count": int(len(pred)),
        "parse_success_rate": float(pred["parse_success"].mean()),
        "comparison_count": int(pred["comparison_name"].nunique()),
        "problem_count": int(pred["prob"].nunique()),
        "total_prompt_tokens": int(pd.to_numeric(pred["prompt_tokens"], errors="coerce").fillna(0).sum()),
        "total_completion_tokens": int(pd.to_numeric(pred["completion_tokens"], errors="coerce").fillna(0).sum()),
        "total_tokens": int(pd.to_numeric(pred["total_tokens"], errors="coerce").fillna(0).sum()),
    }
    return metrics, summary, by_problem


def write_prompt_doc(out_dir: Path, records: Sequence[Mapping[str, Any]], demographic_context: str) -> None:
    example = records[0] if records else {}
    lines = [
        "# Human-Likeness Preference Prompt",
        "",
        f"- prompt_version: `{PROMPT_VERSION}`",
        "",
        "## Demographic Context",
        "",
        demographic_context,
        "",
        "## System",
        "",
        build_system_prompt(demographic_context),
        "",
        "## User Template Example",
        "",
        clean_text(example.get("user_prompt", "")),
        "",
    ]
    (out_dir / "prompt.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["requests", "sandbox"], default="requests")
    parser.add_argument("--baseline_csv", default=str(DEFAULT_DISTILLED_CSV))
    parser.add_argument("--baseline_name", default="distilled_panel25")
    parser.add_argument("--baseline_trace_col", default="generation_text")
    parser.add_argument("--baseline_answer_col", default="pred_answer")
    parser.add_argument("--baseline_id_col", default="")
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help=(
            "Comparison source spec. Format: "
            "name=gemini,path=results/run.csv,trace_col=model_response,answer_col=parsed_answer,id_col=request_id"
        ),
    )
    parser.add_argument("--comparison_trace_col", default="model_response")
    parser.add_argument("--comparison_answer_col", default="parsed_answer")
    parser.add_argument("--comparison_id_col", default="request_id")
    parser.add_argument("--samples_per_problem", type=int, default=25)
    parser.add_argument("--max_pairs_per_comparison", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--demographic_context", default=DEFAULT_DEMOGRAPHIC_CONTEXT)
    parser.add_argument("--judge_model", default="gpt-4.1-mini")
    parser.add_argument("--api_style", choices=["azure", "portkey"], default=os.environ.get("AI_SANDBOX_API_STYLE", "portkey"))
    parser.add_argument("--sandbox_endpoint", default=os.environ.get("AI_SANDBOX_ENDPOINT", DEFAULT_SANDBOX_ENDPOINT))
    parser.add_argument("--portkey_base_url", default=os.environ.get("AI_SANDBOX_PORTKEY_BASE_URL", DEFAULT_PORTKEY_BASE_URL))
    parser.add_argument("--api_version", default=os.environ.get("AI_SANDBOX_API_VERSION", DEFAULT_SANDBOX_API_VERSION))
    parser.add_argument("--api_key_env", default="AI_SANDBOX_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=192)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--retry_base_seconds", type=float, default=2.0)
    parser.add_argument("--request_timeout_seconds", type=float, default=120.0)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args(argv)


def default_comparisons(args: argparse.Namespace) -> list[SourceSpec]:
    specs = []
    if DEFAULT_GEMINI_CSV.exists():
        specs.append(
            SourceSpec(
                name="gemini_2p5_flash",
                path=DEFAULT_GEMINI_CSV,
                trace_col=args.comparison_trace_col,
                answer_col=args.comparison_answer_col,
                id_col=args.comparison_id_col,
            )
        )
    if DEFAULT_GPT_CSV.exists():
        specs.append(
            SourceSpec(
                name="gpt_4p1_mini",
                path=DEFAULT_GPT_CSV,
                trace_col=args.comparison_trace_col,
                answer_col=args.comparison_answer_col,
                id_col=args.comparison_id_col,
            )
        )
    return specs


def write_summary(out_dir: Path, args: argparse.Namespace, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Human-Likeness Preference Run",
        "",
        f"- prompt_version: `{PROMPT_VERSION}`",
        f"- backend: `{args.backend}`",
        f"- judge_model: `{args.judge_model}`",
        f"- samples_per_problem: {args.samples_per_problem}",
        f"- rows: {metrics.get('row_count', 'n/a')}",
        f"- parse_success_rate: {metrics.get('parse_success_rate', 'n/a')}",
        "",
        "The default target population is sixth- and eighth-grade students working",
        "on written fraction arithmetic problems. The judge is asked to choose the",
        "more human-like response, not simply the more correct response.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)

    baseline_spec = SourceSpec(
        name=args.baseline_name,
        path=Path(args.baseline_csv),
        trace_col=args.baseline_trace_col,
        answer_col=args.baseline_answer_col,
        id_col=args.baseline_id_col,
    )
    comparison_specs = [
        parse_source_spec(
            text,
            args.comparison_trace_col,
            args.comparison_answer_col,
            args.comparison_id_col,
        )
        for text in args.comparison
    ]
    if not comparison_specs:
        comparison_specs = default_comparisons(args)
    if not comparison_specs:
        raise ValueError("No comparison sources were provided and no default frontier outputs were found.")

    baseline = load_source(baseline_spec)
    comparison_frames = {spec.name: load_source(spec) for spec in comparison_specs}
    comparisons = make_comparison_rows(
        baseline=baseline,
        comparisons=comparison_frames,
        samples_per_problem=args.samples_per_problem,
        seed=args.seed,
        max_pairs_per_comparison=args.max_pairs_per_comparison,
    )
    if comparisons.empty:
        raise ValueError("No comparison rows could be generated; check problem overlap and source columns.")
    records = make_request_records(comparisons, args)

    comparisons.to_csv(out_dir / "comparisons.csv", index=False)
    write_jsonl(out_dir / "requests.jsonl", records)
    write_prompt_doc(out_dir, records, args.demographic_context)

    source_manifest = {
        "prompt_version": PROMPT_VERSION,
        "backend": args.backend,
        "baseline": baseline_spec.__dict__ | {"path": str(baseline_spec.path), "row_count": int(len(baseline))},
        "comparisons": [
            spec.__dict__ | {"path": str(spec.path), "row_count": int(len(comparison_frames[spec.name]))}
            for spec in comparison_specs
        ],
        "samples_per_problem": args.samples_per_problem,
        "max_pairs_per_comparison": args.max_pairs_per_comparison,
        "seed": args.seed,
        "row_count": int(len(records)),
        "problem_count": int(comparisons["prob"].nunique()),
        "demographic_context": args.demographic_context,
    }
    (out_dir / "manifest.json").write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.backend == "requests":
        metrics = {
            "row_count": len(records),
            "comparison_count": len(comparison_specs),
            "problem_count": int(comparisons["prob"].nunique()),
            "backend": args.backend,
            "judge_model": args.judge_model,
        }
        (out_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_summary(out_dir, args, metrics)
        print(f"Wrote {len(records)} preference requests to {out_dir / 'requests.jsonl'}")
        return 0

    raw_rows = []
    pred_rows = []
    for idx, record in enumerate(records, start=1):
        cached = run_request(record, args, out_dir)
        raw_rows.append(cached)
        pred_rows.append(prediction_row(record, cached))
        if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == len(records)):
            print(f"Completed {idx}/{len(records)} preference judgments", flush=True)
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
            "api_style": args.api_style,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
    )
    preference_summary.to_csv(out_dir / "preference_summary.csv", index=False)
    preference_by_problem.to_csv(out_dir / "preference_by_problem.csv", index=False)
    (out_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(out_dir, args, metrics)
    print(
        "Completed preference judgments: "
        f"rows={metrics['row_count']} parse_success={metrics['parse_success_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
