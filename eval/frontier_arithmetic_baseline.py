"""Run frontier arithmetic baselines for SP2013 fractions and BSS2021 decimals.

The runner prompts only with the arithmetic problem and asks for visible
student-style work. Raw responses are cached so parsing can be repaired later
without paying for another model call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd
import requests as http_requests


EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

FRACTION_PROBLEMS_CSV = EVAL_DIR / "data" / "fraction_problems.csv"
DECIMAL_PROBLEMS_CSV = EVAL_DIR / "data" / "decimal_problems.csv"
DEFAULT_OUT_DIR = EVAL_DIR / "runs" / "frontier_arithmetic_baselines"
DEFAULT_PORTKEY_BASE_URL = "https://api.portkey.ai/v1"
COMPLETION_TOKEN_PARAM_MODELS = ("gpt-5", "o1", "o3", "o4")

FRACTION_PROBLEM_RE = re.compile(
    r"\s*(-?\d+)\s*/\s*(-?\d+)\s*([+\-*:÷/])\s*(-?\d+)\s*/\s*(-?\d+)\s*"
)
DECIMAL_PROBLEM_RE = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*([+*])\s*(-?\d+(?:\.\d+)?)\s*")
LATEX_FRAC_RE = re.compile(r"\\frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}")
ASCII_FRAC_RE = re.compile(r"-?\d+\s*/\s*-?\d+")
NUMBER_RE = re.compile(r"-?(?:\d+\.\d+|\d+)")


@dataclass(frozen=True)
class ProblemRecord:
    problem: str
    operation: str
    operands: str
    correct_answer: str


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def optional_float(value: str) -> Optional[float]:
    if value.lower() in {"none", "null", "omit"}:
        return None
    return float(value)


def stable_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_fraction_str(value: Fraction) -> str:
    if value.denominator < 0:
        value = Fraction(-value.numerator, -value.denominator)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def parse_fraction_problem(problem: str) -> tuple[Fraction, str, Fraction]:
    match = FRACTION_PROBLEM_RE.fullmatch(str(problem).strip())
    if not match:
        raise ValueError(f"Could not parse fraction problem: {problem!r}")
    a_num, a_den, op, b_num, b_den = match.groups()
    op = ":" if op in {":", "÷", "/"} else op
    return Fraction(int(a_num), int(a_den)), op, Fraction(int(b_num), int(b_den))


def solve_fraction(problem: str) -> Fraction:
    left, op, right = parse_fraction_problem(problem)
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if right == 0:
        raise ZeroDivisionError(problem)
    return left / right


def fraction_metadata(problem: str) -> ProblemRecord:
    left, op, right = parse_fraction_problem(problem)
    operation = {"+": "Add", "-": "Sub", "*": "Mul", ":": "Div"}[op]
    operands = "ED" if left.denominator == right.denominator else "UD"
    return ProblemRecord(
        problem=problem,
        operation=operation,
        operands=operands,
        correct_answer=canonical_fraction_str(solve_fraction(problem)),
    )


def decimal_places(value: str) -> int:
    return len(value.split(".", 1)[1]) if "." in value else 0


def decimal_metadata(problem: str) -> ProblemRecord:
    match = DECIMAL_PROBLEM_RE.fullmatch(str(problem).strip())
    if not match:
        raise ValueError(f"Could not parse decimal problem: {problem!r}")
    left_text, op, right_text = match.groups()
    left = Decimal(left_text)
    right = Decimal(right_text)
    answer = left + right if op == "+" else left * right
    operation = "Add" if op == "+" else "Mul"
    left_places = decimal_places(left_text)
    right_places = decimal_places(right_text)
    if left_places == 0 or right_places == 0:
        operands = "D-W"
    elif left_places == right_places:
        operands = "EDD"
    else:
        operands = "UDD"
    return ProblemRecord(
        problem=problem,
        operation=operation,
        operands=operands,
        correct_answer=format(answer, "f"),
    )


def load_problems(domain: str) -> list[ProblemRecord]:
    path = FRACTION_PROBLEMS_CSV if domain == "fraction" else DECIMAL_PROBLEMS_CSV
    df = pd.read_csv(path)
    builder = fraction_metadata if domain == "fraction" else decimal_metadata
    return [builder(clean_text(problem)) for problem in df["prob"].tolist()]


def normalize_fraction_answer(text: str) -> Optional[str]:
    body = clean_text(text)
    if not body:
        return None
    frac_matches = LATEX_FRAC_RE.findall(body)
    if frac_matches:
        num, den = frac_matches[-1]
        den_int = int(den)
        if den_int == 0:
            return None
        return canonical_fraction_str(Fraction(int(num), den_int))
    ascii_matches = ASCII_FRAC_RE.findall(body)
    if ascii_matches:
        num, den = ascii_matches[-1].replace(" ", "").split("/", 1)
        den_int = int(den)
        if den_int == 0:
            return None
        return canonical_fraction_str(Fraction(int(num), den_int))
    number_matches = NUMBER_RE.findall(body)
    if number_matches:
        raw = number_matches[-1]
        try:
            if "." in raw:
                return canonical_fraction_str(Fraction(Decimal(raw)))
            return canonical_fraction_str(Fraction(int(raw), 1))
        except (InvalidOperation, ValueError):
            return None
    return None


def normalize_decimal_answer(text: str) -> Optional[str]:
    body = clean_text(text)
    if not body:
        return None
    frac_matches = LATEX_FRAC_RE.findall(body)
    if frac_matches:
        num, den = frac_matches[-1]
        den_int = int(den)
        if den_int == 0:
            return None
        return format(Decimal(int(num)) / Decimal(den_int), "f")
    ascii_matches = ASCII_FRAC_RE.findall(body)
    if ascii_matches:
        num, den = ascii_matches[-1].replace(" ", "").split("/", 1)
        den_int = int(den)
        if den_int == 0:
            return None
        return format(Decimal(int(num)) / Decimal(den_int), "f")
    number_matches = NUMBER_RE.findall(body.replace(",", ""))
    if not number_matches:
        return None
    try:
        return format(Decimal(number_matches[-1]), "f")
    except InvalidOperation:
        return None


def parse_answer(response_text: str, domain: str) -> Optional[str]:
    body = clean_text(response_text)
    if "### answer:" in body.lower():
        marker = re.search(r"###\s*answer\s*:", body, flags=re.IGNORECASE)
        if marker:
            body = body[marker.end() :]
    if domain == "fraction":
        return normalize_fraction_answer(body)
    return normalize_decimal_answer(body)


def answers_match(parsed: Optional[str], correct: str, domain: str) -> bool:
    if parsed is None:
        return False
    try:
        if domain == "fraction":
            return Fraction(parsed) == Fraction(correct)
        return Decimal(parsed).normalize() == Decimal(correct).normalize()
    except Exception:
        return False


def system_prompt(domain: str) -> str:
    if domain == "fraction":
        return (
            "You are a 7th grader solving fraction arithmetic problems. "
            "Show your work in a few short sentences and give your final answer. "
            "Do not use a calculator."
        )
    return (
        "You are a 7th grader solving decimal arithmetic problems. "
        "Show your work in a few short sentences and give your final answer. "
        "Do not use a calculator."
    )


def user_prompt(problem: str) -> str:
    return "\n".join(
        [
            f"Solve: {problem}",
            "",
            "End with exactly one final-answer line:",
            "### answer: <final answer>",
        ]
    )


def resolve_token_limit_param(args: argparse.Namespace) -> str:
    if args.token_limit_param != "auto":
        return args.token_limit_param
    if args.model.lower().startswith(COMPLETION_TOKEN_PARAM_MODELS):
        return "max_completion_tokens"
    return "max_tokens"


def make_requests(problems: Sequence[ProblemRecord], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    token_limit_param = resolve_token_limit_param(args)
    sys_prompt = system_prompt(args.domain)
    for problem_idx, problem in enumerate(problems):
        for sample_idx in range(args.samples_per_problem):
            request = {
                "request_id": f"{args.domain}:{problem_idx:03d}:sample:{sample_idx:03d}",
                "domain": args.domain,
                "problem_idx": problem_idx,
                "problem": problem.problem,
                "operation": problem.operation,
                "operands": problem.operands,
                "correct_answer": problem.correct_answer,
                "model": args.model,
                "model_label": args.model_label,
                "sample_idx": sample_idx,
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt(problem.problem),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "token_limit_param": token_limit_param,
                "reasoning_effort": args.reasoning_effort or None,
            }
            request["request_key"] = stable_hash(request)
            rows.append(request)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    return rows


def sandbox_payload(record: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": record["system_prompt"]},
            {"role": "user", "content": record["user_prompt"]},
        ],
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    payload[resolve_token_limit_param(args)] = args.max_tokens
    if args.reasoning_effort:
        payload["reasoning"] = {"effort": args.reasoning_effort}
    return payload


def extract_chat_text(response_json: Mapping[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(clean_text(item.get("text", "")))
            else:
                parts.append(clean_text(item))
        return "\n".join(part for part in parts if part)
    return clean_text(content)


def call_sandbox(record: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"{args.api_key_env} is not set.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{args.portkey_base_url.rstrip('/')}/chat/completions"
    payload = sandbox_payload(record, args)
    response = http_requests.post(url, headers=headers, json=payload, timeout=args.request_timeout_seconds)
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1200]}")
    response_json = response.json()
    return {"payload": payload, "response_json": response_json, "response_text": extract_chat_text(response_json)}


def make_gemini_config(record: Mapping[str, Any], args: argparse.Namespace) -> Any:
    from google.genai import types

    kwargs: dict[str, Any] = {
        "system_instruction": record["system_prompt"],
        "max_output_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.gemini_thinking_level:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=args.gemini_thinking_level)
    return types.GenerateContentConfig(**kwargs)


def call_gemini(record: Mapping[str, Any], args: argparse.Namespace, client: Any) -> dict[str, Any]:
    config = make_gemini_config(record, args)
    response = client.models.generate_content(
        model=args.model,
        contents=record["user_prompt"],
        config=config,
    )
    response_json = response.model_dump(mode="json", exclude_none=True)
    return {"payload": response_json.get("sdk_http_request", {}), "response_json": response_json, "response_text": clean_text(response.text)}


def call_model(record: Mapping[str, Any], args: argparse.Namespace, gemini_client: Any) -> dict[str, Any]:
    if args.backend == "sandbox":
        return call_sandbox(record, args)
    if gemini_client is None:
        raise RuntimeError("Gemini client was not initialized.")
    return call_gemini(record, args, gemini_client)


def usage_from_response(response_json: Mapping[str, Any], backend: str) -> dict[str, int]:
    if backend == "sandbox":
        usage = response_json.get("usage") or {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
    usage = response_json.get("usage_metadata") or response_json.get("usageMetadata") or {}
    prompt = int(usage.get("prompt_token_count") or usage.get("promptTokenCount") or 0)
    output = int(usage.get("candidates_token_count") or usage.get("candidatesTokenCount") or 0)
    total = int(usage.get("total_token_count") or usage.get("totalTokenCount") or prompt + output)
    return {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": total}


def cached_request(record: Mapping[str, Any], args: argparse.Namespace, out_dir: Path, gemini_client: Any) -> dict[str, Any]:
    cache_path = out_dir / "cache" / f"{record['request_key']}.json"
    if cache_path.exists() and not args.force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    request_error = ""
    response_json: Optional[dict[str, Any]] = None
    payload: dict[str, Any] = {}
    response_text = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            result = call_model(record, args, gemini_client)
            payload = result["payload"]
            response_json = result["response_json"]
            response_text = result["response_text"]
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
    }
    cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
    return cached


def prediction_row(record: Mapping[str, Any], cached: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    response_text = clean_text(cached.get("response_text", ""))
    parsed = parse_answer(response_text, args.domain)
    usage = usage_from_response(cached.get("response_json") or {}, args.backend)
    common = {
        "problem": record["problem"],
        "operation": record["operation"],
        "correct_answer": record["correct_answer"],
        "model": record["model_label"],
        "sample_idx": record["sample_idx"],
        "parsed_answer": parsed,
        "is_correct": answers_match(parsed, record["correct_answer"], args.domain),
        "model_response": response_text,
        "request_id": record["request_id"],
        "request_key": record["request_key"],
        "request_error": clean_text(cached.get("request_error", "")),
        **usage,
    }
    if args.domain == "fraction":
        common["denom_type"] = record["operands"]
    else:
        common["operands"] = record["operands"]
    return common


def write_outputs(predictions: pd.DataFrame, raw_rows: list[dict[str, Any]], requests: list[dict[str, Any]], args: argparse.Namespace, out_dir: Path) -> None:
    write_jsonl(out_dir / "requests.jsonl", requests)
    write_jsonl(out_dir / "responses_raw.jsonl", raw_rows)
    predictions.to_csv(out_dir / "parsed_samples.csv", index=False)
    if args.domain == "fraction":
        compat_cols = [
            "problem",
            "operation",
            "denom_type",
            "correct_answer",
            "model",
            "sample_idx",
            "parsed_answer",
            "is_correct",
            "model_response",
        ]
        predictions[compat_cols].to_csv(out_dir / "llm_outputs_sp2013_fractions.csv", index=False)
    else:
        compat_cols = [
            "problem",
            "operation",
            "operands",
            "correct_answer",
            "model",
            "sample_idx",
            "parsed_answer",
            "is_correct",
            "model_response",
        ]
        predictions[compat_cols].to_csv(out_dir / "llm_outputs_bss2021_decimals.csv", index=False)
    group_col = "denom_type" if args.domain == "fraction" else "operands"
    cell_accuracy = (
        predictions.groupby(["operation", group_col], as_index=False)
        .agg(rows=("is_correct", "size"), parse_success_rate=("parsed_answer", lambda x: x.notna().mean()), accuracy=("is_correct", "mean"))
        .sort_values(["operation", group_col])
    )
    cell_accuracy.to_csv(out_dir / "cell_accuracy.csv", index=False)
    metrics = {
        "backend": args.backend,
        "domain": args.domain,
        "model": args.model,
        "model_label": args.model_label,
        "row_count": int(len(predictions)),
        "problem_count": int(predictions["problem"].nunique()) if not predictions.empty else 0,
        "samples_per_problem": args.samples_per_problem,
        "parse_success_rate": float(predictions["parsed_answer"].notna().mean()) if not predictions.empty else None,
        "accuracy": float(predictions["is_correct"].mean()) if not predictions.empty else None,
        "request_error_count": int(predictions["request_error"].astype(bool).sum()) if not predictions.empty else 0,
        "prompt_tokens": int(pd.to_numeric(predictions["prompt_tokens"], errors="coerce").fillna(0).sum()) if not predictions.empty else 0,
        "completion_tokens": int(pd.to_numeric(predictions["completion_tokens"], errors="coerce").fillna(0).sum()) if not predictions.empty else 0,
        "total_tokens": int(pd.to_numeric(predictions["total_tokens"], errors="coerce").fillna(0).sum()) if not predictions.empty else 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "token_limit_param": resolve_token_limit_param(args),
        "reasoning_effort": args.reasoning_effort or None,
        "gemini_thinking_level": args.gemini_thinking_level or None,
    }
    (out_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["fraction", "decimal"], required=True)
    parser.add_argument("--backend", choices=["sandbox", "gemini"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--samples_per_problem", type=int, required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--temperature", type=optional_float, default=1.0)
    parser.add_argument("--top_p", type=optional_float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--token_limit_param", choices=["auto", "max_tokens", "max_completion_tokens"], default="auto")
    parser.add_argument("--reasoning_effort", choices=["", "low", "medium", "high", "xhigh"], default="")
    parser.add_argument("--gemini_thinking_level", default="minimal")
    parser.add_argument("--api_key_env", default="AI_SANDBOX_KEY")
    parser.add_argument("--gemini_api_key_env", default="GEMINI_API_KEY")
    parser.add_argument("--gemini_api_version", default="v1alpha")
    parser.add_argument("--portkey_base_url", default=os.environ.get("AI_SANDBOX_PORTKEY_BASE_URL", DEFAULT_PORTKEY_BASE_URL))
    parser.add_argument("--request_timeout_seconds", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--retry_base_seconds", type=float, default=2.0)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)

    problems = load_problems(args.domain)
    requests = make_requests(problems, args)
    gemini_client = None
    if args.backend == "gemini":
        from google import genai

        api_key = os.environ.get(args.gemini_api_key_env)
        if not api_key:
            raise EnvironmentError(f"{args.gemini_api_key_env} is not set.")
        gemini_client = genai.Client(api_key=api_key, http_options={"api_version": args.gemini_api_version})

    raw_rows = []
    pred_rows = []
    for idx, record in enumerate(requests, start=1):
        cached = cached_request(record, args, out_dir, gemini_client)
        raw_rows.append(cached)
        pred_rows.append(prediction_row(record, cached, args))
        if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == len(requests)):
            print(f"Completed {idx}/{len(requests)} requests", flush=True)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    predictions = pd.DataFrame(pred_rows)
    write_outputs(predictions, raw_rows, requests, args, out_dir)
    error_count = int(predictions["request_error"].astype(bool).sum()) if not predictions.empty else 0
    print(
        f"Completed {args.domain} {args.model_label}: rows={len(predictions)} "
        f"errors={error_count} out_dir={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
