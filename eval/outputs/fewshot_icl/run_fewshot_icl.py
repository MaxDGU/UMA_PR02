#!/usr/bin/env python3
"""Few-shot in-context learning baselines for SP2013 fractions and BSS2021 decimals.

For each held-out problem, K human Q/A pairs are sampled from the train split
(excluding any rows whose `prob` matches the held-out problem). The same K
examples are reused across all 100 rollouts for that (model, K, problem).

Reuses the answer parser in `frontier_arithmetic_baseline.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import requests as http_requests

# Make the existing baseline module importable so we reuse its parser.
EXISTING_EVAL_DIR = Path(
    "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/uma_pr02_push/eval"
)
if str(EXISTING_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXISTING_EVAL_DIR.parent))
from eval import frontier_arithmetic_baseline as fab  # noqa: E402


FRACTION_HELDOUT_CSV = Path(
    "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs/fraction_4b/qwen3_4b_base.csv"
)
DECIMAL_HELDOUT_CSV = Path(
    "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs/decimal_4b/qwen3_4b_base.csv"
)
FRACTION_TRAIN_CSV = Path(
    "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/data/human_ft/data_train_nlp.csv"
)
DECIMAL_TRAIN_CSV = Path(
    "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/UMA_PR02_feat_humanft/data/human_ft/data_bss2021_train_nlp.csv"
)

OUT_ROOT = Path(
    "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/fewshot_icl_baselines"
)


# Mirror MODEL_PRESETS from run_inference.py, restricted to the three frontier models.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "gpt_5_5_low": {
        "backend": "sandbox",
        "model": "gpt-5.5",
        "label": "GPT-5.5 low",
        "temperature": None,
        "top_p": None,
        "reasoning_effort": "low",
        "gemini_thinking_level": "",
    },
    "claude_sonnet_4_6": {
        "backend": "sandbox",
        "model": "claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6",
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning_effort": "",
        "gemini_thinking_level": "",
    },
    "gemini_3_flash": {
        "backend": "gemini",
        "model": "gemini-3-flash-preview",
        "label": "Gemini 3 Flash",
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning_effort": "",
        "gemini_thinking_level": "minimal",
    },
}


def load_heldout(domain: str) -> pd.DataFrame:
    src = FRACTION_HELDOUT_CSV if domain == "fraction" else DECIMAL_HELDOUT_CSV
    df = pd.read_csv(src)
    keep = ["problem", "operation", "correct_answer"]
    keep.append("denom_type" if domain == "fraction" else "operands")
    df = df[keep].drop_duplicates(subset=["problem"]).reset_index(drop=True)
    return df


def load_train(domain: str) -> pd.DataFrame:
    src = FRACTION_TRAIN_CSV if domain == "fraction" else DECIMAL_TRAIN_CSV
    df = pd.read_csv(src)
    # Keep what we need verbatim
    df = df[["prob", "instruction_nl", "response_nl"]].dropna()
    return df.reset_index(drop=True)


def sample_examples(
    train_df: pd.DataFrame, held_out_prob: str, k: int, seed: int
) -> pd.DataFrame:
    """Sample K examples uniformly at random from train, excluding the held-out problem."""
    pool = train_df[train_df["prob"] != held_out_prob].reset_index(drop=True)
    if len(pool) < k:
        raise ValueError(
            f"Train pool has only {len(pool)} rows after excluding {held_out_prob!r}; need {k}."
        )
    rng = random.Random(seed)
    idx = rng.sample(range(len(pool)), k)
    return pool.iloc[idx].reset_index(drop=True)


def build_user_prompt(domain: str, examples: pd.DataFrame, held_out_prob: str) -> str:
    noun = "fraction" if domain == "fraction" else "decimal"
    if len(examples) == 0:
        # k=0 control: identical scaffolding minus the examples block.
        return "\n".join([
            f"Solve this {noun} problem: {held_out_prob}=?",
            "",
            "End with exactly one final-answer line:",
            "### answer: <final answer>",
        ])
    parts = [f"Here are some examples of how students solve {noun} problems:", ""]
    for _, row in examples.iterrows():
        parts.append(row["instruction_nl"].strip())
        parts.append(row["response_nl"].strip())
        parts.append("")
    parts.append(f"Now solve this problem in the same style:")
    parts.append("")
    parts.append(f"Solve this {noun} problem: {held_out_prob}=?")
    parts.append("")
    parts.append("End with exactly one final-answer line:")
    parts.append("### answer: <final answer>")
    return "\n".join(parts)


SYSTEM_PROMPTS = {
    "fraction": (
        "You are a 7th grader solving fraction arithmetic problems. "
        "Show your work in a few short sentences and give your final answer. "
        "Do not use a calculator."
    ),
    "decimal": (
        "You are a 7th grader solving decimal arithmetic problems. "
        "Show your work in a few short sentences and give your final answer. "
        "Do not use a calculator."
    ),
}


COMPLETION_TOKEN_PARAM_MODELS = ("gpt-5", "o1", "o3", "o4")


def token_limit_param(model: str) -> str:
    return "max_completion_tokens" if model.lower().startswith(COMPLETION_TOKEN_PARAM_MODELS) else "max_tokens"


def call_sandbox(
    preset: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    api_key: str,
    base_url: str,
    timeout: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": preset["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if preset["temperature"] is not None:
        payload["temperature"] = preset["temperature"]
    if preset["top_p"] is not None:
        payload["top_p"] = preset["top_p"]
    payload[token_limit_param(preset["model"])] = max_tokens
    if preset.get("reasoning_effort"):
        payload["reasoning"] = {"effort": preset["reasoning_effort"]}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/chat/completions"
    response = http_requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:800]}")
    response_json = response.json()
    return {"response_text": fab.extract_chat_text(response_json), "response_json": response_json}


def call_gemini(
    preset: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    client: Any,
) -> dict[str, Any]:
    from google.genai import types

    kwargs: dict[str, Any] = {
        "system_instruction": system_prompt,
        "max_output_tokens": max_tokens,
    }
    if preset["temperature"] is not None:
        kwargs["temperature"] = preset["temperature"]
    if preset["top_p"] is not None:
        kwargs["top_p"] = preset["top_p"]
    if preset.get("gemini_thinking_level"):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=preset["gemini_thinking_level"])
    config = types.GenerateContentConfig(**kwargs)
    response = client.models.generate_content(
        model=preset["model"], contents=user_prompt, config=config
    )
    response_json = response.model_dump(mode="json", exclude_none=True)
    text = fab.clean_text(response.text) if response.text is not None else ""
    return {"response_text": text, "response_json": response_json}


def call_model_with_retries(
    preset: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    api_key: str,
    base_url: str,
    timeout: float,
    gemini_client: Any,
    max_retries: int,
    retry_base_seconds: float,
) -> tuple[str, dict[str, Any], str]:
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            if preset["backend"] == "sandbox":
                out = call_sandbox(preset, system_prompt, user_prompt, max_tokens, api_key, base_url, timeout)
            else:
                out = call_gemini(preset, system_prompt, user_prompt, max_tokens, gemini_client)
            return out["response_text"], out["response_json"], ""
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(retry_base_seconds * (2 ** (attempt - 1)))
    return "", {}, last_err


def process_one(args, preset, system_prompt, user_prompt, api_key, base_url, gemini_client, record):
    """Worker: run one rollout and return a dict ready for the output CSV."""
    text, _resp_json, err = call_model_with_retries(
        preset,
        system_prompt,
        user_prompt,
        args.max_tokens,
        api_key,
        base_url,
        args.request_timeout_seconds,
        gemini_client,
        args.max_retries,
        args.retry_base_seconds,
    )
    parsed = fab.parse_answer(text, args.domain)
    return {
        **record,
        "model_response": text,
        "parsed_answer": parsed,
        "is_correct": fab.answers_match(parsed, str(record["correct_answer"]), args.domain),
        "request_error": err,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["fraction", "decimal"], required=True)
    p.add_argument("--model_slug", choices=list(MODEL_PRESETS.keys()), required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--samples_per_problem", type=int, default=100)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--retry_base_seconds", type=float, default=2.0)
    p.add_argument("--request_timeout_seconds", type=float, default=120.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out_dir", default=str(OUT_ROOT))
    p.add_argument("--api_key_env", default="AI_SANDBOX_KEY")
    p.add_argument("--gemini_api_key_env", default="GEMINI_API_KEY")
    p.add_argument("--portkey_base_url", default=os.environ.get("AI_SANDBOX_PORTKEY_BASE_URL", fab.DEFAULT_PORTKEY_BASE_URL))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    preset = MODEL_PRESETS[args.model_slug]

    out_dir = Path(args.out_dir) / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.model_slug}_k{args.k}.csv"
    if out_csv.exists():
        print(f"Output already exists: {out_csv} — skipping (delete to rerun)", flush=True)
        return 0

    heldout = load_heldout(args.domain)
    train = load_train(args.domain)
    # Sanity check: held-out and train disjoint by problem after we exclude per-problem.
    # (The decimals BSS2021 train has overlap; we exclude per held-out problem below.)
    print(f"Held-out: {len(heldout)} problems; train pool: {len(train)} rows / {train['prob'].nunique()} problems", flush=True)

    sys_prompt = SYSTEM_PROMPTS[args.domain]
    system_seed = args.seed + 1000 * args.k + (hash(args.model_slug) % 1000)

    # Build the (problem, example_set, prompt) plan first
    plan = []
    for problem_idx, row in heldout.iterrows():
        prob = row["problem"]
        per_problem_seed = system_seed + problem_idx
        examples = sample_examples(train, prob, args.k, per_problem_seed)
        user_prompt = build_user_prompt(args.domain, examples, prob)
        for sample_idx in range(args.samples_per_problem):
            base = {
                "problem": prob,
                "operation": row["operation"],
                ("denom_type" if args.domain == "fraction" else "operands"): row["denom_type" if args.domain == "fraction" else "operands"],
                "correct_answer": row["correct_answer"],
                "model": preset["label"],
                "sample_idx": sample_idx,
                "k_shot": args.k,
            }
            plan.append((base, sys_prompt, user_prompt))
    print(f"Total rollouts: {len(plan)} (k={args.k})", flush=True)

    # Initialize clients/keys
    gemini_client = None
    api_key = ""
    if preset["backend"] == "sandbox":
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise EnvironmentError(f"{args.api_key_env} is not set.")
    else:
        from google import genai

        gkey = os.environ.get(args.gemini_api_key_env, "")
        if not gkey:
            raise EnvironmentError(f"{args.gemini_api_key_env} is not set.")
        gemini_client = genai.Client(api_key=gkey, http_options={"api_version": "v1alpha"})

    rows: list[dict[str, Any]] = []
    completed = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = []
        for base, sp, up in plan:
            futures.append(ex.submit(process_one, args, preset, sp, up, api_key, args.portkey_base_url, gemini_client, base))
        for fut in as_completed(futures):
            rows.append(fut.result())
            completed += 1
            if completed % 100 == 0 or completed == len(plan):
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                eta_s = (len(plan) - completed) / rate if rate > 0 else 0
                print(f"  {completed}/{len(plan)}  rate={rate:.2f}/s  eta={eta_s/60:.1f}min", flush=True)

    df = pd.DataFrame(rows)
    # Order columns to match existing baselines + new k_shot/request_error fields.
    base_cols = ["problem", "operation"]
    base_cols.append("denom_type" if args.domain == "fraction" else "operands")
    base_cols += ["correct_answer", "model", "sample_idx", "parsed_answer", "is_correct", "model_response", "k_shot", "request_error"]
    df = df.sort_values(["problem", "sample_idx"]).reset_index(drop=True)
    df[base_cols].to_csv(out_csv, index=False)
    err_n = int(df["request_error"].astype(bool).sum())
    print(f"Wrote {out_csv} (rows={len(df)}, errors={err_n})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
