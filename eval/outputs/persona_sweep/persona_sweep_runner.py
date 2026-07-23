"""Persona sweep for frontier LLM baselines on SP2013 fractions and BSS2021 decimals.

For each (model, persona, domain), rolls out N samples per problem and writes a
CSV that matches the schema of fraction_baselines_100samples/*.csv. The same
answer parsers used by the existing baselines are reused (fractions:
eval_llm_baseline.parse_answer + check_correct; decimals:
frontier_arithmetic_baseline.normalize_decimal_answer + answers_match) so this
script does NOT introduce new scoring code.

Output layout:
    persona_sweep_baselines/fraction/{model_slug}_{persona_slug}.csv
    persona_sweep_baselines/decimal/{model_slug}_{persona_slug}.csv

Models (routed through Princeton AI Sandbox Portkey gateway for Claude/GPT,
and via google.genai for Gemini, mirroring the fewshot-ICL runner so this
script works from della compute nodes that can't reach api.anthropic.com or
api.openai.com directly):
    claude_sonnet_4_6  (claude-sonnet-4-6, Sandbox/Portkey OpenAI-compat)
    gemini_3_flash     (gemini-3-flash-preview, google.genai)
    gpt_5_5_low        (gpt-5.5 with reasoning_effort=low, Sandbox/Portkey)

Skipped per paper decision: gpt_4_1_mini, gemini_2_5_flash.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests as http_requests

# --- import the existing fraction parser (parse_answer) ---
sys.path.insert(0, "/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions/finetune")
import eval_llm_baseline as ev  # fraction parser only — make_client is NOT used here

# --- import the existing decimal parser from frontier_arithmetic_baseline ---
sys.path.insert(0, "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/uma_pr02_push/eval")
import frontier_arithmetic_baseline as fab  # decimal parser + extract_chat_text + DEFAULT_PORTKEY_BASE_URL


# Backend label per model slug. The fewshot-ICL runner uses the same routing.
BACKEND_BY_SLUG = {
    "claude_sonnet_4_6": "sandbox",
    "gpt_5_5_low":       "sandbox",
    "gemini_3_flash":    "gemini",
}

COMPLETION_TOKEN_PARAM_MODELS = ("gpt-5", "o1", "o3", "o4")


def _token_limit_param(model_id: str) -> str:
    return (
        "max_completion_tokens"
        if model_id.lower().startswith(COMPLETION_TOKEN_PARAM_MODELS)
        else "max_tokens"
    )


# --- problem sets (match the existing fraction_baselines_100samples and BSS2021) ---
FRACTION_PROBLEMS_CSV = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/uma_pr02_push/eval/data/fraction_problems.csv"
DECIMAL_PROBLEMS_CSV = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/uma_pr02_push/eval/data/decimal_problems.csv"


def _load_fraction_problems() -> list[dict]:
    """Return list of dicts: {problem, operation, denom_type, correct_answer}.
    `problem` uses the same display form (÷ for division) as the existing baseline CSVs.
    """
    df = pd.read_csv(FRACTION_PROBLEMS_CSV)
    out = []
    for row in df.itertuples(index=False):
        prob = row.prob.replace(":", "÷")
        out.append({
            "problem": prob,
            "operation": row.operation,
            "denom_type": row.denom_type,
            "correct_answer": str(row.correct_answer),
        })
    return out


def _load_decimal_problems() -> list[dict]:
    """Return list of dicts: {problem, operation, operands, correct_answer}."""
    df = pd.read_csv(DECIMAL_PROBLEMS_CSV)
    out = []
    for row in df.itertuples(index=False):
        out.append({
            "problem": row.prob,
            "operation": row.operation,
            "operands": row.operands,
            "correct_answer": str(row.correct_answer),
        })
    return out


# --- persona prompt registry ---
# Each entry: slug -> (fraction_system_prompt, decimal_system_prompt).
# An empty string means "no system prompt at all" (control).

def _persona_prompts() -> dict[str, tuple[str, str]]:
    def pair(stem: str) -> tuple[str, str]:
        # Common closing instruction kept identical across personas so prompt
        # length / format doesn't confound the result.
        tail = (
            " Show your work in a few short sentences and give your final answer."
            " Do not use a calculator."
        )
        f = stem.format(domain="fraction") + tail
        d = stem.format(domain="decimal") + tail
        return f, d

    personas: dict[str, tuple[str, str]] = {}

    # 1. Original paper prompt (control)
    personas["paper_baseline"] = pair(
        "You are a 7th grader solving {domain} arithmetic problems."
    )

    # 2. Tier-targeted: average 7th grader
    personas["avg_7th"] = pair(
        "You are an average 7th grader solving {domain} arithmetic problems."
    )

    # 3. Tier-targeted: high-achieving 8th grader
    personas["high_achieving_8th"] = pair(
        "You are a high-achieving 8th grader solving {domain} arithmetic problems."
    )

    # 4. Tier-targeted: struggling 6th grader (extra clause for fractions only)
    personas["struggling_6th"] = (
        ("You are a struggling 6th grader who hasn't yet mastered finding a"
         " common denominator, solving fraction arithmetic problems."
         " Show your work in a few short sentences and give your final answer."
         " Do not use a calculator."),
        ("You are a struggling 6th grader who hasn't yet mastered place-value"
         " alignment for decimals, solving decimal arithmetic problems."
         " Show your work in a few short sentences and give your final answer."
         " Do not use a calculator."),
    )

    # 5. Error-procedure: independent-components subtraction (fractions)
    #    and place-value-misalignment (decimals)
    personas["err_indep_components"] = (
        ("You are a student who tends to operate on numerators and denominators"
         " independently when adding or subtracting fractions."
         " Show your work in a few short sentences and give your final answer."
         " Do not use a calculator."),
        ("You are a student who tends to add or multiply decimal digits without"
         " carefully aligning the decimal point."
         " Show your work in a few short sentences and give your final answer."
         " Do not use a calculator."),
    )

    # 6. Error-procedure: keep-change-flip / multiplication-as-repeated-addition
    personas["err_forgot_kcf"] = (
        ("You are a student who often forgets the keep-change-flip rule when"
         " dividing fractions."
         " Show your work in a few short sentences and give your final answer."
         " Do not use a calculator."),
        ("You are a student who often forgets to count the total number of"
         " decimal places when multiplying decimals."
         " Show your work in a few short sentences and give your final answer."
         " Do not use a calculator."),
    )

    # 7. Style: quick, no checking
    personas["style_quick"] = pair(
        "You are a student answering {domain} arithmetic problems quickly without checking your work."
    )

    # 8. Style: careful, every step
    personas["style_careful"] = pair(
        "You are a careful student who shows every step of {domain} arithmetic."
    )

    # 9. Empty / no system message (control)
    personas["no_persona"] = ("", "")

    # 10. Cohort-anchored
    personas["cohort_anchored"] = pair(
        "You are a representative sixth-through-eighth grader from a US middle"
        " school math study (SP2013/BSS2021). Solve {domain} arithmetic problems"
        " as that population would, including the typical errors that age group makes."
    )

    # --- Wide 15-persona set (fractions only): the same personas used for the
    # off-the-shelf Qwen sweep (offshelf_wide_persona.py), converted from the
    # first-person "I am ..." prefixes to the "You are ..." system-prompt form
    # used throughout this frontier protocol. Slugs are prefixed wide_ so these
    # never collide with the original ten's output files. The decimal slot
    # duplicates the fraction text; only --domain fraction is intended.
    tail = (
        " Show your work in a few short sentences and give your final answer."
        " Do not use a calculator."
    )
    wide = {
        "wide_barely_started":     "You are a young student who has barely started learning fractions and usually gets them wrong.",
        "wide_very_struggling":    "You are a student who struggles badly with fractions and rarely gets the right answer.",
        "wide_struggling_5th":     "You are a struggling 5th grader who is just beginning to learn fraction arithmetic.",
        "wide_struggling_6th":     "You are a struggling 6th grader who hasn't yet mastered finding a common denominator.",
        "wide_below_average_6th":  "You are a below-average 6th grader solving fraction arithmetic problems.",
        "wide_average_6th":        "You are an average 6th grader solving fraction arithmetic problems.",
        "wide_average_7th":        "You are an average 7th grader solving fraction arithmetic problems.",
        "wide_above_average_7th":  "You are an above-average 7th grader who is fairly comfortable with fractions.",
        "wide_solid_8th":          "You are a solid 8th grader who understands fraction arithmetic well.",
        "wide_high_achieving_8th": "You are a high-achieving 8th grader solving fraction arithmetic problems.",
        "wide_top_student":        "You are a top math student who almost always solves fraction problems correctly.",
        "wide_math_expert":        "You are a mathematics expert who never makes mistakes on fraction arithmetic.",
        "wide_err_indep":          "You are a student who operates on numerators and denominators independently when adding or subtracting fractions.",
        "wide_err_no_kcf":         "You are a student who often forgets the keep-change-flip rule when dividing fractions.",
        "wide_err_cross":          "You are a student who sometimes cross-multiplies the wrong way when working with fractions.",
    }
    for slug, stem in wide.items():
        personas[slug] = (stem + tail, stem + tail)

    return personas


PERSONAS = _persona_prompts()


# --- model registry ---
MODEL_SPECS = {
    # slug -> (api_model_id, pretty_label)
    "claude_sonnet_4_6": ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    "gemini_3_flash":    ("gemini-3-flash-preview", "Gemini 3 Flash"),
    "gpt_5_5_low":       ("gpt-5.5", "GPT-5.5 low"),
}


def _make_client(model_slug: str) -> dict:
    """Build the per-slug context needed to issue requests.

    For sandbox-routed models (Claude, GPT), returns the Portkey base URL +
    bearer key. For Gemini, returns an initialized google.genai client.
    Mirrors `main()` of run_fewshot_icl.py (lines ~322-335).
    """
    backend = BACKEND_BY_SLUG[model_slug]
    if backend == "sandbox":
        api_key = os.environ.get("AI_SANDBOX_KEY", "")
        if not api_key:
            raise EnvironmentError("AI_SANDBOX_KEY is not set.")
        base_url = os.environ.get(
            "AI_SANDBOX_PORTKEY_BASE_URL", fab.DEFAULT_PORTKEY_BASE_URL
        )
        return {"backend": "sandbox", "api_key": api_key, "base_url": base_url}
    elif backend == "gemini":
        gkey = os.environ.get("GEMINI_API_KEY", "")
        if not gkey:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        from google import genai
        gemini_client = genai.Client(api_key=gkey, http_options={"api_version": "v1alpha"})
        return {"backend": "gemini", "client": gemini_client}
    raise ValueError(f"Unknown backend for slug {model_slug}")


def _call_sandbox(ctx: Mapping[str, Any], model_id: str, system_prompt: str,
                  user_prompt: str, temperature: float, max_tokens: int,
                  timeout: float = 120.0) -> str:
    """HTTP POST to {portkey_base_url}/chat/completions, mirroring
    fewshot_icl.run_fewshot_icl.call_sandbox (lines ~152-180).
    Empty system_prompt => no system message at all.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload: dict[str, Any] = {"model": model_id, "messages": messages}
    payload[_token_limit_param(model_id)] = max_tokens
    if model_id.lower().startswith("gpt-5"):
        # gpt-5/o1/o3/o4 reasoning models: low effort, no temperature/top_p.
        payload["reasoning"] = {"effort": "low"}
    else:
        payload["temperature"] = temperature

    url = f"{ctx['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {ctx['api_key']}",
        "Content-Type": "application/json",
    }
    # Retry with exponential backoff on rate-limit / overloaded (429, 529, 503)
    delays = [2, 4, 8, 16, 32]
    for attempt, delay in enumerate([0] + delays):
        if delay:
            time.sleep(delay)
        resp = http_requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code < 400:
            return fab.extract_chat_text(resp.json())
        if resp.status_code in (429, 503, 529) and attempt < len(delays):
            continue
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")
    raise RuntimeError("unreachable")


def _call_gemini(ctx: Mapping[str, Any], model_id: str, system_prompt: str,
                 user_prompt: str, temperature: float, max_tokens: int) -> str:
    """google.genai generate_content, mirroring fewshot_icl.run_fewshot_icl.call_gemini
    (lines ~189-215). Empty system_prompt => omit system_instruction.
    """
    from google.genai import types

    kwargs: dict[str, Any] = {"max_output_tokens": max_tokens, "temperature": temperature}
    if system_prompt:
        kwargs["system_instruction"] = system_prompt
    # Match the fewshot runner's thinking_level=minimal preset for gemini-3-flash-preview.
    kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="minimal")
    config = types.GenerateContentConfig(**kwargs)
    resp = ctx["client"].models.generate_content(
        model=model_id, contents=user_prompt, config=config,
    )
    return fab.clean_text(resp.text) if resp.text is not None else ""


def _query_one(client, model_id: str, system_prompt: str, user_prompt: str,
               temperature: float = 1.0, max_tokens: int = 2048) -> str:
    """Single chat completion. Returns raw response text. Routes through Sandbox
    (Claude+GPT) or google.genai (Gemini) so the call works from della compute
    nodes. Empty system_prompt => no system message at all.
    """
    backend = client["backend"]
    if backend == "sandbox":
        return _call_sandbox(client, model_id, system_prompt, user_prompt,
                             temperature, max_tokens)
    elif backend == "gemini":
        return _call_gemini(client, model_id, system_prompt, user_prompt,
                            temperature, max_tokens)
    raise ValueError(f"Unknown backend {backend!r}")


def _user_prompt(problem: str, domain: str) -> str:
    if domain == "fraction":
        return f"Solve: {problem}"
    return f"Solve: {problem}"


def _score_fraction(text: str, correct: str) -> tuple[str | None, bool]:
    try:
        parsed = ev.parse_answer(text or "")  # returns Fraction or None
    except (ZeroDivisionError, ValueError):
        return None, False
    if parsed is None:
        return None, False
    # canonical "num/den" string for parsed_answer column
    parsed_str = f"{parsed.numerator}/{parsed.denominator}"
    try:
        correct_frac = Fraction(correct)
    except (ZeroDivisionError, ValueError):
        return parsed_str, False
    return parsed_str, (parsed == correct_frac)


def _score_decimal(text: str, correct: str) -> tuple[str | None, bool]:
    parsed = fab.normalize_decimal_answer(text or "")
    is_correct = fab.answers_match(parsed, correct, "decimal")
    return parsed, is_correct


def _run_sample(client, model_id: str, system_prompt: str, problem: str,
                domain: str, sample_idx: int, correct: str,
                temperature: float, max_tokens: int) -> dict:
    try:
        text = _query_one(client, model_id, system_prompt,
                          _user_prompt(problem, domain),
                          temperature=temperature, max_tokens=max_tokens)
    except Exception as e:  # noqa: BLE001
        text = f"[ERROR] {type(e).__name__}: {e}"
    if domain == "fraction":
        parsed, is_correct = _score_fraction(text, correct)
    else:
        parsed, is_correct = _score_decimal(text, correct)
    return {
        "problem": problem,
        "correct_answer": correct,
        "sample_idx": sample_idx,
        "parsed_answer": parsed,
        "is_correct": bool(is_correct),
        "model_response": text,
    }


def run_one_cell(model_slug: str, persona_slug: str, domain: str,
                 n_samples: int, max_workers: int, out_dir: Path,
                 temperature: float, max_tokens: int,
                 force: bool) -> Path:
    model_id, pretty = MODEL_SPECS[model_slug]
    sys_prompts = PERSONAS[persona_slug]
    system_prompt = sys_prompts[0] if domain == "fraction" else sys_prompts[1]
    problems = _load_fraction_problems() if domain == "fraction" else _load_decimal_problems()

    cell_dir = out_dir / domain
    cell_dir.mkdir(parents=True, exist_ok=True)
    out_path = cell_dir / f"{model_slug}_{persona_slug}.csv"
    if out_path.exists() and not force:
        print(f"  SKIP existing {out_path}")
        return out_path

    client = _make_client(model_slug)
    futures = []
    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for prob_meta in problems:
            for s in range(n_samples):
                futures.append(pool.submit(
                    _run_sample, client, model_id, system_prompt,
                    prob_meta["problem"], domain, s, prob_meta["correct_answer"],
                    temperature, max_tokens,
                ))
        n_total = len(futures)
        done = 0
        for f in as_completed(futures):
            results.append(f.result())
            done += 1
            if done % 200 == 0 or done == n_total:
                print(f"    {done}/{n_total} ({time.time()-t0:.0f}s)", flush=True)

    # Re-attach the per-problem metadata so output matches existing baseline schema.
    meta_lookup = {p["problem"]: p for p in problems}
    rows = []
    for r in results:
        meta = meta_lookup[r["problem"]]
        row = {
            "problem": r["problem"],
            "operation": meta["operation"],
        }
        if domain == "fraction":
            row["denom_type"] = meta["denom_type"]
        else:
            row["operands"] = meta["operands"]
        row.update({
            "correct_answer": r["correct_answer"],
            "model": pretty,
            "persona": persona_slug,
            "sample_idx": r["sample_idx"],
            "parsed_answer": r["parsed_answer"],
            "is_correct": r["is_correct"],
            "model_response": r["model_response"],
        })
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["problem", "sample_idx"]).reset_index(drop=True)
    df["sample_idx"] = df.groupby("problem").cumcount()
    df.to_csv(out_path, index=False)
    mean_acc = df["is_correct"].mean()
    print(f"  wrote {out_path}  rows={len(df)}  mean_acc={mean_acc:.3f}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_slug", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--domain", required=True, choices=["fraction", "decimal"])
    ap.add_argument("--personas", nargs="+", default=list(PERSONAS),
                    help="Subset of persona slugs to run; default = all 10.")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_workers", type=int, default=20)
    ap.add_argument("--out_dir", default="/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/73c8dadc/persona_sweep_baselines")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="1 sample per problem, single persona, single model — debug only.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    personas = list(args.personas)
    if args.smoke:
        personas = personas[:1]
        args.n_samples = 1

    for persona_slug in personas:
        if persona_slug not in PERSONAS:
            raise SystemExit(f"unknown persona {persona_slug!r}")
        print(f"\n=== {args.model_slug} | {persona_slug} | {args.domain} ===", flush=True)
        run_one_cell(
            model_slug=args.model_slug,
            persona_slug=persona_slug,
            domain=args.domain,
            n_samples=args.n_samples,
            max_workers=args.max_workers,
            out_dir=out_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            force=args.force,
        )


if __name__ == "__main__":
    main()
