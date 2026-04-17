#!/usr/bin/env python3
"""
Shared utilities for human strategy-matching experiments.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linprog
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


DEFAULT_LABEL_ORDER = ["KDON", "CDON", "ONOD", "ICDM", "CROP"]
DEFAULT_KL_EPS = 1e-6
NUMERIC_TOKEN_RE = r"[+\-]?(?:\d+(?:/\d+)?|\d*\.\d+)"
PROB_RE = re.compile(rf"\s*({NUMERIC_TOKEN_RE})\s*([+\-*:])\s*({NUMERIC_TOKEN_RE})\s*")
PROMPT_PROB_RE = re.compile(r"Solve this fraction problem:\s*(.*?)\s*=\?\s*$", flags=re.IGNORECASE)


@dataclass
class GeneratedRollout:
    prob: str
    prompt_text: str
    query_token_ids: List[int]
    response_token_ids: List[int]
    full_token_ids: List[int]
    response_text: str


def save_json(path: str | os.PathLike[str], payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_flexible(path: str | os.PathLike[str]) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def clean_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace("\r\n", "\n", regex=False)
        .str.replace("\r", "\n", regex=False)
        .str.strip()
    )


def pick_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return ""


def normalize_to_nlp_frame(
    df: pd.DataFrame,
    prompt_col: str = "instruction_nl",
    response_col: str = "response_nl",
) -> pd.DataFrame:
    if prompt_col in df.columns and response_col in df.columns:
        out = df.copy()
        out[prompt_col] = clean_series(out[prompt_col])
        out[response_col] = clean_series(out[response_col])
        if "prob" in out.columns:
            out["prob"] = clean_series(out["prob"])
        else:
            out["prob"] = ""
        if "strategy" not in out.columns:
            out["strategy"] = ""
        else:
            out["strategy"] = clean_series(out["strategy"])
        out = out[(out[prompt_col] != "") & (out[response_col] != "") & (out["prob"] != "")].reset_index(drop=True)
        return out

    prob_col = pick_first_existing(df.columns, ["prob", "prob.1"])
    resp_col = pick_first_existing(df.columns, ["resp", "resp.1"])
    reasoning_col = pick_first_existing(df.columns, ["strategy", "Exp", "Comments"])
    if not prob_col or not resp_col or not reasoning_col:
        raise ValueError(
            "Could not normalize human CSV to NLP format. "
            f"Found columns={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["prob"] = clean_series(df[prob_col])
    out["instruction_nl"] = out["prob"].map(lambda prob: f"Solve this fraction problem: {prob}=?")
    reasoning = clean_series(df[reasoning_col])
    resp = clean_series(df[resp_col])
    out["response_nl"] = reasoning.where(reasoning != "", resp)
    has_answer = out["response_nl"].str.contains(r"###\s*answer\s*:", case=False, regex=True, na=False)
    out.loc[~has_answer, "response_nl"] = (
        out.loc[~has_answer, "response_nl"] + "\n### answer: " + resp.loc[~has_answer]
    ).str.strip()
    out["strategy"] = clean_series(df[reasoning_col])
    out = out[(out["prob"] != "") & (out["response_nl"] != "")].reset_index(drop=True)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp_dtype(mode: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda" or mode == "none":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return torch.autocast(device_type="cpu", enabled=False)


def ensure_tokenizer_has_pad(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer is missing both pad_token_id and eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token


def strip_answer_lines(text: str) -> str:
    kept: List[str] = []
    for line in str(text).splitlines():
        lower = line.strip().lower()
        if lower.startswith("### answer:") or lower.startswith("### correctness:"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def build_classifier_text(
    prob: str,
    response_nl: str,
    instruction_nl: str = "",
    text_mode: str = "prob_response",
    drop_answer_line: bool = True,
) -> str:
    response = strip_answer_lines(response_nl) if drop_answer_line else str(response_nl).strip()
    if text_mode == "response":
        return response
    if text_mode == "prob_response":
        return f"Problem: {prob}=?\nResponse:\n{response}".strip()
    if text_mode == "instruction_response":
        prompt = str(instruction_nl).strip() if str(instruction_nl).strip() else f"Solve this fraction problem: {prob}=?"
        return f"Prompt:\n{prompt}\nResponse:\n{response}".strip()
    raise ValueError(f"Unsupported text_mode: {text_mode}")


def prompt_for_problem(prob: str) -> str:
    return f"Solve this fraction problem: {prob}=?"


def load_strategy_classifier(
    checkpoint_dir: str | os.PathLike[str],
    device: torch.device,
    local_files_only: bool = True,
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, List[str]]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, local_files_only=local_files_only)
    model.to(device)
    model.eval()
    id2label = getattr(model.config, "id2label", None) or {}
    label_order = [id2label[idx] for idx in sorted(id2label.keys())]
    if not label_order:
        label_order = list(DEFAULT_LABEL_ORDER)
    return tokenizer, model, label_order


def score_texts_with_classifier(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
    amp_dtype: Optional[torch.dtype] = None,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with autocast_context(device, amp_dtype):
                logits = model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1)
            outputs.append(probs.detach().cpu().numpy())
    if not outputs:
        return np.zeros((0, int(model.config.num_labels)), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def score_prob_response_pairs(
    probs: Sequence[str],
    responses: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
    amp_dtype: Optional[torch.dtype] = None,
    text_mode: str = "prob_response",
    drop_answer_line: bool = True,
) -> np.ndarray:
    texts = [
        build_classifier_text(
            prob=prob,
            response_nl=response,
            text_mode=text_mode,
            drop_answer_line=drop_answer_line,
        )
        for prob, response in zip(probs, responses)
    ]
    return score_texts_with_classifier(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        amp_dtype=amp_dtype,
    )


def distribution_to_named_dict(dist: Sequence[float], label_order: Sequence[str]) -> Dict[str, float]:
    return {label: float(value) for label, value in zip(label_order, dist)}


def named_dict_to_distribution(named_dist: Dict[str, Any], label_order: Sequence[str]) -> np.ndarray:
    return np.asarray([float(named_dist[label]) for label in label_order], dtype=np.float64)


def normalize_distribution(dist: Sequence[float], eps: float = 0.0) -> np.ndarray:
    arr = np.asarray(dist, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D distribution, got shape={arr.shape}")
    if eps > 0.0:
        arr = arr + float(eps)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError(f"Distribution has non-positive mass: {arr}")
    return arr / total


def total_variation_distance(p: Sequence[float], q: Sequence[float]) -> float:
    p_arr = normalize_distribution(p)
    q_arr = normalize_distribution(q)
    return float(0.5 * np.abs(p_arr - q_arr).sum())


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = DEFAULT_KL_EPS) -> float:
    p_arr = normalize_distribution(p, eps=eps)
    q_arr = normalize_distribution(q, eps=eps)
    return float(np.sum(p_arr * (np.log(p_arr) - np.log(q_arr))))


def build_wasserstein_cost_matrix(label_order: Sequence[str]) -> np.ndarray:
    label_set = set(label_order)
    if len(label_set) != len(label_order):
        raise ValueError(f"Duplicate labels in label_order={label_order}")
    cost = np.full((len(label_order), len(label_order)), 2.0, dtype=np.float64)
    np.fill_diagonal(cost, 0.0)
    special_pairs = {frozenset({"KDON", "CDON"}), frozenset({"ONOD", "CROP"})}
    for i, left in enumerate(label_order):
        for j, right in enumerate(label_order):
            if i == j:
                continue
            if frozenset({left, right}) in special_pairs:
                cost[i, j] = 1.0
    return cost


def _transport_constraints(num_labels: int) -> Tuple[np.ndarray, np.ndarray]:
    a_eq = np.zeros((2 * num_labels, num_labels * num_labels), dtype=np.float64)
    b_template = np.zeros(2 * num_labels, dtype=np.float64)
    row = 0
    for i in range(num_labels):
        a_eq[row, i * num_labels : (i + 1) * num_labels] = 1.0
        row += 1
    for j in range(num_labels):
        a_eq[row, j::num_labels] = 1.0
        row += 1
    return a_eq, b_template


def wasserstein_distance(
    p: Sequence[float],
    q: Sequence[float],
    cost_matrix: Optional[np.ndarray] = None,
) -> float:
    p_arr = normalize_distribution(p)
    q_arr = normalize_distribution(q)
    if p_arr.shape != q_arr.shape:
        raise ValueError(f"Distribution shape mismatch: {p_arr.shape} vs {q_arr.shape}")
    num_labels = int(p_arr.shape[0])
    if cost_matrix is None:
        cost_matrix = build_wasserstein_cost_matrix(DEFAULT_LABEL_ORDER[:num_labels])
    if cost_matrix.shape != (num_labels, num_labels):
        raise ValueError(f"Cost matrix shape mismatch: {cost_matrix.shape}")
    a_eq, b_template = _transport_constraints(num_labels)
    b_eq = b_template.copy()
    b_eq[:num_labels] = p_arr
    b_eq[num_labels:] = q_arr
    result = linprog(
        c=cost_matrix.reshape(-1),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Wasserstein LP failed: {result.message}")
    return float(result.fun)


def compute_divergence(
    p: Sequence[float],
    q: Sequence[float],
    loss_name: str,
    cost_matrix: Optional[np.ndarray] = None,
    eps: float = DEFAULT_KL_EPS,
) -> float:
    if loss_name == "tv":
        return total_variation_distance(p, q)
    if loss_name == "kl":
        return kl_divergence(p, q, eps=eps)
    if loss_name == "wasserstein":
        return wasserstein_distance(p, q, cost_matrix=cost_matrix)
    raise ValueError(f"Unsupported loss: {loss_name}")


def compute_all_divergences(
    p: Sequence[float],
    q: Sequence[float],
    label_order: Sequence[str],
    eps: float = DEFAULT_KL_EPS,
) -> Dict[str, float]:
    cost_matrix = build_wasserstein_cost_matrix(label_order)
    return {
        "tv": total_variation_distance(p, q),
        "kl": kl_divergence(p, q, eps=eps),
        "wasserstein": wasserstein_distance(p, q, cost_matrix=cost_matrix),
    }


def parse_numeric_token(token: str) -> Fraction:
    tok = str(token).strip()
    if "/" in tok:
        left, right = tok.split("/", 1)
        return Fraction(int(left), int(right))
    return Fraction(tok)


def compute_correct_answer(prob: str) -> str:
    match = PROB_RE.fullmatch(str(prob))
    if not match:
        return "?"
    left_s, op, right_s = match.group(1), match.group(2), match.group(3)
    try:
        left = parse_numeric_token(left_s)
        right = parse_numeric_token(right_s)
        if op == "+":
            ans = left + right
        elif op == "-":
            ans = left - right
        elif op == "*":
            ans = left * right
        elif op == ":":
            if right == 0:
                return "?"
            ans = left / right
        else:
            return "?"
    except Exception:
        return "?"
    if ans.denominator == 1:
        return str(ans.numerator)
    return f"{ans.numerator}/{ans.denominator}"


def normalize_answer(ans: str) -> str:
    out = str(ans).strip().replace(" ", "")
    return out.rstrip(".,;:!?")


def answer_to_float(ans: str) -> Optional[float]:
    token = normalize_answer(ans)
    if token == "" or token in {"?", "nan", "None"}:
        return None
    try:
        if "/" in token:
            left, right = token.split("/", 1)
            return float(int(left) / int(right))
        return float(token)
    except Exception:
        try:
            return float(Fraction(token))
        except Exception:
            return None


def answers_match(pred: str, correct: str, tol: float = 1e-6) -> bool:
    pred_val = answer_to_float(pred)
    correct_val = answer_to_float(correct)
    if pred_val is None or correct_val is None:
        return False
    return abs(pred_val - correct_val) < tol


def extract_final_answer(generation: str) -> str:
    txt = str(generation)
    match = re.search(r"###\s*answer\s*:\s*([^\n\r]+)", txt, flags=re.IGNORECASE)
    if match:
        candidate = normalize_answer(match.group(1).strip().split()[0])
        if candidate and answer_to_float(candidate) is not None:
            return candidate
    frac_matches = re.findall(r"-?\d+/-?\d+", txt)
    for token in reversed(frac_matches):
        candidate = normalize_answer(token)
        if answer_to_float(candidate) is not None:
            return candidate
    int_matches = re.findall(r"-?\d+", txt)
    for token in reversed(int_matches):
        candidate = normalize_answer(token)
        if answer_to_float(candidate) is not None:
            return candidate
    return "?"


def analyze_generated_response(
    response_text: str,
    min_reasoning_chars: int,
    empty_response_penalty: float = 0.0,
    short_reasoning_penalty: float = 0.0,
    unparseable_answer_penalty: float = 0.0,
) -> Dict[str, Any]:
    response = str(response_text)
    stripped_response = response.strip()
    empty_response = stripped_response == ""
    reasoning_text = strip_answer_lines(response).strip()
    reasoning_chars = int(len(reasoning_text))
    short_reasoning = (not empty_response) and (reasoning_chars < int(min_reasoning_chars))
    pred_answer = extract_final_answer(response)
    parseable_answer = answer_to_float(pred_answer) is not None
    valid_for_strategy = (not empty_response) and (reasoning_chars >= int(min_reasoning_chars)) and parseable_answer
    penalty = (
        float(empty_response_penalty) * float(empty_response)
        + float(short_reasoning_penalty) * float(short_reasoning)
        + float(unparseable_answer_penalty) * float(not parseable_answer)
    )
    return {
        "empty_response": bool(empty_response),
        "reasoning_text": reasoning_text,
        "reasoning_chars": int(reasoning_chars),
        "short_reasoning": bool(short_reasoning),
        "pred_answer": pred_answer,
        "parseable_answer": bool(parseable_answer),
        "valid_for_strategy": bool(valid_for_strategy),
        "penalty": float(penalty),
    }


def summarize_validity_records(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n_total = int(len(records))
    n_valid = int(sum(1 for row in records if row.get("valid_for_strategy", False)))
    n_empty = int(sum(1 for row in records if row.get("empty_response", False)))
    n_short = int(sum(1 for row in records if row.get("short_reasoning", False)))
    n_unparseable = int(sum(1 for row in records if not row.get("parseable_answer", False)))
    penalties = [float(row.get("penalty", 0.0)) for row in records]
    return {
        "n_valid_for_strategy": n_valid,
        "n_empty_response": n_empty,
        "n_short_reasoning": n_short,
        "n_unparseable_answer": n_unparseable,
        "valid_rollout_fraction": float(n_valid / n_total) if n_total > 0 else 0.0,
        "empty_response_fraction": float(n_empty / n_total) if n_total > 0 else 0.0,
        "short_reasoning_fraction": float(n_short / n_total) if n_total > 0 else 0.0,
        "unparseable_answer_fraction": float(n_unparseable / n_total) if n_total > 0 else 0.0,
        "mean_penalty": float(np.mean(penalties)) if penalties else 0.0,
    }


def compute_strategy_rollout_distribution(
    posterior: np.ndarray,
    rollout_records: Sequence[Dict[str, Any]],
    label_order: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if int(posterior.shape[0]) != int(len(rollout_records)):
        raise ValueError(
            f"Posterior/rollout length mismatch: posterior={posterior.shape[0]} rollouts={len(rollout_records)}"
        )
    valid_indices = [idx for idx, row in enumerate(rollout_records) if row.get("valid_for_strategy", False)]
    if valid_indices:
        rollout_dist = posterior[valid_indices].mean(axis=0)
        used_uniform_fallback = False
    else:
        rollout_dist = np.full((len(label_order),), 1.0 / float(len(label_order)), dtype=np.float64)
        used_uniform_fallback = True
    rollout_dist = normalize_distribution(rollout_dist)
    return rollout_dist, {
        "n_valid_for_strategy": int(len(valid_indices)),
        "used_uniform_fallback": bool(used_uniform_fallback),
    }


def op_from_prob(prob: str) -> str:
    text = str(prob)
    if "+" in text:
        return "+"
    if "-" in text:
        return "-"
    if "*" in text:
        return "*"
    if ":" in text:
        return ":"
    return "?"


def extract_prob_from_prompt(prompt: str) -> str:
    match = PROMPT_PROB_RE.search(str(prompt).strip())
    if not match:
        return ""
    return match.group(1).strip()


def dedupe_sp2013_human_problems(sp2013_df: pd.DataFrame) -> pd.DataFrame:
    if "prob" not in sp2013_df.columns:
        raise ValueError("sp2013_human.csv is missing a 'prob' column.")
    deduped = (
        sp2013_df.copy()
        .assign(prob=clean_series(sp2013_df["prob"]))
        .drop_duplicates(subset=["prob"])
        .sort_values("prob")
        .reset_index(drop=True)
    )
    return deduped


def trim_generated_response_tokens(
    response_token_ids: Sequence[int],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> List[int]:
    tokens = list(int(token) for token in response_token_ids)
    if not tokens:
        return []
    if eos_token_id is not None:
        for idx, token in enumerate(tokens):
            if token == eos_token_id:
                return tokens[: idx + 1]
    if pad_token_id is not None and pad_token_id != eos_token_id:
        for idx, token in enumerate(tokens):
            if token == pad_token_id:
                return tokens[:idx]
    return tokens


def sample_rollouts_for_problem(
    model_for_generation: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prob: str,
    prompt_text: str,
    num_rollouts: int,
    rollout_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: torch.device,
    min_new_tokens: int = 0,
) -> List[GeneratedRollout]:
    ensure_tokenizer_has_pad(tokenizer)
    base_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    query_ids = base_enc["input_ids"][0].tolist()
    query_tensor = base_enc["input_ids"].to(device)
    query_mask = base_enc["attention_mask"].to(device)
    outputs: List[GeneratedRollout] = []

    while len(outputs) < num_rollouts:
        cur_batch = min(int(rollout_batch_size), int(num_rollouts - len(outputs)))
        input_ids = query_tensor.repeat(cur_batch, 1)
        attention_mask = query_mask.repeat(cur_batch, 1)
        generate_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if int(min_new_tokens) > 0:
            generate_kwargs["min_new_tokens"] = int(min_new_tokens)
        generated = model_for_generation.generate(**generate_kwargs)
        generated = generated.detach().cpu()
        for row in generated:
            full_ids = row.tolist()
            response_ids = trim_generated_response_tokens(
                response_token_ids=full_ids[len(query_ids) :],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            outputs.append(
                GeneratedRollout(
                    prob=prob,
                    prompt_text=prompt_text,
                    query_token_ids=list(query_ids),
                    response_token_ids=response_ids,
                    full_token_ids=list(query_ids) + list(response_ids),
                    response_text=response_text,
                )
            )
    return outputs


def rollout_batch_to_tensors(
    rollouts: Sequence[GeneratedRollout],
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, Any]:
    if not rollouts:
        raise ValueError("Expected at least one rollout.")
    max_total_len = max(len(item.full_token_ids) for item in rollouts)
    input_ids = torch.full((len(rollouts), max_total_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rollouts), max_total_len), dtype=torch.long)
    query_lens: List[int] = []
    response_lens: List[int] = []
    for row_idx, item in enumerate(rollouts):
        token_ids = torch.tensor(item.full_token_ids, dtype=torch.long)
        seq_len = int(token_ids.numel())
        input_ids[row_idx, :seq_len] = token_ids
        attention_mask[row_idx, :seq_len] = 1
        query_lens.append(int(len(item.query_token_ids)))
        response_lens.append(int(len(item.response_token_ids)))
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "query_lens": query_lens,
        "response_lens": response_lens,
    }


def gather_response_logprobs_and_values(
    logits: torch.Tensor,
    values: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    query_lens: Sequence[int],
    response_lens: Sequence[int],
    max_resp_len: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    batch_size = int(input_ids.shape[0])
    max_resp_len = int(max_resp_len) if max_resp_len is not None else max(int(length) for length in response_lens)
    logprobs = torch.zeros((batch_size, max_resp_len), dtype=torch.float32, device=input_ids.device)
    gathered_values = None
    if values is not None:
        gathered_values = torch.zeros((batch_size, max_resp_len), dtype=torch.float32, device=input_ids.device)
    mask = torch.zeros((batch_size, max_resp_len), dtype=torch.bool, device=input_ids.device)

    for row_idx in range(batch_size):
        query_len = int(query_lens[row_idx])
        resp_len = int(response_lens[row_idx])
        if resp_len <= 0:
            continue
        target_tokens = input_ids[row_idx, query_len : query_len + resp_len]
        step_logits = logits[row_idx, query_len - 1 : query_len - 1 + resp_len].float()
        step_logprobs = F.log_softmax(step_logits, dim=-1).gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        logprobs[row_idx, :resp_len] = step_logprobs
        if gathered_values is not None:
            step_values = values[row_idx, query_len - 1 : query_len - 1 + resp_len].float()
            gathered_values[row_idx, :resp_len] = step_values
        mask[row_idx, :resp_len] = True

    return logprobs, gathered_values, mask


def summarize_accuracy_records(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n_total = int(len(records))
    n_parseable = int(sum(1 for row in records if row["pred_answer_parseable"]))
    n_correct = int(sum(1 for row in records if row["is_correct"]))
    return {
        "n_total": n_total,
        "n_parseable": n_parseable,
        "n_correct": n_correct,
        "parseable_coverage": float(n_parseable / n_total) if n_total > 0 else 0.0,
        "accuracy_all": float(n_correct / n_total) if n_total > 0 else 0.0,
        "accuracy_parseable": float(n_correct / n_parseable) if n_parseable > 0 else float("nan"),
    }


def evaluate_rollouts_against_target(
    prob: str,
    rollouts: Sequence[GeneratedRollout],
    target_distribution: Sequence[float],
    label_order: Sequence[str],
    classifier_tokenizer: PreTrainedTokenizerBase,
    classifier_model: AutoModelForSequenceClassification,
    classifier_device: torch.device,
    min_reasoning_chars: int = 8,
    selection_loss: str = "tv",
    best_id_invalid_coef: float = 0.5,
    classifier_batch_size: int = 64,
    classifier_max_length: int = 256,
    classifier_amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    responses = [item.response_text for item in rollouts]
    rollout_analyses = [
        analyze_generated_response(
            response_text=item.response_text,
            min_reasoning_chars=min_reasoning_chars,
        )
        for item in rollouts
    ]
    posterior = score_prob_response_pairs(
        probs=[prob] * len(rollouts),
        responses=responses,
        tokenizer=classifier_tokenizer,
        model=classifier_model,
        device=classifier_device,
        batch_size=classifier_batch_size,
        max_length=classifier_max_length,
        amp_dtype=classifier_amp_dtype,
        text_mode="prob_response",
        drop_answer_line=True,
    )
    rollout_dist, strategy_summary = compute_strategy_rollout_distribution(
        posterior=posterior,
        rollout_records=rollout_analyses,
        label_order=label_order,
    )
    divergence = compute_all_divergences(target_distribution, rollout_dist, label_order)

    correct_answer = compute_correct_answer(prob)
    records: List[Dict[str, Any]] = []
    for row_idx, (item, post, analysis) in enumerate(zip(rollouts, posterior, rollout_analyses)):
        records.append(
            {
                "prob": prob,
                "rollout_idx": int(row_idx),
                "prompt_text": item.prompt_text,
                "response_text": item.response_text,
                "pred_answer": str(analysis["pred_answer"]),
                "pred_answer_parseable": bool(analysis["parseable_answer"]),
                "correct_answer": correct_answer,
                "is_correct": bool(answers_match(str(analysis["pred_answer"]), correct_answer)),
                "empty_response": bool(analysis["empty_response"]),
                "reasoning_chars": int(analysis["reasoning_chars"]),
                "short_reasoning": bool(analysis["short_reasoning"]),
                "parseable_answer": bool(analysis["parseable_answer"]),
                "valid_for_strategy": bool(analysis["valid_for_strategy"]),
                "penalty": float(analysis["penalty"]),
                "classifier_argmax": str(label_order[int(np.argmax(post))]),
                **{f"classifier_prob_{label}": float(prob_value) for label, prob_value in zip(label_order, post)},
            }
        )
    accuracy = summarize_accuracy_records(records)
    validity = summarize_validity_records(records)
    selected_distance = float(divergence[selection_loss])
    selection_metric = float(selected_distance + float(best_id_invalid_coef) * (1.0 - validity["valid_rollout_fraction"]))
    metrics = {
        "prob": prob,
        "n_rollouts": int(len(rollouts)),
        "rollout_distribution": distribution_to_named_dict(rollout_dist, label_order),
        "target_distribution": distribution_to_named_dict(target_distribution, label_order),
        "selected_loss": str(selection_loss),
        "selected_distance": selected_distance,
        "selection_metric": selection_metric,
        "used_uniform_fallback": bool(strategy_summary["used_uniform_fallback"]),
        **{f"{name}_distance": float(value) for name, value in divergence.items()},
        **accuracy,
        **validity,
    }
    return metrics, records


def run_strategy_matching_unit_checks() -> Dict[str, Any]:
    label_order = list(DEFAULT_LABEL_ORDER)
    cost_matrix = build_wasserstein_cost_matrix(label_order)
    p = np.asarray([0.25, 0.15, 0.20, 0.30, 0.10], dtype=np.float64)
    zero_checks = {
        "tv_zero": total_variation_distance(p, p),
        "kl_zero": kl_divergence(p, p),
        "wasserstein_zero": wasserstein_distance(p, p, cost_matrix=cost_matrix),
    }
    for name, value in zero_checks.items():
        if abs(value) > 1e-8:
            raise AssertionError(f"{name} expected 0, got {value}")

    idx = {label: i for i, label in enumerate(label_order)}
    if cost_matrix[idx["KDON"], idx["CDON"]] != 1.0 or cost_matrix[idx["CDON"], idx["KDON"]] != 1.0:
        raise AssertionError("KDON<->CDON Wasserstein cost must be 1.")
    if cost_matrix[idx["ONOD"], idx["CROP"]] != 1.0 or cost_matrix[idx["CROP"], idx["ONOD"]] != 1.0:
        raise AssertionError("ONOD<->CROP Wasserstein cost must be 1.")
    if cost_matrix[idx["KDON"], idx["ICDM"]] != 2.0:
        raise AssertionError("All other off-diagonal Wasserstein costs must be 2.")

    empty = analyze_generated_response(
        response_text="",
        min_reasoning_chars=8,
        empty_response_penalty=1.0,
        short_reasoning_penalty=0.5,
        unparseable_answer_penalty=0.5,
    )
    if empty["valid_for_strategy"]:
        raise AssertionError("Empty response must be invalid for strategy.")
    if abs(float(empty["penalty"]) - 1.5) > 1e-8:
        raise AssertionError(f"Empty-response penalty must equal 1.5, got {empty['penalty']}")

    short = analyze_generated_response(
        response_text="ok\n### answer: 1/2",
        min_reasoning_chars=8,
        empty_response_penalty=1.0,
        short_reasoning_penalty=0.5,
        unparseable_answer_penalty=0.5,
    )
    if short["valid_for_strategy"]:
        raise AssertionError("Short reasoning must be excluded from strategy aggregation.")
    if abs(float(short["penalty"]) - 0.5) > 1e-8:
        raise AssertionError(f"Short-reasoning penalty must equal 0.5, got {short['penalty']}")

    valid = analyze_generated_response(
        response_text="I multiplied the numerators and denominators to get 3/10.\n### answer: 3/10",
        min_reasoning_chars=8,
        empty_response_penalty=1.0,
        short_reasoning_penalty=0.5,
        unparseable_answer_penalty=0.5,
    )
    if not valid["valid_for_strategy"]:
        raise AssertionError("Long parseable response must be valid for strategy aggregation.")

    posterior = np.asarray(
        [
            [0.10, 0.10, 0.10, 0.10, 0.60],
            [0.60, 0.10, 0.10, 0.10, 0.10],
            [0.05, 0.05, 0.05, 0.80, 0.05],
        ],
        dtype=np.float64,
    )
    q_valid, q_valid_info = compute_strategy_rollout_distribution(
        posterior=posterior,
        rollout_records=[empty, short, valid],
        label_order=label_order,
    )
    if q_valid_info["n_valid_for_strategy"] != 1:
        raise AssertionError("Expected exactly one valid rollout in q_valid_info.")
    if not np.allclose(q_valid, posterior[2]):
        raise AssertionError("Strategy distribution must average only valid posteriors.")

    q_fallback, q_fallback_info = compute_strategy_rollout_distribution(
        posterior=posterior[:2],
        rollout_records=[empty, short],
        label_order=label_order,
    )
    expected_uniform = np.full((len(label_order),), 1.0 / float(len(label_order)), dtype=np.float64)
    if not q_fallback_info["used_uniform_fallback"]:
        raise AssertionError("All-invalid rollouts must trigger uniform fallback.")
    if not np.allclose(q_fallback, expected_uniform):
        raise AssertionError("All-invalid rollout set must fall back to the uniform distribution.")

    return {
        "zero_checks": {key: float(value) for key, value in zero_checks.items()},
        "wasserstein_cost_matrix": cost_matrix.tolist(),
        "validity_checks": {
            "empty_response_valid_for_strategy": bool(empty["valid_for_strategy"]),
            "short_reasoning_valid_for_strategy": bool(short["valid_for_strategy"]),
            "long_parseable_valid_for_strategy": bool(valid["valid_for_strategy"]),
            "empty_response_penalty": float(empty["penalty"]),
            "short_reasoning_penalty": float(short["penalty"]),
            "uniform_fallback_distribution": q_fallback.tolist(),
        },
    }
