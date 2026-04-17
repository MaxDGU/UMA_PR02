#!/usr/bin/env python3
"""
Evaluate human strategy-matching models on ID strategy metrics and OOD answer accuracy.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

from strategy_matching_utils import (
    dedupe_sp2013_human_problems,
    ensure_tokenizer_has_pad,
    evaluate_rollouts_against_target,
    extract_final_answer,
    load_json,
    load_strategy_classifier,
    named_dict_to_distribution,
    op_from_prob,
    read_csv_flexible,
    resolve_amp_dtype,
    resolve_device,
    sample_rollouts_for_problem,
    save_json,
    summarize_accuracy_records,
)


DEFAULT_CLASSIFIER_CKPT = os.path.join(
    "results",
    "transformer_replication",
    "strategy_classifier_distilbert_n160000_e3_final",
    "best",
)
DEFAULT_HUMAN_TARGET_JSON = os.path.join(
    "results",
    "human",
    "human_strategy_targets_classifier160k.json",
)
DEFAULT_SP2013_HUMAN_CSV = "sp2013_human.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a human strategy-matching checkpoint.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--classifier_ckpt", type=str, default=DEFAULT_CLASSIFIER_CKPT)
    parser.add_argument("--human_target_json", type=str, default=DEFAULT_HUMAN_TARGET_JSON)
    parser.add_argument("--sp2013_human_csv", type=str, default=DEFAULT_SP2013_HUMAN_CSV)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--id_rollouts_per_problem", type=int, default=128)
    parser.add_argument("--ood_rollouts_per_problem", type=int, default=128)
    parser.add_argument("--rollout_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_new_tokens", type=int, default=12)
    parser.add_argument("--min_reasoning_chars", type=int, default=8)
    parser.add_argument("--selection_loss", choices=["tv", "kl", "wasserstein"], default="tv")
    parser.add_argument("--best_id_invalid_coef", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--classifier_device", type=str, default="auto")
    parser.add_argument(
        "--amp_dtype",
        choices=["auto", "bf16", "fp16", "none"],
        default="auto",
        help="Mixed precision mode for the policy model on CUDA.",
    )
    parser.add_argument(
        "--classifier_amp_dtype",
        choices=["auto", "bf16", "fp16", "none"],
        default="auto",
        help="Mixed precision mode for the classifier on CUDA.",
    )
    parser.add_argument("--classifier_batch_size", type=int, default=64)
    parser.add_argument("--classifier_max_length", type=int, default=256)
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load all models/tokenizers from local cache only.",
    )
    return parser.parse_args()


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_output_dir(raw: str, model_path: str) -> Path:
    if raw.strip():
        return Path(raw).resolve()
    model_name = Path(model_path).resolve().name
    return Path("results", "transformer_replication", f"human_strategy_eval_{model_name}_{now_ts()}").resolve()


def load_policy_model(
    model_path: str,
    device: torch.device,
    local_files_only: bool,
) -> Tuple[AutoTokenizer, AutoModelForCausalLMWithValueHead]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    ensure_tokenizer_has_pad(tokenizer)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, local_files_only=local_files_only)
    model.to(device)
    model.eval()
    model.pretrained_model.config.use_cache = True
    return tokenizer, model


def load_human_target_payload(path: str) -> Tuple[Dict[str, Any], List[str]]:
    payload = load_json(path)
    label_order = list(payload["classifier_label_order"])
    if len(payload["problems"]) != 8:
        raise ValueError(f"Expected 8 ID problems in human target JSON, found {len(payload['problems'])}")
    return payload, label_order


def evaluate_id_split(
    model_for_generation,
    tokenizer,
    human_target_payload: Dict[str, Any],
    label_order: Sequence[str],
    classifier_tokenizer,
    classifier_model,
    generation_device: torch.device,
    classifier_device: torch.device,
    rollouts_per_problem: int,
    rollout_batch_size: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_reasoning_chars: int,
    selection_loss: str,
    best_id_invalid_coef: float,
    classifier_batch_size: int,
    classifier_max_length: int,
    classifier_amp_dtype,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    per_problem_metrics: List[Dict[str, Any]] = []
    rollout_records: List[Dict[str, Any]] = []

    for problem_payload in sorted(human_target_payload["problems"], key=lambda row: row["prob"]):
        prob = str(problem_payload["prob"])
        prompt_text = str(problem_payload["prompt_text"])
        target_dist = named_dict_to_distribution(problem_payload["target_distribution"], label_order)
        rollouts = sample_rollouts_for_problem(
            model_for_generation=model_for_generation,
            tokenizer=tokenizer,
            prob=prob,
            prompt_text=prompt_text,
            num_rollouts=rollouts_per_problem,
            rollout_batch_size=rollout_batch_size,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            device=generation_device,
        )
        metrics, records = evaluate_rollouts_against_target(
            prob=prob,
            rollouts=rollouts,
            target_distribution=target_dist,
            label_order=label_order,
            classifier_tokenizer=classifier_tokenizer,
            classifier_model=classifier_model,
            classifier_device=classifier_device,
            min_reasoning_chars=min_reasoning_chars,
            selection_loss=selection_loss,
            best_id_invalid_coef=best_id_invalid_coef,
            classifier_batch_size=classifier_batch_size,
            classifier_max_length=classifier_max_length,
            classifier_amp_dtype=classifier_amp_dtype,
        )
        per_problem_metrics.append(metrics)
        rollout_records.extend(records)

    problem_df = pd.DataFrame(per_problem_metrics)
    rollout_df = pd.DataFrame(rollout_records)
    summary = {
        "problem_count": int(problem_df.shape[0]),
        "rollouts_per_problem": int(rollouts_per_problem),
        "mean_tv_distance": float(problem_df["tv_distance"].mean()),
        "mean_kl_distance": float(problem_df["kl_distance"].mean()),
        "mean_wasserstein_distance": float(problem_df["wasserstein_distance"].mean()),
        "valid_rollout_fraction": float(problem_df["valid_rollout_fraction"].mean()),
        "empty_response_fraction": float(problem_df["empty_response_fraction"].mean()),
        "short_reasoning_fraction": float(problem_df["short_reasoning_fraction"].mean()),
        "unparseable_answer_fraction": float(problem_df["unparseable_answer_fraction"].mean()),
        "selected_loss": str(selection_loss),
        "selected_distance": float(problem_df["selected_distance"].mean()),
        "selection_metric": float(problem_df["selection_metric"].mean()),
        **summarize_accuracy_records(rollout_records),
    }
    return summary, problem_df, rollout_df


def evaluate_ood_split(
    model_for_generation,
    tokenizer,
    sp2013_human_csv: str,
    generation_device: torch.device,
    rollouts_per_problem: int,
    rollout_batch_size: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    sp2013_df = dedupe_sp2013_human_problems(read_csv_flexible(sp2013_human_csv))
    per_problem_metrics: List[Dict[str, Any]] = []
    rollout_records: List[Dict[str, Any]] = []

    for prob in sorted(sp2013_df["prob"].astype(str).str.strip().tolist()):
        prompt_text = f"Solve this fraction problem: {prob}=?"
        rollouts = sample_rollouts_for_problem(
            model_for_generation=model_for_generation,
            tokenizer=tokenizer,
            prob=prob,
            prompt_text=prompt_text,
            num_rollouts=rollouts_per_problem,
            rollout_batch_size=rollout_batch_size,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            device=generation_device,
        )
        correct_answer = None
        prob_records: List[Dict[str, Any]] = []
        for row_idx, rollout in enumerate(rollouts):
            if correct_answer is None:
                from strategy_matching_utils import compute_correct_answer, answer_to_float, answers_match

                correct_answer = compute_correct_answer(prob)
            pred_answer = extract_final_answer(rollout.response_text)
            pred_parseable = answer_to_float(pred_answer) is not None
            prob_records.append(
                {
                    "prob": prob,
                    "op": op_from_prob(prob),
                    "rollout_idx": int(row_idx),
                    "prompt_text": rollout.prompt_text,
                    "response_text": rollout.response_text,
                    "pred_answer": pred_answer,
                    "pred_answer_parseable": bool(pred_parseable),
                    "correct_answer": correct_answer,
                    "is_correct": bool(answers_match(pred_answer, correct_answer)),
                }
            )
        accuracy = summarize_accuracy_records(prob_records)
        per_problem_metrics.append(
            {
                "prob": prob,
                "op": op_from_prob(prob),
                "n_rollouts": int(len(prob_records)),
                **accuracy,
            }
        )
        rollout_records.extend(prob_records)

    problem_df = pd.DataFrame(per_problem_metrics)
    rollout_df = pd.DataFrame(rollout_records)
    by_op: Dict[str, Any] = {}
    for op, group in rollout_df.groupby("op", sort=True):
        by_op[str(op)] = summarize_accuracy_records(group.to_dict(orient="records"))
    summary = {
        "problem_count": int(problem_df.shape[0]),
        "rollouts_per_problem": int(rollouts_per_problem),
        **summarize_accuracy_records(rollout_records),
        "by_operation": by_op,
    }
    return summary, problem_df, rollout_df


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    classifier_device = resolve_device(args.classifier_device)
    classifier_amp_dtype = resolve_amp_dtype(args.classifier_amp_dtype, classifier_device)
    output_dir = resolve_output_dir(args.output_dir, args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_policy_model(
        model_path=args.model_path,
        device=device,
        local_files_only=bool(args.local_files_only),
    )
    classifier_tokenizer, classifier_model, label_order = load_strategy_classifier(
        checkpoint_dir=args.classifier_ckpt,
        device=classifier_device,
        local_files_only=bool(args.local_files_only),
    )
    human_target_payload, target_label_order = load_human_target_payload(args.human_target_json)
    if list(label_order) != list(target_label_order):
        raise ValueError(
            f"Classifier label order mismatch: checkpoint={label_order} target_json={target_label_order}"
        )

    id_summary, id_problem_df, id_rollout_df = evaluate_id_split(
        model_for_generation=model.pretrained_model,
        tokenizer=tokenizer,
        human_target_payload=human_target_payload,
        label_order=label_order,
        classifier_tokenizer=classifier_tokenizer,
        classifier_model=classifier_model,
        generation_device=device,
        classifier_device=classifier_device,
        rollouts_per_problem=args.id_rollouts_per_problem,
        rollout_batch_size=args.rollout_batch_size,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_reasoning_chars=args.min_reasoning_chars,
        selection_loss=args.selection_loss,
        best_id_invalid_coef=args.best_id_invalid_coef,
        classifier_batch_size=args.classifier_batch_size,
        classifier_max_length=args.classifier_max_length,
        classifier_amp_dtype=classifier_amp_dtype,
    )
    ood_summary, ood_problem_df, ood_rollout_df = evaluate_ood_split(
        model_for_generation=model.pretrained_model,
        tokenizer=tokenizer,
        sp2013_human_csv=args.sp2013_human_csv,
        generation_device=device,
        rollouts_per_problem=args.ood_rollouts_per_problem,
        rollout_batch_size=args.rollout_batch_size,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    eval_summary = {
        "model_path": os.path.abspath(args.model_path),
        "classifier_checkpoint": os.path.abspath(args.classifier_ckpt),
        "human_target_json": os.path.abspath(args.human_target_json),
        "sp2013_human_csv": os.path.abspath(args.sp2013_human_csv),
        "id_metrics": id_summary,
        "ood_metrics": ood_summary,
    }
    save_json(output_dir / "eval_args.json", vars(args))
    save_json(output_dir / "eval_summary.json", eval_summary)
    id_problem_df.to_csv(output_dir / "id_per_problem.csv", index=False)
    id_rollout_df.to_csv(output_dir / "id_rollouts.csv", index=False)
    ood_problem_df.to_csv(output_dir / "ood_per_problem.csv", index=False)
    ood_rollout_df.to_csv(output_dir / "ood_rollouts.csv", index=False)

    print(f"Saved evaluation artifacts to {output_dir}")
    print(
        "ID mean distances | "
        f"TV={id_summary['mean_tv_distance']:.4f} "
        f"KL={id_summary['mean_kl_distance']:.4f} "
        f"W={id_summary['mean_wasserstein_distance']:.4f}"
    )
    print(
        "Accuracy | "
        f"ID={id_summary['accuracy_all']:.4f} "
        f"OOD={ood_summary['accuracy_all']:.4f}"
    )


if __name__ == "__main__":
    main()
