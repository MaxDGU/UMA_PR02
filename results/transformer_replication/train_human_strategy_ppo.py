#!/usr/bin/env python3
"""
Train a distilled model to match human strategy distributions with PPO.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType
from torch.optim import AdamW
from trl import AutoModelForCausalLMWithValueHead
from trl.trainer.ppo_trainer import masked_mean, masked_whiten
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from eval_human_strategy_ppo import evaluate_id_split, evaluate_ood_split
from strategy_matching_utils import (
    analyze_generated_response,
    autocast_context,
    compute_all_divergences,
    compute_strategy_rollout_distribution,
    distribution_to_named_dict,
    ensure_tokenizer_has_pad,
    gather_response_logprobs_and_values,
    load_json,
    load_strategy_classifier,
    named_dict_to_distribution,
    resolve_amp_dtype,
    resolve_device,
    rollout_batch_to_tensors,
    run_strategy_matching_unit_checks,
    sample_rollouts_for_problem,
    score_prob_response_pairs,
    save_json,
    set_seed,
    summarize_validity_records,
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
    parser = argparse.ArgumentParser(description="PPO strategy-distribution matching on human fraction data.")
    parser.add_argument("--init_model_dir", type=str, required=True)
    parser.add_argument("--classifier_ckpt", type=str, default=DEFAULT_CLASSIFIER_CKPT)
    parser.add_argument("--human_target_json", type=str, default=DEFAULT_HUMAN_TARGET_JSON)
    parser.add_argument("--sp2013_human_csv", type=str, default=DEFAULT_SP2013_HUMAN_CSV)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--loss", choices=["tv", "kl", "wasserstein"], default="tv")
    parser.add_argument("--rollouts_per_prompt", type=int, default=32)
    parser.add_argument("--prompt_batch_size", type=int, default=4)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument("--policy_forward_batch_size", type=int, default=16)
    parser.add_argument("--train_micro_batch_size", type=int, default=8)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_new_tokens", type=int, default=12)
    parser.add_argument("--min_reasoning_chars", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--total_updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--cliprange_value", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--empty_response_penalty", type=float, default=1.0)
    parser.add_argument("--short_reasoning_penalty", type=float, default=0.5)
    parser.add_argument("--unparseable_answer_penalty", type=float, default=0.5)
    parser.add_argument("--best_id_invalid_coef", type=float, default=0.5)
    parser.add_argument(
        "--whiten_advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whiten PPO advantages using the valid response-token mask.",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing on the trainable policy backbone.",
    )
    parser.add_argument("--id_eval_every", type=int, default=10)
    parser.add_argument("--id_eval_rollouts_per_problem", type=int, default=128)
    parser.add_argument("--ood_eval_rollouts_per_problem", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--classifier_batch_size", type=int, default=64)
    parser.add_argument("--classifier_max_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--classifier_device", type=str, default="cpu")
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
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load all models/tokenizers from local cache only.",
    )
    parser.add_argument(
        "--run_unit_checks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the divergence/cost-matrix unit checks before training.",
    )
    return parser.parse_args()


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(message: str) -> None:
    print(message, flush=True)


def resolve_output_dir(raw: str, init_model_dir: str, loss_name: str) -> Path:
    if raw.strip():
        return Path(raw).resolve()
    model_name = Path(init_model_dir).resolve().name
    return Path(
        "results",
        "transformer_replication",
        f"human_strategy_ppo_{model_name}_{loss_name}_{now_ts()}",
    ).resolve()


def parse_target_modules(text: str) -> List[str]:
    modules = [item.strip() for item in str(text).split(",") if item.strip()]
    if not modules:
        raise ValueError("--lora_target_modules must not be empty.")
    return modules


def load_human_targets(path: str) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
    payload = load_json(path)
    label_order = list(payload["classifier_label_order"])
    problems = sorted(payload["problems"], key=lambda row: row["prob"])
    if len(problems) != 8:
        raise ValueError(f"Expected 8 human target problems, found {len(problems)}")
    return payload, label_order, problems


def load_policy_and_reference_models(
    init_model_dir: str,
    device: torch.device,
    local_files_only: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str],
    gradient_checkpointing: bool,
) -> Tuple[AutoTokenizer, AutoModelForCausalLMWithValueHead, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(init_model_dir, local_files_only=local_files_only)
    ensure_tokenizer_has_pad(tokenizer)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(target_modules),
        bias="none",
    )
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        init_model_dir,
        peft_config=lora_config,
        is_trainable=True,
        local_files_only=local_files_only,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(init_model_dir, local_files_only=local_files_only)

    policy.to(device)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    if gradient_checkpointing:
        policy.pretrained_model.gradient_checkpointing_enable()
        policy.pretrained_model.config.use_cache = False
    else:
        policy.pretrained_model.config.use_cache = True
    ref_model.config.use_cache = True
    return tokenizer, policy, ref_model


def trainable_param_count(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        total += int(param.numel())
        if param.requires_grad:
            trainable += int(param.numel())
    return trainable, total


def compute_rollout_stats(
    rollouts: Sequence[Any],
    policy: AutoModelForCausalLMWithValueHead,
    ref_model: AutoModelForCausalLM,
    pad_token_id: int,
    device: torch.device,
    amp_dtype,
    forward_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_resp_len = max(int(len(item.response_token_ids)) for item in rollouts)
    old_logprobs_list: List[torch.Tensor] = []
    old_values_list: List[torch.Tensor] = []
    ref_logprobs_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []

    policy.eval()
    with torch.no_grad():
        for start in range(0, len(rollouts), forward_batch_size):
            batch_rollouts = list(rollouts[start : start + forward_batch_size])
            batch = rollout_batch_to_tensors(batch_rollouts, pad_token_id=pad_token_id, device=device)
            with autocast_context(device, amp_dtype):
                policy_logits, _, policy_values = policy(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                ref_logits = ref_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits

            old_logprobs, old_values, valid_mask = gather_response_logprobs_and_values(
                logits=policy_logits,
                values=policy_values,
                input_ids=batch["input_ids"],
                query_lens=batch["query_lens"],
                response_lens=batch["response_lens"],
                max_resp_len=max_resp_len,
            )
            ref_logprobs, _, _ = gather_response_logprobs_and_values(
                logits=ref_logits,
                values=None,
                input_ids=batch["input_ids"],
                query_lens=batch["query_lens"],
                response_lens=batch["response_lens"],
                max_resp_len=max_resp_len,
            )
            old_logprobs_list.append(old_logprobs.detach())
            old_values_list.append(old_values.detach())
            ref_logprobs_list.append(ref_logprobs.detach())
            mask_list.append(valid_mask.detach())

    policy.train()
    return (
        torch.cat(old_logprobs_list, dim=0),
        torch.cat(old_values_list, dim=0),
        torch.cat(ref_logprobs_list, dim=0),
        torch.cat(mask_list, dim=0),
    )


def build_rewards_and_advantages(
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    terminal_scores: torch.Tensor,
    gamma: float,
    lam: float,
    kl_coef: float,
    whiten_advantages: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_mask_f = valid_mask.float()
    token_kl = old_logprobs - ref_logprobs
    rewards = (-float(kl_coef) * token_kl) * valid_mask_f
    values = values * valid_mask_f

    last_token_idx = valid_mask.long().sum(dim=1) - 1
    for row_idx in range(rewards.shape[0]):
        token_idx = int(last_token_idx[row_idx].item())
        if token_idx >= 0:
            rewards[row_idx, token_idx] += terminal_scores[row_idx]

    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(rewards.shape[0], dtype=torch.float32, device=rewards.device)
    for t in reversed(range(rewards.shape[1])):
        nextvalues = values[:, t + 1] if t < rewards.shape[1] - 1 else torch.zeros_like(lastgaelam)
        delta = rewards[:, t] + float(gamma) * nextvalues - values[:, t]
        lastgaelam = delta + float(gamma) * float(lam) * lastgaelam
        lastgaelam = lastgaelam * valid_mask_f[:, t]
        advantages[:, t] = lastgaelam
    returns = advantages + values

    if whiten_advantages and bool(valid_mask.any()):
        advantages = masked_whiten(advantages, valid_mask)
    advantages = torch.where(valid_mask, advantages, torch.zeros_like(advantages))
    returns = torch.where(valid_mask, returns, torch.zeros_like(returns))
    return rewards, advantages, returns, token_kl


def save_policy_checkpoint(
    checkpoint_dir: Path,
    policy: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    metadata: Dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    save_json(checkpoint_dir / "checkpoint_metadata.json", metadata)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, args.init_model_dir, args.loss)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.run_unit_checks:
        save_json(output_dir / "unit_checks.json", run_strategy_matching_unit_checks())

    device = resolve_device(args.device)
    classifier_device = resolve_device(args.classifier_device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    classifier_amp_dtype = resolve_amp_dtype(args.classifier_amp_dtype, classifier_device)

    human_target_payload, label_order, target_problems = load_human_targets(args.human_target_json)
    classifier_tokenizer, classifier_model, classifier_label_order = load_strategy_classifier(
        checkpoint_dir=args.classifier_ckpt,
        device=classifier_device,
        local_files_only=bool(args.local_files_only),
    )
    if list(label_order) != list(classifier_label_order):
        raise ValueError(
            f"Classifier label order mismatch: target_json={label_order} classifier={classifier_label_order}"
        )

    tokenizer, policy, ref_model = load_policy_and_reference_models(
        init_model_dir=args.init_model_dir,
        device=device,
        local_files_only=bool(args.local_files_only),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_target_modules(args.lora_target_modules),
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    trainable_params, total_params = trainable_param_count(policy)

    optimizer = AdamW(
        [param for param in policy.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_optimizer_steps = max(
        1,
        int(args.total_updates)
        * int(args.ppo_epochs)
        * math.ceil((args.rollouts_per_prompt * args.prompt_batch_size) / max(1, args.train_micro_batch_size)),
    )
    warmup_steps = int(round(float(args.warmup_ratio) * float(total_optimizer_steps)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    save_json(
        output_dir / "train_args.json",
        {
            **vars(args),
            "resolved_output_dir": str(output_dir),
            "policy_device": str(device),
            "classifier_device": str(classifier_device),
            "classifier_label_order": label_order,
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
        },
    )
    log(
        f"[setup] output_dir={output_dir} "
        f"trainable_params={trainable_params} total_params={total_params}"
    )

    rng = random.Random(args.seed)
    history: List[Dict[str, Any]] = []
    best_id_metric = float("inf")
    best_id_summary: Dict[str, Any] | None = None
    best_update = None
    update_start_time = time.time()

    for update_idx in range(1, args.total_updates + 1):
        log(f"[update {update_idx}] sampling {args.prompt_batch_size} problems")
        sampled_problem_payloads = rng.sample(target_problems, k=args.prompt_batch_size)
        group_infos: List[Dict[str, Any]] = []
        all_rollouts: List[Any] = []

        policy.eval()
        for item in sampled_problem_payloads:
            prob = str(item["prob"])
            prompt_text = str(item["prompt_text"])
            target_dist = named_dict_to_distribution(item["target_distribution"], label_order)
            log(f"[update {update_idx}] generating rollouts for {prob}")
            rollouts = sample_rollouts_for_problem(
                model_for_generation=policy.pretrained_model,
                tokenizer=tokenizer,
                prob=prob,
                prompt_text=prompt_text,
                num_rollouts=args.rollouts_per_prompt,
                rollout_batch_size=args.rollout_batch_size,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=device,
            )
            log(f"[update {update_idx}] classifier scoring for {prob}")
            responses = [rollout.response_text for rollout in rollouts]
            rollout_validity = [
                analyze_generated_response(
                    response_text=response,
                    min_reasoning_chars=args.min_reasoning_chars,
                    empty_response_penalty=args.empty_response_penalty,
                    short_reasoning_penalty=args.short_reasoning_penalty,
                    unparseable_answer_penalty=args.unparseable_answer_penalty,
                )
                for response in responses
            ]
            posterior = score_prob_response_pairs(
                probs=[prob] * len(rollouts),
                responses=responses,
                tokenizer=classifier_tokenizer,
                model=classifier_model,
                device=classifier_device,
                batch_size=args.classifier_batch_size,
                max_length=args.classifier_max_length,
                amp_dtype=classifier_amp_dtype,
                text_mode="prob_response",
                drop_answer_line=True,
            )
            rollout_dist, rollout_strategy_summary = compute_strategy_rollout_distribution(
                posterior=posterior,
                rollout_records=rollout_validity,
                label_order=label_order,
            )
            divergence = compute_all_divergences(target_dist, rollout_dist, label_order)
            validity_summary = summarize_validity_records(rollout_validity)
            mean_penalty = float(validity_summary["mean_penalty"])
            raw_score = -float(divergence[args.loss]) - mean_penalty
            group_infos.append(
                {
                    "prob": prob,
                    "prompt_text": prompt_text,
                    "rollouts": rollouts,
                    "target_distribution": target_dist,
                    "rollout_distribution": rollout_dist,
                    "divergence": divergence,
                    "raw_score": raw_score,
                    "validity_summary": validity_summary,
                    "mean_penalty": mean_penalty,
                    "used_uniform_fallback": bool(rollout_strategy_summary["used_uniform_fallback"]),
                }
            )
            all_rollouts.extend(rollouts)

        log(f"[update {update_idx}] computing old policy/ref stats")
        batch_score_mean = float(np.mean([info["raw_score"] for info in group_infos]))
        terminal_scores: List[float] = []
        rollout_prob_list: List[str] = []
        rollout_raw_scores: List[float] = []
        rollout_group_divergence: List[float] = []
        for info in group_infos:
            centered_score = float(info["raw_score"] - batch_score_mean)
            info["centered_score"] = centered_score
            info["selected_loss_distance"] = float(info["divergence"][args.loss])
            terminal_scores.extend([centered_score] * len(info["rollouts"]))
            rollout_prob_list.extend([info["prob"]] * len(info["rollouts"]))
            rollout_raw_scores.extend([float(info["raw_score"])] * len(info["rollouts"]))
            rollout_group_divergence.extend([float(info["selected_loss_distance"])] * len(info["rollouts"]))

        old_logprobs, old_values, ref_logprobs, valid_mask = compute_rollout_stats(
            rollouts=all_rollouts,
            policy=policy,
            ref_model=ref_model,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
            amp_dtype=amp_dtype,
            forward_batch_size=args.policy_forward_batch_size,
        )
        log(f"[update {update_idx}] building rewards and running PPO")
        terminal_scores_tensor = torch.tensor(terminal_scores, dtype=torch.float32, device=device)
        rewards, advantages, returns, token_kl = build_rewards_and_advantages(
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            values=old_values,
            valid_mask=valid_mask,
            terminal_scores=terminal_scores_tensor,
            gamma=args.gamma,
            lam=args.lam,
            kl_coef=args.kl_coef,
            whiten_advantages=bool(args.whiten_advantages),
        )

        max_resp_len = int(valid_mask.shape[1])
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        approxkl_values: List[float] = []
        pg_loss_values: List[float] = []
        vf_loss_values: List[float] = []
        ratio_values: List[float] = []
        clipfrac_values: List[float] = []

        batch_indices = np.arange(len(all_rollouts))
        for _ppo_epoch in range(args.ppo_epochs):
            rng.shuffle(batch_indices)
            for mini_start in range(0, len(batch_indices), args.mini_batch_size):
                mini_inds = batch_indices[mini_start : mini_start + args.mini_batch_size]
                for micro_start in range(0, len(mini_inds), args.train_micro_batch_size):
                    micro_inds = mini_inds[micro_start : micro_start + args.train_micro_batch_size]
                    micro_rollouts = [all_rollouts[int(idx)] for idx in micro_inds]
                    micro_batch = rollout_batch_to_tensors(
                        micro_rollouts,
                        pad_token_id=tokenizer.pad_token_id,
                        device=device,
                    )
                    mb_old_logprobs = old_logprobs[micro_inds]
                    mb_old_values = old_values[micro_inds]
                    mb_advantages = advantages[micro_inds]
                    mb_returns = returns[micro_inds]
                    mb_mask = valid_mask[micro_inds]

                    with autocast_context(device, amp_dtype):
                        new_logits, _, new_values_raw = policy(
                            input_ids=micro_batch["input_ids"],
                            attention_mask=micro_batch["attention_mask"],
                        )
                    new_logprobs, new_values, _ = gather_response_logprobs_and_values(
                        logits=new_logits,
                        values=new_values_raw,
                        input_ids=micro_batch["input_ids"],
                        query_lens=micro_batch["query_lens"],
                        response_lens=micro_batch["response_lens"],
                        max_resp_len=max_resp_len,
                    )
                    new_logprobs = torch.where(mb_mask, new_logprobs, torch.zeros_like(new_logprobs))
                    new_values = torch.where(mb_mask, new_values, torch.zeros_like(new_values))

                    logprob_diff = new_logprobs - mb_old_logprobs
                    ratio = torch.exp(logprob_diff)
                    pg_losses1 = -mb_advantages * ratio
                    pg_losses2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                    pg_loss = masked_mean(torch.max(pg_losses1, pg_losses2), mb_mask)

                    vpred_clipped = torch.clamp(
                        new_values,
                        mb_old_values - args.cliprange_value,
                        mb_old_values + args.cliprange_value,
                    )
                    vf_losses1 = torch.square(new_values - mb_returns)
                    vf_losses2 = torch.square(vpred_clipped - mb_returns)
                    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mb_mask)
                    loss = pg_loss + float(args.vf_coef) * vf_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        clipfrac = masked_mean(
                            (torch.abs(ratio - 1.0) > args.cliprange).float(),
                            mb_mask,
                        )
                        approxkl = 0.5 * masked_mean(torch.square(logprob_diff), mb_mask)
                        approxkl_values.append(float(approxkl.item()))
                        pg_loss_values.append(float(pg_loss.item()))
                        vf_loss_values.append(float(vf_loss.item()))
                        ratio_values.append(float(masked_mean(ratio, mb_mask).item()))
                        clipfrac_values.append(float(clipfrac.item()))

        update_record = {
            "update": int(update_idx),
            "sampled_probs": [info["prob"] for info in group_infos],
            "elapsed_sec": float(time.time() - update_start_time),
            "mean_batch_raw_reward": float(np.mean([info["raw_score"] for info in group_infos])),
            "mean_batch_centered_reward": float(np.mean([info["centered_score"] for info in group_infos])),
            "mean_batch_tv_distance": float(np.mean([info["divergence"]["tv"] for info in group_infos])),
            "mean_batch_kl_distance": float(np.mean([info["divergence"]["kl"] for info in group_infos])),
            "mean_batch_wasserstein_distance": float(np.mean([info["divergence"]["wasserstein"] for info in group_infos])),
            "selected_loss": args.loss,
            "mean_batch_selected_distance": float(np.mean([info["selected_loss_distance"] for info in group_infos])),
            "mean_batch_valid_fraction": float(np.mean([info["validity_summary"]["valid_rollout_fraction"] for info in group_infos])),
            "mean_batch_empty_fraction": float(np.mean([info["validity_summary"]["empty_response_fraction"] for info in group_infos])),
            "mean_batch_short_reasoning_fraction": float(np.mean([info["validity_summary"]["short_reasoning_fraction"] for info in group_infos])),
            "mean_batch_unparseable_answer_fraction": float(np.mean([info["validity_summary"]["unparseable_answer_fraction"] for info in group_infos])),
            "mean_batch_penalty": float(np.mean([info["mean_penalty"] for info in group_infos])),
            "mean_token_ref_kl": float(masked_mean(token_kl, valid_mask).item()),
            "mean_terminal_score": float(terminal_scores_tensor.mean().item()),
            "ppo_approxkl": float(np.mean(approxkl_values)) if approxkl_values else 0.0,
            "ppo_pg_loss": float(np.mean(pg_loss_values)) if pg_loss_values else 0.0,
            "ppo_vf_loss": float(np.mean(vf_loss_values)) if vf_loss_values else 0.0,
            "ppo_ratio": float(np.mean(ratio_values)) if ratio_values else 1.0,
            "ppo_clipfrac": float(np.mean(clipfrac_values)) if clipfrac_values else 0.0,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }

        if args.id_eval_every > 0 and (update_idx % args.id_eval_every == 0 or update_idx == args.total_updates):
            policy.eval()
            log(f"[update {update_idx}] running ID eval")
            id_summary, id_problem_df, _ = evaluate_id_split(
                model_for_generation=policy.pretrained_model,
                tokenizer=tokenizer,
                human_target_payload=human_target_payload,
                label_order=label_order,
                classifier_tokenizer=classifier_tokenizer,
                classifier_model=classifier_model,
                generation_device=device,
                classifier_device=classifier_device,
                rollouts_per_problem=args.id_eval_rollouts_per_problem,
                rollout_batch_size=args.rollout_batch_size,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_reasoning_chars=args.min_reasoning_chars,
                selection_loss=args.loss,
                best_id_invalid_coef=args.best_id_invalid_coef,
                classifier_batch_size=args.classifier_batch_size,
                classifier_max_length=args.classifier_max_length,
                classifier_amp_dtype=classifier_amp_dtype,
            )
            update_record.update(
                {
                    "id_eval_mean_tv_distance": float(id_summary["mean_tv_distance"]),
                    "id_eval_mean_kl_distance": float(id_summary["mean_kl_distance"]),
                    "id_eval_mean_wasserstein_distance": float(id_summary["mean_wasserstein_distance"]),
                    "id_eval_accuracy_all": float(id_summary["accuracy_all"]),
                    "id_eval_valid_rollout_fraction": float(id_summary["valid_rollout_fraction"]),
                    "id_eval_empty_response_fraction": float(id_summary["empty_response_fraction"]),
                    "id_eval_short_reasoning_fraction": float(id_summary["short_reasoning_fraction"]),
                    "id_eval_unparseable_answer_fraction": float(id_summary["unparseable_answer_fraction"]),
                    "id_eval_selection_metric": float(id_summary["selection_metric"]),
                }
            )
            current_id_metric = float(id_summary["selection_metric"])
            if current_id_metric < best_id_metric:
                best_id_metric = current_id_metric
                best_id_summary = id_summary
                best_update = int(update_idx)
                save_policy_checkpoint(
                    checkpoint_dir=output_dir / "best_id",
                    policy=policy,
                    tokenizer=tokenizer,
                    metadata={
                        "best_update": int(update_idx),
                        "selection_loss": args.loss,
                        "selection_metric": float(current_id_metric),
                        "selected_distance": float(id_summary["selected_distance"]),
                        "id_eval_summary": id_summary,
                    },
                )
                id_problem_df.to_csv(output_dir / "best_id_per_problem.csv", index=False)
                save_json(output_dir / "best_id_eval_summary.json", id_summary)

        history.append(update_record)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        save_json(output_dir / "history.json", {"history": history})

        if args.save_every > 0 and update_idx % args.save_every == 0:
            save_policy_checkpoint(
                checkpoint_dir=output_dir / f"update_{update_idx}",
                policy=policy,
                tokenizer=tokenizer,
                metadata={"update": int(update_idx), "history_record": update_record},
            )

        print(
            f"[update {update_idx}/{args.total_updates}] "
            f"loss={args.loss} batch_div={update_record['mean_batch_selected_distance']:.4f} "
            f"reward={update_record['mean_batch_raw_reward']:.4f} "
            f"valid={update_record['mean_batch_valid_fraction']:.4f} "
            f"ref_kl={update_record['mean_token_ref_kl']:.4f} "
            f"ppo_approxkl={update_record['ppo_approxkl']:.4f}"
        )

    save_policy_checkpoint(
        checkpoint_dir=output_dir / "final",
        policy=policy,
        tokenizer=tokenizer,
        metadata={
            "total_updates": int(args.total_updates),
            "best_update": best_update,
            "best_id_metric": None if not math.isfinite(best_id_metric) else float(best_id_metric),
        },
    )

    policy.eval()
    log("[final] running ID eval")
    final_id_summary, final_id_problem_df, final_id_rollout_df = evaluate_id_split(
        model_for_generation=policy.pretrained_model,
        tokenizer=tokenizer,
        human_target_payload=human_target_payload,
        label_order=label_order,
        classifier_tokenizer=classifier_tokenizer,
        classifier_model=classifier_model,
        generation_device=device,
        classifier_device=classifier_device,
        rollouts_per_problem=args.id_eval_rollouts_per_problem,
        rollout_batch_size=args.rollout_batch_size,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_reasoning_chars=args.min_reasoning_chars,
        selection_loss=args.loss,
        best_id_invalid_coef=args.best_id_invalid_coef,
        classifier_batch_size=args.classifier_batch_size,
        classifier_max_length=args.classifier_max_length,
        classifier_amp_dtype=classifier_amp_dtype,
    )
    log("[final] running OOD eval")
    final_ood_summary, final_ood_problem_df, final_ood_rollout_df = evaluate_ood_split(
        model_for_generation=policy.pretrained_model,
        tokenizer=tokenizer,
        sp2013_human_csv=args.sp2013_human_csv,
        generation_device=device,
        rollouts_per_problem=args.ood_eval_rollouts_per_problem,
        rollout_batch_size=args.rollout_batch_size,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    final_eval_dir = output_dir / "final_eval"
    final_eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        final_eval_dir / "eval_summary.json",
        {
            "id_metrics": final_id_summary,
            "ood_metrics": final_ood_summary,
            "best_id_summary": best_id_summary,
        },
    )
    final_id_problem_df.to_csv(final_eval_dir / "id_per_problem.csv", index=False)
    final_id_rollout_df.to_csv(final_eval_dir / "id_rollouts.csv", index=False)
    final_ood_problem_df.to_csv(final_eval_dir / "ood_per_problem.csv", index=False)
    final_ood_rollout_df.to_csv(final_eval_dir / "ood_rollouts.csv", index=False)

    save_json(
        output_dir / "run_summary.json",
        {
            "init_model_dir": os.path.abspath(args.init_model_dir),
            "classifier_ckpt": os.path.abspath(args.classifier_ckpt),
            "human_target_json": os.path.abspath(args.human_target_json),
            "loss": args.loss,
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "best_update": best_update,
            "best_id_metric": None if not math.isfinite(best_id_metric) else float(best_id_metric),
            "best_id_summary": best_id_summary,
            "final_id_summary": final_id_summary,
            "final_ood_summary": final_ood_summary,
        },
    )
    print(f"Saved PPO run to {output_dir}")
    print(
        f"Final ID mean {args.loss}: "
        f"{final_id_summary['mean_' + ('wasserstein' if args.loss == 'wasserstein' else args.loss) + '_distance']:.4f}"
    )
    print(f"Final OOD accuracy: {final_ood_summary['accuracy_all']:.4f}")


if __name__ == "__main__":
    main()
