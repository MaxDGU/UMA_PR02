#!/usr/bin/env python3
"""
Build human strategy target distributions from train+val human NLP data.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from strategy_matching_utils import (
    dedupe_sp2013_human_problems,
    distribution_to_named_dict,
    load_strategy_classifier,
    normalize_to_nlp_frame,
    read_csv_flexible,
    resolve_amp_dtype,
    resolve_device,
    save_json,
    score_texts_with_classifier,
)


DEFAULT_CLASSIFIER_CKPT = os.path.join(
    "results",
    "transformer_replication",
    "strategy_classifier_distilbert_n160000_e3_final",
    "best",
)
DEFAULT_TRAIN_CSV = os.path.join("results", "human", "data_train_nlp.csv")
DEFAULT_VAL_CSV = os.path.join("results", "human", "data_val_nlp.csv")
DEFAULT_SP2013_HUMAN_CSV = "sp2013_human.csv"
DEFAULT_OUTPUT_JSON = os.path.join(
    "results",
    "human",
    "human_strategy_targets_classifier160k.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build posterior-averaged human strategy targets.")
    parser.add_argument("--classifier_ckpt", type=str, default=DEFAULT_CLASSIFIER_CKPT)
    parser.add_argument("--train_csv", type=str, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val_csv", type=str, default=DEFAULT_VAL_CSV)
    parser.add_argument("--sp2013_human_csv", type=str, default=DEFAULT_SP2013_HUMAN_CSV)
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--prompt_col", type=str, default="instruction_nl")
    parser.add_argument("--response_col", type=str, default="response_nl")
    parser.add_argument("--text_mode", choices=["response", "prob_response", "instruction_response"], default="prob_response")
    parser.add_argument(
        "--drop_answer_line",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove answer/correctness lines before classifier scoring.",
    )
    parser.add_argument("--classifier_batch_size", type=int, default=64)
    parser.add_argument("--classifier_max_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--amp_dtype",
        choices=["auto", "bf16", "fp16", "none"],
        default="auto",
        help="Mixed precision mode for classifier scoring on CUDA.",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load classifier/tokenizer from local cache only.",
    )
    return parser.parse_args()


def build_problem_payloads(
    frame: pd.DataFrame,
    posterior: np.ndarray,
    label_order: List[str],
    prompt_col: str,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    posterior_df = pd.DataFrame(posterior, columns=label_order)
    work = pd.concat([frame.reset_index(drop=True), posterior_df], axis=1)

    for prob, group in work.groupby("prob", sort=True):
        prompts = sorted(set(str(value).strip() for value in group[prompt_col].tolist() if str(value).strip()))
        if len(prompts) != 1:
            raise ValueError(f"Expected one prompt per problem, got {len(prompts)} for prob={prob}: {prompts}")
        dist = group[label_order].mean(axis=0).to_numpy(dtype=np.float64)
        argmax_counts = (
            group[label_order]
            .idxmax(axis=1)
            .value_counts()
            .reindex(label_order, fill_value=0)
            .astype(int)
            .to_dict()
        )
        native_counts = group["strategy"].fillna("").astype(str).value_counts().astype(int).to_dict()
        payloads.append(
            {
                "prob": prob,
                "prompt_text": prompts[0],
                "n_rows": int(len(group)),
                "subject_count": int(group["subjid"].nunique()) if "subjid" in group.columns else None,
                "target_distribution": distribution_to_named_dict(dist, label_order),
                "target_distribution_vector": [float(x) for x in dist.tolist()],
                "argmax_classifier_counts": {label: int(argmax_counts.get(label, 0)) for label in label_order},
                "native_strategy_counts": native_counts,
            }
        )
    return payloads


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)

    train_df = normalize_to_nlp_frame(read_csv_flexible(args.train_csv), prompt_col=args.prompt_col, response_col=args.response_col)
    val_df = normalize_to_nlp_frame(read_csv_flexible(args.val_csv), prompt_col=args.prompt_col, response_col=args.response_col)
    human_df = pd.concat([train_df, val_df], ignore_index=True)
    human_df["prob"] = human_df["prob"].astype(str).str.strip()
    human_df[args.prompt_col] = human_df[args.prompt_col].astype(str).str.strip()
    human_df[args.response_col] = human_df[args.response_col].astype(str).str.strip()

    problem_counts = human_df["prob"].value_counts().sort_index()
    if int(problem_counts.shape[0]) != 8:
        raise ValueError(f"Expected exactly 8 unique human-train problems, found {int(problem_counts.shape[0])}.")
    if not (problem_counts == 48).all():
        raise ValueError(f"Expected 48 rows per problem, found counts={problem_counts.to_dict()}")

    tokenizer, classifier, label_order = load_strategy_classifier(
        checkpoint_dir=args.classifier_ckpt,
        device=device,
        local_files_only=bool(args.local_files_only),
    )
    texts = [
        (
            f"Problem: {prob}=?\nResponse:\n"
            + (
                "\n".join(
                    line
                    for line in str(response).splitlines()
                    if not line.strip().lower().startswith("### answer:")
                    and not line.strip().lower().startswith("### correctness:")
                ).strip()
                if args.drop_answer_line
                else str(response).strip()
            )
        ).strip()
        if args.text_mode == "prob_response"
        else None
        for prob, response in zip(human_df["prob"].tolist(), human_df[args.response_col].tolist())
    ]
    if args.text_mode != "prob_response":
        from strategy_matching_utils import build_classifier_text

        texts = [
            build_classifier_text(
                prob=prob,
                response_nl=response,
                instruction_nl=prompt,
                text_mode=args.text_mode,
                drop_answer_line=bool(args.drop_answer_line),
            )
            for prob, prompt, response in zip(
                human_df["prob"].tolist(),
                human_df[args.prompt_col].tolist(),
                human_df[args.response_col].tolist(),
            )
        ]

    posterior = score_texts_with_classifier(
        texts=texts,
        tokenizer=tokenizer,
        model=classifier,
        device=device,
        batch_size=args.classifier_batch_size,
        max_length=args.classifier_max_length,
        amp_dtype=amp_dtype,
    )
    if posterior.shape[0] != len(human_df):
        raise RuntimeError(f"Posterior row mismatch: posterior={posterior.shape}, rows={len(human_df)}")

    problem_payloads = build_problem_payloads(
        frame=human_df,
        posterior=posterior,
        label_order=label_order,
        prompt_col=args.prompt_col,
    )
    problem_to_target = {
        item["prob"]: {
            "prompt_text": item["prompt_text"],
            "n_rows": item["n_rows"],
            "subject_count": item["subject_count"],
            "target_distribution": item["target_distribution"],
        }
        for item in problem_payloads
    }

    sp2013_df = dedupe_sp2013_human_problems(read_csv_flexible(args.sp2013_human_csv))
    train_probs = sorted(problem_to_target.keys())
    ood_probs = sorted(str(prob).strip() for prob in sp2013_df["prob"].tolist())
    shared_probs = sorted(set(train_probs) & set(ood_probs))

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "classifier_checkpoint": os.path.abspath(args.classifier_ckpt),
        "classifier_label_order": list(label_order),
        "classifier_text_mode": args.text_mode,
        "classifier_drop_answer_line": bool(args.drop_answer_line),
        "classifier_batch_size": int(args.classifier_batch_size),
        "classifier_max_length": int(args.classifier_max_length),
        "human_source_files": {
            "train_csv": os.path.abspath(args.train_csv),
            "val_csv": os.path.abspath(args.val_csv),
        },
        "source_counts": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "combined_rows": int(len(human_df)),
            "unique_train_val_problems": int(problem_counts.shape[0]),
            "rows_per_problem": {str(k): int(v) for k, v in problem_counts.to_dict().items()},
        },
        "problems": problem_payloads,
        "problem_to_target": problem_to_target,
        "train_problem_list": train_probs,
        "sp2013_overlap_report": {
            "sp2013_human_csv": os.path.abspath(args.sp2013_human_csv),
            "sp2013_unique_problem_count": int(len(ood_probs)),
            "sp2013_unique_problem_list": ood_probs,
            "shared_problem_count": int(len(shared_probs)),
            "shared_problem_list": shared_probs,
            "id_problem_count": int(len(train_probs)),
            "id_problem_list": train_probs,
        },
    }
    save_json(args.output_json, payload)
    print(f"Saved human strategy targets to {os.path.abspath(args.output_json)}")
    print(f"Human-train problems: {len(train_probs)} | rows per problem: {sorted(set(problem_counts.tolist()))}")
    print(f"SP2013 unique problems: {len(ood_probs)} | overlap: {len(shared_probs)}")


if __name__ == "__main__":
    main()
