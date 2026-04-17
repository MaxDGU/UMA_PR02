#!/usr/bin/env python3
"""
Standalone SP2013 rollout evaluator for a fixed student tuple.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from train_transformer_hf import (
    evaluate_sp2013_final_answer,
    load_eval_model,
    load_saved_train_args_if_exists,
    pick_saved_checkpoint_for_reload,
    resolve_model_source,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    tr_dir = root / "results" / "transformer_replication"
    uma_dir = root / "results" / "UMA_replication"
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on SP2013 with a fixed student tuple.")
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Training output directory containing best/final checkpoints. Mutually exclusive with --checkpoint_dir.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Explicit checkpoint directory to load. If omitted, --run_dir is used to resolve the best checkpoint.",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="",
        help="Base model name or path. Defaults to the saved train_args model_name or SmolLM2-135M.",
    )
    parser.add_argument("--g", type=float, required=True)
    parser.add_argument("--d", type=float, required=True)
    parser.add_argument("--rt_mu", type=float, required=True)
    parser.add_argument("--ice", type=float, required=True)
    parser.add_argument("--sp2013_csv", type=Path, default=uma_dir / "sp2013.csv")
    parser.add_argument(
        "--target_csv",
        type=Path,
        required=True,
        help="CSV with same-tuple UMA outputs used as the distribution target.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--num_rollouts", type=int, default=1000)
    parser.add_argument("--rollout_batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--sample_seed", type=int, default=123)
    parser.add_argument("--progress_every", type=int, default=1000)
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resolve model/tokenizer from local cache only.",
    )
    return parser.parse_args()


def resolve_checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir is not None and args.run_dir is not None:
        raise ValueError("Use only one of --checkpoint_dir or --run_dir.")
    if args.checkpoint_dir is not None:
        ckpt = args.checkpoint_dir.resolve()
    elif args.run_dir is not None:
        ckpt = Path(pick_saved_checkpoint_for_reload(str(args.run_dir.resolve())))
    else:
        raise ValueError("One of --checkpoint_dir or --run_dir is required.")
    if not ckpt or not ckpt.is_dir():
        raise FileNotFoundError(f"Could not resolve checkpoint directory: {ckpt}")
    return ckpt


def main() -> None:
    args = parse_args()
    checkpoint_dir = resolve_checkpoint_dir(args)
    train_args = load_saved_train_args_if_exists(str(checkpoint_dir))
    base_model_name = args.base_model_name or str(train_args.get("model_name", "HuggingFaceTB/SmolLM2-135M"))
    model_source = resolve_model_source(base_model_name, local_files_only=args.local_files_only)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_eval_model(
        checkpoint_dir=str(checkpoint_dir),
        base_model_name_or_path=str(model_source),
        device=device,
        local_files_only=args.local_files_only,
    )
    model.eval()

    metrics, out_df = evaluate_sp2013_final_answer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sp2013_csv=str(args.sp2013_csv),
        target_csv=str(args.target_csv),
        max_new_tokens=args.max_new_tokens,
        target_mode="uma_distribution",
        use_param_grid=False,
        num_rollouts=args.num_rollouts,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        sample_seed=args.sample_seed,
        rollout_batch_size=args.rollout_batch_size,
        progress_every=args.progress_every,
        progress_label="sp2013_fixed_tuple_eval",
        fixed_student_prompt_tuple=(args.g, args.d, args.rt_mu, args.ice),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "sp2013_fixed_tuple_rollouts.csv.gz"
    out_json = args.output_dir / "sp2013_fixed_tuple_metrics.json"
    out_df.to_csv(out_csv, index=False, compression="gzip")

    payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "base_model_name": base_model_name,
        "tuple": {
            "g": float(args.g),
            "d": float(args.d),
            "rt_mu": float(args.rt_mu),
            "ice": float(args.ice),
        },
        "metrics": metrics,
        "n_rows": int(len(out_df)),
        "n_problems": int(out_df["prob"].nunique()) if len(out_df) > 0 else 0,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
