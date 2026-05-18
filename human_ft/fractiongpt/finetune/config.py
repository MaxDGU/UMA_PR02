"""Configuration for GRPO fine-tuning."""

import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_CONFIG = {
    # Model
    "checkpoint": os.path.join(PROJECT_DIR, "phase3_output", "best_model_exhaustive_heldout.pt"),
    "metadata_checkpoint": os.path.join(PROJECT_DIR, "phase3_output", "phase3_model_exhaustive_heldout.pt"),

    # Task switch: "fractions" (Siegler 2011 + SP2013) or "decimals" (BSS2021)
    "task": "fractions",

    # Fraction human data
    "siegler2011_path": os.path.join(PROJECT_DIR, "siegler_data", "data and strategies - ALL - revised.csv"),
    "sp2013_path": os.path.join(PROJECT_DIR, "data", "siegler_fraction_human.csv"),

    # Decimal human data (BSS2021 / DAX6)
    "bss2021_path": os.path.join(PROJECT_DIR, "data", "bss2021_human_responses.csv"),
    "held_out_problems": [],

    # UMA trace files used for SFT reweighting (per task)
    "uma_traces_path": os.path.join(PROJECT_DIR, "data", "uma_traces_human_problems_all_students.csv"),
    "uma_traces_decimal_path": os.path.join(PROJECT_DIR, "data", "uma_traces_decimal_human_problems.csv"),

    # GRPO
    "group_size": 8,        # G rollouts per problem
    "clip_epsilon": 0.2,    # PPO clip range
    "kl_beta": 0.1,         # KL penalty coefficient
    "temperature": 1.0,     # Generation temperature during rollouts
    "max_new_tokens": 60,   # Max tokens to generate per rollout

    # Training
    "lr": 1e-6,
    "epochs": 50,
    "grad_clip": 1.0,
    "seed": 42,
    "eval_every": 1,
    "patience": 10,

    # Student configs (coarse bins only)
    "student_configs": None,  # Built at runtime

    # Output
    "output_dir": os.path.join(PROJECT_DIR, "finetune", "output"),
    "tag": "grpo_v1",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str)
    p.add_argument("--group_size", type=int)
    p.add_argument("--clip_epsilon", type=float)
    p.add_argument("--kl_beta", type=float)
    p.add_argument("--temperature", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--grad_clip", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--eval_every", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--max_new_tokens", type=int)
    p.add_argument("--tag", type=str)
    p.add_argument("--eval_temperature", type=float, default=1.3,
                   help="Temperature for evaluation (baseline=1.3 for Siegler)")
    args = p.parse_args()
    cfg = dict(DEFAULT_CONFIG)
    for k, v in vars(args).items():
        if v is not None and k in cfg:
            cfg[k] = v
    cfg["eval_temperature"] = args.eval_temperature
    return cfg
