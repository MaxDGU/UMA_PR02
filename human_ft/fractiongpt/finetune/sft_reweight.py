#!/usr/bin/env python3
"""
Importance-weighted SFT: fine-tune FractionGPT by reweighting UMA traces
so the answer distribution matches the human distribution.

For each of the 24 human-eval problems:
  w(trace) = p_human(answer) / p_uma(answer)

Traces whose answers humans give more often than UMA get upweighted.
Traces whose answers humans give less often get downweighted.
Answers in human data but not in UMA get a smoothed floor.
"""

import os
import sys
import json
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from train_transformer_phase3 import (
    FractionTokenizer, FractionGPT, SyntheticTracesDataset, make_collate_fn
)
from finetune.config import DEFAULT_CONFIG
from finetune.data import normalize_answer, load_human_data
from finetune.evaluate import evaluate_model


def build_importance_weights(uma_df, human_problems, raw_data, smooth_alpha=0.05):
    """Compute per-trace importance weights for the 24 human-eval problems.

    Args:
        uma_df: DataFrame of UMA traces (must have 'prob', 'answer' columns)
        human_problems: dict of problem -> {N, source}
        raw_data: dict of problem -> {answers: {answer: freq}}
        smooth_alpha: fraction of probability mass to redistribute uniformly
                      over all answers (Laplace-style smoothing)

    Returns:
        weights: array of per-trace weights, same length as uma_df
        stats: dict with coverage and weight diagnostics
    """
    weights = np.ones(len(uma_df), dtype=np.float64)
    stats = {"problems": {}}

    for prob in human_problems:
        human_dist = raw_data[prob]["answers"]
        mask = uma_df["prob"] == prob
        prob_df = uma_df[mask]

        if len(prob_df) == 0:
            continue

        # Count UMA answer frequencies for this problem
        uma_counts = Counter()
        for idx, row in prob_df.iterrows():
            norm = normalize_answer(str(row["answer"]).strip())
            uma_counts[norm] = uma_counts.get(norm, 0) + 1

        uma_total = sum(uma_counts.values())
        uma_freq = {a: c / uma_total for a, c in uma_counts.items()}

        # All answers in either distribution
        all_answers = set(human_dist.keys()) | set(uma_freq.keys())

        # Smooth both distributions
        n_answers = len(all_answers)
        uniform = 1.0 / n_answers if n_answers > 0 else 0.0

        smoothed_human = {}
        smoothed_uma = {}
        for a in all_answers:
            smoothed_human[a] = (1 - smooth_alpha) * human_dist.get(a, 0.0) + smooth_alpha * uniform
            smoothed_uma[a] = (1 - smooth_alpha) * uma_freq.get(a, 0.0) + smooth_alpha * uniform

        # Compute importance weight for each answer
        answer_weights = {}
        for a in all_answers:
            answer_weights[a] = smoothed_human[a] / smoothed_uma[a]

        # Assign weights to traces
        covered_mass = sum(human_dist.get(a, 0) for a in uma_freq if a in human_dist)
        prob_weights = []
        for idx, row in prob_df.iterrows():
            norm = normalize_answer(str(row["answer"]).strip())
            w = answer_weights.get(norm, 1.0)
            weights[uma_df.index.get_loc(idx)] = w
            prob_weights.append(w)

        stats["problems"][prob] = {
            "n_traces": len(prob_df),
            "n_uma_answers": len(uma_freq),
            "n_human_answers": len(human_dist),
            "coverage": covered_mass,
            "weight_mean": np.mean(prob_weights),
            "weight_std": np.std(prob_weights),
            "weight_min": np.min(prob_weights),
            "weight_max": np.max(prob_weights),
        }

    stats["overall_coverage"] = np.mean([s["coverage"] for s in stats["problems"].values()])
    stats["mean_weight"] = weights[weights != 1.0].mean() if (weights != 1.0).any() else 1.0

    return weights, stats


class WeightedTracesDataset(Dataset):
    """SyntheticTracesDataset with per-sample importance weights."""

    def __init__(self, base_dataset, weights):
        """
        Args:
            base_dataset: SyntheticTracesDataset instance
            weights: array of per-sample weights (before filtering by base_dataset)
        """
        self.base = base_dataset
        # base_dataset.samples is a list of (full_ids, context_len)
        # We need to map these back to the original dataframe indices
        # Since SyntheticTracesDataset filters out some rows, we store weights
        # during construction
        self.weights = weights

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        input_ids, target_ids, context_len = self.base[idx]
        w = self.weights[idx]
        return input_ids, target_ids, context_len, torch.tensor(w, dtype=torch.float32)


def build_weighted_dataset(uma_path, human_problems, raw_data, tokenizer,
                            max_len=100, smooth_alpha=0.05):
    """Build a weighted dataset from UMA traces for the 24 human problems.

    Returns:
        dataset: WeightedTracesDataset
        stats: importance weight diagnostics
    """
    print("Loading UMA traces...")
    uma_full = pd.read_csv(uma_path, dtype={"answer": str}, low_memory=False)

    # Filter to just the 24 human problems
    prob_list = list(human_problems.keys())
    uma_subset = uma_full[uma_full["prob"].isin(prob_list)].reset_index(drop=True)
    print(f"  Filtered to {len(uma_subset):,} traces on {len(prob_list)} problems")

    # Compute importance weights
    print("Computing importance weights...")
    raw_weights, stats = build_importance_weights(uma_subset, human_problems, raw_data, smooth_alpha)

    # Build base dataset (tokenizes all traces)
    print("Tokenizing traces...")
    base_dataset = SyntheticTracesDataset(uma_subset, tokenizer, max_len=max_len)
    print(f"  {len(base_dataset)} samples after filtering by max_len")

    # The base dataset may have dropped some rows (those exceeding max_len).
    # SyntheticTracesDataset iterates with iterrows and appends samples that fit.
    # We need to figure out which rows survived to assign correct weights.
    # Reconstruct by re-iterating the same way.
    surviving_weights = []
    i = 0
    for _, row in uma_subset.iterrows():
        resp = str(row.get('answer', row.get('resp', '?'))).strip()
        if resp == '?' or resp == 'nan' or not resp:
            i += 1
            continue
        # Reconstruct the full text to check length
        from train_transformer_phase3 import bin_g, bin_d, bin_rt_mu, bin_ice
        g_bin = bin_g(row['g'], fine=False)
        d_bin = bin_d(row['d'], fine=False)
        rt_bin = bin_rt_mu(row['rt_mu']) if pd.notna(row.get('rt_mu')) else 'rt_4'
        ice_bin = bin_ice(row['ice'])
        strategy = str(row['strategy'])
        goals_str = str(row['goals']) if pd.notna(row.get('goals')) and row['goals'] else ''
        exec_str = str(row['exec']) if pd.notna(row.get('exec')) and row['exec'] else ''
        if not exec_str or exec_str == 'nan':
            exec_str = 'no_exec'

        context_text = f"<student> {g_bin} {d_bin} {rt_bin} {ice_bin} </student> <problem> {row['prob']} </problem>"
        target_text = (f"<strategy> {strategy} </strategy> "
                       f"<goals> {goals_str} </goals> "
                       f"<exec> {exec_str} </exec> "
                       f"<answer> {resp} </answer>")
        full_text = context_text + " " + target_text
        full_ids = tokenizer.encode(full_text)

        if len(full_ids) <= max_len:
            surviving_weights.append(raw_weights[i])
        i += 1

    surviving_weights = np.array(surviving_weights)
    assert len(surviving_weights) == len(base_dataset), \
        f"Weight count {len(surviving_weights)} != dataset size {len(base_dataset)}"

    # Clamp max weight to prevent outlier answers from dominating
    max_weight = cfg.get("max_weight", 5.0)
    surviving_weights = np.clip(surviving_weights, 0, max_weight)

    # Normalize weights to mean=1.0 so overall loss scale matches standard CE
    w_mean = surviving_weights.mean()
    if w_mean > 0:
        surviving_weights = surviving_weights / w_mean

    dataset = WeightedTracesDataset(base_dataset, surviving_weights)
    print(f"  Weight stats (clamped to {max_weight}, normalized): mean={surviving_weights.mean():.3f}, "
          f"std={surviving_weights.std():.3f}, "
          f"min={surviving_weights.min():.3f}, max={surviving_weights.max():.3f}")

    return dataset, stats


def weighted_collate_fn(batch):
    """Collate with weights."""
    inputs, targets, context_lens, weights = zip(*batch)

    max_len = max(len(x) for x in inputs)
    B = len(inputs)

    padded_inputs = torch.zeros(B, max_len, dtype=torch.long)
    padded_targets = torch.zeros(B, max_len, dtype=torch.long)
    loss_mask = torch.zeros(B, max_len, dtype=torch.float)
    weight_tensor = torch.stack(weights)

    for i in range(B):
        seq_len = len(inputs[i])
        padded_inputs[i, :seq_len] = inputs[i]
        padded_targets[i, :seq_len] = targets[i]
        ctx_len = context_lens[i]
        loss_mask[i, ctx_len-1:seq_len] = 1.0

    return padded_inputs, padded_targets, loss_mask, weight_tensor


def train_sft(cfg):
    """Importance-weighted SFT training loop."""
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Load model
    tokenizer = FractionTokenizer()
    state_dict = torch.load(cfg["checkpoint"], map_location=device)
    # Strip torch.compile() prefix if the checkpoint was saved from a compiled model
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    vocab_size = state_dict["embedding.weight"].shape[0]
    # Infer architecture from the checkpoint so the loader supports both the
    # 1.3M baseline (d=128, L=5) and the 50M-class scale-up (d=640, L=8).
    hidden_size = state_dict["embedding.weight"].shape[1]
    layer_indices = [int(k.split(".")[2]) for k in state_dict
                     if k.startswith("decoder.layers.")]
    nlayers = (max(layer_indices) + 1) if layer_indices else cfg.get("n_layers", 5)
    nhead = cfg.get("n_heads", 8)
    print(f"Inferred architecture: d_model={hidden_size}, nlayers={nlayers}, nhead={nhead}")
    model = FractionGPT(
        vocab_size=vocab_size, hidden_size=hidden_size, nlayers=nlayers, nhead=nhead,
        dropout_p=0.1, pad_idx=tokenizer.pad_token_id, legacy_decoder=False,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Loaded model ({sum(p.numel() for p in model.parameters()):,} params)")

    # Cap tokenizer to model vocab size — the current tokenizer has Phase 4
    # tokens (1194) that the pretrained model (1174) doesn't know
    unk_id = tokenizer.unk_token_id
    for tok, tid in list(tokenizer.token_to_id.items()):
        if tid >= vocab_size:
            tokenizer.token_to_id[tok] = unk_id

    # Frozen reference for KL penalty
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    kl_beta = cfg.get("kl_beta", 0.5)
    print(f"KL penalty beta={kl_beta}")

    # Load human data (also sets the task on the data module so the right
    # answer normalizer is used everywhere downstream)
    reward_table, problems, raw_data = load_human_data(cfg)
    task = cfg.get("task", "fractions")

    # Hold-out problems are kept in `problems`/`raw_data` for evaluation but
    # excluded from the SFT reweighting signal.
    held_out = set(str(p).strip() for p in (cfg.get("held_out_problems") or []))
    train_problems = {p: info for p, info in problems.items() if p not in held_out}
    train_raw_data = {p: data for p, data in raw_data.items() if p not in held_out}
    if held_out:
        print(f"Held-out problems ({len(held_out)}): {sorted(held_out)}")
        print(f"Train problems ({len(train_problems)}): {sorted(train_problems.keys())}")

    # Pick the right UMA trace file for this task
    if task == "decimals":
        uma_path = cfg.get("uma_traces_decimal_path")
    else:
        uma_path = cfg.get("uma_traces_path",
                           os.path.join(PROJECT_DIR, "data", "uma_traces_human_problems_all_students.csv"))

    dataset, weight_stats = build_weighted_dataset(
        uma_path, train_problems, train_raw_data, tokenizer,
        max_len=100, smooth_alpha=cfg.get("smooth_alpha", 0.05)
    )

    # Print coverage report
    print(f"\nCoverage report (smooth_alpha={cfg.get('smooth_alpha', 0.05)}):")
    for prob in sorted(weight_stats["problems"].keys()):
        s = weight_stats["problems"][prob]
        print(f"  {prob:16s}: coverage={s['coverage']*100:.1f}%, "
              f"w=[{s['weight_min']:.2f}, {s['weight_max']:.2f}], "
              f"mean={s['weight_mean']:.2f}")
    print(f"  Overall coverage: {weight_stats['overall_coverage']*100:.1f}%")

    dataloader = DataLoader(
        dataset, batch_size=cfg.get("batch_size", 64),
        shuffle=True, collate_fn=weighted_collate_fn,
        num_workers=0, pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("sft_lr", 1e-5))

    # Eval setup
    g_bins = ["g_low", "g_mid", "g_high"]
    d_bins = ["d_low", "d_mid", "d_high"]
    eval_configs = [
        (g, d, rt, ice)
        for g in g_bins for d in d_bins
        for rt in ["rt_3", "rt_6"] for ice in ["ice_0", "ice_50", "ice_100"]
    ]

    best_mae = float("inf")
    patience_counter = 0
    history = []
    n_epochs = cfg.get("sft_epochs", 20)
    eval_every = cfg.get("eval_every", 5)

    print(f"\n{'='*60}")
    print(f"Starting importance-weighted SFT + KL (tag={cfg['tag']})")
    print(f"  {len(dataset)} samples, batch_size={cfg.get('batch_size', 64)}, "
          f"lr={cfg.get('sft_lr', 1e-5)}, kl_beta={kl_beta}, epochs={n_epochs}")
    print(f"{'='*60}\n")

    for epoch in range(1, n_epochs + 1):
        # Use train mode — dropout acts as implicit regularization alongside the
        # KL penalty. The dropout mismatch with ref_model (eval mode) inflates KL
        # by ~3 nats, but empirically this produces better results than eval mode
        # by preventing overfitting on the small reweighted dataset.
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0
        n_batches = 0

        for inputs, targets, loss_mask, weights in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss_mask = loss_mask.to(device)
            weights = weights.to(device)

            logits = model(inputs)
            B, T, V = logits.shape

            # Per-token cross-entropy
            loss_per_token = F.cross_entropy(
                logits.view(-1, V), targets.view(-1),
                ignore_index=0, reduction='none'
            ).view(B, T)

            # Weighted CE on completion tokens
            weighted_loss = loss_per_token * loss_mask * weights.unsqueeze(1)
            n_tokens = loss_mask.sum() + 1e-8
            ce_loss = weighted_loss.sum() / n_tokens

            # KL(model || reference) on completion tokens
            # Prevents forgetting pretrained representations
            with torch.no_grad():
                ref_logits = ref_model(inputs)
            log_p = F.log_softmax(logits, dim=-1)
            log_q = F.log_softmax(ref_logits, dim=-1)
            # Per-position KL: sum_v p(v) * (log p(v) - log q(v))
            kl_per_pos = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # (B, T)
            kl_loss = (kl_per_pos * loss_mask).sum() / n_tokens

            loss = ce_loss + kl_beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        mean_ce = total_ce / max(n_batches, 1)
        mean_kl = total_kl / max(n_batches, 1)
        print(f"Epoch {epoch:3d} | loss={mean_loss:.4f} | CE={mean_ce:.4f} | KL={mean_kl:.4f}")

        if epoch % eval_every == 0:
            # Quick eval (9 configs, T=1.3 for all)
            eval_temp = cfg.get("eval_temperature", 1.3)
            eval_configs_quick = [
                (g, d, rt, ice)
                for g in g_bins for d in d_bins
                for rt in ["rt_4"] for ice in ["ice_50"]
            ]
            summary, details = evaluate_model(
                model, tokenizer, device, reward_table, problems, raw_data,
                eval_configs_quick, temperature=eval_temp,
                max_new_tokens=cfg.get("max_new_tokens", 60)
            )

            overall_mae = summary["overall"]["mae"]
            if task == "decimals":
                tr_mae = summary.get("bss2021_train", {}).get("mae", float("nan"))
                ho_mae = summary.get("bss2021_held_out", {}).get("mae", float("nan"))
                print(f"  EVAL (T={eval_temp}) | BSS2021 train MAE={tr_mae:.2f}pp | "
                      f"held-out MAE={ho_mae:.2f}pp | Overall MAE={overall_mae:.2f}pp")
                history.append({
                    "epoch": epoch, "loss": mean_loss,
                    "bss2021_train_mae": tr_mae, "bss2021_held_out_mae": ho_mae,
                    "overall_mae": overall_mae,
                })
            else:
                s11_mae = summary.get("siegler2011", {}).get("mae", float("nan"))
                sp_mae = summary.get("sp2013", {}).get("mae", float("nan"))
                print(f"  EVAL (T={eval_temp}) | Siegler2011 MAE={s11_mae:.2f}pp | "
                      f"SP2013 MAE={sp_mae:.2f}pp | Overall MAE={overall_mae:.2f}pp")
                history.append({
                    "epoch": epoch, "loss": mean_loss,
                    "s11_mae": s11_mae, "sp_mae": sp_mae, "overall_mae": overall_mae,
                })

            if overall_mae < best_mae:
                best_mae = overall_mae
                patience_counter = 0
                ckpt_path = os.path.join(cfg["output_dir"], f"best_model_{cfg['tag']}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ** New best MAE={best_mae:.2f}pp -> saved {ckpt_path}")
            else:
                patience_counter += 1

            if patience_counter >= cfg.get("patience", 10):
                print(f"Early stopping at epoch {epoch}")
                break

    # Save history
    hist_path = os.path.join(cfg["output_dir"], f"history_{cfg['tag']}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # Fair eval at the end
    print(f"\n{'='*60}")
    print("FAIR EVAL: 54 configs x 5 rollouts (270/problem)")
    print(f"{'='*60}")
    if task == "decimals":
        eval_temp_final = cfg.get("eval_temperature", 1.0)
        eval_splits = [
            ("BSS2021 train",    "bss2021_train",    eval_temp_final),
            ("BSS2021 held-out", "bss2021_held_out", eval_temp_final),
        ]
    else:
        eval_splits = [
            ("Siegler 2011", "siegler2011", 1.3),
            ("SP2013",       "sp2013",      0.5),
        ]

    for label, src_key, temp in eval_splits:
        prob_subset = {p: info for p, info in problems.items() if info["source"] == src_key}
        if not prob_subset:
            continue
        raw_subset = {p: raw_data[p] for p in prob_subset}
        summary, details = evaluate_model(
            model, tokenizer, device, reward_table, prob_subset, raw_subset,
            eval_configs, temperature=temp, n_rollouts_per_config=5,
            max_new_tokens=cfg.get("max_new_tokens", 60)
        )
        mae = summary[src_key]["mae"]
        tv = summary[src_key]["tv"]
        print(f"  {label} (T={temp}): MAE={mae:.2f}pp  TV={tv:.3f}")
        for prob in sorted(details.keys()):
            d = details[prob]
            print(f"    {prob:16s} model={d['model_acc']*100:5.1f}% human={d['human_acc']*100:5.1f}% "
                  f"AE={d['acc_ae']*100:5.1f}pp TV={d['tv_distance']:.3f}")

    print(f"\nDone. Best MAE={best_mae:.2f}pp")
    return best_mae


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_lr", type=float, default=1e-5)
    p.add_argument("--sft_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--smooth_alpha", type=float, default=0.05)
    p.add_argument("--kl_beta", type=float, default=0.5)
    p.add_argument("--max_weight", type=float, default=5.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--tag", type=str, default="sft_reweight_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", type=str, default="fractions",
                   choices=["fractions", "decimals"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Pretrained distilled-model checkpoint to fine-tune")
    p.add_argument("--held_out_problems", type=str, nargs="*", default=None,
                   help="Problems to exclude from the SFT signal (kept for eval)")
    p.add_argument("--uma_traces_path", type=str, default=None,
                   help="Override UMA traces path (default: per-task)")
    p.add_argument("--eval_temperature", type=float, default=None)
    args = p.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    for k, v in vars(args).items():
        if v is None:
            continue
        cfg[k] = v
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train_sft(cfg)
