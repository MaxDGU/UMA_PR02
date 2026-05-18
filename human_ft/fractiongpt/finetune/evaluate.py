#!/usr/bin/env python3
"""
Evaluate FractionGPT against human data: accuracy MAE, TV distance, per-problem breakdown.

Usage:
    python evaluate.py --checkpoint path/to/model.pt
    python evaluate.py --checkpoint path/to/model.pt --temperature 0.5
"""

import os
import sys
import copy
import argparse
from fractions import Fraction

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from train_transformer_phase3 import FractionTokenizer, FractionGPT, extract_region
from finetune.data import normalize_answer, load_human_data, get_reward
from finetune.generate import build_prompt, generate_rollout
from finetune.config import DEFAULT_CONFIG


# ============================================================================
# Evaluation helpers
# ============================================================================

def _compute_correct_val(prob):
    """Return the numerical value of the correct answer (as float), or None."""
    try:
        if ":" in prob:
            left, right = prob.split(":")
            return float(Fraction(left.strip()) / Fraction(right.strip()))
        else:
            return float(eval(prob))
    except Exception:
        return None


def _ans_to_val(ans_str):
    """Convert a surface-form answer string to float, or None."""
    try:
        parts = ans_str.split("/")
        if len(parts) == 2:
            return float(parts[0]) / float(parts[1])
        return float(ans_str)
    except Exception:
        return None


def _sum_correct_freq(dist, correct_val):
    """Sum frequencies across all surface forms that are numerically correct."""
    if correct_val is None:
        return 0.0
    total = 0.0
    for ans, freq in dist.items():
        v = _ans_to_val(ans)
        if v is not None and abs(v - correct_val) < 1e-6:
            total += freq
    return total


def compute_tv_distance(model_dist, human_dist):
    """TV distance = 0.5 * sum |p(a) - q(a)| over union of answers."""
    all_answers = set(model_dist.keys()) | set(human_dist.keys())
    tv = 0.0
    for a in all_answers:
        tv += abs(model_dist.get(a, 0.0) - human_dist.get(a, 0.0))
    return 0.5 * tv


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate_model(model, tokenizer, device, reward_table, problems, raw_data,
                   student_configs, temperature, n_rollouts_per_config=1,
                   max_new_tokens=60):
    """Evaluate model against human data. Returns MAE and diagnostics."""
    model.eval()

    # Collect model answer distribution per problem
    model_answers = {prob: {} for prob in problems}
    strategy_counts = {prob: {} for prob in problems}

    for prob in problems:
        total_rollouts = 0
        for sc in student_configs:
            for _ in range(n_rollouts_per_config):
                _, answer, text = generate_rollout(
                    model, build_prompt(prob, sc, tokenizer),
                    tokenizer, device, temperature, max_new_tokens
                )
                if answer:
                    norm = normalize_answer(answer)
                    if norm:
                        model_answers[prob][norm] = model_answers[prob].get(norm, 0) + 1

                # Track strategy
                strat = extract_region(text, "<strategy>", "</strategy>").strip()
                if strat:
                    strategy_counts[prob][strat] = strategy_counts[prob].get(strat, 0) + 1

                total_rollouts += 1

        # Normalize to frequencies
        total = sum(model_answers[prob].values())
        if total > 0:
            for ans in model_answers[prob]:
                model_answers[prob][ans] /= total

    # Compute MAE per problem: |model_acc - human_acc|
    results = {}
    for prob, pinfo in problems.items():
        source = pinfo["source"]
        human_dist = raw_data[prob]["answers"]

        op = "+" if "+" in prob else "-" if "-" in prob else "*" if "*" in prob else ":"
        correct_val = _compute_correct_val(prob)

        # Accuracy: sum frequencies of all surface forms that are numerically correct
        model_acc = _sum_correct_freq(model_answers[prob], correct_val)
        human_acc = _sum_correct_freq(human_dist, correct_val)

        # Mean reward
        mean_reward = 0.0
        for ans, freq in model_answers[prob].items():
            r = reward_table.get((prob, ans), 0.0)
            mean_reward += freq * r

        # TV distance
        tv = compute_tv_distance(model_answers[prob], human_dist)

        results[prob] = {
            "source": source,
            "operation": op,
            "model_acc": model_acc,
            "human_acc": human_acc,
            "acc_ae": abs(model_acc - human_acc),
            "mean_reward": mean_reward,
            "tv_distance": tv,
            "model_dist": model_answers[prob],
            "human_dist": human_dist,
            "strategies": strategy_counts[prob],
        }

    # Aggregate
    by_source = {}
    for prob, r in results.items():
        src = r["source"]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(r)

    summary = {}
    for src, items in by_source.items():
        mae = np.mean([r["acc_ae"] for r in items])
        mean_reward = np.mean([r["mean_reward"] for r in items])
        mean_tv = np.mean([r["tv_distance"] for r in items])
        summary[src] = {"mae": mae * 100, "mean_reward": mean_reward,
                        "tv": mean_tv, "n_problems": len(items)}

        # By operation
        ops = {}
        for r in items:
            op = r["operation"]
            if op not in ops:
                ops[op] = []
            ops[op].append(r)
        for op, op_items in ops.items():
            op_mae = np.mean([r["acc_ae"] for r in op_items])
            summary[f"{src}_{op}"] = {"mae": op_mae * 100, "n_problems": len(op_items)}

    overall_mae = np.mean([r["acc_ae"] for r in results.values()]) * 100
    overall_reward = np.mean([r["mean_reward"] for r in results.values()])
    overall_tv = np.mean([r["tv_distance"] for r in results.values()])
    summary["overall"] = {"mae": overall_mae, "mean_reward": overall_reward, "tv": overall_tv}

    return summary, results


def load_single_model(checkpoint_path, device):
    """Load a single FractionGPT from checkpoint."""
    tokenizer = FractionTokenizer()
    state_dict = torch.load(checkpoint_path, map_location=device)
    vocab_size = state_dict["embedding.weight"].shape[0]
    model = FractionGPT(
        vocab_size=vocab_size, hidden_size=128, nlayers=5, nhead=8,
        dropout_p=0.1, pad_idx=tokenizer.pad_token_id, legacy_decoder=False,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    return model, tokenizer


# ============================================================================
# Oracle config selection
# ============================================================================

def load_per_student_data(cfg):
    """Load raw per-student answers from both datasets.

    Returns list of dicts: [{subjid, source, grade, age, pssa_math, pssa_verb,
                             answers: {prob: raw_answer_str}}]
    """
    import pandas as pd
    students = []

    s11 = pd.read_csv(cfg["siegler2011_path"], encoding="latin-1")
    for sid in s11["subjid"].unique():
        sub = s11[s11["subjid"] == sid]
        row0 = sub.iloc[0]
        answers = {}
        for _, row in sub.iterrows():
            prob = str(row["prob"]).strip()
            norm = normalize_answer(row["resp"])
            if norm is not None:
                answers[prob] = norm
        students.append({
            "subjid": str(sid), "source": "siegler2011",
            "grade": int(row0["grade"]) if pd.notna(row0.get("grade")) else None,
            "age": float(row0["age"]) if pd.notna(row0.get("age")) else None,
            "pssa_math": float(row0["PSSAmath"]) if pd.notna(row0.get("PSSAmath")) else None,
            "pssa_verb": float(row0["PSSAverb"]) if pd.notna(row0.get("PSSAverb")) else None,
            "answers": answers,
        })

    sp = pd.read_csv(cfg["sp2013_path"])
    for sid in sp["subjid"].unique():
        sub = sp[sp["subjid"] == sid]
        row0 = sub.iloc[0]
        answers = {}
        for _, row in sub.iterrows():
            prob = str(row["prob"]).strip()
            norm = normalize_answer(row["resp"])
            if norm is not None:
                answers[prob] = norm
        students.append({
            "subjid": str(sid), "source": "sp2013",
            "grade": int(row0["grade"]) if pd.notna(row0.get("grade")) else None,
            "age": None, "pssa_math": None, "pssa_verb": None,
            "answers": answers,
        })

    return students


def cache_model_distributions(model, tokenizer, device, all_problems, all_configs,
                               temperature, n_rollouts=5, max_new_tokens=60):
    """Generate and cache model answer distributions for all (problem, config) pairs.

    Returns: dict mapping (prob, config_str) -> {answer: frequency}
    """
    model.eval()
    cache = {}
    total = len(all_problems) * len(all_configs)
    done = 0

    for prob in all_problems:
        for sc in all_configs:
            config_str = "_".join(sc)
            counts = {}
            for _ in range(n_rollouts):
                prompt_ids = build_prompt(prob, sc, tokenizer)
                _, answer, _ = generate_rollout(
                    model, prompt_ids, tokenizer, device, temperature, max_new_tokens
                )
                if answer:
                    norm = normalize_answer(answer)
                    if norm:
                        counts[norm] = counts.get(norm, 0) + 1
            total_r = sum(counts.values())
            dist = {a: c / total_r for a, c in counts.items()} if total_r > 0 else {}
            cache[(prob, config_str)] = dist
            done += 1

        print(f"  Cached {done}/{total} pairs ({prob})")

    return cache


def oracle_config_selection(students, all_configs, cache, reward_table):
    """For each student, find the config that maximizes reward against their answers.

    Args:
        students: list of student dicts with 'answers' field
        all_configs: list of (g, d, rt, ice) tuples
        cache: dict from cache_model_distributions
        reward_table: from load_human_data

    Returns: list of dicts with oracle results per student
    """
    results = []

    for student in students:
        student_answers = student["answers"]  # {prob: normalized_answer}
        if not student_answers:
            continue

        best_config = None
        best_reward = -1.0

        per_config_rewards = {}
        for sc in all_configs:
            config_str = "_".join(sc)
            total_reward = 0.0
            n_probs = 0

            for prob, student_ans in student_answers.items():
                dist = cache.get((prob, config_str), {})
                # Reward: P(model generates this student's answer | config)
                total_reward += dist.get(student_ans, 0.0)
                n_probs += 1

            mean_reward = total_reward / max(n_probs, 1)
            per_config_rewards[config_str] = mean_reward

            if mean_reward > best_reward:
                best_reward = mean_reward
                best_config = sc

        results.append({
            "subjid": student["subjid"],
            "source": student["source"],
            "grade": student["grade"],
            "age": student["age"],
            "pssa_math": student["pssa_math"],
            "best_config": best_config,
            "best_config_str": "_".join(best_config) if best_config else None,
            "oracle_reward": best_reward,
            "n_problems": len(student_answers),
            "answers": student_answers,
        })

    return results


def compute_oracle_metrics(oracle_results, cache, raw_data):
    """Compute MAE and TV under oracle config vs averaged baseline per student.

    Returns per-student and aggregate metrics.
    """
    per_student = []

    for student in oracle_results:
        answers = student["answers"]
        best_config_str = student["best_config_str"]
        source = student["source"]
        if not answers or not best_config_str:
            continue

        oracle_maes = []
        oracle_tvs = []
        baseline_maes = []
        baseline_tvs = []

        for prob, student_ans in answers.items():
            human_dist = raw_data[prob]["answers"]
            correct_val = _compute_correct_val(prob)

            # Oracle: single best config
            oracle_dist = cache.get((prob, best_config_str), {})
            oracle_acc = _sum_correct_freq(oracle_dist, correct_val)
            human_acc = _sum_correct_freq(human_dist, correct_val)
            oracle_maes.append(abs(oracle_acc - human_acc))
            oracle_tvs.append(compute_tv_distance(oracle_dist, human_dist))

            # Baseline: average over all configs for this problem
            all_config_keys = [k for k in cache if k[0] == prob]
            if all_config_keys:
                avg_dist = {}
                for key in all_config_keys:
                    for ans, freq in cache[key].items():
                        avg_dist[ans] = avg_dist.get(ans, 0.0) + freq / len(all_config_keys)
                baseline_acc = _sum_correct_freq(avg_dist, correct_val)
                baseline_maes.append(abs(baseline_acc - human_acc))
                baseline_tvs.append(compute_tv_distance(avg_dist, human_dist))

        op_map = {}
        for prob in answers:
            op = "+" if "+" in prob else "-" if "-" in prob else "*" if "*" in prob else ":"
            if op not in op_map:
                op_map[op] = {"oracle_mae": [], "baseline_mae": [],
                              "oracle_tv": [], "baseline_tv": []}

        idx = 0
        for prob in answers:
            op = "+" if "+" in prob else "-" if "-" in prob else "*" if "*" in prob else ":"
            if idx < len(oracle_maes):
                op_map[op]["oracle_mae"].append(oracle_maes[idx])
                op_map[op]["oracle_tv"].append(oracle_tvs[idx])
            if idx < len(baseline_maes):
                op_map[op]["baseline_mae"].append(baseline_maes[idx])
                op_map[op]["baseline_tv"].append(baseline_tvs[idx])
            idx += 1

        per_student.append({
            "subjid": student["subjid"],
            "source": source,
            "grade": student.get("grade"),
            "pssa_math": student.get("pssa_math"),
            "best_config": student["best_config_str"],
            "oracle_mae": np.mean(oracle_maes) if oracle_maes else None,
            "oracle_tv": np.mean(oracle_tvs) if oracle_tvs else None,
            "baseline_mae": np.mean(baseline_maes) if baseline_maes else None,
            "baseline_tv": np.mean(baseline_tvs) if baseline_tvs else None,
            "by_op": {op: {k: np.mean(v) if v else None for k, v in vals.items()}
                      for op, vals in op_map.items()},
        })

    return per_student


def run_oracle_experiment(args):
    """Run the full oracle config selection experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = dict(DEFAULT_CONFIG)
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint

    model, tokenizer = load_single_model(cfg["checkpoint"], device)
    reward_table, problems, raw_data = load_human_data(cfg)

    g_bins = ["g_low", "g_mid", "g_high"]
    d_bins = ["d_low", "d_mid", "d_high"]
    all_configs = [
        (g, d, rt, ice)
        for g in g_bins for d in d_bins
        for rt in ["rt_3", "rt_4", "rt_5", "rt_6"]
        for ice in ["ice_0", "ice_25", "ice_50", "ice_75", "ice_100"]
    ]
    print(f"All configs: {len(all_configs)} (full coarse grid)")

    temperature = args.temperature
    n_rollouts = args.n_rollouts
    print(f"Temperature: {temperature}, Rollouts per (prob, config): {n_rollouts}")

    # Step 1: Load per-student data
    print("\nLoading per-student data...")
    students = load_per_student_data(cfg)
    s11_students = [s for s in students if s["source"] == "siegler2011"]
    sp_students = [s for s in students if s["source"] == "sp2013"]
    print(f"  Siegler 2011: {len(s11_students)} students")
    print(f"  SP2013: {len(sp_students)} students")

    # Step 2: Cache model distributions
    all_problems = sorted(problems.keys())
    print(f"\nCaching model distributions for {len(all_problems)} problems x {len(all_configs)} configs...")
    cache = cache_model_distributions(
        model, tokenizer, device, all_problems, all_configs,
        temperature, n_rollouts, cfg["max_new_tokens"]
    )
    print(f"  Cached {len(cache)} (problem, config) distributions")

    # Step 3: Oracle selection
    print("\nRunning oracle config selection...")
    oracle_results = oracle_config_selection(students, all_configs, cache, reward_table)

    # Step 4: Compute metrics
    print("Computing oracle vs baseline metrics...")
    per_student = compute_oracle_metrics(oracle_results, cache, raw_data)

    # Step 5: Report
    print(f"\n{'='*80}")
    print("ORACLE CONFIG SELECTION RESULTS")
    print(f"{'='*80}")

    for source_name, source_key in [("Siegler 2011", "siegler2011"), ("SP2013", "sp2013")]:
        subset = [s for s in per_student if s["source"] == source_key]
        if not subset:
            continue

        oracle_maes = [s["oracle_mae"] for s in subset if s["oracle_mae"] is not None]
        baseline_maes = [s["baseline_mae"] for s in subset if s["baseline_mae"] is not None]
        oracle_tvs = [s["oracle_tv"] for s in subset if s["oracle_tv"] is not None]
        baseline_tvs = [s["baseline_tv"] for s in subset if s["baseline_tv"] is not None]

        print(f"\n--- {source_name} ({len(subset)} students) ---")
        print(f"  {'Metric':12s} | {'Baseline':>10s} | {'Oracle':>10s} | {'Delta':>10s}")
        print(f"  {'-'*50}")

        if oracle_maes and baseline_maes:
            bm, om = np.mean(baseline_maes), np.mean(oracle_maes)
            print(f"  {'Acc MAE':12s} | {bm*100:9.2f}pp | {om*100:9.2f}pp | {(om-bm)*100:+9.2f}pp")
        if oracle_tvs and baseline_tvs:
            bt, ot = np.mean(baseline_tvs), np.mean(oracle_tvs)
            print(f"  {'TV distance':12s} | {bt:10.3f} | {ot:10.3f} | {ot-bt:+10.3f}")

        # By operation
        ops = {}
        for s in subset:
            for op, vals in s.get("by_op", {}).items():
                if op not in ops:
                    ops[op] = {"oracle_mae": [], "baseline_mae": [],
                               "oracle_tv": [], "baseline_tv": []}
                for k in ["oracle_mae", "baseline_mae", "oracle_tv", "baseline_tv"]:
                    if vals.get(k) is not None:
                        ops[op][k].append(vals[k])

        if ops:
            print(f"\n  By operation:")
            for op in sorted(ops.keys()):
                v = ops[op]
                if v["oracle_mae"] and v["baseline_mae"]:
                    bm = np.mean(v["baseline_mae"])
                    om = np.mean(v["oracle_mae"])
                    bt = np.mean(v["baseline_tv"]) if v["baseline_tv"] else 0
                    ot = np.mean(v["oracle_tv"]) if v["oracle_tv"] else 0
                    print(f"    {op}: MAE {bm*100:.1f}pp -> {om*100:.1f}pp ({(om-bm)*100:+.1f}pp) | "
                          f"TV {bt:.3f} -> {ot:.3f} ({ot-bt:+.3f})")

        # Config clustering
        from collections import Counter
        config_counts = Counter(s["best_config"] for s in subset if s["best_config"])
        print(f"\n  Most common oracle configs:")
        for config, cnt in config_counts.most_common(5):
            print(f"    {config}: {cnt}/{len(subset)} students ({cnt/len(subset)*100:.0f}%)")

        # Grade breakdown
        grades = {}
        for s in subset:
            g = s.get("grade")
            if g is not None:
                if g not in grades:
                    grades[g] = {"oracle_mae": [], "baseline_mae": []}
                if s["oracle_mae"] is not None:
                    grades[g]["oracle_mae"].append(s["oracle_mae"])
                if s["baseline_mae"] is not None:
                    grades[g]["baseline_mae"].append(s["baseline_mae"])

        if grades:
            print(f"\n  By grade:")
            for g in sorted(grades.keys()):
                v = grades[g]
                if v["oracle_mae"] and v["baseline_mae"]:
                    bm = np.mean(v["baseline_mae"])
                    om = np.mean(v["oracle_mae"])
                    print(f"    Grade {g} (n={len(v['oracle_mae'])}): "
                          f"MAE {bm*100:.1f}pp -> {om*100:.1f}pp ({(om-bm)*100:+.1f}pp)")

        # PSSA breakdown (Siegler 2011 only)
        pssa_students = [s for s in subset if s.get("pssa_math") is not None]
        if pssa_students:
            median_pssa = np.median([s["pssa_math"] for s in pssa_students])
            low = [s for s in pssa_students if s["pssa_math"] <= median_pssa]
            high = [s for s in pssa_students if s["pssa_math"] > median_pssa]
            print(f"\n  By PSSA math (median={median_pssa:.0f}, n={len(pssa_students)}):")
            for label, group in [("low", low), ("high", high)]:
                if group:
                    bm = np.mean([s["baseline_mae"] for s in group if s["baseline_mae"] is not None])
                    om = np.mean([s["oracle_mae"] for s in group if s["oracle_mae"] is not None])
                    print(f"    {label} (n={len(group)}): "
                          f"MAE {bm*100:.1f}pp -> {om*100:.1f}pp ({(om-bm)*100:+.1f}pp)")

    # Overall summary
    all_oracle_mae = [s["oracle_mae"] for s in per_student if s["oracle_mae"] is not None]
    all_baseline_mae = [s["baseline_mae"] for s in per_student if s["baseline_mae"] is not None]
    all_oracle_tv = [s["oracle_tv"] for s in per_student if s["oracle_tv"] is not None]
    all_baseline_tv = [s["baseline_tv"] for s in per_student if s["baseline_tv"] is not None]

    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    if all_oracle_mae and all_baseline_mae:
        bm, om = np.mean(all_baseline_mae), np.mean(all_oracle_mae)
        print(f"  Acc MAE:     {bm*100:.2f}pp (baseline) -> {om*100:.2f}pp (oracle) [{(om-bm)*100:+.2f}pp]")
    if all_oracle_tv and all_baseline_tv:
        bt, ot = np.mean(all_baseline_tv), np.mean(all_oracle_tv)
        print(f"  TV distance: {bt:.3f} (baseline) -> {ot:.3f} (oracle) [{ot-bt:+.3f}]")
    print(f"{'='*80}")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    # Standard eval
    eval_p = sub.add_parser("eval", help="Standard evaluation")
    eval_p.add_argument("--checkpoint", type=str, required=True)
    eval_p.add_argument("--temperature", type=float, default=1.3)

    # Oracle config selection
    oracle_p = sub.add_parser("oracle", help="Oracle config selection experiment")
    oracle_p.add_argument("--checkpoint", type=str, default=None)
    oracle_p.add_argument("--temperature", type=float, default=1.3)
    oracle_p.add_argument("--n_rollouts", type=int, default=5,
                          help="Rollouts per (problem, config) pair for caching")

    args = p.parse_args()

    # Default to eval if no subcommand (backward compat)
    if args.command is None or args.command == "eval":
        if not hasattr(args, "checkpoint") or args.checkpoint is None:
            p.error("--checkpoint required for eval")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = dict(DEFAULT_CONFIG)
        model, tokenizer = load_single_model(args.checkpoint, device)
        reward_table, problems, raw_data = load_human_data(cfg)

        g_bins = ["g_low", "g_mid", "g_high"]
        d_bins = ["d_low", "d_mid", "d_high"]
        eval_configs = [
            (g, d, rt, ice)
            for g in g_bins for d in d_bins
            for rt in ["rt_3", "rt_6"] for ice in ["ice_0", "ice_50", "ice_100"]
        ]

        print(f"Checkpoint: {args.checkpoint}")
        print(f"Temperature: {args.temperature}, Configs: {len(eval_configs)}")

        summary, details = evaluate_model(
            model, tokenizer, device, reward_table, problems, raw_data,
            eval_configs, temperature=args.temperature
        )

        s11 = summary.get("siegler2011", {})
        sp = summary.get("sp2013", {})
        overall = summary["overall"]

        print(f"\nSiegler2011: MAE={s11.get('mae', 0):.2f}pp  TV={s11.get('tv', 0):.3f}")
        print(f"SP2013:      MAE={sp.get('mae', 0):.2f}pp  TV={sp.get('tv', 0):.3f}")
        print(f"Overall:     MAE={overall['mae']:.2f}pp  TV={overall['tv']:.3f}")

        print(f"\n{'Problem':16s} | {'Src':10s} | {'Mod%':>5s} {'Hum%':>5s} {'MAE':>6s} | {'TV':>5s} | {'Reward':>6s}")
        print("-" * 70)
        for prob in sorted(details.keys()):
            d = details[prob]
            print(f"{prob:16s} | {d['source']:10s} | {d['model_acc']*100:4.1f}% {d['human_acc']*100:4.1f}% "
                  f"{d['acc_ae']*100:5.1f}pp | {d['tv_distance']:.3f} | {d['mean_reward']:.3f}")

    elif args.command == "oracle":
        run_oracle_experiment(args)


if __name__ == "__main__":
    main()
