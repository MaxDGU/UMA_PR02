#!/usr/bin/env python3
"""
Run saved UMA models on a specified problem set CSV and emit trace-style rows.

Default use targets full GoMath curriculum:
  1. Model/Problem Sets Training/go_math.csv

Output schema is compatible with results/UMA_replication/uma_traces_all.csv:
  subjid, prob, operation, strategy, goals, exec, answer, correct, is_correct,
  g, d, rt_mu, ice

Optional detailed trace fields can also be emitted:
  trace_rules_full, trace_step_count, trace_steps_json
"""

import argparse
import contextlib
import csv
import gc
import glob
import gzip
import io
import json
import os
import pickle
import re
import sys
import time
from fractions import Fraction

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
UMA_MODEL_DIR = os.path.join(REPO_ROOT, "1. Model")
DEFAULT_MODELS_DIR = os.path.join(SCRIPT_DIR, "trained_models")
DEFAULT_PROBLEM_CSV = os.path.join(UMA_MODEL_DIR, "Problem Sets Training", "go_math.csv")
DEFAULT_OUTCSV = os.path.join(SCRIPT_DIR, "uma_traces_all_np_changho.csv")

sys.path.insert(0, UMA_MODEL_DIR)
cwd0 = os.getcwd()
os.chdir(UMA_MODEL_DIR)
from uma import State, myEval  # noqa: E402
from models import all_rules, RA_rules  # noqa: E402
os.chdir(cwd0)


SKIP_RULES = {"deferred_action", "pop_goal", "finish_problem", "larger_first_add_mul", "cannot_simplify"}
GOAL_RULES = {
    "operate_nums",
    "operate_dens",
    "pass_den",
    "convert_CD",
    "convert_CD_LCM",
    "get_LCM",
    "invert_op2",
    "div_to_mul",
    "check_simplify",
    "skip_simplify",
    "get_GCD",
    "simplify_fraction",
    "convert_fra_to_den",
}
EXEC_RULES = {
    "add_fact",
    "sub_fact",
    "mul_fact",
    "div_calculator",
    "convert_CD_omit_nums",
    "invert_rand",
    "invert_fail",
    "div_to_mul_denied",
    "acc_skip",
    "acc_extra",
    "sub_LbS",
    "div_LbS",
    "div_LbS_drop_rem",
    "div_drop_rem",
}
STRATEGIES = [
    "KDON_AS",
    "KDON_OG",
    "CDON_AS",
    "CDON_OG",
    "ONOD_M",
    "ONOD_OG",
    "CROP_M",
    "ICDM_D",
    "ICDM_OG",
]


BASE_FIELDNAMES = [
    "subjid",
    "prob",
    "operation",
    "strategy",
    "goals",
    "exec",
    "answer",
    "correct",
    "is_correct",
    "g",
    "d",
    "rt_mu",
    "ice",
]
TRACE_FIELDNAMES = [
    "trace_rules_full",
    "trace_step_count",
    "trace_steps_json",
]


@contextlib.contextmanager
def suppress_output():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def get_strategy(trace):
    for state, rule in trace:
        if rule and rule.name in STRATEGIES:
            return rule.name
    return "OTHER"


def get_rules(trace):
    goals, execs = [], []
    for state, rule in trace:
        if rule is None:
            continue
        name = rule.name
        if name in SKIP_RULES:
            continue
        if name in GOAL_RULES:
            goals.append(name)
        elif name in EXEC_RULES:
            execs.append(name)
    return goals, execs


def get_answer(trace):
    if not trace:
        return "?"
    final_state = trace[-1][0]
    if not final_state.ws:
        return "?"
    try:
        answer = final_state.ws[0].features["answer"]
        if answer is None:
            return "?"
        subtype = answer.features.get("subtype")
        if subtype == "fraction":
            num = answer.features.get("num")
            den = answer.features.get("den")
            if num is not None and den is not None:
                return f"{str(num)}/{str(den)}"
        return str(answer)
    except Exception:
        return "?"


def number_to_text(number_obj):
    if number_obj is None:
        return "?"
    try:
        feats = getattr(number_obj, "features", None)
        if not isinstance(feats, dict):
            return str(number_obj)
        subtype = feats.get("subtype")
        if subtype == "fraction":
            num = feats.get("num")
            den = feats.get("den")
            if num is None or den is None:
                return "?"
            return f"{num}/{den}"
        if subtype == "whole":
            digits = feats.get("digits")
            if isinstance(digits, list):
                return "".join(str(d) for d in digits)
        return str(number_obj)
    except Exception:
        return str(number_obj)


def problem_obj_to_dict(problem_obj):
    try:
        feats = getattr(problem_obj, "features", {})
        op = str(feats.get("operator", "?"))
        op1 = number_to_text(feats.get("operand1"))
        op2 = number_to_text(feats.get("operand2"))
        ans = number_to_text(feats.get("answer"))
        return {"op": op, "op1": op1, "op2": op2, "ans": ans}
    except Exception:
        return {"op": "?", "op1": "?", "op2": "?", "ans": "?"}


def extract_trace_payload(trace, max_subproblems_per_step=6):
    steps = []
    rule_names = []
    for idx, (state, rule) in enumerate(trace, start=1):
        rule_name = rule.name if rule is not None else ""
        if rule_name != "":
            rule_names.append(rule_name)

        ws = getattr(state, "ws", None)
        if ws is None:
            steps.append({"i": idx, "rule": rule_name, "main": {"op": "?", "op1": "?", "op2": "?", "ans": "?"}})
            continue

        problems = [problem_obj_to_dict(obj) for obj in ws if obj.__class__.__name__ == "Problem"]
        main_prob = problems[0] if len(problems) > 0 else {"op": "?", "op1": "?", "op2": "?", "ans": "?"}
        answered_subs = [p for p in problems[1:] if p.get("ans", "?") not in {"?", "None", ""}]
        if max_subproblems_per_step is not None and max_subproblems_per_step > 0:
            answered_subs = answered_subs[:max_subproblems_per_step]

        step_payload = {"i": idx, "rule": rule_name, "main": main_prob}
        if len(answered_subs) > 0:
            step_payload["subs"] = answered_subs
        steps.append(step_payload)

    return {
        "trace_rules_full": " ".join(rule_names),
        "trace_step_count": len(steps),
        "trace_steps_json": json.dumps(steps, separators=(",", ":"), ensure_ascii=True),
    }


def get_op(prob):
    p = str(prob)
    if "*" in p:
        return "*"
    if ":" in p:
        return ":"
    if "+" in p:
        return "+"
    if "-" in p:
        return "-"
    return "?"


def compute_correct(prob):
    try:
        val = myEval(prob)
        frac = Fraction(val).limit_denominator(1000)
        if frac.denominator == 1:
            return str(frac.numerator)
        return f"{frac.numerator}/{frac.denominator}"
    except Exception:
        return "?"


def as_float(expr):
    if expr is None:
        return None
    s = str(expr).strip()
    if s == "" or s in {"?", "FRACTION", "None", "nan", "-99999"}:
        return None
    s = s.replace("−", "-").replace("÷", ":")
    try:
        return float(myEval(s))
    except Exception:
        return None


def answers_match(a, b, tol=1e-6):
    x = as_float(a)
    y = as_float(b)
    if x is None or y is None:
        return False
    return abs(x - y) < tol


def normalize_prob(prob):
    p = str(prob or "").strip()
    p = p.replace("−", "-").replace("÷", ":")
    p = re.sub(r"\s+", "", p)
    return p


def parse_subjid_from_path(path):
    m = re.search(r"model_subjid_(\d+)\.pkl\.gz$", os.path.basename(path))
    return int(m.group(1)) if m else None


def discover_model_paths(models_dir, start_subjid=None, end_subjid=None):
    paths = sorted(glob.glob(os.path.join(models_dir, "model_subjid_*.pkl.gz")))
    filtered = []
    for path in paths:
        subjid = parse_subjid_from_path(path)
        if subjid is None:
            continue
        if start_subjid is not None and subjid < start_subjid:
            continue
        if end_subjid is not None and subjid > end_subjid:
            continue
        filtered.append((subjid, path))
    return filtered


def build_student_from_payload(payload):
    proc_mem = payload["proc_mem"]
    ans_mem = payload["ans_mem"]
    params = payload["params"]
    rule_names = payload.get("rule_names", list(proc_mem.columns))

    rule_lookup = {}
    for rule in list(all_rules) + list(RA_rules):
        rule_lookup[rule.name] = rule
    missing = [name for name in rule_names if name not in rule_lookup]
    if missing:
        raise ValueError(f"Missing rule definitions for: {missing[:10]}")

    rules = [rule_lookup[name] for name in rule_names]

    from uma import UMA  # local import avoids path/circular surprises

    student = UMA(rules=rules, params=params)
    student.proc_mem = proc_mem
    student.ans_mem = ans_mem
    student.trace = []
    return student


def parse_args():
    parser = argparse.ArgumentParser(description="Run saved UMA models on a problem set CSV.")
    parser.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_DIR, help="Directory of saved model files.")
    parser.add_argument("--problem-csv", type=str, default=DEFAULT_PROBLEM_CSV, help="CSV containing a 'prob' column.")
    parser.add_argument("--problem-col", type=str, default="prob", help="Problem column name in --problem-csv.")
    parser.add_argument(
        "--dedup-problems",
        action="store_true",
        help="Evaluate each unique problem once per model (otherwise preserve row duplicates).",
    )
    parser.add_argument("--out-csv", type=str, default=DEFAULT_OUTCSV, help="Output CSV path.")
    parser.add_argument("--start-subjid", type=int, default=None, help="Optional lower model subjid bound.")
    parser.add_argument("--end-subjid", type=int, default=None, help="Optional upper model subjid bound.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume writing to an existing --out-csv by skipping subjid values that already have "
            "a full row block. Partial subjid blocks are discarded and recomputed."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --out-csv if it exists.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N models.")
    parser.add_argument(
        "--include-trace-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include detailed step-level state transition payload columns in output.",
    )
    parser.add_argument(
        "--trace-max-subproblems-per-step",
        type=int,
        default=6,
        help="Maximum answered sub-problems to serialize per step in trace_steps_json.",
    )
    return parser.parse_args()


def get_output_fieldnames(include_trace_steps: bool):
    fields = list(BASE_FIELDNAMES)
    if include_trace_steps:
        fields.extend(TRACE_FIELDNAMES)
    return fields


def read_csv_header_fields(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        return reader.fieldnames


def load_problem_rows(problem_csv: str, problem_col: str, dedup: bool):
    df = pd.read_csv(problem_csv)
    if problem_col not in df.columns:
        raise ValueError(f"Missing problem column '{problem_col}' in {problem_csv}")

    probs = [normalize_prob(x) for x in df[problem_col].tolist()]
    rows = []
    for i, prob in enumerate(probs):
        if prob == "":
            continue
        rows.append({"row_idx": i, "prob": prob})

    if dedup:
        seen = set()
        uniq = []
        for r in rows:
            p = r["prob"]
            if p in seen:
                continue
            seen.add(p)
            uniq.append(r)
        rows = uniq

    return rows


def inspect_resume_output(out_csv: str, expected_rows_per_model: int, fieldnames):
    """
    Inspect an existing output CSV for resume mode.

    Returns:
      completed_subjids: set[int] with full row blocks
      dropped_rows: count of rows removed due to partial/invalid blocks
      rewritten: whether file was rewritten to remove invalid rows
    """
    if not os.path.exists(out_csv):
        return set(), 0, False

    if os.path.getsize(out_csv) == 0:
        # Timeout/cancellation can leave a 0-byte output. Treat as empty.
        os.remove(out_csv)
        return set(), 0, False

    counts = {}
    bad_rows = 0
    with open(out_csv, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            os.remove(out_csv)
            return set(), 0, False
        if "subjid" not in reader.fieldnames:
            raise ValueError(f"Cannot resume: missing 'subjid' column in existing output {out_csv}")

        for row in reader:
            raw = str(row.get("subjid", "")).strip()
            if raw == "":
                bad_rows += 1
                continue
            try:
                sid = int(raw)
            except Exception:
                bad_rows += 1
                continue
            counts[sid] = counts.get(sid, 0) + 1

    completed_subjids = {sid for sid, n in counts.items() if n == expected_rows_per_model}
    incomplete_subjids = {sid for sid, n in counts.items() if n != expected_rows_per_model}

    needs_rewrite = bad_rows > 0 or bool(incomplete_subjids)
    if not needs_rewrite:
        return completed_subjids, 0, False

    tmp_csv = out_csv + ".resume_tmp"
    dropped_rows = 0
    with open(out_csv, "r", newline="", encoding="utf-8") as f_in, open(
        tmp_csv, "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            writer = csv.writer(f_out)
            writer.writerow(fieldnames)
        else:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                raw = str(row.get("subjid", "")).strip()
                try:
                    sid = int(raw)
                except Exception:
                    dropped_rows += 1
                    continue
                if sid in completed_subjids:
                    writer.writerow(row)
                else:
                    dropped_rows += 1

    os.replace(tmp_csv, out_csv)
    return completed_subjids, dropped_rows, True


def main():
    args = parse_args()

    out_csv = os.path.abspath(args.out_csv)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")

    fieldnames = get_output_fieldnames(include_trace_steps=args.include_trace_steps)

    model_items = discover_model_paths(
        os.path.abspath(args.models_dir),
        start_subjid=args.start_subjid,
        end_subjid=args.end_subjid,
    )
    if not model_items:
        # Some subjid shards may be empty if no trained models exist in that range.
        # Emit a header-only CSV so shard merges can proceed, and exit cleanly.
        with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
        print("[INFO] No model files found for the requested subjid range; wrote header-only shard.")
        print(f"output: {out_csv}")
        return

    problem_rows = load_problem_rows(
        problem_csv=os.path.abspath(args.problem_csv),
        problem_col=args.problem_col,
        dedup=args.dedup_problems,
    )
    if not problem_rows:
        raise ValueError(f"No valid problems loaded from {args.problem_csv}")

    rows_per_model = len(problem_rows)
    completed_subjids = set()
    dropped_resume_rows = 0
    rewritten_resume = False
    if os.path.exists(out_csv):
        if args.overwrite:
            pass
        elif args.resume:
            header_fields = read_csv_header_fields(out_csv)
            if header_fields is not None and header_fields != fieldnames:
                raise ValueError(
                    "Cannot resume because existing output schema differs from current settings. "
                    "Use --overwrite or match --include-trace-steps."
                )
            completed_subjids, dropped_resume_rows, rewritten_resume = inspect_resume_output(
                out_csv=out_csv,
                expected_rows_per_model=rows_per_model,
                fieldnames=fieldnames,
            )
        else:
            raise FileExistsError(f"{out_csv} exists. Use --overwrite to replace it.")

    total_models_requested = len(model_items)
    if completed_subjids:
        model_items = [(sid, p) for sid, p in model_items if sid not in completed_subjids]
    skipped_models = total_models_requested - len(model_items)

    unique_probs = sorted({r["prob"] for r in problem_rows})
    correct_by_prob = {p: compute_correct(p) for p in unique_probs}

    if not model_items:
        print("=" * 70)
        print("UMA PROBLEM-SET TRACE GENERATION")
        print("=" * 70)
        print("[INFO] Resume found all requested models already completed; nothing to do.")
        print(f"models requested: {total_models_requested}")
        print(f"models skipped: {skipped_models}")
        print(f"problem rows per model: {rows_per_model}")
        print(f"output: {out_csv}")
        return

    print("=" * 70)
    print("UMA PROBLEM-SET TRACE GENERATION")
    print("=" * 70)
    print(f"models requested: {total_models_requested}")
    print(f"models to run: {len(model_items)}")
    print(f"models skipped (resume): {skipped_models}")
    print(f"problem csv: {os.path.abspath(args.problem_csv)}")
    print(f"problem rows: {len(problem_rows)}")
    print(f"unique probs: {len(unique_probs)}")
    print(f"output: {out_csv}")
    print(f"include_trace_steps: {args.include_trace_steps}")
    if args.resume:
        print(
            f"resume: enabled, rewritten_existing={int(rewritten_resume)}, "
            f"dropped_partial_rows={dropped_resume_rows}"
        )
    elif args.overwrite:
        print("resume: disabled (overwrite enabled)")
    else:
        print("resume: disabled")
    print("")

    t0 = time.time()
    written = 0
    failed = 0

    write_mode = "a" if args.resume and os.path.exists(out_csv) and not args.overwrite else "w"
    write_header = write_mode == "w" or (write_mode == "a" and os.path.getsize(out_csv) == 0)

    with open(out_csv, write_mode, newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, (subjid, model_path) in enumerate(model_items, start=1):
            try:
                with gzip.open(model_path, "rb") as f_model:
                    payload = pickle.load(f_model)
                student = build_student_from_payload(payload)
            except Exception as exc:
                failed += 1
                print(f"[WARN] load failed for subjid={subjid}: {exc}")
                continue

            params = payload["params"]

            per_prob = {}
            for prob in unique_probs:
                try:
                    with suppress_output():
                        student.run(State(prob), learn=False, verbose=False)
                    trace = student.trace if student.trace else []
                    strategy = get_strategy(trace)
                    goals, execs = get_rules(trace)
                    answer = get_answer(trace)
                    correct = correct_by_prob[prob]
                    per_prob[prob] = {
                        "operation": get_op(prob),
                        "answer": answer,
                        "correct": correct,
                        "is_correct": answers_match(answer, correct),
                        "strategy": strategy,
                        "goals": " ".join(goals),
                        "exec": " ".join(execs),
                    }
                    if args.include_trace_steps:
                        per_prob[prob].update(
                            extract_trace_payload(
                                trace,
                                max_subproblems_per_step=args.trace_max_subproblems_per_step,
                            )
                        )
                except Exception:
                    per_prob[prob] = {
                        "operation": get_op(prob),
                        "answer": "?",
                        "correct": correct_by_prob[prob],
                        "is_correct": False,
                        "strategy": "OTHER",
                        "goals": "",
                        "exec": "",
                    }
                    if args.include_trace_steps:
                        per_prob[prob].update(
                            {
                                "trace_rules_full": "",
                                "trace_step_count": 0,
                                "trace_steps_json": "[]",
                            }
                        )
                finally:
                    student.trace = []

            for pr in problem_rows:
                pred = per_prob[pr["prob"]]
                out_row = {
                    "subjid": subjid,
                    "prob": pr["prob"],
                    "operation": pred["operation"],
                    "strategy": pred["strategy"],
                    "goals": pred["goals"],
                    "exec": pred["exec"],
                    "answer": pred["answer"],
                    "correct": pred["correct"],
                    "is_correct": int(pred["is_correct"]),
                    "g": params.get("g", ""),
                    "d": params.get("d", ""),
                    "rt_mu": params.get("rt_mu", ""),
                    "ice": params.get("ice", ""),
                }
                if args.include_trace_steps:
                    out_row.update(
                        {
                            "trace_rules_full": pred.get("trace_rules_full", ""),
                            "trace_step_count": pred.get("trace_step_count", 0),
                            "trace_steps_json": pred.get("trace_steps_json", "[]"),
                        }
                    )
                writer.writerow(out_row)
                written += 1

            del student
            gc.collect()

            if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(model_items)):
                elapsed = time.time() - t0
                print(
                    f"[{i}/{len(model_items)}] rows_written={written} failed_models={failed} "
                    f"elapsed={elapsed/60:.1f} min"
                )

    elapsed = time.time() - t0
    print("")
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"rows written: {written}")
    print(f"failed models: {failed}")
    print(f"elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
