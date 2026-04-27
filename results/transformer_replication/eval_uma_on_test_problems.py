"""Run trained UMA students on the 145 held-out test problems (sd_add + sd_mul).

Mirrors the cohort used for distillation (subjids 0..59) so the UMA accuracy
reported here is directly comparable to base / distilled SmolLM2 on the same
problems.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
UMA_MODEL_DIR = os.path.join(PROJECT_ROOT, 'UMA_PR02_fork', '1. Model')
MODEL_OUTPUT_DIR = os.path.join(UMA_MODEL_DIR, 'Output', 'UMA_PARALLEL')
TEST_DIR = os.path.join(UMA_MODEL_DIR, 'Problem Sets Testing')

os.chdir(UMA_MODEL_DIR)
sys.path.insert(0, UMA_MODEL_DIR)

from uma import UMA, State  # noqa: E402
from models import all_rules  # noqa: E402


def compute_correct(prob: str) -> int:
    import re
    m = re.fullmatch(r"\s*(-?\d+)\s*([+\-*])\s*(-?\d+)\s*", prob)
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    return {'+': a + b, '-': a - b, '*': a * b}[op]


def load_problems():
    add = pd.read_csv(os.path.join(TEST_DIR, 'sd_add.csv'))
    mul = pd.read_csv(os.path.join(TEST_DIR, 'sd_mul.csv'))
    rows = []
    for _, r in add.iterrows():
        p = str(r['prob']).strip()
        rows.append({'prob': p, 'set': 'sd_add', 'key': compute_correct(p)})
    for _, r in mul.iterrows():
        p = str(r['prob']).strip()
        rows.append({'prob': p, 'set': 'sd_mul', 'key': compute_correct(p)})
    return rows


def get_answer(trace):
    if not trace:
        return None
    final_state = trace[-1][0]
    if not getattr(final_state, 'ws', None):
        return None
    try:
        answer = final_state.ws[0].features['answer']
        if answer is None:
            return None
        return str(answer)
    except Exception:
        return None


def load_student(subjid):
    fpath = os.path.join(MODEL_OUTPUT_DIR, f"sim subjid_{subjid} grade_6 model.xlsx")
    if not os.path.exists(fpath):
        return None
    subj_df = pd.read_excel(fpath, sheet_name='subj', index_col=0)
    proc_df = pd.read_excel(fpath, sheet_name='proc', index_col=0)
    ans_df = pd.read_excel(fpath, sheet_name='ans', index_col=0)
    params = json.loads(subj_df.loc[0, 'params'])
    return params, proc_df, ans_df


_PROBLEMS = None


def _init_worker(problems):
    global _PROBLEMS
    _PROBLEMS = problems
    os.chdir(UMA_MODEL_DIR)


def _run_one_student(subjid):
    data = load_student(subjid)
    if data is None:
        return subjid, []
    params, proc_df, ans_df = data
    os.chdir(UMA_MODEL_DIR)
    model = UMA(rules=all_rules, params=params)
    model.proc_mem = proc_df.copy()
    model.ans_mem = ans_df.copy()
    rows = []
    for prob_info in _PROBLEMS:
        try:
            model.run(State(prob_info['prob']), learn=False, verbose=False)
            ans = get_answer(model.trace)
            try:
                ans_int = int(ans) if ans is not None else None
            except (TypeError, ValueError):
                ans_int = None
            rows.append({
                'subjid': subjid,
                'prob': prob_info['prob'],
                'set': prob_info['set'],
                'key': prob_info['key'],
                'answer': ans_int,
                'is_correct': bool(ans_int is not None and ans_int == prob_info['key']),
            })
            model.trace = []
        except Exception:
            rows.append({
                'subjid': subjid, 'prob': prob_info['prob'], 'set': prob_info['set'],
                'key': prob_info['key'], 'answer': None, 'is_correct': False,
            })
    return subjid, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=59)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    problems = load_problems()
    print(f"Loaded {len(problems)} test problems "
          f"({sum(1 for p in problems if p['set']=='sd_add')} add, "
          f"{sum(1 for p in problems if p['set']=='sd_mul')} mul)")

    subjids = list(range(args.start, args.end + 1))
    print(f"Running on subjids {args.start}..{args.end} ({len(subjids)} students) "
          f"× {len(problems)} problems = {len(subjids)*len(problems)} trials")

    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(problems,)) as ex:
        futures = {ex.submit(_run_one_student, s): s for s in subjids}
        done = 0
        for fut in as_completed(futures):
            sid, rows = fut.result()
            all_rows.extend(rows)
            done += 1
            if done % 5 == 0 or done == len(subjids):
                print(f"  done {done}/{len(subjids)} students")

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows -> {args.out}")
    print(f"\nUMA accuracy by set: " + str(df.groupby('set')['is_correct'].mean().round(3).to_dict()))
    print(f"UMA overall accuracy: {df['is_correct'].mean():.3f}")


if __name__ == '__main__':
    main()
