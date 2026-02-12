#!/usr/bin/env python3
"""
Replicate UMA simulation from Braithwaite & Siegler (2018).

Phase 1: Paper Replication
- Train 1000 students on FULL GoMath curriculum (3,222 problems across grades 1-6)
  * Whole numbers: arithmetic facts and multidigit operations
  * Fractions: 405 problems
  * Decimals: 307 problems
- Test on 16 Siegler & Pyke 2013 problems
- Verify accuracy matches paper (~60% add, ~70% sub, ~45% mul, ~20% div)

Phase 2: Synthetic Trace Generation
- Test trained students on 1000 synthetic problems (250 per operation)
- Generate ~1M traces for transformer training

Parameter grid (1000 students):
- g:     [.01, .02, .03, .04, .05, .06, .07, .08, .09, .10]  (10 values)
- d:     [.1, .3, .5, .7, .9]                                 (5 values)
- c:     [5]                                                  (1 value)
- rt_mu: [3, 4, 5, 6]                                         (4 values)
- rt_sd: [1]                                                  (1 value)
- ice:   [0, 25, 50, 75, 100]                                 (5 values)
Total: 10 x 5 x 4 x 5 = 1000 unique students

Output:
- uma_paper_replication.csv (16,000 traces - paper verification)
- uma_synthetic_traces.csv (~1,000,000 traces - transformer training)
"""

import os
import sys
import gc
import io
import contextlib
import warnings

import numpy as np
import pandas as pd
from fractions import Fraction

warnings.filterwarnings('ignore')

# Paths - relative to repo structure: UMA_PR02/results/UMA_replication/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
UMA_MODEL_DIR = os.path.join(REPO_ROOT, '1. Model')
PAPER_OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'uma_paper_replication.csv')
SYNTHETIC_OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'uma_synthetic_traces.csv')

# Change to UMA directory and import
os.chdir(UMA_MODEL_DIR)
sys.path.insert(0, UMA_MODEL_DIR)

from uma import State, UMA, Problem, myEval
from models import RA_rules, R_acc_once
from simulators import Course, Cohort

os.chdir(SCRIPT_DIR)

print(f"Loaded {len(RA_rules)} UMA rules")


# =============================================================================
# Helper Functions
# =============================================================================

@contextlib.contextmanager
def suppress_output():
    """Suppress stdout/stderr during UMA operations."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


# Rule classification (from cot_generator.py)
SKIP_RULES = {'deferred_action', 'pop_goal', 'finish_problem', 'larger_first_add_mul', 'cannot_simplify'}

GOAL_RULES = {
    'operate_nums', 'operate_dens', 'pass_den',
    'convert_CD', 'convert_CD_LCM', 'get_LCM',
    'invert_op2', 'div_to_mul',
    'check_simplify', 'skip_simplify', 'get_GCD', 'simplify_fraction',
    'convert_fra_to_den',
}

EXEC_RULES = {
    'add_fact', 'sub_fact', 'mul_fact', 'div_calculator',
    'convert_CD_omit_nums', 'invert_rand', 'invert_fail',
    'div_to_mul_denied', 'acc_skip', 'acc_extra',
    'sub_LbS', 'div_LbS', 'div_LbS_drop_rem', 'div_drop_rem',
}


def get_strategy(trace):
    """Extract main strategy from UMA trace."""
    strategies = ['KDON_AS', 'KDON_OG', 'CDON_AS', 'CDON_OG',
                  'ONOD_M', 'ONOD_OG', 'CROP_M', 'ICDM_D', 'ICDM_OG']
    for state, rule in trace:
        if rule and rule.name in strategies:
            return rule.name
    return 'OTHER'


def get_rules(trace):
    """Extract goal-level and execution-level rules from trace."""
    goals, execs = [], []
    for state, rule in trace:
        if rule is None:
            continue
        name = rule.name
        if name in SKIP_RULES:
            continue
        elif name in GOAL_RULES:
            goals.append(name)
        elif name in EXEC_RULES:
            execs.append(name)
    return goals, execs


def get_answer(trace):
    """Extract final answer from UMA trace as string.

    Follows the pattern from simulators.py:107-114.
    """
    if not trace:
        return '?'

    final_state = trace[-1][0]
    if not final_state.ws:
        return '?'

    # Access answer directly from first workspace item (the Problem being solved)
    try:
        answer = final_state.ws[0].features['answer']
        if answer is None:
            return '?'

        subtype = answer.features.get('subtype')

        if subtype == 'fraction':
            num = answer.features.get('num')
            den = answer.features.get('den')
            if num is not None and den is not None:
                return f"{str(num)}/{str(den)}"
        elif subtype in ['whole', 'negint']:
            return str(answer)
        elif subtype == 'decimal':
            return str(answer)
        else:
            # Fallback: try to convert to string
            return str(answer)
    except Exception:
        return '?'

    return '?'


def get_op(prob):
    """Extract operation from problem string."""
    prob = str(prob)
    if '*' in prob:
        return '*'
    if ':' in prob:
        return ':'
    if '+' in prob:
        return '+'
    if '-' in prob:
        return '-'
    return '?'


def compute_correct(prob):
    """Compute correct answer for a fraction problem."""
    try:
        # Use myEval to get numeric answer
        val = myEval(prob)
        # Convert to simplified fraction
        frac = Fraction(val).limit_denominator(1000)
        if frac.denominator == 1:
            return str(frac.numerator)
        return f"{frac.numerator}/{frac.denominator}"
    except Exception:
        return '?'


def answers_match(uma_ans, correct_ans):
    """Check if UMA answer matches correct answer (handles equivalence)."""
    if uma_ans == '?' or correct_ans == '?':
        return False
    try:
        # Parse both as fractions and compare values
        if '/' in uma_ans:
            parts = uma_ans.split('/')
            uma_val = float(parts[0]) / float(parts[1])
        else:
            uma_val = float(uma_ans)

        if '/' in correct_ans:
            parts = correct_ans.split('/')
            correct_val = float(parts[0]) / float(parts[1])
        else:
            correct_val = float(correct_ans)

        return abs(uma_val - correct_val) < 1e-6
    except Exception:
        return False


def generate_synthetic_problems(n_per_op=250, seed=42):
    """Generate synthetic fraction problems for each operation.

    Args:
        n_per_op: Number of problems per operation (will generate 4x this many total)
        seed: Random seed for reproducibility

    Returns:
        List of problem strings (n_per_op * 4 problems)
    """
    np.random.seed(seed)
    problems = []
    denoms = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]

    for _ in range(n_per_op):
        # Random fractions a/b op c/d
        a = np.random.randint(1, 10)
        b = np.random.choice(denoms)
        c = np.random.randint(1, 10)
        d = np.random.choice(denoms)

        problems.append(f"{a}/{b}+{c}/{d}")  # Addition
        problems.append(f"{a}/{b}-{c}/{d}")  # Subtraction
        problems.append(f"{a}/{b}*{c}/{d}")  # Multiplication
        problems.append(f"{a}/{b}:{c}/{d}")  # Division

    return problems  # 4 * n_per_op problems


# =============================================================================
# Main Replication
# =============================================================================

def main():
    print("=" * 70)
    print("UMA PAPER REPLICATION & SYNTHETIC TRACE GENERATION")
    print("Braithwaite & Siegler (2018)")
    print("=" * 70)

    # Parameter grid (exactly as in paper)
    params_sets = {
        'g':     [.01, .02, .03, .04, .05, .06, .07, .08, .09, .10],
        'd':     [.1, .3, .5, .7, .9],
        'c':     [5],
        'rt_mu': [3, 4, 5, 6],
        'rt_sd': [1],
        'ice':   [0, 25, 50, 75, 100]
    }

    # Create cohort of 1000 students
    os.chdir(UMA_MODEL_DIR)
    cohort = Cohort(rules=RA_rules, params_sets=params_sets, N=1)
    print(f"\nCreated cohort of {len(cohort.students)} students")

    # Create curriculum course (full GoMath grades 1-6: whole numbers, fractions, decimals)
    course = Course('go_math', mode='FIRST_COURSE')
    print(f"Curriculum: {len(course.P)} training problems")

    # Load sp2013 test problems (16 problems)
    test_df = pd.read_csv('Problem Sets Testing/sp2013.csv')
    sp2013_probs = test_df['prob'].tolist()
    print(f"SP2013 test problems: {len(sp2013_probs)}")

    os.chdir(SCRIPT_DIR)

    # Generate synthetic problems (1000 problems: 250 per operation)
    synthetic_probs = generate_synthetic_problems(n_per_op=250, seed=42)
    print(f"Synthetic test problems: {len(synthetic_probs)}")

    # Pre-compute correct answers
    sp2013_correct = {prob: compute_correct(prob) for prob in sp2013_probs}
    synthetic_correct = {prob: compute_correct(prob) for prob in synthetic_probs}

    print("\nCorrect answers for sp2013:")
    for prob, ans in sp2013_correct.items():
        print(f"  {prob} = {ans}")

    # Expected trace counts
    print(f"\nExpected traces:")
    print(f"  Paper replication (sp2013): {len(cohort.students)} x {len(sp2013_probs)} = {len(cohort.students) * len(sp2013_probs)}")
    print(f"  Synthetic: {len(cohort.students)} x {len(synthetic_probs)} = {len(cohort.students) * len(synthetic_probs)}")

    # Initialize output files (remove existing)
    for path in [PAPER_OUTPUT_PATH, SYNTHETIC_OUTPUT_PATH]:
        if os.path.exists(path):
            os.remove(path)

    # Tracking variables
    paper_results = []
    synthetic_results = []
    paper_total = 0
    synthetic_total = 0
    paper_first_write = True
    synthetic_first_write = True

    print(f"\n{'=' * 70}")
    print("TRAINING AND TESTING")
    print(f"{'=' * 70}\n")

    for subjid in range(len(cohort.students)):
        student = cohort.students[subjid]
        params = cohort.params[subjid]

        # Train on GoMath curriculum
        os.chdir(UMA_MODEL_DIR)
        try:
            with suppress_output():
                course.trainModel(student)
        except Exception as e:
            print(f"  Warning: Training failed for student {subjid}: {e}")
            os.chdir(SCRIPT_DIR)
            continue

        # === Phase 1: Test on sp2013 (paper replication) ===
        for prob in sp2013_probs:
            try:
                with suppress_output():
                    student.run(State(prob), learn=False, verbose=False)

                if not student.trace:
                    student.trace = []
                    continue

                strategy = get_strategy(student.trace)
                goals, execs = get_rules(student.trace)
                answer = get_answer(student.trace)
                correct = sp2013_correct[prob]
                is_correct = answers_match(answer, correct)

                paper_results.append({
                    'subjid': subjid,
                    'prob': prob,
                    'operation': get_op(prob),
                    'strategy': strategy,
                    'goals': ' '.join(goals),
                    'exec': ' '.join(execs),
                    'answer': answer,
                    'correct': correct,
                    'is_correct': is_correct,
                    'g': params['g'],
                    'd': params['d'],
                    'rt_mu': params['rt_mu'],
                    'ice': params['ice']
                })

            except Exception:
                pass

            student.trace = []

        # === Phase 2: Test on synthetic problems ===
        for prob in synthetic_probs:
            try:
                with suppress_output():
                    student.run(State(prob), learn=False, verbose=False)

                if not student.trace:
                    student.trace = []
                    continue

                strategy = get_strategy(student.trace)
                goals, execs = get_rules(student.trace)
                answer = get_answer(student.trace)
                correct = synthetic_correct[prob]
                is_correct = answers_match(answer, correct)

                # Only save if we got a valid answer
                if answer != '?' and answer != 'FRACTION':
                    synthetic_results.append({
                        'subjid': subjid,
                        'prob': prob,
                        'operation': get_op(prob),
                        'strategy': strategy,
                        'goals': ' '.join(goals),
                        'exec': ' '.join(execs),
                        'answer': answer,
                        'correct': correct,
                        'is_correct': is_correct,
                        'g': params['g'],
                        'd': params['d'],
                        'rt_mu': params['rt_mu'],
                        'ice': params['ice']
                    })

            except Exception:
                pass

            student.trace = []

        os.chdir(SCRIPT_DIR)

        # Clear student's memory after testing
        cohort.students[subjid] = None
        gc.collect()

        # Incremental save every 20 students (memory efficient)
        if (subjid + 1) % 20 == 0:
            # Save paper results
            if paper_results:
                df_batch = pd.DataFrame(paper_results)
                df_batch.to_csv(PAPER_OUTPUT_PATH, mode='a', index=False, header=paper_first_write)
                paper_total += len(paper_results)
                paper_first_write = False
                paper_results = []

            # Save synthetic results
            if synthetic_results:
                df_batch = pd.DataFrame(synthetic_results)
                df_batch.to_csv(SYNTHETIC_OUTPUT_PATH, mode='a', index=False, header=synthetic_first_write)
                synthetic_total += len(synthetic_results)
                synthetic_first_write = False
                synthetic_results = []

            gc.collect()

        # Progress updates every 100 students
        if (subjid + 1) % 100 == 0:
            print(f"  [{subjid + 1}/1000] students complete | "
                  f"paper: {paper_total} | synthetic: {synthetic_total}")

    # Final batch save
    if paper_results:
        df_batch = pd.DataFrame(paper_results)
        df_batch.to_csv(PAPER_OUTPUT_PATH, mode='a', index=False, header=paper_first_write)
        paper_total += len(paper_results)

    if synthetic_results:
        df_batch = pd.DataFrame(synthetic_results)
        df_batch.to_csv(SYNTHETIC_OUTPUT_PATH, mode='a', index=False, header=synthetic_first_write)
        synthetic_total += len(synthetic_results)

    gc.collect()

    # ==========================================================================
    # Summary Statistics
    # ==========================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 1: PAPER REPLICATION RESULTS (sp2013)")
    print(f"{'=' * 70}")

    paper_df = pd.read_csv(PAPER_OUTPUT_PATH)
    print(f"\nTotal traces: {len(paper_df)}")
    print(f"Expected: 16,000 (1000 students x 16 problems)")

    # Accuracy by operation
    print(f"\nAccuracy by operation:")
    op_names = {'+': 'Addition', '-': 'Subtraction', '*': 'Multiplication', ':': 'Division'}
    for op in ['+', '-', '*', ':']:
        op_df = paper_df[paper_df['operation'] == op]
        if len(op_df) > 0:
            acc = op_df['is_correct'].mean() * 100
            print(f"  {op_names[op]:15s}: {acc:5.1f}% ({op_df['is_correct'].sum()}/{len(op_df)})")

    # Overall accuracy
    overall_acc = paper_df['is_correct'].mean() * 100
    print(f"\n  Overall:        {overall_acc:.1f}%")

    # Strategy distribution
    print(f"\nStrategy distribution:")
    strat_counts = paper_df['strategy'].value_counts()
    for strat, count in strat_counts.items():
        pct = count / len(paper_df) * 100
        print(f"  {strat:15s}: {count:5d} ({pct:5.1f}%)")

    # Strategy by operation
    print(f"\nStrategy by operation:")
    for op in ['+', '-', '*', ':']:
        op_df = paper_df[paper_df['operation'] == op]
        if len(op_df) > 0:
            top_strat = op_df['strategy'].value_counts().head(3)
            print(f"  {op}:")
            for strat, count in top_strat.items():
                pct = count / len(op_df) * 100
                print(f"    {strat:15s}: {pct:5.1f}%")

    print(f"\nSaved to: {PAPER_OUTPUT_PATH}")

    # ==========================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 2: SYNTHETIC TRACE GENERATION")
    print(f"{'=' * 70}")

    synthetic_df = pd.read_csv(SYNTHETIC_OUTPUT_PATH)
    print(f"\nTotal traces: {len(synthetic_df)}")
    print(f"Expected: ~1,000,000 (1000 students x 1000 problems)")

    # Operation distribution
    print(f"\nOperation distribution:")
    op_counts = synthetic_df['operation'].value_counts()
    for op, count in op_counts.items():
        pct = count / len(synthetic_df) * 100
        print(f"  {op_names.get(op, op):15s}: {count:7d} ({pct:5.1f}%)")

    # Accuracy by operation
    print(f"\nAccuracy by operation:")
    for op in ['+', '-', '*', ':']:
        op_df = synthetic_df[synthetic_df['operation'] == op]
        if len(op_df) > 0:
            acc = op_df['is_correct'].mean() * 100
            print(f"  {op_names[op]:15s}: {acc:5.1f}% ({op_df['is_correct'].sum()}/{len(op_df)})")

    # Strategy distribution
    print(f"\nStrategy distribution:")
    strat_counts = synthetic_df['strategy'].value_counts()
    for strat, count in strat_counts.head(10).items():
        pct = count / len(synthetic_df) * 100
        print(f"  {strat:15s}: {count:7d} ({pct:5.1f}%)")

    print(f"\nSaved to: {SYNTHETIC_OUTPUT_PATH}")

    print(f"\n{'=' * 70}")
    print("SIMULATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
