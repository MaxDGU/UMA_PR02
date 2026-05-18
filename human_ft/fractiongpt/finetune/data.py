"""Human data loading, answer normalization, and reward computation."""

import math
import pandas as pd
from fractions import Fraction


_TASK = "fractions"


def set_task(task):
    """Configure module-level task switch. Routes normalize_answer and load_human_data."""
    global _TASK
    if task not in ("fractions", "decimals"):
        raise ValueError(f"Unknown task: {task!r}")
    _TASK = task


def normalize_answer(ans_str):
    """Dispatch to the task-specific answer normalizer set via set_task()."""
    if _TASK == "decimals":
        return _normalize_answer_decimal(ans_str)
    return _normalize_answer_fraction(ans_str)


def _normalize_answer_fraction(ans_str):
    """Normalize a human or model answer to a canonical fraction surface form.

    Preserves the numerator/denominator as written (no reduction to lowest terms)
    because for cognitive modeling, 5/10 and 1/2 are behaviorally distinct answers.
    Only handles formatting noise: whitespace, mixed numbers -> improper fractions,
    decimals-in-fractions -> integer num/den.

    Returns a/b string or None for unparseable answers.
    """
    s = str(ans_str).strip()
    if not s or s == "nan" or "?" in s:
        return None

    # Mixed number: "1 1/5" -> convert to improper fraction preserving denominator
    if " " in s and "/" in s:
        parts = s.split()
        if len(parts) == 2:
            try:
                whole = int(float(parts[0]))
                frac_parts = parts[1].split("/")
                num, den = int(float(frac_parts[0])), int(float(frac_parts[1]))
                if den == 0:
                    return None
                # w a/b -> (w*b + a)/b, preserving denominator
                return f"{whole * den + num}/{den}"
            except (ValueError, ZeroDivisionError, IndexError):
                return None

    # Plain fraction a/b (possibly with decimals like 1.3/5)
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0])
                den = float(parts[1])
                if den == 0:
                    return None
                # If both are integers, preserve as-is
                if num == int(num) and den == int(den):
                    return f"{int(num)}/{int(den)}"
                # Decimals in fractions (e.g. 1.3/5): round to nearest integer
                return f"{round(num)}/{round(den)}"
            except (ValueError, ZeroDivisionError):
                return None

    # Whole number -> n/1
    try:
        val = float(s)
        if val == int(val):
            return f"{int(val)}/1"
        return f"{round(val)}/1"
    except ValueError:
        return None


def _normalize_answer_decimal(ans_str, precision=4):
    """Canonicalize a decimal answer string.

    Output is a stripped fixed-precision decimal: trailing zeros and trailing
    decimal point are dropped, so 0.1271 stays "0.1271", 6.56 stays "6.56", and
    191.0 collapses to "191". Fraction surface forms ("2/1") are folded to the
    decimal value. Returns None for unparseable inputs.
    """
    s = str(ans_str).strip()
    if not s or s.lower() == "nan" or "?" in s:
        return None

    try:
        val = float(s)
    except ValueError:
        # Fraction surface form like "2/1" or "3/2" — fold to decimal
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2:
                try:
                    num, den = float(parts[0]), float(parts[1])
                    if den == 0:
                        return None
                    val = num / den
                except ValueError:
                    return None
            else:
                return None
        else:
            return None

    if not math.isfinite(val):
        return None

    out = f"{val:.{precision}f}"
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return out if out else "0"


def normalize_prob_format(prob, source):
    """Ensure problem string uses UMA notation (: for division)."""
    return str(prob).strip()


def _load_one_dataset(df, prob_col, resp_col, source_name):
    """Accumulate per-problem answer counts from a dataframe.

    Returns:
        prob_counts: dict mapping prob_key -> {answer: count}
        prob_N: dict mapping prob_key -> total responses (including unparseable)
    """
    prob_counts = {}
    prob_N = {}
    for prob in df[prob_col].unique():
        sub = df[df[prob_col] == prob]
        prob_key = str(prob).strip()
        prob_N[prob_key] = len(sub)
        counts = {}
        for _, row in sub.iterrows():
            norm = normalize_answer(row[resp_col])
            if norm is None:
                continue
            counts[norm] = counts.get(norm, 0) + 1
        prob_counts[prob_key] = counts
    return prob_counts, prob_N


def load_human_data(cfg):
    """Load human data and build a reward table. Routes by cfg['task'].

    Also sets the module-level task so normalize_answer dispatches correctly
    for the rest of the pipeline.

    Returns:
        reward_table: dict mapping (problem, normalized_answer) -> frequency
        problems: dict mapping problem -> {"N": int, "source": str}
        raw_data: dict with per-problem answer distributions for evaluation
    """
    task = cfg.get("task", "fractions")
    set_task(task)
    if task == "decimals":
        return _load_decimal_data(cfg)
    return _load_fraction_data(cfg)


def _load_fraction_data(cfg):
    """Load Siegler 2011 + SP2013 fraction human responses."""
    reward_table = {}
    problems = {}
    raw_data = {}

    datasets = [
        (pd.read_csv(cfg["siegler2011_path"], encoding="latin-1"),
         "prob", "resp", "siegler2011"),
        (pd.read_csv(cfg["sp2013_path"]),
         "prob", "resp", "sp2013"),
    ]

    for df, prob_col, resp_col, source in datasets:
        prob_counts, prob_N = _load_one_dataset(df, prob_col, resp_col, source)
        for prob_key, counts in prob_counts.items():
            N = prob_N[prob_key]
            problems[prob_key] = {"N": N, "source": source}
            freqs = {ans: cnt / N for ans, cnt in counts.items()}
            raw_data[prob_key] = {"source": source, "N": N, "answers": freqs}
            for ans, freq in freqs.items():
                reward_table[(prob_key, ans)] = freq

    return reward_table, problems, raw_data


def _load_decimal_data(cfg):
    """Load BSS2021 decimal human responses.

    Problems in cfg['held_out_problems'] get source='bss2021_held_out';
    the rest get source='bss2021_train'. evaluate_model groups results by
    source so the held-out set is reported as a separate generalization metric.
    The caller (e.g. sft_reweight.train_sft) is responsible for excluding
    held-out problems from the SFT reweighting signal.
    """
    bss_path = cfg["bss2021_path"]
    df = pd.read_csv(bss_path)
    held_out = set(str(p).strip() for p in (cfg.get("held_out_problems") or []))

    reward_table = {}
    problems = {}
    raw_data = {}

    prob_counts, prob_N = _load_one_dataset(df, "prob", "resp", "bss2021")
    for prob_key, counts in prob_counts.items():
        N = prob_N[prob_key]
        source = "bss2021_held_out" if prob_key in held_out else "bss2021_train"
        problems[prob_key] = {"N": N, "source": source}
        freqs = {ans: cnt / N for ans, cnt in counts.items()}
        raw_data[prob_key] = {"source": source, "N": N, "answers": freqs}
        for ans, freq in freqs.items():
            reward_table[(prob_key, ans)] = freq

    return reward_table, problems, raw_data


def get_reward(prob, model_answer, reward_table):
    """Look up reward for a (problem, answer) pair. Returns 0 if unseen."""
    norm = normalize_answer(model_answer)
    if norm is None:
        return 0.0
    return reward_table.get((prob, norm), 0.0)
