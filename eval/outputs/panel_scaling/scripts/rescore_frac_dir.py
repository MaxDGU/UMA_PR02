#!/usr/bin/env python3
"""Re-score fraction CSVs in a directory with the mixed-number-aware final-answer
parser. Updates is_correct in place; backs up originals to *_v1.csv once."""
import sys, re, shutil
from pathlib import Path
from fractions import Fraction
import pandas as pd


def extract_final_answer(text):
    if not isinstance(text, str):
        return set()
    m_fa = list(re.finditer(r"(?:final\s+answer|answer)[:\s]", text, re.I))
    m_bx = list(re.finditer(r"\\boxed", text))
    m_mk = list(re.finditer(r"###\s*answer:", text, re.I))
    starts = [m[-1].start() for m in (m_fa, m_bx, m_mk) if m]
    region = text[max(starts):] if starts else text[-200:]
    cs = set()
    for m in re.finditer(r"\\boxed\{[^{}]*(-?\d+)\s*\\d?frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}[^{}]*\}", region):
        w, n, d = map(int, m.groups())
        if d: cs.add(Fraction(w*d + (n if w >= 0 else -n), d))
    for m in re.finditer(r"\\d?frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}", region):
        n, d = map(int, m.groups())
        if d: cs.add(Fraction(n, d))
    for m in re.finditer(r"(-?\d+)\s*\\d?frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}", region):
        w, n, d = map(int, m.groups())
        if d: cs.add(Fraction(w*d + (n if w >= 0 else -n), d))
    for m in re.finditer(r"(?<!/)(?<!\d)(-?\d+)\s+(\d+)\s*/\s*(\d+)(?!\d)", region):
        w, n, d = map(int, m.groups())
        if d: cs.add(Fraction(w*d + (n if w >= 0 else -n), d))
    for m in re.finditer(r"(-?\d+)\s*/\s*(\d+)", region):
        n, d = map(int, m.groups())
        if d: cs.add(Fraction(n, d))
    for m in re.finditer(r"(?<![\d./{])(-?\d+\.\d+)(?![\d./}])", region):
        try: cs.add(Fraction(float(m.group(1))).limit_denominator(10000))
        except Exception: pass
    return cs


def to_frac(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if not s:
        return None
    if "/" in s:
        try:
            n, d = s.split("/", 1)
            return Fraction(int(float(n)), int(float(d)))
        except Exception:
            pass
    try:
        return Fraction(float(s)).limit_denominator(10000)
    except Exception:
        return None


def main():
    d = Path(sys.argv[1])
    csvs = sorted(d.glob("*.csv"))
    for src in csvs:
        df = pd.read_csv(src)
        if "model_response" not in df.columns or "correct_answer" not in df.columns:
            continue
        bak = src.with_name(src.stem + "_v1.csv")
        if not bak.exists():
            shutil.copy(src, bak)
        old = df["is_correct"].mean() * 100 if "is_correct" in df.columns else float("nan")
        correct = df["correct_answer"].apply(to_frac)
        new_ok = [int(c is not None and c in extract_final_answer(r))
                  for r, c in zip(df["model_response"], correct)]
        df["is_correct"] = new_ok
        df.to_csv(src, index=False)
        print(f"{src.name:44s} old={old:5.1f}% new={sum(new_ok)/len(new_ok)*100:5.1f}%")


if __name__ == "__main__":
    main()
