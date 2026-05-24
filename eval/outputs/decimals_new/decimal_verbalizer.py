#!/usr/bin/env python3
"""Prototype: synthesize first-person decimal reasoning traces from the coded
BSS2021 human data, to mirror the SP2013 fraction `strategy` think-alouds and
supply a reasoning target for decimal humanFT.

Each trace is built deterministically from the structured fields, so it faithfully
narrates the human's actual strategy and misconception (not a fabricated one):
  - work_operand_* vs op*  -> did they strip the decimal points?
  - operation + strat_add/strat_mul -> which procedure (incl. overgeneralized add)
  - dplace (Add/Mul/None)  -> how they placed the decimal point
  - strat_conf / conc_*    -> conceptual error language
"""
import argparse, random, re
import numpy as np, pandas as pd

def fmt(x):
    if pd.isna(x): return None
    s = f"{float(x):g}"
    return s

def stripped(op, work):
    """True if the written operand drops the decimal (e.g. 0.41 -> 41)."""
    if pd.isna(op) or pd.isna(work): return False
    if float(op) == float(work): return False
    digits = str(op).replace('.', '').lstrip('0') or '0'
    return str(int(abs(work))).lstrip('0') == digits.lstrip('0')

def verbalize(row, rng):
    op = row['operation']; o1, o2 = fmt(row['op1']), fmt(row['op2'])
    w1, w2, wa = fmt(row['work_operand_1']), fmt(row['work_operand_2']), row.get('work_answer')
    resp = row['resp']; dplace = row['dplace']
    verb = "added" if op == 'Add' else "multiplied"
    parts = []

    strip1 = stripped(row['op1'], row['work_operand_1'])
    strip2 = stripped(row['op2'], row['work_operand_2'])
    used_w1, used_w2 = (w1, w2) if (w1 and w2) else (o1, o2)
    did_strip = strip1 or strip2

    # whole-number product (faithful pre-placement value) when decimals stripped.
    # Only narrate it when its digits match the human's final answer, so we never
    # state an intermediate that contradicts their reported result.
    whole_prod = None
    if did_strip and op == 'Mul':
        try:
            wp = str(int(round(float(row['work_operand_1']) * float(row['work_operand_2']))))
            resp_digits = re.sub(r'\D', '', str(resp)).lstrip('0')
            if wp.lstrip('0') == resp_digits:
                whole_prod = wp
        except (TypeError, ValueError):
            whole_prod = None
    got = (str(wa).strip() if (wa is not None and str(wa).strip()) else None)

    # 1) setup / digit handling
    if did_strip:
        prod = f" and got {whole_prod}" if whole_prod else (f" and got {got}" if got else "")
        parts.append(rng.choice([
            f"I ignored the decimal points and {verb} {used_w1} and {used_w2} as whole numbers{prod}",
            (f"I dropped the decimals and {verb} {used_w1} by {used_w2}{prod}" if op=='Mul'
             else f"I dropped the decimals and {verb} {used_w1} and {used_w2}{prod}"),
            f"I treated them like whole numbers, so I {verb} {used_w1} and {used_w2}{prod}",
        ]))
    else:
        if op == 'Add' and row.get('work_alignment') == 3.0:
            base = rng.choice([
                f"I lined up the decimal points and {verb} {used_w1} and {used_w2}",
                f"I stacked them with the decimals aligned and {verb} {used_w1} and {used_w2}",
            ])
        else:
            base = rng.choice([
                f"I {verb} {used_w1} and {used_w2}",
                f"I took {used_w1} and {used_w2} and {verb} them",
            ])
        if got: base += f" and got {got}"
        parts.append(base)

    # 2) decimal-point placement (own sentence, no leading 'and')
    if dplace == 'Mul':
        tail = f" to get {resp}" if (did_strip and whole_prod) else ""
        parts.append(rng.choice([
            f"then I counted the decimal places in both numbers and moved the point over that many spots{tail}",
            f"then I added up the digits after each decimal and placed the point{tail}",
        ]))
    elif dplace == 'Add':
        if op == 'Mul':  # overgeneralized addition rule = the classic error
            parts.append(rng.choice([
                "I put the decimal point straight down like I do for addition",
                "I lined the decimal point up under the others the way I would when adding",
            ]))
        else:
            parts.append("the decimal point lined up straight down")
    elif dplace == 'None':
        parts.append(rng.choice(["I wasn't sure where the decimal point went",
                                  "I left the decimal point where it looked right"]))

    # 3) misconception / confidence color
    if row.get('strat_conf') == 1.0:
        parts.append(rng.choice(["I think I mixed up the steps",
                                 "I wasn't sure if that was the right method"]))
    elif row.get('conc_answer_mag') == 1.0:
        parts.append("it seemed off because multiplying should make the number bigger")
    elif row.get('conc_place_val') == 1.0:
        parts.append("I had trouble with the place values")
    elif row.get('acc') == 1.0 and rng.random() < 0.2:
        parts.append(rng.choice(["that was my answer", "so that was what I got"]))

    sents = [p.strip().rstrip('.') for p in parts if p.strip()]
    sents = [s[0].upper() + s[1:] for s in sents]
    return ". ".join(sents) + "."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/bss2021_human_responses.csv")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(a.in_csv, encoding='latin-1')
    df = df[df['resp'].notna()].copy()
    rng = random.Random(a.seed)
    df['response_nl'] = [verbalize(r, rng) for _, r in df.iterrows()]
    df['target'] = df['response_nl'] + "\n### answer: " + df['resp'].astype(str)
    if a.out_csv:
        df[['subjid','prob','operation','operands','resp','acc','dplace',
            'response_nl','target']].to_csv(a.out_csv, index=False)
        print("wrote", a.out_csv, "rows:", len(df))
    if a.show:
        # show a spread across strategy patterns
        for key, sub in [("Add correct (aligned)", df[(df.operation=='Add')&(df.acc==1)]),
                         ("Mul correct", df[(df.operation=='Mul')&(df.acc==1)]),
                         ("Mul overgeneralized-add error", df[(df.operation=='Mul')&(df.dplace=='Add')]),
                         ("strategy conflation (strat_conf=1)", df[df.strat_conf==1])]:
            print(f"\n### {key}  (n={len(sub)})")
            for _, r in sub.head(a.show).iterrows():
                print(f"  [{r['prob']}={r['resp']} acc={int(r['acc'])}] {r['response_nl']}")

if __name__ == "__main__":
    main()
