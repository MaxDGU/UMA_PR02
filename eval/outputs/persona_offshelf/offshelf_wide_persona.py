#!/usr/bin/env python3
"""Wide-persona sweep on the OFF-THE-SHELF Qwen3-4B models (no distillation, no
human fine-tuning): can persona prompting alone bring a raw model to the human
accuracy profile, without our two-stage pipeline?

Same 15 personas as the fine-tuned-checkpoint sweep (wide_persona_profctrl.py).
Base uses a bare-text prompt; Instruct uses its native chat template. Both get
the frontier-protocol tail ("Show your work ... final answer.") and are scored
with the region candidate parser used for the frontier persona sweep.
"""
from __future__ import annotations
import argparse, re
from fractions import Fraction
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_HF = "Qwen/Qwen3-4B-Base"
INST_HF = "Qwen/Qwen3-4B-Instruct-2507"
FRAC_PROBS_SRC = "/scratch/gpfs/GRIFFITHS/mg7411/.claude/jobs/5702f84f/UMA_PR02/eval/outputs/fraction_4b/qwen3_4b_base.csv"

PERSONAS = {
    "barely_started":    "I am a young student who has barely started learning fractions and usually gets them wrong.",
    "very_struggling":   "I am a student who struggles badly with fractions and rarely gets the right answer.",
    "struggling_5th":    "I am a struggling 5th grader who is just beginning to learn fraction arithmetic.",
    "struggling_6th":    "I am a struggling 6th grader who hasn't yet mastered finding a common denominator.",
    "below_average_6th": "I am a below-average 6th grader solving fraction arithmetic problems.",
    "average_6th":       "I am an average 6th grader solving fraction arithmetic problems.",
    "average_7th":       "I am an average 7th grader solving fraction arithmetic problems.",
    "above_average_7th": "I am an above-average 7th grader who is fairly comfortable with fractions.",
    "solid_8th":         "I am a solid 8th grader who understands fraction arithmetic well.",
    "high_achieving_8th":"I am a high-achieving 8th grader solving fraction arithmetic problems.",
    "top_student":       "I am a top math student who almost always solves fraction problems correctly.",
    "math_expert":       "I am a mathematics expert who never makes mistakes on fraction arithmetic.",
    "err_indep":         "I am a student who operates on numerators and denominators independently when adding or subtracting fractions.",
    "err_no_kcf":        "I am a student who often forgets the keep-change-flip rule when dividing fractions.",
    "err_cross":         "I am a student who sometimes cross-multiplies the wrong way when working with fractions.",
}

TAIL = "Show your work in a few short sentences and give your final answer. Do not use a calculator."
N_SAMPLES = 60
BATCH = 32
MAX_NEW = 256


def to_frac(s):
    if s is None:
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


def extract_candidates(text):
    """Region candidate parser (same family as the frontier persona scoring)."""
    if not isinstance(text, str):
        return set()
    m_fa = list(re.finditer(r"(?:final\s+answer|answer)[:\s]", text, re.I))
    m_bx = list(re.finditer(r"\\boxed", text))
    starts = [m[-1].start() for m in (m_fa, m_bx) if m]
    region = text[max(starts):] if starts else text[-200:]
    cs = set()
    for m in re.finditer(r"\\d?frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}", region):
        n, d = map(int, m.groups())
        if d:
            cs.add(Fraction(n, d))
    for m in re.finditer(r"(?<!/)(?<!\d)(-?\d+)\s+(\d+)\s*/\s*(\d+)(?!\d)", region):
        w, n, d = map(int, m.groups())
        if d:
            cs.add(Fraction(w * d + (n if w >= 0 else -n), d))
    for m in re.finditer(r"(-?\d+)\s*/\s*(\d+)", region):
        n, d = map(int, m.groups())
        if d:
            cs.add(Fraction(n, d))
    for m in re.finditer(r"(?<![\d./{])(-?\d+\.\d+)(?![\d./}])", region):
        try:
            cs.add(Fraction(float(m.group(1))).limit_denominator(10000))
        except Exception:
            pass
    return cs


def build_prompts(tok, use_chat, pdesc, problems):
    body = [f"{pdesc}\nSolve this fraction problem: {p}=? {TAIL}" for p in problems]
    if not use_chat:
        return body
    return [tok.apply_chat_template([{"role": "user", "content": b}],
                                    tokenize=False, add_generation_prompt=True)
            for b in body]


def run_one(slug, hf_id, use_chat, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"\n>>> loading {hf_id} (off-the-shelf, chat={use_chat})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype).to(device).eval()

    probs = (pd.read_csv(FRAC_PROBS_SRC)[["problem", "correct_answer", "operation", "denom_type"]]
             .drop_duplicates(subset=["problem"]).reset_index(drop=True))
    trues = [to_frac(str(a)) for a in probs["correct_answer"]]

    rows_all, summary = [], []
    for pslug, pdesc in PERSONAS.items():
        prompts = build_prompts(tok, use_chat, pdesc, probs["problem"])
        print(f"\n=== {slug} / {pslug} ===", flush=True)
        outs = [[] for _ in prompts]
        for r in range(N_SAMPLES):
            if r % 20 == 0:
                print(f"    round {r}/{N_SAMPLES}", flush=True)
            for start in range(0, len(prompts), BATCH):
                b = prompts[start: start + BATCH]
                enc = tok(b, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                                         max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for i, t in enumerate(texts):
                    outs[start + i].append(t)
        n_ok = 0
        for i in range(len(probs)):
            for s, resp in enumerate(outs[i]):
                ok = int(trues[i] is not None and trues[i] in extract_candidates(resp))
                n_ok += ok
                rows_all.append({
                    "model": slug, "persona": pslug,
                    "problem": probs.loc[i, "problem"],
                    "operation": probs.loc[i, "operation"],
                    "denom_type": probs.loc[i, "denom_type"],
                    "correct_answer": probs.loc[i, "correct_answer"],
                    "sample_idx": s, "is_correct": ok, "model_response": resp,
                })
        acc = n_ok / (len(probs) * N_SAMPLES)
        summary.append({"model": slug, "persona": pslug, "acc": acc})
        print(f"  {pslug} acc = {acc*100:.1f}%", flush=True)
        pd.DataFrame(summary).to_csv(out_dir / f"offshelf_persona_summary_{slug}.csv", index=False)
        pd.DataFrame(rows_all).to_csv(out_dir / f"offshelf_persona_rollouts_{slug}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", choices=["base_4b_offshelf", "instruct_4bi_offshelf"], required=True)
    ap.add_argument("--only", default="", help="comma-separated persona slugs to run")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.only:
        keep = set(args.only.split(","))
        for k in list(PERSONAS):
            if k not in keep:
                del PERSONAS[k]
    if args.model == "base_4b_offshelf":
        run_one("base_4b_offshelf", BASE_HF, False, out_dir)
    else:
        run_one("instruct_4bi_offshelf", INST_HF, True, out_dir)


if __name__ == "__main__":
    main()
