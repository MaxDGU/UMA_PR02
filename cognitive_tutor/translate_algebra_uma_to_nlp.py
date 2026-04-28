#!/usr/bin/env python3
"""Translate algebra UMA traces to observable NLP supervision.

The translated dataset deliberately hides symbolic teacher metadata such as
strategy, goals, and execution rules. The model sees only the problem, optional
student parameters, observable step states, and the produced answer.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from fractions import Fraction
from typing import TextIO


DEFAULT_INPUT_CSV = Path("results/UMA_replication/algebra_uma_k100_v2/algebra_uma_master_k100_v2.csv")
DEFAULT_OUTPUT_CSV = Path(
    "results/UMA_replication/algebra_uma_k100_v2/algebra_uma_master_k100_v2_nlp_observable.csv.gz"
)

OUTPUT_COLUMNS = [
    "subjid",
    "prob",
    "equation_type",
    "variable",
    "answer",
    "correct",
    "g",
    "d",
    "c",
    "rt_mu",
    "ice",
    "instruction_nl",
    "response_nl",
]

TEMPLATE_ID_COLUMN = "template_id"
TEMPLATED_OUTPUT_COLUMNS = OUTPUT_COLUMNS + [TEMPLATE_ID_COLUMN]

CONNECTED_TEMPLATES = [
    "step_bridge",
    "attempt_narrative",
    "compact_chain",
    "tutor_log",
]
RICH_TEMPLATES = [
    "worked_solution_rich",
    "first_attempt_rich",
    "student_reasoning_rich",
    "checkpoint_rich",
]
HUMAN_TEMPLATES = [
    "human_scratch",
    "human_uncertain",
    "human_wrong_guess",
    "human_quiet_work",
]
ALL_CONNECTED_TEMPLATES = CONNECTED_TEMPLATES + RICH_TEMPLATES + HUMAN_TEMPLATES
RESPONSE_TEMPLATES = ["plain", *ALL_CONNECTED_TEMPLATES, "mixed", "rich_mixed", "human_mixed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate algebra UMA traces to observable NLP rows.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--response-template",
        choices=RESPONSE_TEMPLATES,
        default="plain",
        help="Surface template for response_nl. The default preserves the original Step N format.",
    )
    parser.add_argument(
        "--expand-templates",
        nargs="+",
        choices=ALL_CONNECTED_TEMPLATES,
        default=None,
        help="Emit one row per listed template. Adds a template_id column.",
    )
    return parser.parse_args()


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def safe_text(value: object) -> str:
    return str(value or "").strip()


def observable_steps(row: dict[str, str]) -> list[str]:
    raw = safe_text(row.get("trace_steps_json"))
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
        steps = [
            safe_text(item.get("state", ""))
            for item in parsed
            if isinstance(item, dict) and safe_text(item.get("state", ""))
        ]
        if steps:
            return steps
    answer = safe_text(row.get("answer"))
    return [answer] if answer else []


def is_correct_row(row: dict[str, str]) -> bool:
    return safe_text(row.get("correct", row.get("is_correct", ""))) == "1"


def stable_template_choice(row: dict[str, str], templates: list[str] | None = None) -> str:
    choices = templates or ALL_CONNECTED_TEMPLATES
    key = "|".join(
        [
            safe_text(row.get("subjid")),
            safe_text(row.get("prob")),
            safe_text(row.get("answer")),
            safe_text(row.get("correct", row.get("is_correct", ""))),
            safe_text(row.get("trace_steps_json")),
        ]
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return choices[int(digest[:8], 16) % len(choices)]


def build_response(row: dict[str, str], template_id: str = "plain") -> str:
    steps = observable_steps(row)
    if template_id == "mixed":
        template_id = stable_template_choice(row)
    if template_id == "rich_mixed":
        template_id = stable_template_choice(row, RICH_TEMPLATES)
    if template_id == "human_mixed":
        template_id = stable_template_choice(row, HUMAN_TEMPLATES)
    if template_id == "plain":
        return build_plain_response(row, steps)
    if template_id == "step_bridge":
        return build_step_bridge_response(row, steps)
    if template_id == "attempt_narrative":
        return build_attempt_narrative_response(row, steps)
    if template_id == "compact_chain":
        return build_compact_chain_response(row, steps)
    if template_id == "tutor_log":
        return build_tutor_log_response(row, steps)
    if template_id == "worked_solution_rich":
        return build_worked_solution_rich_response(row, steps)
    if template_id == "first_attempt_rich":
        return build_first_attempt_rich_response(row, steps)
    if template_id == "student_reasoning_rich":
        return build_student_reasoning_rich_response(row, steps)
    if template_id == "checkpoint_rich":
        return build_checkpoint_rich_response(row, steps)
    if template_id == "human_scratch":
        return build_human_scratch_response(row, steps)
    if template_id == "human_uncertain":
        return build_human_uncertain_response(row, steps)
    if template_id == "human_wrong_guess":
        return build_human_wrong_guess_response(row, steps)
    if template_id == "human_quiet_work":
        return build_human_quiet_work_response(row, steps)
    raise ValueError(f"Unknown response template: {template_id}")


def build_plain_response(row: dict[str, str], steps: list[str]) -> str:
    lines = [f"Step {idx}: {step}" for idx, step in enumerate(steps, start=1)]
    lines.append(f"### answer: {safe_text(row.get('answer'))}")
    return "\n".join(lines)


def build_step_bridge_response(row: dict[str, str], steps: list[str]) -> str:
    answer = safe_text(row.get("answer"))
    correct = is_correct_row(row)
    lines: list[str] = []
    if steps:
        lines.append(f"Step 1: Start with {steps[0]}.")
        for idx, step in enumerate(steps[1:], start=2):
            lines.append(f"Step {idx}: Then rewrite it as {step}.")
    lines.append(final_sentence(answer, correct))
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def build_attempt_narrative_response(row: dict[str, str], steps: list[str]) -> str:
    answer = safe_text(row.get("answer"))
    correct = is_correct_row(row)
    lines: list[str] = []
    if steps:
        lines.append(f"I begin with {steps[0]}.")
        connectors = ["This gives", "Next I get", "Then I have", "So I have"]
        for offset, step in enumerate(steps[1:]):
            lines.append(f"{connectors[offset % len(connectors)]} {step}.")
    lines.append(final_sentence(answer, correct))
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def build_compact_chain_response(row: dict[str, str], steps: list[str]) -> str:
    answer = safe_text(row.get("answer"))
    correct = is_correct_row(row)
    lines: list[str] = []
    if steps:
        lines.append(steps[0])
        for step in steps[1:]:
            lines.append(f"=> {step}")
    lines.append(f"{'Final answer' if correct else 'Stopped at'}: {answer}")
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def build_tutor_log_response(row: dict[str, str], steps: list[str]) -> str:
    answer = safe_text(row.get("answer"))
    correct = is_correct_row(row)
    lines: list[str] = []
    for idx, step in enumerate(steps, start=1):
        label = "Current equation" if idx == 1 else "Next equation"
        lines.append(f"{label}: {step}")
    lines.append(f"{'Answer entered' if correct else 'Last entry'}: {answer}")
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def build_worked_solution_rich_response(row: dict[str, str], steps: list[str]) -> str:
    answer = safe_text(row.get("answer"))
    correct = is_correct_row(row)
    lines: list[str] = []
    if steps:
        lines.append(f"Start from the equation {steps[0]}.")
        for idx, (prev, step) in enumerate(pairwise_steps(steps), start=1):
            lines.append(f"{describe_transition(prev, step, row, idx)}")
    lines.append(rich_final_sentence(answer, correct))
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def build_first_attempt_rich_response(row: dict[str, str], steps: list[str]) -> str:
    answer = safe_text(row.get("answer"))
    correct = is_correct_row(row)
    lines: list[str] = []
    if steps:
        lines.append(f"Student work:")
        lines.append(f"1. The equation is {steps[0]}.")
        for idx, (prev, step) in enumerate(pairwise_steps(steps), start=2):
            lines.append(f"{idx}. {describe_transition(prev, step, row, idx - 1)}")
    if correct:
        lines.append(f"The trace reaches the isolated value, so the entered answer is {answer}.")
    else:
        lines.append(f"The work stops at {answer}, so that is the entered answer.")
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def build_student_reasoning_rich_response(row: dict[str, str], steps: list[str]) -> str:
    answer = safe_text(row.get("answer"))
    correct = is_correct_row(row)
    lines: list[str] = []
    if steps:
        lines.append(f"I read the equation as {steps[0]}.")
        for idx, (prev, step) in enumerate(pairwise_steps(steps), start=1):
            lines.append(first_person_transition(prev, step, row, idx))
    if correct:
        lines.append(f"That leaves the variable value, so I enter {answer}.")
    else:
        lines.append(f"I am not sure how to proceed, so I enter {answer}.")
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def build_checkpoint_rich_response(row: dict[str, str], steps: list[str]) -> str:
    answer = safe_text(row.get("answer"))
    correct = is_correct_row(row)
    lines: list[str] = []
    if steps:
        lines.append(f"Checkpoint 1: {steps[0]}")
        for idx, (prev, step) in enumerate(pairwise_steps(steps), start=2):
            lines.append(f"Checkpoint {idx}: {short_transition_label(prev, step, row)} -> {step}")
    lines.append(f"{'Final checkpoint answer' if correct else 'Stopped checkpoint'}: {answer}")
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def build_human_scratch_response(row: dict[str, str], steps: list[str]) -> str:
    answer = response_answer(row, steps, "human_scratch")
    correct = is_correct_row(row)
    lines: list[str] = []
    if steps:
        lines.append(f"I start with {steps[0]}.")
        for idx, (prev, step) in enumerate(pairwise_steps(steps), start=1):
            lines.append(human_transition(prev, step, row, idx))
    if correct:
        lines.append(f"So I get {answer}.")
    else:
        lines.append(f"I am not sure how to keep going from here, so I put {answer}.")
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def build_human_uncertain_response(row: dict[str, str], steps: list[str]) -> str:
    answer = response_answer(row, steps, "human_uncertain")
    correct = is_correct_row(row)
    lines: list[str] = []
    if steps:
        lines.append(f"{steps[0]}")
        for idx, (prev, step) in enumerate(pairwise_steps(steps), start=1):
            lines.append(human_short_transition(prev, step, row, idx))
    if correct:
        lines.append(f"That makes the answer {answer}.")
    else:
        last_step = steps[-1] if steps else answer
        lines.append(f"I get to {last_step}, but I am not sure what to do next.")
        lines.append(f"My answer is {answer}.")
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def build_human_wrong_guess_response(row: dict[str, str], steps: list[str]) -> str:
    answer = response_answer(row, steps, "human_wrong_guess")
    correct = is_correct_row(row)
    lines: list[str] = []
    if steps:
        lines.append(f"I write down {steps[0]}.")
        for idx, (prev, step) in enumerate(pairwise_steps(steps), start=1):
            lines.append(human_transition(prev, step, row, idx))
    if correct:
        lines.append(f"The variable is isolated, so my answer is {answer}.")
    else:
        guess_state = format_variable_guess(row, answer)
        lines.append(f"I think the last step should give {guess_state}.")
        lines.append(f"So I answer {answer}.")
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def build_human_quiet_work_response(row: dict[str, str], steps: list[str]) -> str:
    answer = response_answer(row, steps, "human_quiet_work")
    correct = is_correct_row(row)
    lines: list[str] = []
    if steps:
        lines.append(steps[0])
        for idx, (prev, step) in enumerate(pairwise_steps(steps), start=1):
            lines.append(f"{human_short_transition(prev, step, row, idx)}")
    if correct:
        lines.append(f"answer = {answer}")
    else:
        lines.append(f"not totally sure, but I enter {answer}")
    lines.append(f"### answer: {answer}")
    return "\n".join(lines)


def final_sentence(answer: str, correct: bool) -> str:
    if correct:
        return f"So the final answer is {answer}."
    return f"I stop at {answer}."


def rich_final_sentence(answer: str, correct: bool) -> str:
    if correct:
        return f"The equation is solved, and the final answer is {answer}."
    return f"The work stops at {answer}, so that is the submitted response."


def response_answer(row: dict[str, str], steps: list[str], template_id: str) -> str:
    if template_id == "human_wrong_guess" and not is_correct_row(row):
        return plausible_wrong_answer(row, steps)
    return safe_text(row.get("answer"))


def plausible_wrong_answer(row: dict[str, str], steps: list[str]) -> str:
    solution = parse_fraction_text(row.get("solution", ""))
    if solution is None:
        solution = parse_fraction_text(row.get("final_answer", ""))
    if solution is None:
        solution = solution_like_from_last_step(row, steps)
    if solution is None:
        return safe_text(row.get("answer"))

    choices = wrong_fraction_choices(solution)
    if not choices:
        return safe_text(row.get("answer"))
    key = "|".join(
        [
            safe_text(row.get("subjid")),
            safe_text(row.get("prob")),
            safe_text(row.get("answer")),
            safe_text(row.get("trace_steps_json")),
            "wrong-answer",
        ]
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    wrong = choices[int(digest[:8], 16) % len(choices)]
    return format_fraction_text(wrong)


def parse_fraction_text(value: object) -> Fraction | None:
    text = safe_text(value)
    if not text:
        return None
    if "=" in text:
        text = text.split("=", 1)[-1].strip()
    try:
        return Fraction(text)
    except ValueError:
        return None


def solution_like_from_last_step(row: dict[str, str], steps: list[str]) -> Fraction | None:
    if not steps:
        return None
    variable = safe_text(row.get("variable")) or infer_variable(" ".join(steps))
    last = clean_equation_text(steps[-1])
    pattern = rf"(?:^|=){re.escape(variable)}=(-?\d+(?:/\d+)?)$|^(-?\d+(?:/\d+)?)={re.escape(variable)}(?:$|=)"
    match = re.search(pattern, last)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return parse_fraction_text(value)


def wrong_fraction_choices(solution: Fraction) -> list[Fraction]:
    choices = [
        -solution,
        solution + 1,
        solution - 1,
        solution * 2,
    ]
    if solution != 0:
        choices.append(Fraction(1, 1) / solution)
    return [choice for choice in choices if choice != solution]


def format_fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_variable_guess(row: dict[str, str], answer: str) -> str:
    variable = safe_text(row.get("variable")) or infer_variable(safe_text(row.get("prob")))
    return f"{variable} = {answer}" if variable else answer


def pairwise_steps(steps: list[str]) -> list[tuple[str, str]]:
    return list(zip(steps, steps[1:]))


def clean_equation_text(text: str) -> str:
    return re.sub(r"\s+", "", safe_text(text))


def equation_sides(text: str) -> tuple[str, str] | None:
    cleaned = clean_equation_text(text)
    if cleaned.count("=") != 1:
        return None
    left, right = cleaned.split("=", 1)
    return left, right


def has_variable_alone(text: str, variable: str) -> bool:
    sides = equation_sides(text)
    if not sides or not variable:
        return False
    return any(side == variable for side in sides)


def variable_side_count(text: str, variable: str) -> int:
    sides = equation_sides(text)
    if not sides or not variable:
        return 0
    return sum(1 for side in sides if variable in side)


def contains_fraction_like(text: str) -> bool:
    return "/" in clean_equation_text(text)


def contains_grouping(text: str) -> bool:
    cleaned = clean_equation_text(text)
    return "(" in cleaned or ")" in cleaned


def describe_transition(prev: str, step: str, row: dict[str, str], transition_idx: int) -> str:
    label = short_transition_label(prev, step, row)
    if label == "clear fractions and isolate variable":
        return f"Clear the denominator and isolate the variable, giving {step}."
    if label == "clear fractions":
        return f"Clear the fraction or denominator so the equation becomes {step}."
    if label == "expand or simplify":
        return f"Expand or simplify the grouped expression to get {step}."
    if label == "collect variable terms":
        return f"Collect the variable terms onto one side, giving {step}."
    if label == "move constants":
        return f"Move the constant terms away from the variable term: {step}."
    if label == "isolate variable":
        return f"Isolate the variable, producing {step}."
    if label == "combine terms":
        return f"Combine the visible terms and rewrite the equation as {step}."
    return f"Rewrite the equation as {step}."


def first_person_transition(prev: str, step: str, row: dict[str, str], transition_idx: int) -> str:
    label = short_transition_label(prev, step, row)
    if label == "clear fractions and isolate variable":
        return f"I clear the denominator and finish with {step}."
    if label == "clear fractions":
        return f"I clear the denominator and get {step}."
    if label == "expand or simplify":
        return f"I simplify the grouped expression, giving {step}."
    if label == "collect variable terms":
        return f"I move the variable terms together and get {step}."
    if label == "move constants":
        return f"I move the numbers away from the variable term, so I have {step}."
    if label == "isolate variable":
        return f"I finish isolating the variable as {step}."
    if label == "combine terms":
        return f"I combine like terms and get {step}."
    return f"I rewrite it as {step}."


def human_transition(prev: str, step: str, row: dict[str, str], transition_idx: int) -> str:
    label = short_transition_label(prev, step, row)
    if label == "clear fractions and isolate variable":
        return f"I clear the denominator, so it turns into {step}."
    if label == "clear fractions":
        return f"I multiply through to get rid of the denominator: {step}."
    if label == "expand or simplify":
        return f"I simplify that part and get {step}."
    if label == "collect variable terms":
        return f"I move the variable terms together: {step}."
    if label == "move constants":
        return f"I move the number part over, giving {step}."
    if label == "isolate variable":
        return f"Then I divide to get {step}."
    if label == "combine terms":
        return f"I combine terms and get {step}."
    return f"Then I get {step}."


def human_short_transition(prev: str, step: str, row: dict[str, str], transition_idx: int) -> str:
    label = short_transition_label(prev, step, row)
    if label == "clear fractions and isolate variable":
        return f"clear the denominator -> {step}"
    if label == "clear fractions":
        return f"clear the denominator -> {step}"
    if label == "expand or simplify":
        return f"simplify -> {step}"
    if label == "collect variable terms":
        return f"move the variable terms -> {step}"
    if label == "move constants":
        return f"move the constants -> {step}"
    if label == "isolate variable":
        return f"divide -> {step}"
    if label == "combine terms":
        return f"combine terms -> {step}"
    return f"rewrite -> {step}"


def short_transition_label(prev: str, step: str, row: dict[str, str]) -> str:
    variable = safe_text(row.get("variable")) or infer_variable(prev + step)
    prev_clean = clean_equation_text(prev)
    step_clean = clean_equation_text(step)
    prev_variable_sides = variable_side_count(prev, variable)
    step_variable_sides = variable_side_count(step, variable)

    if contains_fraction_like(prev_clean) and not contains_fraction_like(step_clean) and has_variable_alone(step, variable):
        return "clear fractions and isolate variable"
    if contains_fraction_like(prev_clean) and not contains_fraction_like(step_clean):
        return "clear fractions"
    if has_variable_alone(step, variable):
        return "isolate variable"
    if contains_grouping(prev_clean) and not contains_grouping(step_clean):
        return "expand or simplify"
    if prev_variable_sides > step_variable_sides and step_variable_sides == 1:
        return "collect variable terms"
    if variable and variable in prev_clean and variable in step_clean and constant_part_changed(prev_clean, step_clean, variable):
        return "move constants"
    if term_count(step_clean, variable) < term_count(prev_clean, variable):
        return "combine terms"
    return "rewrite"


def infer_variable(text: str) -> str:
    variables = sorted(set(re.findall(r"[A-Za-z]", text)))
    for variable in variables:
        if variable in {"x", "y"}:
            return variable
    return variables[0] if variables else ""


def constant_part_changed(prev: str, step: str, variable: str) -> bool:
    prev_sides = equation_sides(prev)
    step_sides = equation_sides(step)
    if not prev_sides or not step_sides:
        return False
    prev_variable_side = next((side for side in prev_sides if variable in side), "")
    step_variable_side = next((side for side in step_sides if variable in side), "")
    return strip_variable_terms(prev_variable_side, variable) != strip_variable_terms(step_variable_side, variable)


def strip_variable_terms(text: str, variable: str) -> str:
    return re.sub(rf"[+-]?\d*(?:/\d+)?\*?{re.escape(variable)}", "", text)


def term_count(text: str, variable: str) -> int:
    return len([part for part in re.split(r"(?=[+-])", text) if part and part not in {"=", "+", "-"}])


def translate_row(
    row: dict[str, str],
    response_template: str = "plain",
    include_template_id: bool = False,
) -> dict[str, object]:
    if response_template == "mixed":
        template_id = stable_template_choice(row)
    elif response_template == "rich_mixed":
        template_id = stable_template_choice(row, RICH_TEMPLATES)
    elif response_template == "human_mixed":
        template_id = stable_template_choice(row, HUMAN_TEMPLATES)
    else:
        template_id = response_template
    steps = observable_steps(row)
    response_nl = build_response(row, template_id=template_id)
    output: dict[str, object] = {
        "subjid": safe_text(row.get("subjid")),
        "prob": safe_text(row.get("prob")),
        "equation_type": safe_text(row.get("equation_type")),
        "variable": safe_text(row.get("variable")),
        "answer": response_answer(row, steps, template_id),
        "correct": safe_text(row.get("correct", row.get("is_correct", ""))),
        "g": safe_text(row.get("g")),
        "d": safe_text(row.get("d")),
        "c": safe_text(row.get("c")),
        "rt_mu": safe_text(row.get("rt_mu")),
        "ice": safe_text(row.get("ice")),
        "instruction_nl": f"Solve this algebra equation: {safe_text(row.get('prob'))}.",
        "response_nl": response_nl,
    }
    if include_template_id:
        output[TEMPLATE_ID_COLUMN] = template_id
    return output

def translate_file(
    input_csv: Path,
    output_csv: Path,
    max_rows: int | None = None,
    response_template: str = "plain",
    expand_templates: list[str] | None = None,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    output_count = 0
    templates = expand_templates or []
    use_expansion = bool(templates)
    fieldnames = TEMPLATED_OUTPUT_COLUMNS if use_expansion else OUTPUT_COLUMNS
    with open_text(input_csv, "rt") as in_handle, open_text(output_csv, "wt") as out_handle:
        reader = csv.DictReader(in_handle)
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if use_expansion:
                for template_id in templates:
                    writer.writerow(
                        translate_row(
                            row,
                            response_template=template_id,
                            include_template_id=True,
                        )
                    )
                    output_count += 1
            else:
                writer.writerow(translate_row(row, response_template=response_template))
                output_count += 1
            count += 1
            if max_rows is not None and count >= max_rows:
                break
    return output_count


def main() -> None:
    args = parse_args()
    count = translate_file(
        args.input_csv,
        args.output_csv,
        max_rows=args.max_rows,
        response_template=args.response_template,
        expand_templates=args.expand_templates,
    )
    print(f"wrote_rows={count}", flush=True)
    print(f"output_csv={args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
