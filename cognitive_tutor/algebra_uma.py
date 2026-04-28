#!/usr/bin/env python3
"""Lightweight UMA-style machinery for single-variable algebra equations.

This module intentionally stays separate from the existing arithmetic UMA core.
It provides conservative equation filtering/solving plus a small stochastic
trace simulator that emits the same broad columns used by UMA trace pipelines.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_PANEL_CSV = Path(
    "results/UMA_replication/decimal_param_selection_k100/"
    "selected_params_gd_full_grid_optimized100.csv"
)

UMA_TRACE_COLUMNS = [
    "subjid",
    "seed",
    "prob",
    "operation",
    "equation_type",
    "variable",
    "solution",
    "strategy",
    "goals",
    "exec",
    "answer",
    "correct",
    "is_correct",
    "g",
    "d",
    "c",
    "rt_mu",
    "ice",
    "first_wrong_step",
    "trace_step_count",
    "trace_rules_full",
    "trace_steps_json",
    "source",
]

GOAL_PARSE = "parse equation"
GOAL_SIMPLIFY = "simplify sides"
GOAL_CLEAR_DENOMINATOR = "clear denominators"
GOAL_COLLECT_VARIABLES = "collect variable terms"
GOAL_COLLECT_CONSTANTS = "collect constants"
GOAL_UNDO_COEFFICIENT = "undo coefficient"
GOAL_CHECK = "check solution"

EXEC_DISTRIBUTE = "distribute"
EXEC_COMBINE_LIKE_TERMS = "combine like terms"
EXEC_ADD_SUBTRACT_BOTH_SIDES = "add/subtract both sides"
EXEC_MULTIPLY_BOTH_SIDES = "multiply both sides"
EXEC_DIVIDE_BOTH_SIDES = "divide both sides"
EXEC_SWAP_SIDES = "swap sides"
EXEC_CHECK = "check solution"

ERROR_RULES = [
    "sign error",
    "inverse-op error",
    "dropped term",
    "wrong coefficient",
    "premature stop",
]


class EquationParseError(ValueError):
    """Raised when a problem is outside the conservative v1 grammar."""


@dataclass(frozen=True)
class AlgebraExpr:
    """Expression of the form a*x + b/x + c for one variable."""

    x: Fraction = Fraction(0)
    inv_x: Fraction = Fraction(0)
    const: Fraction = Fraction(0)

    def __add__(self, other: "AlgebraExpr") -> "AlgebraExpr":
        return AlgebraExpr(self.x + other.x, self.inv_x + other.inv_x, self.const + other.const)

    def __sub__(self, other: "AlgebraExpr") -> "AlgebraExpr":
        return AlgebraExpr(self.x - other.x, self.inv_x - other.inv_x, self.const - other.const)

    def scale(self, factor: Fraction) -> "AlgebraExpr":
        return AlgebraExpr(self.x * factor, self.inv_x * factor, self.const * factor)

    def is_constant(self) -> bool:
        return self.x == 0 and self.inv_x == 0

    def is_variable_alone(self) -> bool:
        return self.x == 1 and self.inv_x == 0 and self.const == 0

    def has_variable(self) -> bool:
        return self.x != 0 or self.inv_x != 0


@dataclass(frozen=True)
class EquationMetadata:
    original_prob: str
    equation_text: str
    variable: str
    equation_type: str
    solution: Fraction
    lhs: AlgebraExpr
    rhs: AlgebraExpr


@dataclass(frozen=True)
class CorrectTrace:
    steps: list[str]
    goals: list[str]
    exec_rules: list[str]
    strategy: str


@dataclass
class StudentState:
    subjid: str
    g: float
    d: float
    c: float
    rt_mu: float
    ice: float
    strengths: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_panel_row(cls, row: dict[str, object]) -> "StudentState":
        ice = float(row.get("ice", 0) or 0)
        initial = max(0.0, min(1.0, ice / 100.0))
        return cls(
            subjid=str(row.get("subjid", "")),
            g=float(row.get("g", 0.05) or 0.05),
            d=float(row.get("d", 0.5) or 0.5),
            c=float(row.get("c", 5.0) or 5.0),
            rt_mu=float(row.get("rt_mu", 5.0) or 5.0),
            ice=ice,
            strengths={rule: initial for rule in all_exec_rules()},
        )

    def reinforce(self, rule: str) -> None:
        current = self.strengths.get(rule, max(0.0, min(1.0, self.ice / 100.0)))
        decayed = current * (1.0 - min(0.02, self.d * 0.01))
        self.strengths[rule] = max(0.0, min(1.0, decayed + self.g * (1.0 - decayed)))

    def success_probability(self, rule: str) -> float:
        strength = self.strengths.get(rule, max(0.0, min(1.0, self.ice / 100.0)))
        prob = 0.45 + 0.42 * strength + 0.9 * self.g - 0.18 * self.d
        if rule == EXEC_CHECK:
            prob += 0.05
        return max(0.03, min(0.97, prob))


@dataclass(frozen=True)
class TraceAttempt:
    subjid: str
    seed: int
    prob: str
    operation: str
    equation_type: str
    variable: str
    solution: str
    strategy: str
    goals: list[str]
    exec_rules: list[str]
    answer: str
    correct: int
    g: float
    d: float
    c: float
    rt_mu: float
    ice: float
    first_wrong_step: str
    trace_steps: list[dict[str, object]]
    source: str

    def to_row(self) -> dict[str, object]:
        return {
            "subjid": self.subjid,
            "seed": self.seed,
            "prob": self.prob,
            "operation": self.operation,
            "equation_type": self.equation_type,
            "variable": self.variable,
            "solution": self.solution,
            "strategy": self.strategy,
            "goals": " | ".join(self.goals),
            "exec": " | ".join(self.exec_rules),
            "answer": self.answer,
            "correct": self.correct,
            "is_correct": self.correct,
            "g": self.g,
            "d": self.d,
            "c": self.c,
            "rt_mu": self.rt_mu,
            "ice": self.ice,
            "first_wrong_step": self.first_wrong_step,
            "trace_step_count": len(self.trace_steps),
            "trace_rules_full": " | ".join(self.exec_rules),
            "trace_steps_json": json.dumps(self.trace_steps, separators=(",", ":")),
            "source": self.source,
        }


@dataclass(frozen=True)
class CalibrationBin:
    correct_rate: float
    wrong_steps: list[str]
    rows: int
    wrong_rows: int
    wrong_step_indices: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class HumanCalibration:
    by_equation_type: dict[str, CalibrationBin]
    global_bin: CalibrationBin
    source_csv: str


def all_exec_rules() -> list[str]:
    return [
        EXEC_DISTRIBUTE,
        EXEC_COMBINE_LIKE_TERMS,
        EXEC_ADD_SUBTRACT_BOTH_SIDES,
        EXEC_MULTIPLY_BOTH_SIDES,
        EXEC_DIVIDE_BOTH_SIDES,
        EXEC_SWAP_SIDES,
        EXEC_CHECK,
    ]


def fraction_from_number(value: int | float) -> Fraction:
    return Fraction(str(value)) if isinstance(value, float) else Fraction(value)


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def safe_int_local(value: object, default: int = 0) -> int:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def format_coeff_var(coefficient: Fraction, variable: str) -> str:
    if coefficient == 1:
        return variable
    if coefficient == -1:
        return f"-{variable}"
    return f"{format_fraction(coefficient)}*{variable}"


def extract_equation_text(problem_text: str) -> str:
    """Remove Cognitive Tutor problem id prefixes such as ``EG21``."""
    candidate = str(problem_text).strip()
    for _ in range(3):
        match = re.match(r"^[A-Za-z][A-Za-z0-9_-]*\s+(.+)$", candidate)
        if match and "=" in match.group(1):
            candidate = match.group(1).strip()
        else:
            break
    return candidate


def normalize_algebra_text(text: str) -> str:
    expr = str(text).strip()
    expr = expr.replace("^", "**")
    expr = expr.replace("\u2212", "-")
    expr = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", expr)
    expr = re.sub(r"(?<=\))(?=[A-Za-z0-9(])", "*", expr)
    expr = re.sub(r"(?<=[A-Za-z])(?=\d)", "*", expr)
    expr = re.sub(r"(?<=[A-Za-z])(?=\()", "*", expr)
    return expr


def find_single_variable(equation_text: str) -> str | None:
    if "sqrt" in equation_text.lower():
        return None
    variables = sorted(set(re.findall(r"[A-Za-z]", equation_text)))
    if len(variables) != 1:
        return None
    if variables[0] not in {"x", "y"}:
        return None
    return variables[0]


def parse_expression(text: str, variable: str) -> AlgebraExpr:
    normalized = normalize_algebra_text(text)
    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise EquationParseError(str(exc)) from exc
    return parse_ast_node(parsed.body, variable)


def parse_ast_node(node: ast.AST, variable: str) -> AlgebraExpr:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return AlgebraExpr(const=fraction_from_number(node.value))
    if isinstance(node, ast.Name):
        if node.id != variable:
            raise EquationParseError(f"unexpected variable {node.id!r}")
        return AlgebraExpr(x=Fraction(1))
    if isinstance(node, ast.UnaryOp):
        value = parse_ast_node(node.operand, variable)
        if isinstance(node.op, ast.USub):
            return value.scale(Fraction(-1))
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = parse_ast_node(node.left, variable)
        right = parse_ast_node(node.right, variable)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            if left.is_constant():
                return right.scale(left.const)
            if right.is_constant():
                return left.scale(right.const)
            raise EquationParseError("nonlinear multiplication")
        if isinstance(node.op, ast.Div):
            if right.is_constant() and right.const != 0:
                return left.scale(Fraction(1, 1) / right.const)
            if left.is_constant() and right.inv_x == 0 and right.const == 0 and right.x != 0:
                return AlgebraExpr(inv_x=left.const / right.x)
            raise EquationParseError("unsupported denominator")
    raise EquationParseError(f"unsupported expression: {ast.dump(node)}")


def solve_delta(delta: AlgebraExpr) -> Fraction | None:
    if delta.inv_x == 0 and delta.x != 0:
        return -delta.const / delta.x
    if delta.x == 0 and delta.inv_x != 0 and delta.const != 0:
        return -delta.inv_x / delta.const
    return None


def classify_equation(equation_text: str, lhs: AlgebraExpr, rhs: AlgebraExpr) -> str:
    if lhs.inv_x != 0 or rhs.inv_x != 0:
        return "reciprocal_variable_denominator"
    variable = find_single_variable(equation_text) or "x"
    if re.search(rf"\b{variable}\s*/", equation_text):
        return "linear_with_denominator"
    if lhs.x != 0 and rhs.x != 0:
        return "linear_variable_both_sides"
    variable_side = lhs if lhs.x != 0 else rhs
    if variable_side.const != 0:
        return "linear_two_step"
    return "linear_one_step"


def get_equation_metadata(problem_text: str) -> EquationMetadata | None:
    equation_text = extract_equation_text(problem_text)
    if equation_text.count("=") != 1 or "?" in equation_text:
        return None
    variable = find_single_variable(equation_text)
    if not variable:
        return None
    lhs_text, rhs_text = [piece.strip() for piece in equation_text.split("=", 1)]
    try:
        lhs = parse_expression(lhs_text, variable)
        rhs = parse_expression(rhs_text, variable)
    except EquationParseError:
        return None

    if (lhs.is_variable_alone() and rhs.is_constant()) or (
        rhs.is_variable_alone() and lhs.is_constant()
    ):
        return None

    solution = solve_delta(lhs - rhs)
    if solution is None or not math.isfinite(float(solution)):
        return None
    equation_type = classify_equation(equation_text, lhs, rhs)
    return EquationMetadata(
        original_prob=str(problem_text),
        equation_text=equation_text,
        variable=variable,
        equation_type=equation_type,
        solution=solution,
        lhs=lhs,
        rhs=rhs,
    )


def is_single_variable_equation(problem_text: str) -> bool:
    return get_equation_metadata(problem_text) is not None


def build_correct_trace(meta: EquationMetadata) -> CorrectTrace:
    steps = [meta.equation_text]
    goals = [GOAL_PARSE]
    exec_rules: list[str] = []
    delta = meta.lhs - meta.rhs
    solution_text = f"{meta.variable} = {format_fraction(meta.solution)}"

    if "(" in meta.equation_text and ")" in meta.equation_text:
        goals.append(GOAL_SIMPLIFY)
        exec_rules.append(EXEC_DISTRIBUTE)

    if delta.inv_x != 0:
        goals.append(GOAL_CLEAR_DENOMINATOR)
        exec_rules.append(EXEC_MULTIPLY_BOTH_SIDES)
        steps.append(f"{format_fraction(delta.const)}*{meta.variable} = {format_fraction(-delta.inv_x)}")
        goals.append(GOAL_UNDO_COEFFICIENT)
        exec_rules.append(EXEC_DIVIDE_BOTH_SIDES)
        steps.append(solution_text)
    else:
        if meta.lhs.x != 0 and meta.rhs.x != 0:
            goals.append(GOAL_COLLECT_VARIABLES)
            exec_rules.append(EXEC_ADD_SUBTRACT_BOTH_SIDES)
        if delta.const != 0:
            goals.append(GOAL_COLLECT_CONSTANTS)
            exec_rules.append(EXEC_ADD_SUBTRACT_BOTH_SIDES)
            steps.append(f"{format_coeff_var(delta.x, meta.variable)} = {format_fraction(-delta.const)}")
        if delta.x not in (0, 1):
            goals.append(GOAL_UNDO_COEFFICIENT)
            exec_rules.append(EXEC_DIVIDE_BOTH_SIDES)
        if steps[-1] != solution_text:
            steps.append(solution_text)

    goals.append(GOAL_CHECK)
    exec_rules.append(EXEC_CHECK)
    strategy = strategy_name(meta)
    return CorrectTrace(steps=dedupe_adjacent(steps), goals=dedupe_preserve(goals), exec_rules=exec_rules, strategy=strategy)


def build_ct_flow_trace(meta: EquationMetadata) -> CorrectTrace:
    """Build a problem-local, Cognitive Tutor-like visible equation flow.

    The real tutor logs step names/KCs rather than free-form thoughts. This
    trace keeps that spirit: each visible step is an equation state the student
    is trying to reach. It is more useful for distillation than v2's single
    sampled stuck string, while still keeping strategy/goals/exec hidden from
    the translated NLP data by default.
    """
    steps = [meta.equation_text]
    goals = [GOAL_PARSE]
    exec_rules: list[str] = []
    work_lhs = meta.lhs
    work_rhs = meta.rhs

    if meta.lhs.inv_x != 0 or meta.rhs.inv_x != 0:
        goals.append(GOAL_CLEAR_DENOMINATOR)
        exec_rules.append(EXEC_MULTIPLY_BOTH_SIDES)
        coeff = meta.lhs.const - meta.rhs.const
        const = meta.rhs.inv_x - meta.lhs.inv_x
        steps.append(f"{format_coeff_var(coeff, meta.variable)} = {format_fraction(const)}")
        if coeff not in (0, 1):
            goals.append(GOAL_UNDO_COEFFICIENT)
            exec_rules.append(EXEC_DIVIDE_BOTH_SIDES)
        steps.append(f"{meta.variable} = {format_fraction(meta.solution)}")
        goals.append(GOAL_CHECK)
        exec_rules.append(EXEC_CHECK)
        return CorrectTrace(
            steps=dedupe_adjacent(steps),
            goals=dedupe_preserve(goals),
            exec_rules=exec_rules,
            strategy=strategy_name(meta),
        )

    if needs_canonical_simplification(meta):
        goals.append(GOAL_SIMPLIFY)
        exec_rules.append(EXEC_DISTRIBUTE if "(" in meta.equation_text else EXEC_COMBINE_LIKE_TERMS)
        steps.append(
            f"{format_linear_expr(work_lhs.x, work_lhs.const, meta.variable)} = "
            f"{format_linear_expr(work_rhs.x, work_rhs.const, meta.variable)}"
        )

    if has_fractional_variable_coefficient(work_lhs) or has_fractional_variable_coefficient(work_rhs):
        denominator = common_variable_denominator(work_lhs, work_rhs)
        if denominator > 1:
            goals.append(GOAL_CLEAR_DENOMINATOR)
            exec_rules.append(EXEC_MULTIPLY_BOTH_SIDES)
            work_lhs = work_lhs.scale(Fraction(denominator))
            work_rhs = work_rhs.scale(Fraction(denominator))
            steps.append(
                f"{format_linear_expr(work_lhs.x, work_lhs.const, meta.variable)} = "
                f"{format_linear_expr(work_rhs.x, work_rhs.const, meta.variable)}"
            )

    delta = work_lhs - work_rhs
    if work_lhs.x != 0 and work_rhs.x != 0:
        goals.append(GOAL_COLLECT_VARIABLES)
        exec_rules.append(EXEC_ADD_SUBTRACT_BOTH_SIDES)
        steps.append(f"{format_linear_expr(delta.x, work_lhs.const, meta.variable)} = {format_fraction(work_rhs.const)}")

    if delta.const != 0:
        goals.append(GOAL_COLLECT_CONSTANTS)
        exec_rules.append(EXEC_ADD_SUBTRACT_BOTH_SIDES)
        steps.append(f"{format_coeff_var(delta.x, meta.variable)} = {format_fraction(-delta.const)}")

    if delta.x not in (0, 1):
        goals.append(GOAL_UNDO_COEFFICIENT)
        exec_rules.append(EXEC_DIVIDE_BOTH_SIDES)

    steps.append(f"{meta.variable} = {format_fraction(meta.solution)}")
    goals.append(GOAL_CHECK)
    exec_rules.append(EXEC_CHECK)
    return CorrectTrace(
        steps=dedupe_adjacent(steps),
        goals=dedupe_preserve(goals),
        exec_rules=exec_rules,
        strategy=strategy_name(meta),
    )


def needs_canonical_simplification(meta: EquationMetadata) -> bool:
    return (
        "(" in meta.equation_text
        or ")" in meta.equation_text
        or abs(meta.lhs.x) > 1
        and re.search(rf"{meta.variable}.*{meta.variable}", meta.equation_text)
    )


def has_fractional_variable_coefficient(expr: AlgebraExpr) -> bool:
    return expr.x.denominator != 1


def common_variable_denominator(lhs: AlgebraExpr, rhs: AlgebraExpr) -> int:
    return math.lcm(lhs.x.denominator, rhs.x.denominator)


def format_linear_expr(coefficient: Fraction, const: Fraction, variable: str) -> str:
    pieces: list[str] = []
    if coefficient:
        pieces.append(format_coeff_var(coefficient, variable))
    if const:
        const_text = format_fraction(abs(const))
        if pieces:
            pieces.append(("+" if const > 0 else "-") + const_text)
        else:
            pieces.append(format_fraction(const))
    if not pieces:
        return "0"
    return "".join(pieces)


def strategy_name(meta: EquationMetadata) -> str:
    if meta.equation_type == "reciprocal_variable_denominator":
        return "clear denominator"
    if "(" in meta.equation_text and ")" in meta.equation_text:
        return "distribute then isolate"
    if meta.equation_type == "linear_variable_both_sides":
        return "move variables/constants"
    if meta.equation_type == "linear_with_denominator":
        return "clear denominator"
    return "isolate variable"


def dedupe_adjacent(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if not out or out[-1] != item:
            out.append(item)
    return out


def dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def corrupt_step(correct_step: str, meta: EquationMetadata, error_rule: str, rng: random.Random) -> str:
    if error_rule == "premature stop":
        return correct_step
    solution = meta.solution
    if error_rule == "sign error":
        wrong = -solution if solution != 0 else Fraction(-1)
    elif error_rule == "inverse-op error":
        wrong = solution + rng.choice([Fraction(1), Fraction(-1), Fraction(2)])
    elif error_rule == "dropped term":
        wrong = solution - meta.lhs.const + meta.rhs.const
        if wrong == solution:
            wrong += 1
    else:
        wrong = solution * rng.choice([Fraction(2), Fraction(-1), Fraction(1, 2)])
        if wrong == solution:
            wrong += 1
    return f"{meta.variable} = {format_fraction(wrong)}"


def trace_step_records(steps: Sequence[str], goals: Sequence[str], exec_rules: Sequence[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for idx, step in enumerate(steps):
        rule_idx = max(0, min(idx - 1, len(exec_rules) - 1))
        goal_idx = max(0, min(idx, len(goals) - 1))
        records.append(
            {
                "step_index": idx + 1,
                "state": step,
                "goal": goals[goal_idx] if goals else "",
                "exec_rule": exec_rules[rule_idx] if exec_rules and idx > 0 else "",
                "correct_first_attempt": 1,
            }
        )
    return records


def train_student(
    params: dict[str, object],
    curriculum: Sequence[EquationMetadata],
    seed: int = 1,
) -> StudentState:
    state = StudentState.from_panel_row(params)
    rng = random.Random(seed + int(float(state.subjid or 0)))
    for meta in curriculum:
        trace = build_correct_trace(meta)
        for rule in trace.exec_rules:
            if rng.random() < state.success_probability(rule):
                state.reinforce(rule)
    return state


def simulate_attempt(
    state: StudentState,
    meta: EquationMetadata,
    seed: int = 1,
    force_error_rule: str | None = None,
    source: str = "algebra_uma",
) -> TraceAttempt:
    stable_problem_hash = int(hashlib.md5(meta.equation_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + int(float(state.subjid or 0)) * 1_000_003 + stable_problem_hash)
    correct_trace = build_correct_trace(meta)
    error_rule = force_error_rule
    error_step_index = 0

    if error_rule is None:
        for idx, rule in enumerate(correct_trace.exec_rules, start=1):
            if rng.random() > state.success_probability(rule):
                error_rule = rng.choice(ERROR_RULES)
                error_step_index = min(idx, len(correct_trace.steps) - 1)
                break
    else:
        error_step_index = min(max(1, len(correct_trace.steps) - 1), len(correct_trace.steps) - 1)

    if error_rule:
        if error_rule == "premature stop":
            visible_steps = correct_trace.steps[: max(1, error_step_index)]
            first_wrong_step = visible_steps[-1]
        else:
            correct_step = correct_trace.steps[error_step_index]
            first_wrong_step = corrupt_step(correct_step, meta, error_rule, rng)
            visible_steps = list(correct_trace.steps[:error_step_index]) + [first_wrong_step]
        trace_records = trace_step_records(visible_steps, correct_trace.goals, correct_trace.exec_rules)
        if trace_records:
            trace_records[-1]["correct_first_attempt"] = 0
            trace_records[-1]["error_rule"] = error_rule
        return TraceAttempt(
            subjid=state.subjid,
            seed=seed,
            prob=meta.equation_text,
            operation=meta.equation_type,
            equation_type=meta.equation_type,
            variable=meta.variable,
            solution=format_fraction(meta.solution),
            strategy="premature stop" if error_rule == "premature stop" else correct_trace.strategy,
            goals=correct_trace.goals,
            exec_rules=list(correct_trace.exec_rules) + [error_rule],
            answer=first_wrong_step,
            correct=0,
            g=state.g,
            d=state.d,
            c=state.c,
            rt_mu=state.rt_mu,
            ice=state.ice,
            first_wrong_step=first_wrong_step,
            trace_steps=trace_records,
            source=source,
        )

    answer = format_fraction(meta.solution)
    return TraceAttempt(
        subjid=state.subjid,
        seed=seed,
        prob=meta.equation_text,
        operation=meta.equation_type,
        equation_type=meta.equation_type,
        variable=meta.variable,
        solution=answer,
        strategy=correct_trace.strategy,
        goals=correct_trace.goals,
        exec_rules=correct_trace.exec_rules,
        answer=answer,
        correct=1,
        g=state.g,
        d=state.d,
        c=state.c,
        rt_mu=state.rt_mu,
        ice=state.ice,
        first_wrong_step="",
        trace_steps=trace_step_records(correct_trace.steps, correct_trace.goals, correct_trace.exec_rules),
        source=source,
    )


def load_human_calibration(path: Path) -> HumanCalibration:
    grouped: dict[str, list[dict[str, str]]] = {}
    all_rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            equation_type = str(row.get("equation_type", "") or "")
            grouped.setdefault(equation_type, []).append(row)
            all_rows.append(row)

    if not all_rows:
        raise ValueError(f"No calibration rows found in {path}")

    by_equation_type = {
        equation_type: build_calibration_bin(rows)
        for equation_type, rows in grouped.items()
    }
    return HumanCalibration(
        by_equation_type=by_equation_type,
        global_bin=build_calibration_bin(all_rows),
        source_csv=str(path),
    )


def build_calibration_bin(rows: Sequence[dict[str, str]]) -> CalibrationBin:
    correct_values = [safe_int_local(row.get("correct", row.get("is_correct", ""))) for row in rows]
    wrong_steps = [
        str(row.get("first_wrong_step", "") or row.get("answer", "") or "").strip()
        for row in rows
        if not safe_int_local(row.get("correct", row.get("is_correct", "")))
    ]
    wrong_steps = [step for step in wrong_steps if step]
    wrong_step_indices = [
        safe_int_local(row.get("first_wrong_step_index", ""), 0)
        for row in rows
        if not safe_int_local(row.get("correct", row.get("is_correct", "")))
    ]
    wrong_step_indices = [idx for idx in wrong_step_indices if idx > 0]
    return CalibrationBin(
        correct_rate=sum(correct_values) / len(correct_values) if correct_values else 0.0,
        wrong_steps=wrong_steps,
        rows=len(rows),
        wrong_rows=len(wrong_steps),
        wrong_step_indices=wrong_step_indices,
    )


def ability_adjustment(state: StudentState) -> float:
    """Small panel-driven offset whose mean is near zero for the decimal k=100 panel."""
    ice_centered = (state.ice / 100.0) - 0.5
    g_centered = state.g - 0.055
    d_centered = state.d - 0.5
    return 0.04 * ice_centered + 0.70 * g_centered - 0.04 * d_centered


def clamp_probability(value: float) -> float:
    return max(0.02, min(0.98, value))


def simulate_attempt_v2_calibrated(
    state: StudentState,
    meta: EquationMetadata,
    calibration: HumanCalibration,
    seed: int = 1,
    source: str = "algebra_uma_v2_calibrated",
) -> TraceAttempt:
    """Human-calibrated v2 simulator.

    V1 used per-rule failures and produced too many long error traces. V2 treats
    the Cognitive Tutor target as a one-shot first-attempt trace: per equation
    type, it samples whether the student is correct from a human calibration
    file and samples the stuck step from the observed human stuck distribution.
    """
    stable_problem_hash = int(hashlib.md5(meta.equation_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + int(float(state.subjid or 0)) * 1_000_003 + stable_problem_hash)
    correct_trace = build_correct_trace(meta)
    calibration_bin = calibration.by_equation_type.get(meta.equation_type, calibration.global_bin)
    correct_probability = clamp_probability(calibration_bin.correct_rate + ability_adjustment(state))
    solution = format_fraction(meta.solution)

    if rng.random() < correct_probability:
        trace_steps = [
            {
                "step_index": 1,
                "state": meta.equation_text,
                "goal": GOAL_PARSE,
                "exec_rule": "",
                "correct_first_attempt": 1,
                "calibration_source": calibration.source_csv,
            }
        ]
        return TraceAttempt(
            subjid=state.subjid,
            seed=seed,
            prob=meta.equation_text,
            operation=meta.equation_type,
            equation_type=meta.equation_type,
            variable=meta.variable,
            solution=solution,
            strategy=correct_trace.strategy,
            goals=correct_trace.goals,
            exec_rules=correct_trace.exec_rules,
            answer=solution,
            correct=1,
            g=state.g,
            d=state.d,
            c=state.c,
            rt_mu=state.rt_mu,
            ice=state.ice,
            first_wrong_step="",
            trace_steps=trace_steps,
            source=source,
        )

    wrong_steps = calibration_bin.wrong_steps or calibration.global_bin.wrong_steps
    if wrong_steps:
        first_wrong_step = rng.choice(wrong_steps)
    else:
        first_wrong_step = corrupt_step(f"{meta.variable} = {solution}", meta, rng.choice(ERROR_RULES), rng)
    error_rule = "human-calibrated stuck sample"
    trace_steps = [
        {
            "step_index": 1,
            "state": first_wrong_step,
            "goal": GOAL_PARSE,
            "exec_rule": error_rule,
            "correct_first_attempt": 0,
            "error_rule": error_rule,
            "calibration_source": calibration.source_csv,
        }
    ]
    return TraceAttempt(
        subjid=state.subjid,
        seed=seed,
        prob=meta.equation_text,
        operation=meta.equation_type,
        equation_type=meta.equation_type,
        variable=meta.variable,
        solution=solution,
        strategy="human-calibrated firstshot",
        goals=[GOAL_PARSE],
        exec_rules=[error_rule],
        answer=first_wrong_step,
        correct=0,
        g=state.g,
        d=state.d,
        c=state.c,
        rt_mu=state.rt_mu,
        ice=state.ice,
        first_wrong_step=first_wrong_step,
        trace_steps=trace_steps,
        source=source,
    )


def simulate_attempt_v3_ct_flow(
    state: StudentState,
    meta: EquationMetadata,
    calibration: HumanCalibration,
    seed: int = 1,
    source: str = "algebra_uma_v3_ct_flow",
) -> TraceAttempt:
    """Human-calibrated first-shot simulator with problem-local CT-like steps.

    V2 matched human stuck-step strings directly, which was useful as an upper
    bound but produced visible traces that could come from a different problem.
    V3 keeps the same accuracy calibration and student-panel adjustment, but
    samples only the stuck *position* from human data. The step text itself is
    generated from the current equation, so the response reads like a coherent
    Cognitive Tutor equation-state flow.
    """
    stable_problem_hash = int(hashlib.md5(meta.equation_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + int(float(state.subjid or 0)) * 1_000_003 + stable_problem_hash)
    ct_trace = build_ct_flow_trace(meta)
    calibration_bin = calibration.by_equation_type.get(meta.equation_type, calibration.global_bin)
    correct_probability = clamp_probability(calibration_bin.correct_rate + ability_adjustment(state))
    solution = format_fraction(meta.solution)

    if rng.random() < correct_probability:
        return TraceAttempt(
            subjid=state.subjid,
            seed=seed,
            prob=meta.equation_text,
            operation=meta.equation_type,
            equation_type=meta.equation_type,
            variable=meta.variable,
            solution=solution,
            strategy=ct_trace.strategy,
            goals=ct_trace.goals,
            exec_rules=ct_trace.exec_rules,
            answer=solution,
            correct=1,
            g=state.g,
            d=state.d,
            c=state.c,
            rt_mu=state.rt_mu,
            ice=state.ice,
            first_wrong_step="",
            trace_steps=trace_step_records(ct_trace.steps, ct_trace.goals, ct_trace.exec_rules),
            source=source,
        )

    stuck_index = sample_stuck_step_index(calibration_bin, calibration.global_bin, len(ct_trace.steps), rng)
    visible_steps = ct_trace.steps[:stuck_index]
    first_wrong_step = visible_steps[-1]
    trace_records = trace_step_records(visible_steps, ct_trace.goals, ct_trace.exec_rules)
    if trace_records:
        trace_records[-1]["correct_first_attempt"] = 0
        trace_records[-1]["error_rule"] = "human-calibrated stuck position"
    return TraceAttempt(
        subjid=state.subjid,
        seed=seed,
        prob=meta.equation_text,
        operation=meta.equation_type,
        equation_type=meta.equation_type,
        variable=meta.variable,
        solution=solution,
        strategy="human-calibrated CT flow",
        goals=ct_trace.goals,
        exec_rules=list(ct_trace.exec_rules) + ["human-calibrated stuck position"],
        answer=first_wrong_step,
        correct=0,
        g=state.g,
        d=state.d,
        c=state.c,
        rt_mu=state.rt_mu,
        ice=state.ice,
        first_wrong_step=first_wrong_step,
        trace_steps=trace_records,
        source=source,
    )


def sample_stuck_step_index(
    calibration_bin: CalibrationBin,
    global_bin: CalibrationBin,
    trace_len: int,
    rng: random.Random,
) -> int:
    candidates = calibration_bin.wrong_step_indices or global_bin.wrong_step_indices
    if candidates:
        sampled = rng.choice(candidates)
    else:
        sampled = 1
    return max(1, min(trace_len, sampled))


def generate_synthetic_equations(count: int, seed: int = 1) -> list[EquationMetadata]:
    rng = random.Random(seed)
    metas: list[EquationMetadata] = []
    seen: set[str] = set()
    templates = [
        "ax_eq_b",
        "x_plus_b_eq_c",
        "ax_plus_b_eq_c",
        "ax_plus_b_eq_cx_plus_d",
        "x_over_a_eq_b",
        "a_over_x_eq_b",
        "a_over_x_plus_b_eq_c",
        "a_times_x_plus_b_eq_c",
    ]
    while len(metas) < count:
        variable = rng.choice(["x", "y"])
        x_value = rng.choice([v for v in range(-12, 13) if v != 0])
        a = rng.choice([v for v in range(-9, 10) if v not in (0, 1, -1)])
        b = rng.randint(-12, 12)
        c = rng.randint(-12, 12)
        template = templates[len(metas) % len(templates)]
        if template == "ax_eq_b":
            equation = f"{format_coeff_for_surface(a, variable)} = {a * x_value}"
        elif template == "x_plus_b_eq_c":
            equation = f"{variable}{signed_int(b)} = {x_value + b}"
        elif template == "ax_plus_b_eq_c":
            equation = f"{format_coeff_for_surface(a, variable)}{signed_int(b)} = {a * x_value + b}"
        elif template == "ax_plus_b_eq_cx_plus_d":
            c_coeff = rng.choice([v for v in range(-6, 7) if v not in (0, a)])
            d = (a - c_coeff) * x_value + b
            equation = (
                f"{format_coeff_for_surface(a, variable)}{signed_int(b)} = "
                f"{format_coeff_for_surface(c_coeff, variable)}{signed_int(d)}"
            )
        elif template == "x_over_a_eq_b":
            denom = rng.choice([v for v in range(2, 10)])
            rhs = Fraction(x_value, denom)
            equation = f"{variable}/{denom} = {format_fraction(rhs)}"
        elif template == "a_over_x_eq_b":
            rhs = rng.choice([v for v in range(-8, 9) if v not in (0,)])
            numerator = x_value * rhs
            equation = f"{numerator}/{variable} = {rhs}"
        elif template == "a_over_x_plus_b_eq_c":
            rhs = rng.choice([v for v in range(-8, 9) if v not in (0,)])
            numerator = x_value * (rhs - b)
            equation = f"{numerator}/{variable}{signed_int(b)} = {rhs}"
        else:
            c_value = a * (x_value + b)
            equation = f"{a}*({variable}{signed_int(b)}) = {c_value}"
        if equation in seen:
            continue
        seen.add(equation)
        meta = get_equation_metadata(equation)
        if meta is not None:
            metas.append(meta)
    return metas


def format_coeff_for_surface(coefficient: int, variable: str) -> str:
    if coefficient == 1:
        return variable
    if coefficient == -1:
        return f"-{variable}"
    return f"{coefficient}{variable}"


def signed_int(value: int) -> str:
    if value == 0:
        return ""
    if value > 0:
        return f"+{value}"
    return str(value)


def generate_synthetic_equations_matched(
    reference_metas: Sequence[EquationMetadata],
    count: int,
    seed: int = 1,
) -> list[EquationMetadata]:
    """Generate synthetic equations with reference equation-type proportions.

    The generator samples reference problems with replacement and synthesizes a
    new equation that matches the sampled problem's equation type and variable.
    This is deliberately distributional rather than a clone of any reference
    problem: coefficients, constants, and hidden solutions are resampled.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    references = list(reference_metas)
    if count == 0:
        return []
    if not references:
        raise ValueError("reference_metas must be nonempty")

    rng = random.Random(seed)
    if count <= len(references):
        sampled_refs = rng.sample(references, count)
    else:
        sampled_refs = [references[idx % len(references)] for idx in range(count)]
        rng.shuffle(sampled_refs)

    metas: list[EquationMetadata] = []
    seen: set[str] = set()
    for idx, reference in enumerate(sampled_refs):
        meta = synthesize_equation_like(reference, rng, seen, salt=idx)
        metas.append(meta)
        seen.add(meta.equation_text)
    return metas


def synthesize_equation_like(
    reference: EquationMetadata,
    rng: random.Random,
    seen: set[str],
    salt: int = 0,
) -> EquationMetadata:
    for attempt in range(500):
        equation = synthesize_equation_surface(reference, rng, salt=salt + attempt)
        meta = get_equation_metadata(equation)
        if (
            meta is not None
            and meta.equation_type == reference.equation_type
            and meta.variable == reference.variable
            and meta.equation_text not in seen
        ):
            return meta
    raise RuntimeError(f"Could not synthesize matched equation for {reference.equation_text!r}")


def random_nonzero_int(rng: random.Random, lo: int = -50, hi: int = 50, exclude: set[int] | None = None) -> int:
    excluded = {0}
    if exclude:
        excluded |= set(exclude)
    choices = [value for value in range(lo, hi + 1) if value not in excluded]
    return rng.choice(choices)


def random_solution(rng: random.Random, allow_zero: bool = True) -> int:
    choices = [value for value in range(-200, 201) if allow_zero or value != 0]
    return rng.choice(choices)


def synthesize_equation_surface(reference: EquationMetadata, rng: random.Random, salt: int = 0) -> str:
    variable = reference.variable
    text = reference.equation_text
    x_value = random_solution(rng, allow_zero=reference.equation_type != "reciprocal_variable_denominator")
    a = random_nonzero_int(rng, -12, 12, exclude={1, -1})
    b = rng.randint(-20, 20)

    if reference.equation_type == "linear_one_step":
        return f"{format_coeff_for_surface(a, variable)} = {a * x_value}"

    if reference.equation_type == "linear_two_step":
        if "/" in text and "(" in text:
            denominator = rng.choice([value for value in range(2, 51)])
            rhs = Fraction(a * x_value + b, denominator)
            return f"({format_coeff_for_surface(a, variable)}{signed_int(b)})/{denominator} = {format_fraction(rhs)}"
        if "(" in text:
            rhs = a * (x_value + b)
            return f"{a}*({variable}{signed_int(b)}) = {rhs}"
        return f"{format_coeff_for_surface(a, variable)}{signed_int(b)} = {a * x_value + b}"

    if reference.equation_type == "linear_variable_both_sides":
        c_coeff = random_nonzero_int(rng, -10, 10, exclude={a})
        d = (a - c_coeff) * x_value + b
        return (
            f"{format_coeff_for_surface(a, variable)}{signed_int(b)} = "
            f"{format_coeff_for_surface(c_coeff, variable)}{signed_int(d)}"
        )

    if reference.equation_type == "linear_with_denominator":
        denominator = rng.choice([value for value in range(2, 51)])
        if re.search(rf"{variable}\s*/[^=]+[+-]", text) or re.search(rf"[+-].*{variable}\s*/", text):
            offset = rng.choice([value for value in range(-50, 51) if value != 0])
            rhs = Fraction(x_value, denominator) + offset
            return f"{variable}/{denominator}{signed_int(offset)} = {format_fraction(rhs)}"
        if "*" in text or re.search(rf"\d+{variable}/", text):
            coefficient = random_nonzero_int(rng, -50, 50, exclude={1, -1})
            rhs = Fraction(coefficient * x_value, denominator)
            return f"{coefficient}{variable}/{denominator} = {format_fraction(rhs)}"
        rhs = Fraction(x_value, denominator)
        return f"{variable}/{denominator} = {format_fraction(rhs)}"

    if reference.equation_type == "reciprocal_variable_denominator":
        rhs = random_nonzero_int(rng, -12, 12)
        if "+" in text or re.search(rf"/{variable}\s*-", text):
            offset = rng.choice([value for value in range(-50, 51) if value not in (0, rhs)])
            numerator = x_value * (rhs - offset)
            return f"{numerator}/{variable}{signed_int(offset)} = {rhs}"
        numerator = x_value * rhs
        return f"{numerator}/{variable} = {rhs}"

    # Defensive fallback: this should not be reached for the current v1 scope.
    return f"{format_coeff_for_surface(a, variable)}{signed_int(b)} = {a * x_value + b}"


def read_panel(path: Path = DEFAULT_PANEL_CSV) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_problem_csv(path: Path, problem_col: str = "prob", max_problems: int | None = None) -> list[EquationMetadata]:
    metas: list[EquationMetadata] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if problem_col not in fieldnames and "equation_text" in fieldnames:
            problem_col = "equation_text"
        if problem_col not in fieldnames:
            raise ValueError(f"{path} is missing problem column {problem_col!r}")
        for row in reader:
            meta = get_equation_metadata(row.get(problem_col, ""))
            if meta is None or meta.equation_text in seen:
                continue
            seen.add(meta.equation_text)
            metas.append(meta)
            if max_problems is not None and len(metas) >= max_problems:
                break
    return metas


def write_trace_rows(path: Path, rows: Sequence[dict[str, object]], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UMA_TRACE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
