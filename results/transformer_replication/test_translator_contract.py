from __future__ import annotations

import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import translate_uma_traces_to_nlp as translator
import translator_contract_spec as contract


def _assert_substrings(text: str, required: tuple[str, ...], forbidden: tuple[str, ...] = ()) -> None:
    for phrase in required:
        assert phrase in text
    for phrase in forbidden:
        assert phrase not in text


@pytest.mark.parametrize("case", contract.STRATEGY_BRANCH_CASES, ids=lambda case: case.name)
def test_strategy_branch_contract(case: contract.ChildReasoningCase):
    reasoning, quality = translator.build_child_reasoning(
        prob=case.prob,
        operation=case.operation,
        strategy_code=case.strategy_code,
        goals=list(case.goals),
        exec_rules=list(case.exec_rules),
        answer=case.answer,
        surface_hidden_trace_steps=case.surface_hidden_trace_steps,
    )

    _assert_substrings(reasoning, case.expected_substrings, case.forbidden_substrings)
    if case.expected_verified_claims is not None:
        assert quality["reasoning_verified_claims"] == case.expected_verified_claims
    if case.expected_unverified_claims is not None:
        assert quality["reasoning_unverified_claims"] == case.expected_unverified_claims


@pytest.mark.parametrize("case", contract.DETAIL_REASONING_CASES, ids=lambda case: case.name)
def test_detail_reasoning_contract(case: contract.DetailReasoningCase):
    sentences = translator.build_reasoning_details(
        list(case.goals),
        list(case.exec_rules),
        surface_hidden_trace_steps=case.surface_hidden_trace_steps,
    )
    joined = ". ".join(sentences)
    _assert_substrings(joined, case.expected_sentences)


@pytest.mark.parametrize("case", contract.INTERACTION_CASES, ids=lambda case: case.name)
def test_interaction_contract(case: contract.ChildReasoningCase):
    reasoning, quality = translator.build_child_reasoning(
        prob=case.prob,
        operation=case.operation,
        strategy_code=case.strategy_code,
        goals=list(case.goals),
        exec_rules=list(case.exec_rules),
        answer=case.answer,
        surface_hidden_trace_steps=case.surface_hidden_trace_steps,
    )

    _assert_substrings(reasoning, case.expected_substrings, case.forbidden_substrings)
    if case.expected_verified_claims is not None:
        assert quality["reasoning_verified_claims"] == case.expected_verified_claims
    if case.expected_unverified_claims is not None:
        assert quality["reasoning_unverified_claims"] == case.expected_unverified_claims


@pytest.mark.parametrize("case", contract.IGNORED_RULE_CASES, ids=lambda case: case.name)
def test_ignored_internal_rules_do_not_surface(case: contract.IgnoredRuleCase):
    record = {
        "prob": case.prob,
        "operation": case.operation,
        "strategy": case.strategy_code,
        "goals": " ".join(case.goals),
        "exec": " ".join(case.exec_rules),
        "answer": case.answer,
        "correct": case.answer,
        "is_correct": "1",
        "g": "0.01",
        "d": "0.1",
        "rt_mu": "3",
        "ice": "0",
    }
    out = translator.translate_record(
        record,
        include_outcome_text=False,
        student_prompt_config=translator.StudentPromptConfig(mode="none", drop_prob=0.0, drop_seed=0),
        reasoning_mode="clean_child",
        surface_hidden_trace_steps=case.surface_hidden_trace_steps,
    )

    _assert_substrings(out["response_nl"], (), case.forbidden_substrings)
    assert case.expected_answer_suffix in out["response_nl"]
