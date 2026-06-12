import argparse
import json
from pathlib import Path

import pandas as pd

from eval import evaluate_qwen_humanlike_preference as qhp


def test_normalize_model_csv_canonicalizes_fraction_division(tmp_path: Path) -> None:
    path = tmp_path / "model.csv"
    pd.DataFrame(
        [
            {
                "problem": "2/3 ÷ 3/5",
                "parsed_answer": "10/9",
                "model_response": "invert and multiply\n### answer: 10/9",
                "sample_idx": 7,
            }
        ]
    ).to_csv(path, index=False)

    frame = qhp.normalize_model_csv(path, "demo", "fraction")

    assert frame.loc[0, "prob"] == "2/3:3/5"
    assert frame.loc[0, "entity"] == "demo"
    assert frame.loc[0, "answer"] == "10/9"


def test_optional_low_count_source_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "frontier.csv"
    pd.DataFrame(
        [
            {"problem": "3/5+1/5", "model_response": "a", "parsed_answer": "4/5"},
            {"problem": "3/5+1/5", "model_response": "b", "parsed_answer": "4/5"},
        ]
    ).to_csv(path, index=False)
    sources: dict[str, qhp.EntityFrame] = {}
    inventory: list[dict[str, object]] = []

    qhp.add_source(
        sources,
        inventory,
        entity="frontier",
        domain="fraction",
        path=path,
        loader=lambda source_path: qhp.normalize_model_csv(source_path, "frontier", "fraction"),
        samples_per_problem=3,
        required=False,
    )

    assert "frontier" not in sources
    assert inventory[0]["status"] == "skipped_low_count"


def test_apply_baseline_filter_marks_unselected_inventory() -> None:
    sources = {
        "qwen": qhp.EntityFrame("qwen", "fraction", Path("q.csv"), pd.DataFrame()),
        "claude_sonnet_4_6": qhp.EntityFrame("claude_sonnet_4_6", "fraction", Path("c.csv"), pd.DataFrame()),
        "human": qhp.EntityFrame("human", "fraction", Path("h.csv"), pd.DataFrame()),
    }
    inventory = pd.DataFrame(
        [
            {"entity": "qwen", "status": "included", "reason": ""},
            {"entity": "claude_sonnet_4_6", "status": "included", "reason": ""},
            {"entity": "human", "status": "included", "reason": ""},
        ]
    )

    filtered, filtered_inventory = qhp.apply_baseline_filter(
        qwen_entity="qwen",
        sources=sources,
        inventory=inventory,
        only_baselines="claude_sonnet_4_6",
    )

    assert set(filtered) == {"qwen", "claude_sonnet_4_6"}
    human_status = filtered_inventory.set_index("entity").loc["human", "status"]
    assert human_status == "skipped_by_filter"


def test_load_fraction_sources_honors_qwen_override(tmp_path: Path, monkeypatch) -> None:
    qwen_path = tmp_path / "qwen4b.csv"
    calls: list[dict[str, object]] = []

    def fake_add_source(
        sources,
        inventory,
        *,
        entity,
        domain,
        path,
        loader,
        samples_per_problem,
        sample_with_replacement=False,
        required=True,
    ):
        calls.append({"entity": entity, "path": path})
        frame = pd.DataFrame({"prob": ["3/5+1/5"], "response_text": ["I got 4/5."]})
        inventory.append(
            {
                "domain": domain,
                "entity": entity,
                "source_path": str(path),
                "status": "included",
                "reason": "",
            }
        )
        sources[entity] = qhp.EntityFrame(entity, domain, path, frame, sample_with_replacement, required)

    monkeypatch.setattr(qhp, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(qhp, "add_source", fake_add_source)
    args = argparse.Namespace(
        samples_per_problem=1,
        fraction_qwen_csv=str(qwen_path),
        fraction_qwen_entity="qwen3_4b_distill_humanft",
    )

    qwen_entity, sources, inventory = qhp.load_fraction_sources(args)

    assert qwen_entity == "qwen3_4b_distill_humanft"
    assert calls[0] == {"entity": "qwen3_4b_distill_humanft", "path": qwen_path}
    assert "qwen3_4b_distill_humanft" in sources
    assert inventory.iloc[0]["entity"] == "qwen3_4b_distill_humanft"


def test_mocked_gemini_response_parses_preferred_entity(tmp_path: Path) -> None:
    record = {
        "request_id": "r1",
        "request_key": "k1",
        "user_prompt": "Problem: 3/5+1/5",
        "system_prompt": "system",
        "preference_set": "qwen_vs_human",
        "prob": "3/5+1/5",
        "qwen_entity": "qwen",
        "baseline_entity": "human",
        "qwen_side": "A",
        "a_entity": "qwen",
        "b_entity": "human",
    }
    args = argparse.Namespace(
        judge_model="gemini-3.1-flash-lite",
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        thinking_budget=0,
        max_retries=1,
        retry_base_seconds=0.0,
        force=True,
    )

    class FakeModels:
        def generate_content(self, **payload):
            assert payload["model"] == "gemini-3.1-flash-lite"
            return {
                "text": json.dumps(
                    {
                        "preferred": "A",
                        "confidence": "medium",
                        "reason": "It sounds like a short student answer.",
                    }
                ),
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 7,
                    "totalTokenCount": 12,
                },
            }

    class FakeClient:
        models = FakeModels()

    (tmp_path / "cache").mkdir()
    cached = qhp.run_gemini_request(record, args, tmp_path, FakeClient())
    pred = qhp.prediction_row(record, cached)

    assert pred["preferred"] == "A"
    assert pred["preferred_entity"] == "qwen"
    assert pred["parse_success"] is True
    assert pred["total_tokens"] == 12
