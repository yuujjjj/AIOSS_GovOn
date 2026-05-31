"""Validate the committed LLM persona UAT configuration."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_config(name: str):
    with (PROJECT_ROOT / "configs" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_has_at_least_ten_distinct_llm_user_personas():
    personas = _load_config("llm_user_personas.json")

    assert len(personas) >= 10
    assert len({persona["id"] for persona in personas}) == len(personas)
    assert all(persona["system_prompt"] for persona in personas)
    assert all(persona["evaluation_focus"] for persona in personas)


def test_has_scenarios_covering_all_civil_complaint_categories():
    scenarios = _load_config("uat_scenarios.json")
    categories = {scenario["category"] for scenario in scenarios}

    assert len(scenarios) >= 10
    assert categories == {"건축", "교통", "기타", "복지", "세금", "안전", "행정", "환경"}
    assert all(scenario["expected_elements"] for scenario in scenarios)
