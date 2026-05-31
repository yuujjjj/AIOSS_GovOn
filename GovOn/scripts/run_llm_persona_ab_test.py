#!/usr/bin/env python3
"""Run the GovOn RAG A/B test with ten distinct LLM user personas.

Live mode calls the GovOn runtime and an OpenAI-compatible evaluator LLM.
Dry-run mode only validates storage and reporting with synthetic feedback.
Dry-run output is not valid evidence of LLM evaluation or 14-day operation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.ab_testing import ExperimentStore, render_markdown_report
from src.inference.feature_flags import FeatureFlags


def _load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def _request_json(
    method: str,
    url: str,
    *,
    payload: Optional[dict] = None,
    headers: Optional[dict[str, str]] = None,
) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc


def _auth_headers(api_key: Optional[str]) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _parse_llm_json(content: str) -> dict:
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("evaluator LLM did not return a JSON object")
    return json.loads(content[start : end + 1])


def _evaluate_with_llm(
    *,
    evaluator_base_url: str,
    evaluator_model: str,
    evaluator_api_key: Optional[str],
    persona: dict,
    scenario: dict,
    draft: str,
) -> dict:
    prompt = {
        "persona": persona,
        "scenario": scenario,
        "generated_draft": draft,
        "required_output": {
            "rating": "integer from 1 to 5",
            "task_success": "boolean",
            "comment": "short Korean feedback",
            "final_content": "optional improved draft or null",
        },
    }
    headers = {"Authorization": f"Bearer {evaluator_api_key}"} if evaluator_api_key else {}
    response = _request_json(
        "POST",
        f"{evaluator_base_url.rstrip('/')}/chat/completions",
        headers=headers,
        payload={
            "model": evaluator_model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": persona["system_prompt"]},
                {
                    "role": "user",
                    "content": (
                        "민원 답변 초안을 사용자 관점에서 평가하세요. "
                        "과장하지 말고 JSON 객체만 반환하세요.\n\n"
                        + json.dumps(prompt, ensure_ascii=False)
                    ),
                },
            ],
        },
    )
    result = _parse_llm_json(response["choices"][0]["message"]["content"])
    return {
        "rating": int(result["rating"]),
        "task_success": bool(result["task_success"]),
        "comment": str(result.get("comment", "")),
        "final_content": result.get("final_content"),
    }


def _dry_run_feedback(variant: str, scenario: dict) -> dict:
    rag_enabled = variant == "treatment_rag_on"
    return {
        "rating": 4 if rag_enabled else 3,
        "task_success": rag_enabled,
        "comment": (
            "dry-run synthetic feedback: RAG evidence path validated"
            if rag_enabled
            else "dry-run synthetic feedback: control path validated"
        ),
        "final_content": None,
        "draft": " / ".join(scenario["expected_elements"]),
    }


def _write_report(path: Path, summary: dict, *, dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if dry_run:
        prefix = (
            "> Warning: this is a synthetic dry-run report. It validates code paths only. "
            "It is not evidence of LLM evaluation or 14-day operation.\n\n"
        )
    path.write_text(prefix + render_markdown_report(summary), encoding="utf-8")


def _run_dry_test(args: argparse.Namespace, personas: list[dict], scenarios: list[dict]) -> dict:
    store = ExperimentStore(db_path=args.db_path)
    for index, persona in enumerate(personas):
        scenario = scenarios[index % len(scenarios)]
        participant_id = f"llm-user-{persona['id']}"
        assignment = store.get_or_assign(participant_id, persona_id=persona["id"])
        request_id = f"dry-run-{uuid.uuid4()}"
        flags = FeatureFlags(
            use_rag_pipeline=assignment.use_rag_pipeline,
            model_version="v2_lora",
        )
        feedback = _dry_run_feedback(assignment.variant, scenario)
        store.record_exposure(
            request_id=request_id,
            participant_id=participant_id,
            persona_id=persona["id"],
            scenario_id=scenario["id"],
            endpoint="/v1/generate-civil-response",
            flags=flags,
            latency_ms=120 if assignment.use_rag_pipeline else 80,
            success=True,
            metadata={"evaluation_mode": "dry_run"},
        )
        store.record_feedback(
            request_id=request_id,
            participant_id=participant_id,
            persona_id=persona["id"],
            scenario_id=scenario["id"],
            rating=feedback["rating"],
            task_success=feedback["task_success"],
            comment=feedback["comment"],
            evaluator_model="deterministic-dry-run",
            metadata={"evaluation_mode": "dry_run"},
        )
    return store.summarize(days=args.days)


def _run_live_test(args: argparse.Namespace, personas: list[dict], scenarios: list[dict]) -> dict:
    if not args.evaluator_base_url or not args.evaluator_model:
        raise ValueError("live mode requires --evaluator-base-url and --evaluator-model")
    target_headers = _auth_headers(args.target_api_key)
    for index, persona in enumerate(personas):
        scenario = scenarios[index % len(scenarios)]
        participant_id = f"llm-user-{persona['id']}"
        headers = {
            **target_headers,
            "X-GovOn-Participant-ID": participant_id,
            "X-GovOn-Persona-ID": persona["id"],
            "X-GovOn-Scenario-ID": scenario["id"],
        }
        generation = _request_json(
            "POST",
            f"{args.target_base_url.rstrip('/')}/v1/generate-civil-response",
            headers=headers,
            payload={
                "prompt": scenario["complaint"],
                "complaint_id": scenario["id"],
            },
        )
        feedback = _evaluate_with_llm(
            evaluator_base_url=args.evaluator_base_url,
            evaluator_model=args.evaluator_model,
            evaluator_api_key=args.evaluator_api_key,
            persona=persona,
            scenario=scenario,
            draft=generation["text"],
        )
        _request_json(
            "POST",
            f"{args.target_base_url.rstrip('/')}/feedback/submit",
            headers=target_headers,
            payload={
                "request_id": generation["request_id"],
                "participant_id": participant_id,
                "persona_id": persona["id"],
                "scenario_id": scenario["id"],
                "rating": feedback["rating"],
                "task_success": feedback["task_success"],
                "comment": feedback["comment"],
                "final_content": feedback["final_content"],
                "evaluator_model": args.evaluator_model,
                "metadata": {"evaluation_mode": "llm"},
            },
        )

    query = urllib.parse.urlencode({"days": args.days})
    return _request_json(
        "GET",
        f"{args.target_base_url.rstrip('/')}/v1/experiments/rag-ab/metrics?{query}",
        headers=target_headers,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db-path", default=str(PROJECT_ROOT / ".cache" / "uat-dry-run.sqlite3"))
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--target-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--target-api-key", default=os.getenv("GOVON_API_KEY"))
    parser.add_argument("--evaluator-base-url", default=os.getenv("LLM_EVALUATOR_BASE_URL"))
    parser.add_argument("--evaluator-api-key", default=os.getenv("LLM_EVALUATOR_API_KEY"))
    parser.add_argument("--evaluator-model", default=os.getenv("LLM_EVALUATOR_MODEL"))
    parser.add_argument(
        "--personas",
        type=Path,
        default=PROJECT_ROOT / "configs" / "llm_user_personas.json",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=PROJECT_ROOT / "configs" / "uat_scenarios.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    personas = _load_json(args.personas)
    scenarios = _load_json(args.scenarios)
    if len(personas) < 10:
        raise ValueError("at least 10 distinct LLM personas are required")
    if not scenarios:
        raise ValueError("at least one UAT scenario is required")

    started_at = time.monotonic()
    if args.dry_run:
        summary = _run_dry_test(args, personas, scenarios)
        default_output = (
            PROJECT_ROOT / "docs" / "outputs" / "M4_Testing" / "ab-test-dry-run-report.md"
        )
    else:
        summary = _run_live_test(args, personas, scenarios)
        default_output = PROJECT_ROOT / "docs" / "outputs" / "M4_Testing" / "ab-test-report.md"
    output = args.output or default_output
    _write_report(output, summary, dry_run=args.dry_run)
    print(f"report={output}")
    print(f"elapsed_seconds={time.monotonic() - started_at:.2f}")
    print(json.dumps(summary["completion"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
