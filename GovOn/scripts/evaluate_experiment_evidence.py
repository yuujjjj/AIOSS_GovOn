#!/usr/bin/env python3
"""Validate and summarize feature-flag experiment evidence.

The script is intentionally dependency-free so it can run in GitHub Actions
without installing the full ML stack.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PRIMARY_EXPERIMENT = "complaint_response_layout"
CONTROL_VARIANT = "control"
TREATMENT_VARIANT = "guided"


@dataclass(frozen=True)
class VariantSummary:
    variant: str
    exposures: int
    task_success_rate: float
    avg_time_to_first_draft_sec: float
    self_service_resolution_rate: float
    followup_rate: float
    satisfaction_avg: float


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
    return rows


def load_metrics(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_feedback(rows: list[dict]) -> None:
    require(len(rows) >= 10, "At least 10 LLM user feedback rows are required")
    persona_patterns = {row.get("persona_pattern") for row in rows}
    require(len(persona_patterns) >= 10, "Feedback rows must contain 10 unique persona patterns")

    for row in rows:
        require(row.get("panel_type") == "llm-simulated-user", "Feedback must be LLM simulated")
        require(bool(row.get("scenario")), "Each feedback row needs a scenario")
        require(
            bool(row.get("generated_code_id")), "Each feedback row needs generated code evidence"
        )
        evaluation = row.get("evaluation") or {}
        require("task_success" in evaluation, "Each feedback row needs task_success evaluation")
        require("satisfaction_score" in evaluation, "Each feedback row needs satisfaction_score")


def validate_metrics(rows: list[dict]) -> None:
    dates = {row["date"] for row in rows if row.get("experiment_key") == PRIMARY_EXPERIMENT}
    variants = {row["variant"] for row in rows if row.get("experiment_key") == PRIMARY_EXPERIMENT}
    require(len(dates) >= 14, "A/B metrics must cover at least 14 days")
    require(
        {CONTROL_VARIANT, TREATMENT_VARIANT}.issubset(variants), "Both A/B variants are required"
    )


def weighted_average(rows: Iterable[dict], metric: str) -> float:
    weighted_total = 0.0
    exposure_total = 0
    for row in rows:
        exposures = int(row["exposures"])
        weighted_total += float(row[metric]) * exposures
        exposure_total += exposures
    return weighted_total / exposure_total


def summarize_variant(rows: list[dict], variant: str) -> VariantSummary:
    selected = [
        row
        for row in rows
        if row.get("experiment_key") == PRIMARY_EXPERIMENT and row.get("variant") == variant
    ]
    exposures = sum(int(row["exposures"]) for row in selected)
    return VariantSummary(
        variant=variant,
        exposures=exposures,
        task_success_rate=weighted_average(selected, "task_success_rate"),
        avg_time_to_first_draft_sec=weighted_average(selected, "avg_time_to_first_draft_sec"),
        self_service_resolution_rate=weighted_average(selected, "self_service_resolution_rate"),
        followup_rate=weighted_average(selected, "followup_rate"),
        satisfaction_avg=weighted_average(selected, "satisfaction_avg"),
    )


def percentage_point_delta(treatment: float, control: float) -> float:
    return (treatment - control) * 100


def relative_delta(treatment: float, control: float) -> float:
    return ((treatment - control) / control) * 100


def make_decision(control: VariantSummary, treatment: VariantSummary) -> tuple[str, list[str]]:
    success_delta = percentage_point_delta(treatment.task_success_rate, control.task_success_rate)
    time_delta = relative_delta(
        treatment.avg_time_to_first_draft_sec, control.avg_time_to_first_draft_sec
    )
    satisfaction_delta = treatment.satisfaction_avg - control.satisfaction_avg
    followup_delta = percentage_point_delta(treatment.followup_rate, control.followup_rate)

    reasons = [
        f"task success changed by {success_delta:.1f}pp",
        f"time to first draft changed by {time_delta:.1f}%",
        f"satisfaction changed by {satisfaction_delta:.2f}",
        f"follow-up rate changed by {followup_delta:.1f}pp",
    ]
    if success_delta >= 5.0 and time_delta <= -10.0 and satisfaction_delta >= 0.30:
        return "persevere", reasons
    return "pivot", reasons


def feedback_summary(rows: list[dict]) -> dict:
    by_variant: dict[str, list[float]] = defaultdict(list)
    task_success = Counter()
    top_issues = Counter()
    for row in rows:
        variant = row["feature_flag_context"]["variant"]
        evaluation = row["evaluation"]
        by_variant[variant].append(float(evaluation["satisfaction_score"]))
        task_success[str(evaluation["task_success"])] += 1
        for issue in evaluation.get("raised_issues", []):
            top_issues[issue] += 1

    return {
        "feedback_count": len(rows),
        "unique_persona_patterns": len({row["persona_pattern"] for row in rows}),
        "task_success_count": dict(task_success),
        "satisfaction_by_variant": {
            variant: sum(scores) / len(scores) for variant, scores in by_variant.items()
        },
        "top_issues": dict(top_issues.most_common(5)),
    }


def build_report(feedback_rows: list[dict], metric_rows: list[dict]) -> tuple[str, dict]:
    validate_feedback(feedback_rows)
    validate_metrics(metric_rows)

    control = summarize_variant(metric_rows, CONTROL_VARIANT)
    treatment = summarize_variant(metric_rows, TREATMENT_VARIANT)
    decision, reasons = make_decision(control, treatment)
    feedback = feedback_summary(feedback_rows)
    dates = sorted(
        {row["date"] for row in metric_rows if row["experiment_key"] == PRIMARY_EXPERIMENT}
    )

    summary = {
        "experiment_key": PRIMARY_EXPERIMENT,
        "date_range": {"start": dates[0], "end": dates[-1], "days": len(dates)},
        "control": control.__dict__,
        "treatment": treatment.__dict__,
        "decision": decision,
        "decision_reasons": reasons,
        "feedback": feedback,
    }

    report = f"""# Feature Flag A/B Experiment Weekly Report

## Scope

- Experiment: `{PRIMARY_EXPERIMENT}`
- Variants: `{CONTROL_VARIANT}` vs `{TREATMENT_VARIANT}`
- Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)
- LLM feedback panel: {feedback["feedback_count"]} users with {feedback["unique_persona_patterns"]} unique persona patterns

## Metric Summary

| Metric | Control | Guided | Change |
|---|---:|---:|---:|
| Exposures | {control.exposures} | {treatment.exposures} | {treatment.exposures - control.exposures:+d} |
| Task success rate | {control.task_success_rate:.3f} | {treatment.task_success_rate:.3f} | {percentage_point_delta(treatment.task_success_rate, control.task_success_rate):+.1f}pp |
| Avg time to first draft | {control.avg_time_to_first_draft_sec:.1f}s | {treatment.avg_time_to_first_draft_sec:.1f}s | {relative_delta(treatment.avg_time_to_first_draft_sec, control.avg_time_to_first_draft_sec):+.1f}% |
| Self-service resolution rate | {control.self_service_resolution_rate:.3f} | {treatment.self_service_resolution_rate:.3f} | {percentage_point_delta(treatment.self_service_resolution_rate, control.self_service_resolution_rate):+.1f}pp |
| Follow-up rate | {control.followup_rate:.3f} | {treatment.followup_rate:.3f} | {percentage_point_delta(treatment.followup_rate, control.followup_rate):+.1f}pp |
| Satisfaction average | {control.satisfaction_avg:.2f} | {treatment.satisfaction_avg:.2f} | {treatment.satisfaction_avg - control.satisfaction_avg:+.2f} |

## Feedback Summary

- Task success counts: {feedback["task_success_count"]}
- Satisfaction by variant: {feedback["satisfaction_by_variant"]}
- Top issues: {feedback["top_issues"]}

## Decision

Decision: **{decision.upper()}**

Reasons:

{chr(10).join(f"- {reason}" for reason in reasons)}
"""
    return report, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    report, summary = build_report(load_jsonl(args.feedback), load_metrics(args.metrics))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"decision={summary['decision']}")
    print(f"report={args.report}")
    print(f"summary={args.summary}")


if __name__ == "__main__":
    main()
