"""Feature Flag A/B experiment store tests."""

from src.inference.ab_testing import (
    CONTROL_VARIANT,
    MINIMUM_FEEDBACK_PARTICIPANTS,
    MINIMUM_OPERATION_DAYS,
    TREATMENT_VARIANT,
    ExperimentStore,
    render_markdown_report,
)
from src.inference.feature_flags import FeatureFlags


def _new_store(tmp_path):
    return ExperimentStore(db_path=str(tmp_path / "experiments.sqlite3"), seed="test-seed")


def test_assignment_is_stable_and_applies_rag_flag(tmp_path):
    store = _new_store(tmp_path)

    first = store.get_or_assign("participant-1", persona_id="persona-1")
    second = store.get_or_assign("participant-1", persona_id="persona-other")
    flags = first.apply(FeatureFlags(use_rag_pipeline=not first.use_rag_pipeline))

    assert second.variant == first.variant
    assert second.persona_id == "persona-1"
    assert flags.use_rag_pipeline is first.use_rag_pipeline


def test_new_assignments_stay_balanced_for_small_cohorts(tmp_path):
    store = _new_store(tmp_path)

    assignments = [
        store.get_or_assign(f"participant-{index}").variant
        for index in range(MINIMUM_FEEDBACK_PARTICIPANTS)
    ]

    assert assignments.count(CONTROL_VARIANT) == 5
    assert assignments.count(TREATMENT_VARIANT) == 5


def test_record_feedback_requires_matching_exposure(tmp_path):
    store = _new_store(tmp_path)

    try:
        store.record_feedback(
            request_id="missing-request",
            participant_id="participant-1",
            rating=4,
            task_success=True,
        )
    except ValueError as exc:
        assert "unknown request_id" in str(exc)
    else:
        raise AssertionError("record_feedback must reject unknown request IDs")


def test_summarize_tracks_ten_personas_and_fourteen_days(tmp_path):
    store = _new_store(tmp_path)
    first_timestamp = 1_700_000_000.0
    last_timestamp = first_timestamp + (MINIMUM_OPERATION_DAYS * 86400)
    variants_seen = set()

    for index in range(MINIMUM_FEEDBACK_PARTICIPANTS):
        participant_id = f"participant-{index}"
        persona_id = f"persona-{index}"
        assignment = store.get_or_assign(
            participant_id,
            persona_id=persona_id,
            assigned_at=first_timestamp,
        )
        variants_seen.add(assignment.variant)
        flags = FeatureFlags(use_rag_pipeline=assignment.use_rag_pipeline)
        timestamp = first_timestamp if index == 0 else last_timestamp
        request_id = f"request-{index}"
        store.record_exposure(
            request_id=request_id,
            participant_id=participant_id,
            persona_id=persona_id,
            scenario_id=f"scenario-{index}",
            endpoint="/v1/generate-civil-response",
            flags=flags,
            latency_ms=100 + index,
            success=True,
            created_at=timestamp,
        )
        store.record_feedback(
            request_id=request_id,
            participant_id=participant_id,
            persona_id=persona_id,
            rating=4 if assignment.use_rag_pipeline else 3,
            task_success=assignment.use_rag_pipeline,
            created_at=timestamp,
        )

    summary = store.summarize(days=MINIMUM_OPERATION_DAYS, now=last_timestamp)

    assert variants_seen == {CONTROL_VARIANT, TREATMENT_VARIANT}
    assert summary["completion"]["has_minimum_operation_days"] is True
    assert summary["completion"]["has_minimum_feedback_participants"] is True
    assert summary["completion"]["has_minimum_personas"] is True
    assert summary["delta_treatment_minus_control"]["avg_rating"] == 1


def test_summary_exposes_incomplete_dry_run_state(tmp_path):
    store = _new_store(tmp_path)
    assignment = store.get_or_assign("participant-1", persona_id="persona-1")
    request_id = "request-1"
    store.record_exposure(
        request_id=request_id,
        participant_id=assignment.participant_id,
        persona_id=assignment.persona_id,
        endpoint="/v1/generate-civil-response",
        flags=FeatureFlags(use_rag_pipeline=assignment.use_rag_pipeline),
        latency_ms=42,
        success=True,
    )
    store.record_feedback(
        request_id=request_id,
        participant_id=assignment.participant_id,
        persona_id=assignment.persona_id,
        rating=4,
        task_success=True,
    )

    summary = store.summarize()
    report = render_markdown_report(summary)

    assert summary["completion"]["has_minimum_operation_days"] is False
    assert summary["completion"]["has_minimum_feedback_participants"] is False
    assert "GovOn RAG Feature Flag A/B Test Report" in report
    assert "Minimum 14-day operation: `False`" in report


def test_completion_uses_all_time_operation_duration(tmp_path):
    store = _new_store(tmp_path)
    first_timestamp = 1_700_000_000.0
    last_timestamp = first_timestamp + (MINIMUM_OPERATION_DAYS * 86400) + 1
    assignment = store.get_or_assign("participant-1", persona_id="persona-1")
    flags = FeatureFlags(use_rag_pipeline=assignment.use_rag_pipeline)

    for request_id, timestamp in (
        ("request-first", first_timestamp),
        ("request-last", last_timestamp),
    ):
        store.record_exposure(
            request_id=request_id,
            participant_id=assignment.participant_id,
            persona_id=assignment.persona_id,
            endpoint="/v1/generate-civil-response",
            flags=flags,
            latency_ms=42,
            success=True,
            created_at=timestamp,
        )

    summary = store.summarize(days=MINIMUM_OPERATION_DAYS, now=last_timestamp)

    request_count = sum(item["request_count"] for item in summary["variants"].values())
    assert request_count == 1
    assert summary["observed_days"] > MINIMUM_OPERATION_DAYS
    assert summary["completion"]["has_minimum_operation_days"] is True


def test_duplicate_personas_do_not_satisfy_distinct_persona_requirement(tmp_path):
    store = _new_store(tmp_path)

    for index in range(MINIMUM_FEEDBACK_PARTICIPANTS):
        participant_id = f"participant-{index}"
        assignment = store.get_or_assign(participant_id, persona_id="shared-persona")
        request_id = f"request-{index}"
        store.record_exposure(
            request_id=request_id,
            participant_id=participant_id,
            persona_id="shared-persona",
            endpoint="/v1/generate-civil-response",
            flags=FeatureFlags(use_rag_pipeline=assignment.use_rag_pipeline),
            latency_ms=42,
            success=True,
        )
        store.record_feedback(
            request_id=request_id,
            participant_id=participant_id,
            persona_id="shared-persona",
            rating=4,
            task_success=True,
        )

    summary = store.summarize()

    assert summary["totals"]["feedback_participant_count"] == MINIMUM_FEEDBACK_PARTICIPANTS
    assert summary["totals"]["persona_count"] == 1
    assert summary["completion"]["has_minimum_feedback_participants"] is True
    assert summary["completion"]["has_minimum_personas"] is False
