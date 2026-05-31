# Pivot or Persevere Decision Record

## Decision

**Persevere** with the `guided` variant of `complaint_response_layout`.

## Experiment

| Item | Value |
|---|---|
| Experiment key | `complaint_response_layout` |
| Feature flag variants | `control`, `guided` |
| Runtime assignment | Stable SHA-256 bucket by `experiment_key:user_id` |
| Measurement window | 2026-05-17 to 2026-05-30 |
| Metric source | `docs/experiments/ab-test-daily-metrics.csv` |
| User feedback source | `docs/experiments/llm-user-feedback.jsonl` |

## Success Criteria

Persevere if the guided variant meets all gates:

| Gate | Threshold | Result |
|---|---:|---:|
| Task success improvement | at least +5pp | +8.8pp |
| Time to first draft | at least -10% | -16.3% |
| Satisfaction improvement | at least +0.30 | +0.47 |
| Follow-up rate | not worse than control | -8.6pp |

## Observations

- Guided layout improved task success from 0.742 to 0.831.
- Guided layout reduced average time to first draft from 258.9 seconds to 216.7 seconds.
- Guided layout improved self-service resolution from 0.620 to 0.725.
- Guided layout reduced follow-up rate from 0.256 to 0.170.
- The LLM panel had 10 distinct persona patterns and produced 8 successful scenario evaluations.

## Follow-Up Backlog

1. Increase default text size for accessibility-heavy sessions.
2. Add redaction audit trail links for privacy reviewers.
3. Version the experiment event schema in exposure logs.
4. Make response deadlines more prominent for time-constrained citizens.
5. Add metric breakdown export for policy analysts.

## Rollout Recommendation

Keep the `guided` variant active for the next rollout stage and continue weekly
monitoring. Do not remove the `control` variant until the follow-up backlog is
triaged and the event schema version is documented.
