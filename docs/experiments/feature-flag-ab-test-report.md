# Feature Flag A/B Experiment Weekly Report

## Scope

- Experiment: `complaint_response_layout`
- Variants: `control` vs `guided`
- Date range: 2026-05-17 to 2026-05-30 (14 days)
- LLM feedback panel: 10 users with 10 unique persona patterns
- Assignment: stable SHA-256 bucket in `GovOn/src/inference/feature_flags.py`

## Metric Summary

| Metric | Control | Guided | Change |
|---|---:|---:|---:|
| Exposures | 644 | 669 | +25 |
| Task success rate | 0.742 | 0.831 | +8.8pp |
| Avg time to first draft | 258.9s | 216.7s | -16.3% |
| Self-service resolution rate | 0.620 | 0.725 | +10.5pp |
| Follow-up rate | 0.256 | 0.170 | -8.6pp |
| Satisfaction average | 3.71 | 4.18 | +0.47 |

## Feedback Summary

- LLM user feedback rows: 10
- Unique persona patterns: 10
- Task success count: 8 success, 2 failed
- Satisfaction average by variant: guided 4.33, control 3.45
- Recurring backlog themes: bigger accessibility defaults, redaction audit trail, event schema version, prominent response deadline, exportable metric breakdown

## Decision

Decision: **PERSEVERE**

Reasons:

- Task success improved by 8.8 percentage points.
- Time to first draft decreased by 16.3%.
- Satisfaction improved by 0.47 points.
- Follow-up rate decreased by 8.6 percentage points.

## Evidence Inputs

- LLM feedback data: `docs/experiments/llm-user-feedback.jsonl`
- Two-week metric data: `docs/experiments/ab-test-daily-metrics.csv`
- Generated scenario code: `docs/experiments/generated-user-scenario-tests.md`
- Evaluator: `GovOn/scripts/evaluate_experiment_evidence.py`
- Decision record: `docs/experiments/pivot-or-persevere-decision.md`
