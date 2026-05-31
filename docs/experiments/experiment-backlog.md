# Experiment Backlog

This backlog is mirrored by the GitHub issue template and weekly report
automation in `.github/workflows/experiment-report.yml`.

## Active Experiment

| Field | Value |
|---|---|
| Experiment key | `complaint_response_layout` |
| Hypothesis | A guided complaint response layout improves task success and reduces follow-up contact. |
| Variants | `control`, `guided` |
| Primary metric | Task success rate |
| Guardrail metrics | Time to first draft, follow-up rate, satisfaction, self-service resolution |
| Owner | GovOn experiment lead |
| Review cadence | Weekly report every Monday 09:00 KST |

## Backlog Items

| ID | Source | Item | Priority | Status |
|---|---|---|---|---|
| EXP-001 | LLM-UF-002 | Larger default font for accessibility-first users | High | Open |
| EXP-002 | LLM-UF-005 | Redaction audit trail for privacy reviewers | High | Open |
| EXP-003 | LLM-UF-007 | Add event schema version to exposure logs | Medium | Open |
| EXP-004 | LLM-UF-010 | Make response deadline prominent in control/guided comparison | Medium | Open |
| EXP-005 | LLM-UF-008 | Export metric breakdown for policy analysis | Medium | Open |
| EXP-006 | A/B metrics | Keep guided rollout but retain control until follow-up items close | High | Open |

## GitHub Automation

- Issue template: `.github/ISSUE_TEMPLATE/experiment_backlog.yml`
- Weekly report workflow: `.github/workflows/experiment-report.yml`
- Evaluator script: `GovOn/scripts/evaluate_experiment_evidence.py`
- Report artifact: `feature-flag-ab-test-report`

The workflow validates the 10-person LLM panel, validates the 14-day A/B metric
window, creates a Markdown report, uploads the report/summary artifacts, and
opens or updates a GitHub Issue named `Experiment weekly report - feature-flag AB`.
