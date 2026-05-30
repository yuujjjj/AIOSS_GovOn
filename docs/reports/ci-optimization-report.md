# CI Optimization Report

Date: 2026-04-08

## GitHub Links

- Workflow YAML: https://github.com/yuujjjj/AIOSS_GovOn/blob/main/.github/workflows/ci-cd.yml
- Reusable workflow: https://github.com/yuujjjj/AIOSS_GovOn/blob/main/.github/workflows/reusable-python-ci.yml
- Composite action (Python): https://github.com/yuujjjj/AIOSS_GovOn/blob/main/.github/actions/setup-python-ci/action.yml
- Composite action (Frontend): https://github.com/yuujjjj/AIOSS_GovOn/blob/main/.github/actions/setup-node-frontend/action.yml
- Actions page: https://github.com/yuujjjj/AIOSS_GovOn/actions/workflows/ci-cd.yml
- Baseline run `#4`: https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/24130804722
- Expanded matrix validation run `#8`: https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/24131640781
- Cache benchmark run `#9`: https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/24131711329

## What Changed

- Python CI shared logic was extracted into a reusable workflow.
- Python and frontend setup logic was moved into composite actions to remove repeated install/cache steps.
- Python matrix expanded from `ubuntu + macOS` x `3.10 + 3.11` to `ubuntu + macOS + windows` x `3.10 + 3.11 + 3.12`.
- `plan-pipeline` now detects changed files and decides whether to run Python, frontend, deploy, and benchmark stages.
- Deploy is limited to `push` on `main` or `release/*`, and only when frontend work is actually selected.
- Benchmark runs only when CI files changed on `main` pushes or when `workflow_dispatch(force_benchmark=true)` is used.

## Before/After Summary

| Metric | Baseline | Optimized |
| --- | --- | --- |
| Validation run | `#4` | `#8` |
| Total duration | `84s` | `97s` |
| Matrix cells | `4` | `9` |
| OS coverage | `ubuntu-latest`, `macos-latest` | `ubuntu-latest`, `macos-latest`, `windows-latest` |
| Python coverage | `3.10`, `3.11` | `3.10`, `3.11`, `3.12` |
| Frontend/deploy gating | path-based basic skip | changed-file planning + branch/PR deploy gating |

- Matrix coverage increased from `4` to `9` cells, a `125%` expansion.
- Validation runtime increased from `84s` to `97s`, a `15.48%` increase while covering more than twice as many matrix combinations.

## Cache Benchmark

Source: run `#9` artifact `ci-cache-benchmark-summary`

| Benchmark | Measured install time |
| --- | --- |
| No cache | `4s` |
| Cache hit | `3s` |

- Cache hit restored: `true`
- Improvement: `25.00%`

## Selective Pipeline Rules

- `pull_request` and `push` are limited to `main` and `release/**`.
- `plan-pipeline` diffs `src/**`, `tests/**`, `requirements*.txt`, `.github/workflows/**`, `.github/actions/**`, and `GovOn/frontend/**`.
- `run_python` is enabled when Python files or CI files change, or when `workflow_dispatch.force_python=true`.
- `run_frontend` is enabled only when `GovOn/frontend` exists and frontend or CI files changed, or when `workflow_dispatch.force_frontend=true`.
- `run_deploy` is enabled only for `push` to `main` or `release/*` when the frontend pipeline is selected.
- `run_benchmark` is enabled for CI changes on `main` pushes or when `workflow_dispatch.force_benchmark=true`.

## Stability Note

- Windows matrix failures were caused by tests reading UTF-8 output with the platform default code page.
- The preprocessing save-format tests now read files with explicit `utf-8`, which made all Windows matrix cells pass in run `#8` and run `#9`.
