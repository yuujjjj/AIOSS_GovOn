# Deployment Evidence

Last checked: 2026-05-31 KST.

## Recommended Submission Links

Submit the live URLs first, then include the workflow URLs as supporting
evidence.

1. Frontend live URL: https://yuujjjj.github.io/AIOSS_GovOn/
2. PR preview URL: https://govon-frontend-aqrca5xyy-uz-s-projects2.vercel.app/
3. External cloud serverless URL: https://govon-frontend.vercel.app/
4. Health check URL: https://govon-frontend.vercel.app/api/health

The live URLs above are suitable for unauthenticated external grading. The PR
preview URL returned HTTP 200 after Vercel preview deployment protection was
disabled.

## Frontend Deployment and PR Preview

| Requirement | Evidence |
|---|---|
| Frontend automatic deployment | GitHub Pages deploy is handled by `.github/workflows/ci-cd.yml`. |
| Frontend live URL | https://yuujjjj.github.io/AIOSS_GovOn/ |
| CI/CD workflow run | https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26690051678 |
| PR preview workflow | `.github/workflows/frontend-preview.yml` |
| PR preview workflow run | https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26689690930 |
| Public PR preview URL | https://govon-frontend-aqrca5xyy-uz-s-projects2.vercel.app/ |

Verified status:

```text
https://yuujjjj.github.io/AIOSS_GovOn/ -> HTTP 200
https://govon-frontend-aqrca5xyy-uz-s-projects2.vercel.app/ -> HTTP 200
```

## Docker Deployment Pipeline

| Requirement | Evidence |
|---|---|
| Docker strategy document | `docs/deployment-multi-platform.md` |
| Docker workflow | `.github/workflows/docker-publish.yml` |
| Docker workflow run | https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26686568590 |
| Published image | `ghcr.io/yuujjjj/aioss_govon:sha-2251691` |
| Latest image tag | `ghcr.io/yuujjjj/aioss_govon:latest` |
| Local verification | Workflow pulls the pushed image and runs `GET /health` in a local Docker container. |

## External Cloud Serverless Deployment

| Requirement | Evidence |
|---|---|
| Platform | Vercel Serverless |
| Production deployment workflow | `.github/workflows/deploy-vercel-serverless.yml` |
| Successful deployment workflow run | https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26690051682 |
| Production URL | https://govon-frontend.vercel.app/ |
| Health endpoint | https://govon-frontend.vercel.app/api/health |
| Monitoring workflow | `.github/workflows/monitor-vercel-serverless.yml` |
| Successful monitoring workflow run | https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26690595815 |

Verified health response:

```json
{"status":"healthy","service":"govon-vercel-serverless","runtime":"vercel"}
```

## npm Package Publishing

| Requirement | Evidence |
|---|---|
| GitHub Packages workflow | `.github/workflows/npm-publish.yml` |
| Published package | `@yuujjjj/govon-frontend@1.0.1` |
| Version update | `1.0.0` to `1.0.1` |
| Package metadata | `GovOn/frontend/package.json` |
| Successful publish workflow run | https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26553776392 |
| Publish tag | `frontend-v1.0.1` |

## Dependabot and Security Automation

| Requirement | Evidence |
|---|---|
| Dependabot schedule and groups | `.github/dependabot.yml` |
| Auto-merge policy | `.github/workflows/dependabot-auto-merge.yml` |
| Auto-merge scope | npm and GitHub Actions minor/patch updates only; major updates require manual review. |
| Security scan workflow | `.github/workflows/npm-security.yml` |
| Security workflow run | https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26689735642 |
| npm audit report | Uploaded as `npm-security-reports` workflow artifact. |
| Snyk report | Uploaded as `npm-security-reports` workflow artifact when `SNYK_TOKEN` is configured; otherwise the workflow records Snyk as skipped. |
| Issue automation | Opens or updates a GitHub issue when npm audit or Snyk reports vulnerabilities. |

## Feature Flags, Experiments, and Canary Rollout

| Requirement | Evidence |
|---|---|
| Feature flag code | `GovOn/src/inference/feature_flags.py` |
| Feature flags | `USE_RAG_PIPELINE`, `ENABLE_HYBRID_SEARCH`, `ENABLE_AGENT_TOOLS`, `ENABLE_STREAMING_RESPONSE` |
| Target user toggles | Per-flag `*_TARGET_USERS` and `*_DISABLED_USERS` environment variables resolved from `X-User-Id`. |
| A/B experiments | `complaint_response_layout` and `answer_tone`, each with two variants and stable SHA-256 user assignment. |
| Experiment event log | `docs/experiments/feature-flag-experiment-log.jsonl` |
| Canary rollout config | `GovOn/config/canary-rollout.yml` |
| Canary verification | `GovOn/tests/test_inference/test_feature_flags.py` validates `1% -> 10% -> 50% -> 100%` advance and failed-health rollback to `0%`. |

## Testing, TDD, and E2E

| Requirement | Evidence |
|---|---|
| Unit test framework | Vitest in `GovOn/frontend/vitest.config.ts` |
| Coverage threshold | 80% for statements, branches, functions, and lines |
| Core TDD implementation | `GovOn/frontend/lib/complaint-workflow.ts` |
| Core TDD tests | `GovOn/frontend/lib/complaint-workflow.test.ts` |
| Legacy page tests | `GovOn/frontend/app/page.test.tsx`, `GovOn/frontend/app/main/page.test.tsx`, `GovOn/frontend/app/not-found.test.tsx`, `GovOn/frontend/app/deployment-evidence/page.test.tsx` |
| Serverless route test | `GovOn/frontend/app/api/health/route.test.ts` |
| Playwright E2E | `GovOn/frontend/e2e/deployment-evidence.spec.ts` |
| Failure artifacts | CI uploads `GovOn/frontend/test-results` and `GovOn/frontend/playwright-report` when E2E runs. |
| Testing evidence doc | `docs/testing-evidence.md` |
