# Testing, TDD, and E2E Evidence

## Unit Coverage

Frontend unit tests use Vitest with V8 coverage.

```bash
cd GovOn/frontend
npm run test:coverage
```

Coverage gates in `GovOn/frontend/vitest.config.ts` require at least 80% for
statements, branches, functions, and lines.

## TDD Cycles

The complaint workflow was implemented with Red-Green-Refactor checkpoints for
five core functions.

| Cycle | Red test | Green implementation | Refactor/stability check |
|---|---|---|---|
| 1 | Whitespace/control-character normalization test | `normalizeComplaintText()` | Shared by `buildComplaintWorkItem()` |
| 2 | Sensitive email/phone/resident-id masking test | `maskSensitiveContactInfo()` | Regexes grouped in one pure function |
| 3 | Domain category keyword tests | `detectComplaintCategory()` | Keyword table extracted for maintainability |
| 4 | Critical/high/standard urgency tests | `classifyComplaintUrgency()` | Critical priority evaluated before high |
| 5 | Weekend-aware due-date tests | `calculateResponseDueDate()` | Business-day offsets centralized by urgency |
| 6 | Checklist composition test | `buildResponseChecklist()` | Main page uses the same function output |

Core implementation:

```text
GovOn/frontend/lib/complaint-workflow.ts
```

Unit tests:

```text
GovOn/frontend/lib/complaint-workflow.test.ts
```

## Legacy Refactor Safety

Existing frontend pages are covered so refactoring the main workflow remains
safe.

```text
GovOn/frontend/app/page.test.tsx
GovOn/frontend/app/main/page.test.tsx
GovOn/frontend/app/not-found.test.tsx
GovOn/frontend/app/deployment-evidence/page.test.tsx
GovOn/frontend/app/api/health/route.test.ts
```

## Playwright E2E

Playwright scenario:

```text
GovOn/frontend/e2e/deployment-evidence.spec.ts
```

The CI workflow uploads failure artifacts from:

```text
GovOn/frontend/test-results
GovOn/frontend/playwright-report
```

`playwright.config.ts` enables `screenshot: "only-on-failure"` and
`trace: "retain-on-failure"`.

## CI

The CI/CD workflow runs:

- `npm run lint`
- `npm run test:coverage`
- `npm run build`
- `npm run e2e`

Workflow file:

```text
.github/workflows/ci-cd.yml
```

