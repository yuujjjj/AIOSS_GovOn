# Multi-platform Deployment Automation

This repository uses three deployment surfaces:

- GitHub Pages for the static frontend production site.
- Vercel for pull request preview deployments.
- Vercel Serverless Functions for the externally hosted health endpoint.

This deployment track validates automation and runtime readiness, not full
product completeness. The external cloud target is Vercel Serverless, with
`GET /api/health` used as the deployment and monitoring contract.

## Frontend Production: GitHub Pages

Workflow: `.github/workflows/ci-cd.yml`

The CI/CD workflow builds `GovOn/frontend` as a static Next.js export and deploys
the `out` directory to GitHub Pages on pushes to `main`. The frontend build sets:

- `NEXT_PUBLIC_BASE_PATH=/AIOSS_GovOn`
- `NEXT_PUBLIC_SITE_URL` from repository secrets, when configured
- `NEXT_PUBLIC_API_BASE_URL` from repository secrets, when configured

Production frontend URL:

```text
https://yuujjjj.github.io/AIOSS_GovOn/
```

## Pull Request Preview: Vercel

Workflow: `.github/workflows/frontend-preview.yml`

The preview workflow runs on pull requests to `main` when `GovOn/frontend`
changes. It:

1. Installs dependencies with Node.js 22.
2. Runs frontend lint.
3. Builds a static export artifact for validation.
4. Deploys a Vercel preview if the Vercel secrets are configured.
5. Writes the preview URL back to the pull request conversation.

Required GitHub repository secrets:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

If these secrets are missing, the workflow still validates the frontend build and
records that the preview deploy was skipped.

If external reviewers must open the preview without signing in, disable Vercel
preview deployment protection or grant reviewer access.

## Docker Image Pipeline

Workflow: `.github/workflows/docker-publish.yml`

The Docker workflow builds `GovOn/Dockerfile`, pushes the image to GitHub
Container Registry, pulls the pushed image back to the runner, and verifies local
container execution through `GET /health`.

Image format for this repository:

```text
ghcr.io/yuujjjj/aioss_govon:sha-<commit>
ghcr.io/yuujjjj/aioss_govon:latest
```

The `latest` tag is published on successful pushes to `main`.

## External Cloud Deployment: Vercel Serverless

Workflow: `.github/workflows/deploy-vercel-serverless.yml`

The Vercel Serverless workflow runs on pushes to `main` when frontend/serverless
files change, and can also be started manually. It:

1. Checks the Vercel secrets.
2. Installs frontend dependencies with Node.js 22.
3. Runs frontend lint.
4. Pulls Vercel production project settings.
5. Builds the production Vercel deployment.
6. Deploys the project to Vercel production.
7. Runs a post-deploy health check against `/api/health`.

Serverless health endpoint:

```text
GET /api/health
```

The endpoint returns JSON with `status: "healthy"` and `runtime: "vercel"`.

Required GitHub repository secrets:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

## Health Monitoring

Workflow: `.github/workflows/monitor-vercel-serverless.yml`

The monitoring workflow runs every 10 minutes and can also be started manually.
It:

1. Calls the Vercel `/api/health` endpoint.
2. Records HTTP status, latency, and timestamp in the workflow summary.
3. Opens a GitHub issue labeled `monitoring` and `alert` when the endpoint is unhealthy.
4. Closes the open alert issue when the endpoint recovers.

Repository variable:

- `VERCEL_HEALTH_URL`, optional. Defaults to `https://govon-frontend.vercel.app/api/health`.

If the Vercel production domain differs, set `VERCEL_HEALTH_URL` to the actual
production health endpoint.

## Operational Checklist

Before this automation is fully active:

1. Merge the deployment workflow changes to `main`.
2. Ensure GitHub Pages is enabled with the GitHub Actions source.
3. Configure Vercel secrets: `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`.
4. Push or merge to `main` and confirm:
   - `CI/CD` deploys the Pages artifact.
   - `Frontend PR Preview` comments a Vercel preview URL on PRs.
   - `Docker Image Publish` publishes and smoke-tests the GHCR image.
   - `Deploy Vercel Serverless` deploys production and passes `/api/health`.
   - `Vercel Serverless Monitoring` writes a healthy summary or opens an alert issue.

## Submission Evidence Checklist

Use these items as the minimum evidence package for the deployment requirement:

- Frontend production URL: `https://yuujjjj.github.io/AIOSS_GovOn/` returns HTTP 200.
- `CI/CD` workflow shows `Frontend Build`, `Frontend Smoke Test`, and `Deploy` as successful.
- A pull request shows a `Frontend PR Preview` comment with a Vercel preview URL.
- If the preview must be public, Vercel preview deployment protection is disabled or reviewer access is granted.
- `Docker Image Publish` shows `Build and push Docker image`, `Pull pushed image locally`, and `Verify local container execution` as successful.
- `Deploy Vercel Serverless` shows production deploy and `/api/health` health check as successful.
- The Vercel deploy summary includes the production URL and health endpoint URL.
- `Vercel Serverless Monitoring` shows URL, status, HTTP code, latency, and timestamp.

## Responsibility Split

Codex can maintain the workflow files, Docker strategy documentation, frontend
static export configuration, Vercel serverless endpoint, and local validation.

The repository owner must configure external platform settings:

- GitHub Pages source must be set to GitHub Actions.
- Vercel secrets must be configured: `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`.
- `VERCEL_HEALTH_URL` should be set if the production domain is not `govon-frontend.vercel.app`.
- If reviewers need unauthenticated preview access, Vercel deployment protection must be disabled for preview deployments.
