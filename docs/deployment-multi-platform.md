# Multi-platform Deployment Automation

This repository uses three deployment surfaces:

- GitHub Pages for the static frontend production site.
- Vercel for pull request preview deployments.
- Google Cloud Run for the Dockerized API service.

## Frontend Production: GitHub Pages

Workflow: `.github/workflows/ci-cd.yml`

The CI/CD workflow builds `GovOn/frontend` as a static Next.js export and deploys the `out` directory to GitHub Pages on pushes to `main`. The frontend build sets:

- `NEXT_PUBLIC_BASE_PATH=/AIOSS_GovOn`
- `NEXT_PUBLIC_SITE_URL` from repository secrets, when configured
- `NEXT_PUBLIC_API_BASE_URL` from repository secrets, when configured

The repository is currently expected to serve the frontend at:

```text
https://yuujjjj.github.io/AIOSS_GovOn/
```

## Pull Request Preview: Vercel

Workflow: `.github/workflows/frontend-preview.yml`

The preview workflow runs on pull requests to `main` when `GovOn/frontend` changes. It:

1. Installs dependencies with Node.js 22.
2. Runs frontend lint.
3. Builds the static export without a GitHub Pages base path.
4. Uploads the static build artifact.
5. Deploys a Vercel preview if the Vercel secrets are configured.
6. Writes the preview URL back to the pull request conversation.

Required GitHub repository secrets:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

If these secrets are missing, the workflow still validates the frontend build and records that the preview deploy was skipped.

## Docker Image Pipeline

Workflow: `.github/workflows/docker-publish.yml`

The Docker workflow builds `GovOn/Dockerfile`, pushes the image to GitHub Container Registry, pulls the pushed image back to the runner, and verifies local container execution through `GET /health`.

Image format for this repository:

```text
ghcr.io/yuujjjj/aioss_govon:sha-<commit>
ghcr.io/yuujjjj/aioss_govon:latest
```

The `latest` tag is published on successful pushes to `main`.

## Container Deployment: Google Cloud Run

Workflow: `.github/workflows/deploy-cloud-run.yml`

The Cloud Run workflow starts after `Docker Image Publish` completes successfully on `main`, or manually through `workflow_dispatch`. It:

1. Reads the GHCR image tag to deploy.
2. Authenticates to Google Cloud.
3. Enables required Google Cloud APIs.
4. Creates the Artifact Registry Docker repository if it does not exist.
5. Mirrors the GHCR image into Artifact Registry.
6. Deploys the container to Cloud Run.
7. Runs a post-deploy health check against `/health`.

Required GitHub repository secrets:

- `GCP_SA_KEY`
- `GCP_PROJECT_ID`

Optional GitHub repository variables:

- `GCP_REGION`, default `asia-northeast3`
- `GCP_AR_REPOSITORY`, default `govon-repo`
- `GCP_SERVICE_NAME`, default `govon-api`

Optional GitHub repository secret:

- `GOVON_API_KEY`, passed to the service as `API_KEY` when configured

The service account in `GCP_SA_KEY` should have enough permission to deploy Cloud Run services, push Artifact Registry images, use the runtime service account, and enable required APIs.

## Health Monitoring

Workflow: `.github/workflows/monitor-cloud-run.yml`

The monitoring workflow runs every 10 minutes and can also be started manually. It:

1. Resolves the Cloud Run service URL.
2. Calls `/health`.
3. Records HTTP status, latency, revision, and timestamp in the workflow summary.
4. Opens a GitHub issue labeled `monitoring` and `alert` when the service is unhealthy.
5. Closes the open alert issue when the service recovers.

The monitoring workflow uses the same `GCP_SA_KEY`, `GCP_PROJECT_ID`, `GCP_REGION`, and `GCP_SERVICE_NAME` settings as the deploy workflow.

## Operational Checklist

Before this automation is fully active:

1. Commit and push these workflow changes to GitHub.
2. Configure Vercel secrets if PR preview URLs are required.
3. Configure GCP secrets and variables if Cloud Run deployment is required.
4. Ensure GitHub Pages is enabled with the GitHub Actions source.
5. Push or merge to `main` and confirm:
   - `CI/CD` deploys the Pages artifact.
   - `Docker Image Publish` publishes and smoke-tests the GHCR image.
   - `Deploy Container to Cloud Run` deploys the image and passes `/health`.
   - `Cloud Run Monitoring` writes a healthy summary or opens an alert issue.
