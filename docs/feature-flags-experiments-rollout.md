# Feature Flags, Experiments, and Canary Rollout

## Feature Flags

Runtime implementation: `GovOn/src/inference/feature_flags.py`

The app supports four boolean feature flags and one model-selection flag.

| Flag | Environment variable | Targeting variables | Default |
|---|---|---|---|
| RAG pipeline | `USE_RAG_PIPELINE` | `USE_RAG_PIPELINE_TARGET_USERS`, `USE_RAG_PIPELINE_DISABLED_USERS` | `true` |
| Hybrid search | `ENABLE_HYBRID_SEARCH` | `ENABLE_HYBRID_SEARCH_TARGET_USERS`, `ENABLE_HYBRID_SEARCH_DISABLED_USERS` | `true` |
| Agent tools | `ENABLE_AGENT_TOOLS` | `ENABLE_AGENT_TOOLS_TARGET_USERS`, `ENABLE_AGENT_TOOLS_DISABLED_USERS` | `true` |
| Streaming response | `ENABLE_STREAMING_RESPONSE` | `ENABLE_STREAMING_RESPONSE_TARGET_USERS`, `ENABLE_STREAMING_RESPONSE_DISABLED_USERS` | `true` |
| Model version | `MODEL_VERSION` | Not targeted | `v2_lora` |

Request-level operator overrides are supported through `X-Feature-Flag`.

```text
X-User-Id: pilot-user-001
X-Feature-Flag: USE_RAG_PIPELINE=false,ENABLE_HYBRID_SEARCH=false,MODEL_VERSION=v1_lora
```

Targeted users are resolved in `get_feature_flags()` from `X-User-Id`, and the
resolved flags are used by generation, search, stream, and agent endpoints.

## A/B Experiments

Two experiments are configured in `EXPERIMENT_CONFIGS`.

| Experiment | Variants | Assignment |
|---|---|---|
| `complaint_response_layout` | `control`, `guided` | Stable SHA-256 bucket by `experiment_key:user_id` |
| `answer_tone` | `formal`, `plain_language` | Stable SHA-256 bucket by `experiment_key:user_id` |

Experiment assignment endpoint:

```text
GET /v1/experiments/assignments
X-User-Id: pilot-user-001
```

The endpoint returns deterministic assignments and appends exposure events to a
JSONL event log. Set `FEATURE_EVENT_LOG_PATH` to choose the runtime log path.
Set `FEATURE_EVENT_TRACKING_ENABLED=false` to disable file writes.

Sample evidence log:

```text
docs/experiments/feature-flag-experiment-log.jsonl
```

## Canary Rollout

Rollout configuration:

```text
GovOn/config/canary-rollout.yml
```

The canary stages are `1% -> 10% -> 50% -> 100%`. Each stage must pass:

- `/api/health` returns HTTP 200.
- Error rate is at or below `2%`.
- p95 latency is at or below `2500ms`.

The helper `evaluate_canary_rollout()` advances to the next stage when all
health gates pass, and returns a rollback decision to `0%` traffic when health,
error-rate, or latency gates fail.

## Verification

Focused tests:

```bash
cd GovOn
pytest tests/test_inference/test_feature_flags.py -q
```

The tests cover:

- Environment-variable flag defaults and overrides.
- Target user enable/disable rules.
- Request-level `X-Feature-Flag` overrides.
- Stable A/B experiment assignment.
- JSONL exposure event tracking.
- Canary rollout advance and rollback decisions.

