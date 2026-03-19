# DORA Metrics 수집 및 Grafana Cloud 대시보드

## 개요

DORA(DevOps Research and Assessment) 4대 지표를 GitHub Actions로 자동 수집하고,
**Grafana Cloud**(무료 티어)에서 공개 대시보드로 시각화한다.

## 아키텍처

```
GitHub Actions (매주 월요일 + main push)
    │
    ├── DORA 4대 지표 수집 (gh CLI)
    ├── JSON 아티팩트 저장 (metrics/dora/)
    └── Grafana Cloud Prometheus에 메트릭 전송 (InfluxDB line protocol)
            │
            └── Grafana Cloud Dashboard (공개 URL로 팀원/교수님 공유)
```

## DORA 4대 지표

| 지표 | 측정 방법 | PromQL |
|------|----------|--------|
| **배포 빈도** | 최근 분석 기간 내 merge된 PR 수를 주 단위로 환산 | `dora_deployment_frequency{project="govon"}` |
| **리드 타임** | PR의 첫 커밋 → 머지 평균 시간 | `dora_lead_time_hours{project="govon"}` |
| **변경 실패율** | 분석 기간 내 hotfix/revert 계열 커밋 비율 | `dora_change_failure_rate{project="govon"}` |
| **복구 시간 (MTTR)** | 분석 기간 내 bug 이슈 open → close 평균 시간 | `dora_mttr_hours{project="govon"}` |

## 디렉토리 구조

```
metrics/
├── README.md
├── dora/                        # 수집된 JSON 데이터 (Actions 자동 생성)
│   └── dora-YYYYMMDD.json      # window_days, branch별 메트릭 포함
└── grafana-cloud/
    ├── setup-guide.md           # Grafana Cloud 설정 가이드
    └── dora-dashboard.json      # 대시보드 Import용 JSON
```

## 설정 방법

**상세 가이드**: [grafana-cloud/setup-guide.md](grafana-cloud/setup-guide.md)

### 빠른 시작

1. [Grafana Cloud 무료 가입](https://grafana.com/auth/sign-up/create-user)
2. Prometheus Endpoint URL, Username, Token 확인
3. GitHub Secrets 3개 등록:
   - `GRAFANA_CLOUD_URL` — Remote Write Endpoint
   - `GRAFANA_CLOUD_USER` — Instance ID
   - `GRAFANA_CLOUD_API_KEY` — Access Token
4. Actions → DORA Metrics Collector → Run workflow
5. Grafana Cloud에서 `dora-dashboard.json` Import
6. **Share → Public dashboard** 활성화하여 공유

### 수동 실행 입력값

- `collect_enabled`: 수집 실행 여부
- `publish_to_grafana`: Grafana Cloud 전송 여부
- `window_days`: 분석 기간 (기본 30일)

## 등급 기준

| 등급 | 배포 빈도 | 리드 타임 | 변경 실패율 | MTTR |
|------|----------|----------|-----------|------|
| Elite | 일 1회+ | < 1일 | < 15% | < 1시간 |
| High | 주 1회+ | < 1주 | 15~30% | < 24시간 |
| Medium | 월 1회+ | < 1개월 | 30~45% | < 1주 |
| Low | 월 1회 미만 | > 1개월 | > 45% | > 1주 |
