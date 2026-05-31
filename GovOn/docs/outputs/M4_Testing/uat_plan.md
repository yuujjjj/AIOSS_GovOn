# M4 UAT Plan: LLM Persona Feedback and RAG A/B Test

## Scope

GovOn의 첫 Feature Flag 실험은 동일한 v2 모델에서 RAG 파이프라인 효과를 측정한다.

| Variant | Feature Flag |
|---|---|
| Control | `USE_RAG_PIPELINE=false` |
| Treatment | `USE_RAG_PIPELINE=true` |

모델 버전은 고정한다. 폐기된 v1 모델은 비교군으로 사용하지 않는다. 참가자는 최초 요청 시
두 variant의 인원 차이가 최소가 되도록 배정되며, 이후 같은 pseudonymous ID에는 같은
variant가 유지된다.

## LLM Persona Evaluation

- `configs/llm_user_personas.json`: 서로 다른 평가 패턴을 가진 LLM 사용자 페르소나 10명
- `configs/uat_scenarios.json`: 8개 민원 카테고리를 포함하는 UAT 시나리오 10건
- `scripts/run_llm_persona_ab_test.py`: GovOn 답변 생성 후 OpenAI 호환 평가 LLM으로 피드백 수집

라이브 평가 실행 예시:

```bash
export GOVON_API_KEY=...
export LLM_EVALUATOR_BASE_URL=http://127.0.0.1:9000/v1
export LLM_EVALUATOR_MODEL=evaluator-model
python scripts/run_llm_persona_ab_test.py
```

GPU 없이 저장과 리포트 경로만 검증할 때:

```bash
python scripts/run_llm_persona_ab_test.py --dry-run
```

`--dry-run` 결과는 코드 검증용 합성 데이터다. LLM 사용자 피드백 또는 실제 운영 결과로 제출하지 않는다.

## Runtime Prerequisites

- 현재 workspace의 v2 데이터셋은 `../data/processed/`에 있다. compose 실행 전 `.env`의
  `GOVON_DATA_DIR=../data`를 설정하거나 동일 데이터를 `./data/`에 배치한다.
- 온라인 환경은 `umyunsang/GovOn-EXAONE-AWQ-v2`를 사용한다. 폐쇄망은 AWQ v2 모델을
  `GOVON_MODELS_DIR` 아래에 미리 배치한다.
- RAG 비교를 시작하기 전에 FAISS 및 BM25 인덱스를 생성하고 `/health`에서 로드 상태를 확인한다.
- `GOVON_EXPERIMENT_DB`가 가리키는 SQLite 파일은 14일 동안 유지한다. 운영 중 DB를 초기화하지 않는다.
- 같은 라이브 평가 명령을 운영 기간 중 반복 실행하고, 종료 시 `scripts/generate_ab_report.py`로
  최종 리포트를 생성한다.

## Collected Metrics

| Metric | Purpose |
|---|---|
| 요청 수, 성공률, 오류 수 | 변형별 안정성 비교 |
| 평균 레이턴시 | RAG 추가 비용 확인 |
| 사용자 만족도 `1..5` | 답변 품질 체감 비교 |
| task 성공률 | 사용자 목적 달성 여부 |
| 피드백 참가자 수 | 최소 10명 충족 여부 |
| 페르소나 수 | 서로 다른 평가 패턴 최소 10개 충족 여부 |
| 최초·최종 노출 시각 | 실제 14일 운영 여부 판정 |

## Runtime API

| Endpoint | Purpose |
|---|---|
| `GET /v1/experiments/rag-ab/assignment` | 참가자의 고정 variant 확인 |
| `GET /v1/experiments/rag-ab/metrics` | 최근 기간별 A/B 집계 확인 |
| `POST /feedback/submit` | 생성 요청에 대한 사용자 피드백 저장 |

생성 API 호출 시 아래 헤더를 함께 보낸다.

```text
X-GovOn-Participant-ID: pseudonymous-user-id
X-GovOn-Persona-ID: persona-id
X-GovOn-Scenario-ID: scenario-id
```

실명, 연락처, 원문 민원인의 개인정보를 참가자 ID로 사용하지 않는다.

## Completion Rule

최종 리포트는 아래 조건을 모두 만족해야 완료로 판정한다.

1. 최초 노출과 최종 노출 간격이 14일 이상이다.
2. 피드백 참가자가 10명 이상이다.
3. 서로 다른 페르소나가 10개 이상이다.
4. Control과 Treatment의 핵심 지표 변화를 리포트에 기록한다.
