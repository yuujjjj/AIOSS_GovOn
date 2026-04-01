# GovOn Development Tasks: Orchestrator, Tool Calling, Multi-LoRA & Public Doc Fine-Tuning

**Updated**: 2026-04-01
**Status**: Active (Smolagents Phase 1 기반, ADR-006 v2.0 sync)

## Specification Summary

**Core Requirements**:
1. **Multi-LoRA Fine-Tuning**: 3개의 독립적인 LoRA 어댑터 (Brain, Civil, Public Doc)
2. **Smolagents 기반 Orchestrator**: FastAPI + Smolagents ToolCallingAgent 통합
3. **Tool Integration**: 4개 표준화 Tool (@tool 데코레이터)
4. **Frontend Chat UI**: Tool 실행 인디케이터 + 공문서 렌더링
5. **UI/UX Design System**: Figma 기반 디자인 토큰 + 컴포넌트 라이브러리
6. **Infrastructure**: Multi-LoRA vLLM 컨테이너화 + 폐쇄망 배포

**Architecture References**:
- [ADR-006: 3-Tier Agentic Architecture + Multi-LoRA Serving (v2.0)](docs/architecture/ADR-006-agentic-architecture.md)
- [WORKFLOW: Orchestrator & Tool-Calling (v2.0)](docs/architecture/WORKFLOW-orchestrator-tool-calling.md)

---

## Workstream 1: Data Collection & Multi-LoRA Fine-Tuning

**Initiative**: I-1 (#366)
**Milestone**: M3: 고도화 및 최적화

### [ ] Task 1.1: 공공문서 데이터 수집 및 전처리
**Issue**: #408
**Description**: 행안부 공공데이터포털(`apis.data.go.kr/1741000/publicDoc`) API를 통해 5종 공문서 수집
**Acceptance Criteria**:
- 5개 카테고리에서 각 1,000건 이상 수집 (총 5,000건)
- `src/data_collection_preprocessing/collect_public_docs.py` 스크립트
- 페이지네이션 및 Rate limiting 구현
- JSONL 형식 저장

### [ ] Task 1.2: 목적별 3종 독립 학습 데이터셋 전처리 및 데이터 무결성 검증
**Issue**: #390
**Description**: Task 1.1 데이터와 기존 AI Hub 데이터를 3종 데이터셋으로 정제
**Acceptance Criteria**:
- `dataset_brain.jsonl`: Tool-Calling 의도 파악 (약 1만 건)
- `dataset_civil.jsonl`: 부드러운 대민 답변 (AI Hub 민원)
- `dataset_public_doc.jsonl`: 공문서 양식 학습 (Task 1.1 수집 데이터)
- 데이터 무결성 검증 완료

### [ ] Task 1.3: EXAONE-Deep-7.8B Multi-LoRA 독립 파인튜닝
**Issue**: #391
**Description**: QLoRA를 사용해 3개 LoRA 어댑터를 각각 독립 학습
**Acceptance Criteria**:
- 3종 LoRA 어댑터 생성 (`adapter_brain`, `adapter_civil`, `adapter_public_doc`)
- 각 어댑터의 개별 평가 메트릭 달성 (Brain: JSON accuracy >= 90%, Civil: ROUGE-L >= 0.45, Public Doc: 형식 정확도 >= 85%)
- **LoRA 어댑터 레지스트리 파일 생성**: `config/lora_adapters.yaml` (ADR-006 v2.0 참고)
  - 경로, 대상 모듈, Task, 평가 메트릭 메타데이터 포함
  - vLLM Task 2.1, Smolagents Task 2.3에서 참조할 수 있는 형태

---

## Workstream 2: Orchestrator Architecture (Smolagents 기반)

**Initiative**: I-2 (#367) + I-9 (#406)
**Milestone**: M3: 고도화 및 최적화
**선행 조건**: I-9 Task 9.1 완료 필수

### [ ] Task 2.1: vLLM Multi-LoRA 서빙 인프라 및 설정 구성
**Issue**: #392
**Description**: vLLM 서버를 Multi-LoRA 모드(`--enable-lora`)로 구성하고 3개 어댑터 등록
**Acceptance Criteria**:
- vLLM `--enable-lora` flag 활성화
- `config/lora_adapters.yaml`을 읽어 `--lora-modules` 파라미터 자동 생성
- 3개 어댑터 동시 로드 (RTX 3060 12GB 기준 OOM 없음)
- vLLM 헬스체크 엔드포인트 정상 동작

### [ ] Task 2.2: 오케스트레이터 메인 루프 및 동적 LoRA 라우팅 구현
**Issue**: #393
**Description**: FastAPI 백엔드에서 Smolagents Agent와 vLLM을 통합하고 LoRA 동적 라우팅 구현
**Acceptance Criteria**:
- 세션 관리 (SQLAlchemy ORM, #129 연계)
- 컨텍스트 윈도우 관리
- Tool 요청에 따른 LoRA 동적 스위칭 (adapter_brain, adapter_civil, adapter_public_doc)
- 에러 핸들링 및 타임아웃 관리

### [ ] Task 2.3: Smolagents AgentExecutor 구현
**Issue**: #409
**Description**: Smolagents ToolCallingAgent를 FastAPI와 통합
**Acceptance Criteria**:
- vLLM의 OpenAI-compatible endpoint를 Smolagents `OpenAIServerModel`로 래핑
- Smolagents Agent 초기화 (4개 @tool 레지스트리)
- Brain LoRA(`adapter_brain`) 사용해 의도 파악
- E2E 테스트: 의도 결정 정확도 >= 85%
- 의존성: I-9 Task 9.1 완료 후 진행

### [ ] Task 2.4: LangGraph 전환 준비 (선택적)
**Issue**: #410
**Description**: 향후 EXAONE 4.0 도입 시 LangGraph 전환을 위한 마이그레이션 가이드 작성
**Acceptance Criteria**:
- `docs/architecture/MIGRATION-smolagents-to-langgraph.md` 작성
- Tool 인터페이스 마이그레이션 테이블
- 전환 패턴 및 예상 공수
- 선택적 Task (M3 deadline 불포함)

### [ ] Task 2.5 (부자 이슈): #129 Agent 세션 관리 시스템 구현
**Description**: 대화 이력 영속성, 컨텍스트 윈도우 관리

---

## Workstream 3: Tool Integration (Smolagents @tool 기반)

**Initiative**: I-3 (#368)
**Milestone**: M3: 고도화 및 최적화
**선행 조건**: I-9 Task 9.2 완료 필수

### [ ] Task 3.1: 민원분석 API 클라이언트 Tool 구현
**Issue**: #394
**Description**: `apis.data.go.kr/1140100/minAnalsInfoView5` API를 Smolagents `@tool` 형식으로 구현
**Acceptance Criteria**:
- 민원 키워드 기반 유사 사례 검색
- 10초 타임아웃
- Smolagents @tool 인터페이스 준수 (입력: Pydantic/기본타입, 반환: str/dict)
- 의존성: I-9 Task 9.2 완료 후 진행

### [ ] Task 3.2: 기존 FAISS RAG를 표준화 Tool 인터페이스로 리팩토링
**Issue**: #395
**Description**: `retriever.py` + `hybrid_search.py`를 Smolagents `@tool`로 래핑
**Acceptance Criteria**:
- FAISS/BM25 하이브리드 검색
- Top-k 유사 문서 반환
- Smolagents 인터페이스 준수
- 의존성: I-9 Task 9.2 완료 후 진행

### [ ] Task 3.3: 공문서 검색 Tool 구현
**Issue**: #396
**Description**: 행안부 공문서 API 기반 문서 검색 Tool
**Acceptance Criteria**:
- Task 1.1 수집 데이터에 대한 벡터 검색 (임베딩 모델: multilingual-e5-large)
- Smolagents @tool 형식
- 의존성: I-9 Task 9.2 완료 후 진행

### [ ] Task 3.4: 공문서/민원 답변 생성 Tool (LoRA 동적 연동)
**Issue**: #397
**Description**: vLLM Multi-LoRA 어댑터를 활용한 문서 생성 Tool
**Acceptance Criteria**:
- `adapter_civil`: 민원 답변 생성 (lora_request="adapter_civil")
- `adapter_public_doc`: 공문서 초안 생성 (lora_request="adapter_public_doc")
- Smolagents @tool 형식
- HTML 테이블 및 Markdown 구조 보존
- 의존성: I-1 Task 1.3 + I-9 Task 9.2 완료 후 진행

### [ ] Task 3.5+ (부자 이슈들):
- #155: RRF 통합
- #159: ContextAwareQueryBuilder
- #160: 동적 인덱싱

---

## Workstream 4: Frontend Chat UI & Backend 연동

**Initiative**: I-4 (#369)
**Milestone**: M3: 고도화 및 최적화

### [ ] Task 4.1: Tool 실행 UI 인디케이터 구현
**Issue**: #411
**Description**: Tool 실행 중 사용자에게 시각적 피드백 제공
**Acceptance Criteria**:
- 3가지 이상 Tool 타입별 로딩 상태 표시
- 로딩 스피너 + 상태 텍스트
- Tool 타임아웃 시 에러 상태

### [ ] Task 4.2: HTML 테이블 및 이미지 공문서 렌더링
**Issue**: #412
**Description**: MessageBubble에서 공문서 HTML 테이블 및 이미지 안전하게 렌더링
**Acceptance Criteria**:
- react-markdown + remark-gfm 확장
- DOMPurify 기반 sanitization
- 반응형 테이블 스타일링
- 이미지 placeholder 링크

### [ ] Task 4.3+ (부자 이슈들):
- #132: 스트리밍 UI
- #140: 행정 시스템 연동 사이드 패널
- #141: 반응형 웹 디자인
- #144: API 연동
- #161: 출처 표시 API 연동

---

## Workstream 5: UI/UX 디자인 시스템

**Initiative**: I-5 (#370)
**Milestone**: M3: 고도화 및 최적화

### [ ] 디자인 시스템 구축
- #133: 와이어프레임
- #134: UI/UX 화면 설계서
- #135: 목업 디자인
- #136: 인터랙티브 프로토타입
- #137: 디자인 시스템 및 스타일 가이드
- #63: 동서대 협업

---

## Workstream 6: 웹 UI 구축 (Figma MCP + 컴포넌트)

**Initiative**: I-6 (#371)
**Milestone**: M3: 고도화 및 최적화
**선행 조건**: I-5 완료 권장

### [ ] Figma MCP 기반 웹 UI 구축
- #50: Figma MCP 기반 프론트엔드 웹 UI 구축
- #57: Figma MCP 기반 프론트엔드 고도화
- #138: 공통 UI 컴포넌트 라이브러리
- #139: 메인 랜딩 페이지

---

## Workstream 7: 인프라 및 배포 (Multi-LoRA + vLLM)

**Initiative**: I-7 (#372)
**Milestone**: M3: 고도화 및 최적화

### [ ] Task 7.1: vLLM Multi-LoRA 컨테이너화 구성
**Issue**: #404
**Description**: Docker 이미지 작성 (EXAONE-Deep AWQ + 3개 LoRA)
**Acceptance Criteria**:
- Dockerfile 작성 (NVIDIA CUDA 기반)
- 이미지 빌드 및 GPU 모드 실행 성공
- 의존성: I-1 Task 1.3 완료 후 진행

### [ ] Task 7.2: 폐쇄망 배포(Offline) 스크립트 최적화
**Issue**: #405
**Description**: 인터넷 연결 없이 RTX 3060 서버에 배포하기 위한 스크립트
**Acceptance Criteria**:
- 모델 + LoRA 가중치 오프라인 번들
- 의존성 사전 설치 스크립트
- 배포 자동화 스크립트
- 의존성: Task 7.1 완료 후 진행

---

## Workstream 8: 테스트 및 품질 보증

**Initiative**: I-8 (#373)
**Milestone**: M4: 테스트 및 문서화

### [ ] Task 5.1: 오케스트레이터 E2E 통합 테스트
**Issue**: #413
**Description**: I-2, I-3, I-4 전체 오케스트레이터 루프의 E2E 테스트
**Acceptance Criteria**:
- TC-01~TC-07 전 시나리오 통과 (WORKFLOW v2.0 정의)
- Tool 호출 정확도 >= 90%
- Tool 실행 성공률 >= 95%
- 응답 레이턴시: Tool 없음 < 5초, Tool 포함 < 15초
- Multi-turn 대화 검증

### [ ] Task 5.2: 공문서 품질 평가 메트릭
**Issue**: #414
**Description**: LLM 생성 공문서 초안의 품질 정량화
**Acceptance Criteria**:
- 평가 메트릭 정의 (형식, 구문, 내용 충실도)
- 기준선 평가 (100개 샘플)
- 자동 평가 스크립트
- 평가 보고서

### [ ] 부자 이슈들:
- #59: 통합 테스트 및 성능 벤치마킹
- #61: 사용자 수용 테스트 (UAT)
- #142: 실무 환경 통합 테스트

---

## Workstream 9: Smolagents 에이전트 프레임워크 도입

**Initiative**: I-9 (#406)
**Milestone**: M3: 고도화 및 최적화
**상태**: Phase 1 (선행 Initiative -- I-2, I-3의 의존 조건)

### [ ] Task 9.1: Smolagents 환경 구성 및 EXAONE vLLM 래핑
**Issue**: #415
**Description**: Smolagents 패키지 설치 및 vLLM OpenAI-compatible endpoint 연결
**Acceptance Criteria**:
- `smolagents[vllm] >= 1.11.0` 설치
- vLLM OpenAI-compatible endpoint를 Smolagents `OpenAIServerModel`로 래핑
- EXAONE 채팅 템플릿 호환성 검증
- 단위 테스트 통과

### [ ] Task 9.2: 4개 Tool 구현 (RAG, CivilAPI, 민원답변, 공문서생성)
**Issue**: #416
**Description**: Smolagents `@tool` 데코레이터 기반 4개 도구 구현
**Acceptance Criteria**:
- Tool 1: `search_similar_cases` (FAISS/BM25)
- Tool 2: `search_civil_complaints` (행안부 API)
- Tool 3: `generate_civil_reply` (adapter_civil LoRA)
- Tool 4: `draft_public_document` (adapter_public_doc LoRA)
- `@tool` 인터페이스 준수 (name, description, inputs, output_type)
- 의존성: I-1 Task 1.3 완료 후 진행

### [ ] Task 9.3: Tool-Calling 파이프라인 E2E 검증 및 프로토타입
**Issue**: #417
**Description**: Smolagents ToolCallingAgent + 4개 Tool 통합 E2E 검증
**Acceptance Criteria**:
- `src/agent/orchestrator.py`: Agent + Tool 레지스트리
- FastAPI `/v1/agent/run` 엔드포인트
- E2E 테스트 4건 통과
- 응답 레이턴시 < 15초
- 에러 핸들링

### [ ] Task 9.4: LangGraph 전환 준비 (문서 작성)
**Issue**: #418
**Description**: Smolagents -> LangGraph 마이그레이션 가이드 작성
**Acceptance Criteria**:
- `docs/architecture/MIGRATION-smolagents-to-langgraph.md` 작성
- Tool 및 Agent 아키텍처 전환 가이드
- 예상 공수 및 리스크 분석

---

## Workstream 10: 아키텍처 문서 동기화

**Initiative**: I-10 (#407)
**Milestone**: M3: 고도화 및 최적화
**상태**: Documentation (병렬 진행 가능)

### [x] Task 10.1: ADR-006 하이브리드 아키텍처 업데이트
**Issue**: #410 (Note: 별도의 ADR 수정 이슈 미생성, 직접 ADR-006 파일 수정)
**설명**: ADR-006에 Smolagents Phase 1 / LangGraph Phase 2 명시
**완료**: 2026-04-01 완료
- Discussion #403 링크 추가
- Phase 1/2 구분 명시
- Smolagents 경계 명확화
- LoRA 레지스트리 포맷 정의

### [x] Task 10.2: WORKFLOW 문서 Smolagents 구체화
**Issue**: #411 (Note: 별도의 WORKFLOW 수정 이슈 미생성, 직접 WORKFLOW.md 파일 수정)
**설명**: WORKFLOW v2.0으로 Smolagents 기반 재작성
**완료**: 2026-04-01 완료
- Actors: Smolagents Agent 추가
- STEP 2~4: Smolagents 아키텍처 기반 재구성
- TC-05~TC-07: 추가 테스트 시나리오
- Prerequisites: smolagents[vllm] 명시

### [x] Task 10.3: tasklist.md 동기화
**설명**: 모든 Initiative와 Task를 GitHub 이슈와 동기화
**완료**: 2026-04-01 완료 (현재 파일)

---

## Dependency Graph

```
I-1 Task 1.3 (Multi-LoRA) ──┬──→ I-2 Task 2.1 (vLLM Multi-LoRA)
                            ├──→ I-9 Task 9.2 (4개 Tool @tool)
                            └──→ I-7 Task 7.1 (Docker)

I-9 Task 9.1 ──→ I-2 Task 2.3 (Smolagents AgentExecutor)
I-9 Task 9.2 ──→ I-3 Task 3.1~3.4 (Tool 구현)

I-2 Task 2.3 ──→ I-3 Task 3.1 (민원분석 API Tool)
I-2 Task 2.3 ──→ I-3 Task 3.2 (FAISS RAG Tool)

I-5 (Design) ──→ I-6 (UI 구축)
I-6 ──→ I-4 (Frontend Integration)

I-1, I-2, I-3, I-4, I-5, I-6, I-7, I-9 ──→ I-8 (E2E Testing)
```

---

## Milestone 배치

| Milestone | Initiative | 예상 소요 |
|-----------|-----------|---------|
| **M3**: 고도화 및 최적화 | I-1, I-2, I-3, I-4, I-5, I-6, I-7, I-9 | 8주 |
| **M4**: 테스트 및 문서화 | I-8 | 3주 |
| **M5**: 최종 납품 | I-10 (#375 AIOSS) | 1주 |

---

## Status

- [x] ADR-006 v2.0 (Smolagents Phase 1/2 명시)
- [x] WORKFLOW v2.0 (Smolagents 구체화)
- [x] tasklist.md (모든 Workstream 동기화)
- [x] GitHub 이슈 트리 구조 (11개 Task 이슈 생성)
- [ ] Implementation (2026-04-01 시작 예정)
