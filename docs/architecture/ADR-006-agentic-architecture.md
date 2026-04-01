# ADR-006: 3-Tier Agentic Architecture & Multi-LoRA Serving for GovOn

## Status
Accepted

## Context
GovOn은 기존에 단일 방향의 정적 파이프라인(민원 입력 -> LLM 추론 -> 답변 반환)으로 구축되었습니다. 하지만, 사용자의 요구사항이 복잡해지고(유사 사례 검색, 외부 API를 통한 통계 확인, 공문서 자동 초안 작성 등), 다양한 도구(Tools)의 연동이 필수적이게 되었습니다. 기존 구조로는 이러한 도구들을 유연하게 조합하여 자율적으로 목표를 달성하는 "에이전트(Agent)"로서의 기능을 수행하기 어렵습니다. 따라서 전체 아키텍처를 유연하고 확장 가능한 에이전트 기반 구조로 재설계해야 합니다.

또한, 뇌(Brain) 역할을 하는 모델은 "도구 호출(Tool Calling)"에 최적화되어야 하고, 손과 발(Tools) 역할을 하는 문서 생성 모델은 "민원 답변"이나 "공문서 포맷팅"에 최적화되어야 합니다. 이 이질적인 능력들을 하나의 모델에 모두 파인튜닝(Mixed Training)할 경우 파국적 망각(Catastrophic Forgetting)이 발생할 확률이 높습니다. 하지만 보급형 GPU(RTX 3060급) 환경의 온프레미스 제약상 여러 개의 무거운 독립 모델을 동시에 띄우는 것은 VRAM(OOM 리스크) 및 지연 시간 측면에서 불가능합니다.

### 참고: 에이전트 프레임워크 선택 (Discussion #403)

[Discussion #403: 에이전틱 AI 시스템 기술 검토](../../discussions/403)에서 EXAONE-Deep-7.8B의 네이티브 tool calling 미지원 제약을 분석한 결과, 다음과 같은 기술 선택이 이루어졌습니다:

- **Phase 1**: Smolagents 우선 도입 (코드 에이전트 방식, 네이티브 tool calling 불필요)
- **Phase 2**: EXAONE 4.0 도입 시 LangGraph 전환 (네이티브 tool calling 활용)

이 ADR은 위 의사결정을 Architecture 레벨에서 구체화합니다.

## Decision
시스템을 **3-Tier Agentic Architecture (3계층 에이전틱 아키텍처)**와 **Multi-LoRA 서빙 아키텍처**를 결합하여 전면 개편합니다.
이 아키텍처는 사람의 역할과 AI의 인지/행동 역할을 명확히 분리하며, 다음과 같은 3가지 핵심 구성요소로 이루어집니다.

### 1. 3계층 역할 분리
1. **Human (UI 계층 - 인간)**
   - **역할**: 사용자가 지시(명령)를 내리고 결과를 확인하며 상호작용하는 창구입니다.
   - **구현**: React/Next.js 기반의 에이전트 사이드바 및 메인 화면. 멀티턴 대화를 통해 의도를 명확히 전달하고, 스트리밍된 응답을 받습니다.

2. **Brain (Query Engine / Orchestrator 계층 - 뇌)**
   - **역할**: 사용자의 의도를 분석하고, 목표 달성을 위해 어떤 도구를 사용할지 결정(Planning & Routing)하는 제어 센터입니다.
   - **구현**: FastAPI 백엔드가 사용자의 메시지를 받아 사용 가능한 도구 목록(JSON Schema)과 함께 프롬프트를 구성하고, 도구 호출 여부(Tool Calling)를 판단합니다. 도구 실행 결과를 취합하여 최종적인 자연어 응답을 합성(Synthesis)합니다.

3. **Hands and Feet (Tools 계층 - 손과 발)**
   - **역할**: 뇌(Query Engine)의 지시에 따라 실제로 물리적/논리적 작업을 수행하는 실행 장치들입니다.
   - **구현**:
     - **민원답변 생성 기능**: 자체 파인튜닝된 지식을 바탕으로 문서 초안 생성.
     - **공문서 생성 기능**: 보도자료, 연설문 등 공공 문서 초안을 생성하는 내부 로직.
     - **유사 사례 검색 (RAG)**: FAISS 벡터 DB를 활용한 내부 매뉴얼 및 과거 민원 이력 검색.
     - **외부 API 연동**: 공공데이터포털 등의 실시간 외부 민원 통계/사례 조회.

### 2. Multi-LoRA 동적 서빙 (vLLM)
보급형 GPU 제약을 극복하고 전문성을 유지하기 위해 물리적인 다중 모델 배포 대신 **논리적인 어댑터 교체 방식**을 채택합니다.
- **Base Model (메모리 상주)**: `EXAONE-Deep-7.8B-AWQ` (GPU VRAM에 1번만 로드, 약 5GB)
- **Adapter 1 (Brain 용)**: Tool-Calling 최적화 LoRA
- **Adapter 2 (Tool 용)**: 민원 답변 생성 최적화 LoRA
- **Adapter 3 (Tool 용)**: 공문서 생성(보도자료 등) 최적화 LoRA

**작동 흐름 (Orchestrator Loop)**:
FastAPI 백엔드(Orchestrator)가 추론 단계(Intent 파악 vs 문서 생성)에 맞춰 vLLM 서버에 `lora_request` 파라미터를 동적으로 스위칭하며 API를 호출합니다. 베이스 모델은 고정된 상태에서 수십 MB 크기의 어댑터 연산만 교체되므로 속도 저하(스와핑) 없이 고품질의 결과물을 얻을 수 있습니다.

### 3. 에이전트 프레임워크 (Phase별 전략)

**Problem**: EXAONE-Deep-7.8B은 추론 특화 모델로 네이티브 function calling / tool calling을 지원하지 않습니다. 기존 프롬프트 엔지니어링 + JSON 파싱 방식은 구조화된 도구 호출의 안정성과 확장성에 한계가 있습니다.

**Decision**:

#### Phase 1: Smolagents 기반 코드 에이전트 (현재 → M3)
- **프레임워크**: `smolagents[vllm] >= 1.11.0` (HuggingFace 공식, ~1,000줄 경량 코어)
- **모델 연결**: EXAONE-Deep-7.8B를 vLLM OpenAI-compatible 엔드포인트(`/v1/chat/completions`)를 통해 Smolagents `OpenAIServerModel`로 래핑
- **의도 파악**: Smolagents `ToolCallingAgent` 또는 `CodeAgent`가 User query → Tool 의도 결정 (JSON 함수 호출 불필요)
- **Architecture 매핑**:
  - **Brain (Query Engine)**: Smolagents Agent (Planning → Tool Selection → Execution → Synthesis)
  - **Hands and Feet (Tools)**: 4개 `@tool` 데코레이터 (search_similar_cases, search_civil_complaints, generate_civil_reply, draft_public_document)
  - **FastAPI Backend**: I/O Adapter + 세션 영속성 + 컨텍스트 윈도우 관리
- **Milestone**: M3 (완료 대상)

#### Phase 2: LangGraph 기반 상태 머신 (선택적, EXAONE 4.0 이후)
- **트리거 조건**: EXAONE 4.0 라우터 모델 도입 + 네이티브 tool calling 검증 완료
- **프레임워크**: `langgraph` + `langchain-openai`
- **모델 연결**: EXAONE 4.0을 ChatOpenAI-compatible 엔드포인트로 연결
- **의도 파악**: 네이티브 `tool_choice="auto"` 기반 (Smolagents 코드 에이전트 불필요)
- **이점**:
  - 체크포인팅 (실행 중단 후 재개)
  - Human-in-the-loop (승인 필요 작업 지원)
  - 더 정교한 상태 관리
- **마이그레이션**: `docs/architecture/MIGRATION-smolagents-to-langgraph.md` 참고
- **Milestone**: TBD (미정)

**책임 경계**:
| 계층 | 기술 | 책임 | 소유자 |
|------|------|------|-------|
| **I/O Adapter** | FastAPI 미들웨어 | HTTP 요청/응답 | backend/api_server.py |
| **Reasoning Engine** | Smolagents Agent | 의도 파악 → Tool 선택 | backend/agent/orchestrator.py |
| **Tools** | @tool 데코레이터 | 개별 Tool 구현 | backend/inference/tools/ |
| **Session Management** | SQLAlchemy ORM | 대화 이력 영속성, Context Window 관리 | backend/db/ |
| **Serving** | vLLM Multi-LoRA | 모델 추론 + LoRA 동적 스위칭 | vllm_server (별도 프로세스) |

## Consequences
**장점 (What becomes easier)**:
- **전문성 유지 (No Catastrophic Forgetting)**: Tool-Calling 능력과 도메인 지식(공문서 양식)이 섞이지 않아 양쪽 모두의 품질이 보장됩니다.
- **인프라 제약 극복**: RTX 3060 등 VRAM이 제한적인 보급형 GPU 환경에서도 '에이전틱 구조 + 다중 전문가 모델' 효과를 완벽하게 낼 수 있습니다.
- **확장성 및 유지보수성**: 새로운 문서 양식(예: 회의록)이 추가되어도 베이스 모델 재학습 없이 작은 LoRA 어댑터 1개만 추가 학습하여 서빙에 끼워넣으면 됩니다.
- **Smolagents 경량성**: 코어 로직이 ~1,000줄의 미니말한 코드로 구성되어, 디버깅과 커스터마이제이션이 용이
- **프레임워크 독립성**: Tool 인터페이스가 Smolagents-specific하지 않아, LangGraph로의 마이그레이션이 명확함
- **폐쇄망 호환성**: 네이티브 API 호출 없이 로컬 vLLM 엔드포인트만으로 작동

**단점 (What becomes harder)**:
- **복잡한 MLOps 파이프라인**: 거대한 1개의 데이터셋이 아닌, 목적에 맞게 정제된 3개의 데이터셋을 유지 관리하고 3번의 파인튜닝 스크립트를 관리해야 합니다.
- **백엔드 로직 고도화**: FastAPI 백엔드가 단순 프롬프트 전달을 넘어, 현재 컨텍스트에 맞는 적절한 `lora_name`을 동적으로 라우팅하는 로직을 완벽하게 처리해야 합니다.
- **vLLM 설정 관리**: vLLM 서버 실행 시 `--enable-lora` 설정과 여러 LoRA 어댑터 경로를 명시하는 인프라 구성이 추가됩니다.
- **Smolagents 제약사항**: 체크포인팅과 Human-in-the-loop을 직접 구현해야 함 (LangGraph의 네이티브 기능 부재)
- **프레임워크 전환 비용**: Phase 1(Smolagents)에서 Phase 2(LangGraph)로 전환 시, `@tool` → `BaseTool` 마이그레이션 + Agent 아키텍처 재설계 필요
- **모델 버전 의존성**: EXAONE 4.0 도입을 Phase 2 전환의 선행 조건으로 명확히 해야 함

#### LoRA 어댑터 레지스트리

3개의 LoRA 어댑터 메타데이터와 경로를 관리하기 위해 `config/lora_adapters.yaml` 파일을 정의합니다.

**파일 포맷** (`config/lora_adapters.yaml`):
```yaml
adapters:
  - name: "adapter_brain"
    path: "/path/to/model_brain_lora_weights"
    target_module: "q_proj,v_proj"  # QLoRA 대상 모듈
    task: "intent_classification"
    eval_metric: "json_accuracy"
    eval_score: 0.92
    created_at: "2026-03-15T10:30:00Z"
    
  - name: "adapter_civil"
    path: "/path/to/model_civil_lora_weights"
    target_module: "q_proj,v_proj"
    task: "civil_complaint_reply"
    eval_metric: "rouge_l"
    eval_score: 0.47
    created_at: "2026-03-20T14:45:00Z"
    
  - name: "adapter_public_doc"
    path: "/path/to/model_public_doc_lora_weights"
    target_module: "q_proj,v_proj"
    task: "public_document_drafting"
    eval_metric: "format_accuracy"
    eval_score: 0.88
    created_at: "2026-03-25T09:15:00Z"
```

**사용처**:
- vLLM 시작 시: `--lora-modules` 파라미터 자동 생성 (Task 2.1)
- Smolagents Agent 초기화: 사용 가능한 어댑터 목록 로드 (Task 2.3)
- Tool 구현: 특정 LoRA 어댑터 동적 선택 (Task 3.4, 9.2)
- 모니터링: WandB/Prometheus 메트릭 스크래핑

**책임**: I-1 Task 1.3에서 생성, I-2 Task 2.1에서 참조
