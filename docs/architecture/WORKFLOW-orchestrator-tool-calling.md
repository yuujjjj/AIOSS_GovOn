# WORKFLOW: Agent Orchestrator & Tool Calling
**Version**: 2.0
**Date**: 2026-04-01
**Author**: Workflow Architect
**Status**: Active (Smolagents Phase 1 기반)
**Implements**: GovOn Orchestrator Upgrade with Smolagents Agent (vs v1.1의 Custom Orchestrator)

## Overview
This workflow defines the end-to-end execution path for the newly upgraded GovOn conversational AI system. When a public official sends a message, the system uses an LLM Orchestrator to determine the user's intent. If external data is needed, it calls the appropriate tool (Local RAG or Civil Complaint Analysis API). If the intent is to draft a public document (e.g., press release, speech), the LLM utilizes its fine-tuned knowledge (trained on the Ministry of the Interior and Safety Public Doc dataset) to generate the draft natively without an external API call at runtime.

최종 자연어 응답을 합성(Synthesis)합니다.

### Phase 기술 선택

이 워크플로우는 **Smolagents Phase 1** 기반으로 설계되었습니다.
- EXAONE-Deep-7.8B의 네이티브 tool calling 미지원 제약을 극복하기 위해 Smolagents의 코드 에이전트 방식 채택
- 향후 EXAONE 4.0 도입 시 LangGraph로 전환할 수 있도록 설계 (상세: `docs/architecture/ADR-006-agentic-architecture.md` 참고)

## Actors
| Actor | Role in this workflow |
|---|---|
| Public Official (Customer) | Initiates the action via the Web UI Chat Sidebar |
| FastAPI Backend | Validates requests, manages conversation state, and orchestrates the LLM/Tools |
| LLM Engine (vLLM + Smolagents) | Smolagents ToolCallingAgent는 vLLM의 OpenAI-compatible 엔드포인트를 활용하여 의도 분류, 도구 호출 결정, 응답 생성. Brain LoRA (`adapter_brain`)을 사용해 Tool-Calling 최적화. |
| Smolagents Agent | ToolCallingAgent 또는 CodeAgent로서, 사용자 쿼리를 분석하고 사용 가능한 Tool 목록(JSON Schema)을 참고해 실행할 Tool을 선택. 재시도 로직과 에러 핸들링 포함. |
| Tool: Local RAG (FAISS) | Retrieves internal civil complaint histories and manuals |
| Tool: Civil Complaint API | External API (`apis.data.go.kr/1140100/minAnalsInfoView5`) for real-time similar cases and statistics |

## Prerequisites
- vLLM serving the EXAONE-Deep-7.8B model with Multi-LoRA enabled is healthy.
- Smolagents `smolagents[vllm] >= 1.11.0` 설치 필수
- EXAONE-Deep-7.8B를 vLLM의 `/v1/chat/completions` 엔드포인트로 서빙 중
- 4개 Tool (@tool 데코레이터)이 `src/inference/tools/` 디렉토리에 등록되어 있음
- FastAPI 애플리케이션이 Smolagents Agent를 초기화한 상태
- Conversation history database (SQLAlchemy ORM) operational
- External API Keys (Data.go.kr) configured in `.env`

## Trigger
User submits a message through the chat input in the `PG-001` Main Screen.
**Entry point**: `POST /api/v1/chat/message` (or WebSocket equivalent).

---

## Workflow Tree

### STEP 1: Message Validation & Context Assembly
**Actor**: FastAPI Backend
**Action**: Receives user message, retrieves conversation history, and formats the system prompt with available tool schemas (JSON Schema).
**Timeout**: 2s
**Input**: `{ "session_id": "uuid", "message": "국토부 도로 공사 보도자료 초안 써줘" }`
**Output on SUCCESS**: `{ "context": [...history, new_msg], "tools": [...] }` -> GO TO STEP 2
**Output on FAILURE**:
  - `FAILURE(validation_error)`: Invalid session or empty message -> return 400.

**Observable states during this step**:
  - Customer sees: `TypingIndicator` (Loading spinner)

### STEP 2: Intent & Tool Selection (Smolagents Agent 의도 파악)

**Actor**: Smolagents ToolCallingAgent

**Action**: FastAPI 백엔드가 사용자 메시지와 대화 이력을 Smolagents Agent에 전달합니다. Agent는 다음을 수행합니다:
1. `agent.run(user_input)` 호출 -> 내부적으로 vLLM의 OpenAI-compatible API를 사용하여 Brain LoRA (`lora_request="adapter_brain"`)로 추론
2. Tool 목록(4개 @tool의 JSON Schema)을 분석해 실행할 Tool 결정
3. Tool 의도가 감지되면 Tool 이름과 파라미터를 JSON 형식으로 결정
4. Tool 의도가 없으면 (일반 대화) 직접 텍스트 응답 생성 후 반환

**Timeout**: 5s (vLLM 추론 + Agent 의사결정)

**Output on SUCCESS**:
  - IF `Tool Call`: `{ "tool_name": "search_civil_complaints", "args": {...} }` -> GO TO STEP 3
  - IF `Drafting Document Task`: `{ "tool_name": "draft_public_doc", "args": {"type": "press_release"} }` -> GO TO STEP 4
  - IF `Direct Response`: `{ "type": "text", "content": "안녕하세요! ..." }` -> GO TO STEP 5

**Output on FAILURE**:
  - `FAILURE(generation_error)`: Agent가 유효한 Tool 호출이나 텍스트를 생성하지 못함 -> GO TO ABORT_CLEANUP
  - `FAILURE(timeout)`: vLLM 추론 5초 초과 -> GO TO ABORT_CLEANUP

**Observable states during this step**:
  - Customer sees: `TypingIndicator` (기존과 동일)
  - Backend logs: Agent의 Tool 선택 로직 (debug mode)

### STEP 3: Non-Drafting Tool Execution (Data Fetching)
**Actor**: FastAPI Backend
**Action**: Routes the execution to the specific data-fetching tool handler based on STEP 2 output.

더 구체적으로, Smolagents Agent가 Tool 호출을 결정하면, FastAPI는 다음을 수행합니다:
1. Agent의 Tool 호출 결과를 파싱
2. 실제 Tool 함수(`src/inference/tools/` 디렉토리)를 동기/비동기로 실행
3. Tool 반환값(str 또는 dict)을 정규화해서 Agent에 피드백 -> Agent가 다음 Step으로 진행

**Timeout**: 10s
**Branches**:
  - **Branch 3A: Local RAG (FAISS)**
    - Queries the local vector DB for internal manuals.
  - **Branch 3B: Civil Complaint Analysis API**
    - Calls `apis.data.go.kr/1140100/minAnalsInfoView5` API for similar cases, trending keywords.
**Output on SUCCESS**: `{ "tool_result": "Raw data or summarized JSON" }` -> GO TO STEP 4
**Output on FAILURE**:
  - `FAILURE(api_timeout)`: External API times out -> [recovery: Return graceful error context to LLM: "Tool API timed out"] -> GO TO STEP 4

**Observable states during this step**:
  - Customer sees: UI updates to show `Tool execution in progress... (e.g., "민원분석 API 조회 중...")`

### STEP 4: Final Synthesis & Drafting (LLM 응답 생성)

**Actor**: Smolagents Agent (또는 FastAPI 백엔드)

**Action**: STEP 3의 Tool 결과를 포함해서 최종 응답을 생성합니다:
- Tool 결과가 있다면: Agent는 Tool 결과를 컨텍스트에 추가하고, 적절한 LoRA 어댑터를 선택해서 최종 응답 생성
  - 민원 답변 Task: `lora_request="adapter_civil"` 사용
  - 공문서 생성 Task: `lora_request="adapter_public_doc"` 사용
- Tool 결과가 없다면 (STEP 2에서 직접 응답): Agent가 이미 텍스트를 생성했으므로 이 Step 스킵

**Timeout**: 10s (LoRA 스위칭 + vLLM 추론)

**Output on SUCCESS**: Streaming text chunks `"...final response..."` -> GO TO STEP 5

### STEP 5: Response Delivery & State Persistence
**Actor**: FastAPI Backend
**Action**: Streams the final response to the Frontend and saves the updated conversation history to the database.
**Output on SUCCESS**: HTTP 200 OK (Stream closed), DB updated.
**What customer sees**: Full text response or official document draft (with HTML tables/images) is rendered in the chat UI.

### ABORT_CLEANUP: Error Handling
**Triggered by**: LLM failure, Smolagents Agent failure, unrecoverable system error.
**Actions**:
  1. Save error state to conversation log.
  2. Send standard error message to Frontend.
**What customer sees**: Error message bubble in UI with a "Retry" button.

---

## State Transitions
```text
[idle]
  -> (User sends message)
  -> [agent_reasoning] (Smolagents Agent 실행)
    -> (Tool 의도 감지) -> [executing_tool]
    -> (직접 응답) -> [streaming_response]
  [executing_tool]
    -> (Tool 완료)
    -> [agent_synthesis] (Agent가 최종 응답 생성)
    -> [streaming_response]
  [streaming_response]
    -> (Done)
    -> [idle]
```

## Handoff Contracts

### [Backend] -> [Civil Complaint API]
**Endpoint**: `GET apis.data.go.kr/1140100/minAnalsInfoView5`
**Payload**: `serviceKey`, `searchKeyword`, `target` (similar cases)
**Success response**: JSON containing similar complaint cases, keywords.
**Timeout**: 10s
**On Failure**: Do not crash user request. Pass `{"error": "API failed"}` to Step 4 so LLM can apologize.

## Test Cases
| Test | Trigger | Expected behavior |
|---|---|---|
| TC-01: Direct Chat | "안녕" | Step 2 chooses No Tool. Step 5 returns greeting. |
| TC-02: Public Doc Generation | "국토부 도로 공사 보도자료 초안 써줘" | Step 2 chooses No Tool. LLM drafts the document natively using its fine-tuned knowledge. Step 5 streams the draft. |
| TC-03: Similar Case Search | "소음 민원 유사 사례 찾아봐" | Step 2 chooses `search_civil_complaints`. Step 3B calls API. Step 4 summarizes cases. |
| TC-04: API Timeout Fallback | Step 3 external API takes >10s | LLM receives error context and tells user "현재 민원분석 API 연동 지연으로 조회할 수 없습니다." |
| TC-05: Multi-tool 연쇄 호출 | "RAG로 유사 사례 찾은 후 민원분석 API로 통계 확인" | STEP 2에서 다중 Tool 의도 인식, STEP 3에서 순차 또는 병렬 실행, STEP 4에서 결과 통합 후 응답 |
| TC-06: Tool 실패 후 Agent 대체 | STEP 3에서 Tool 타임아웃 (예: API 응답 10초 초과) | Smolagents가 Tool 실패 컨텍스트를 수신 -> STEP 4에서 "현재 조회 불가" 메시지 생성 (graceful degradation) |
| TC-07: LoRA 스위칭 성능 | adapter_brain -> adapter_civil 전환 시 레이턴시 측정 | vLLM Multi-LoRA 동적 스위칭 < 1초 (GPU 벤치마크 기준) |
