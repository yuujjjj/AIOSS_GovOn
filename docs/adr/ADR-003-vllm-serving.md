# ADR-003: vLLM 추론 서빙 엔진 선정

## Status

Accepted

## Context

GovOn 시스템은 AWQ 양자화된 EXAONE-Deep-7.8B 모델을 FastAPI 기반 API 서버로 서빙해야 한다. 추론 서빙 엔진 선택 시 다음 요구사항을 충족해야 한다.

1. **AWQ 모델 네이티브 지원**: AWQ W4A16g128 양자화 모델을 별도 변환 없이 직접 로드하고 추론할 수 있어야 한다.
2. **비동기 스트리밍**: FastAPI와 통합하여 토큰 단위 스트리밍 응답을 제공해야 한다. 민원 답변이 길어질 수 있으므로, 첫 토큰 응답 시간(TTFT)을 최소화해야 한다.
3. **메모리 효율성**: 16~24GB GPU VRAM 환경에서 모델 로딩과 KV 캐시를 효율적으로 관리해야 한다.
4. **동시 요청 처리**: 여러 공무원이 동시에 민원 분석을 요청할 수 있으므로, 배치 처리(continuous batching)를 지원해야 한다.
5. **Python 생태계 호환**: FastAPI, Pydantic 등 기존 Python 스택과 자연스럽게 통합되어야 한다.

## 검토 후보

| 후보 | AWQ 지원 | 비동기 엔진 | 스트리밍 | 메모리 관리 | Python 통합 |
|------|----------|-----------|---------|------------|------------|
| **vLLM** | 네이티브 | AsyncLLMEngine | SSE 스트리밍 | PagedAttention | Python API 제공 |
| Ollama | GGUF 변환 필요 | 자체 서버 | 지원 | 자체 관리 | HTTP API만 제공 |
| TGI (Text Generation Inference) | 제한적 | 자체 서버 | 지원 | Flash Attention | Rust 기반, Docker 필수 |
| TorchServe | 직접 구현 필요 | Java 기반 | 직접 구현 | 직접 관리 | Python 핸들러 |

### 상세 비교

**vLLM**
- PagedAttention 기술로 KV 캐시를 OS의 가상 메모리처럼 페이지 단위로 관리하여, 메모리 단편화를 최소화하고 동시 처리 용량을 극대화한다.
- `AsyncLLMEngine`을 Python 코드에서 직접 임포트하여 사용할 수 있으므로, FastAPI의 비동기 엔드포인트와 자연스럽게 통합된다.
- AWQ 모델을 `model` 경로만 지정하면 자동으로 인식하고 최적화된 GEMM 커널로 추론한다.
- Continuous batching으로 도착하는 요청을 즉시 처리하여 GPU 활용률을 높인다.
- `gpu_memory_utilization` 파라미터로 VRAM 사용 비율을 세밀하게 제어할 수 있다.

**Ollama**
- 사용이 간편하나, AWQ 모델을 직접 지원하지 않고 GGUF 형식으로 변환해야 한다. AWQ에서 GGUF로의 변환은 양자화 품질 손실을 수반할 수 있다.
- 자체 HTTP 서버를 띄우는 구조로, FastAPI와 통합하려면 HTTP 프록시 계층이 필요하여 레이턴시가 추가된다.
- EXAONE 커스텀 모델 코드(`trust_remote_code`)를 지원하지 않아, 별도 모델 변환 작업이 필요하다.

**TGI (Text Generation Inference)**
- Hugging Face가 개발한 Rust 기반 서빙 엔진으로, Flash Attention과 continuous batching을 지원한다.
- Docker 컨테이너 기반 배포가 표준이며, FastAPI와 통합하려면 gRPC 또는 HTTP 클라이언트를 통해 연동해야 한다.
- AWQ 지원이 vLLM 대비 제한적이며, EXAONE 커스텀 모델의 `trust_remote_code` 호환성이 불확실했다.

**TorchServe**
- PyTorch 공식 서빙 프레임워크이나, LLM 특화 기능(PagedAttention, continuous batching, 토큰 스트리밍)이 내장되어 있지 않아 모두 직접 구현해야 한다.
- AWQ 모델 서빙을 위한 커스텀 핸들러를 작성해야 하며, 개발 및 유지보수 비용이 크다.

## Decision

**vLLM**을 추론 서빙 엔진으로 선정한다.

통합 구조는 다음과 같다.

### 엔진 관리: vLLMEngineManager

`src/inference/api_server.py`에 `vLLMEngineManager` 클래스를 구현하여, `AsyncLLMEngine`과 `CivilComplaintRetriever`(RAG), `MultiIndexManager`의 생명주기를 통합 관리한다.

```
vLLMEngineManager
├── AsyncLLMEngine (vLLM 추론 엔진)
├── CivilComplaintRetriever (FAISS 기반 유사 민원 검색)
└── MultiIndexManager (CASE/LAW/MANUAL/NOTICE 다중 인덱스)
```

### 핵심 설정

| 항목 | 값 | 근거 |
|------|----|------|
| `gpu_memory_utilization` | 0.8 | 16GB GPU 기준 KV 캐시 여유 확보 |
| `max_model_len` | 8192 | 민원 텍스트 + RAG 컨텍스트 + 답변 생성에 필요한 토큰 수 |
| `trust_remote_code` | True | EXAONE 커스텀 모델 코드 로드 |
| `enforce_eager` | True | 패치된 모델에서의 안정성 확보 (CUDA graph 비사용) |
| `dtype` | float16 | AWQ 모델의 연산 정밀도 |

### EXAONE 호환성 패치

vLLM이 EXAONE 모델을 로드할 때 transformers 버전 간 호환성 문제가 발생할 수 있다. `src/inference/vllm_stabilizer.py`에서 다음 런타임 패치를 적용한다.

- `RopeParameters` 클래스 주입 (transformers 버전 차이 대응)
- `check_model_inputs` 함수 주입
- `ALL_ATTENTION_FUNCTIONS` 더미 객체 주입

이 패치들은 FastAPI 앱 시작 시 `apply_transformers_patch()`를 호출하여 vLLM 엔진 초기화 전에 적용된다.

## Consequences

### 긍정적 영향

- PagedAttention으로 16GB GPU 환경에서도 여러 동시 요청을 효율적으로 처리할 수 있다.
- `AsyncLLMEngine`이 FastAPI의 `async/await` 패턴과 직접 통합되어, 별도 프록시 계층 없이 토큰 스트리밍이 가능하다.
- AWQ 모델을 네이티브로 지원하므로, 양자화 파이프라인(ADR-002)의 출력물을 변환 없이 바로 서빙할 수 있다.
- `SamplingParams`를 요청별로 다르게 설정할 수 있어, 민원 유형에 따라 temperature, top_p 등을 조정할 수 있다.
- Continuous batching으로 요청 도착 시점에 즉시 처리를 시작하여 대기 시간을 최소화한다.

### 부정적 영향

- vLLM은 CUDA에 강하게 의존하므로, CPU 전용 환경이나 AMD GPU 환경에서는 사용할 수 없다. 개발 환경(로컬 테스트)에서도 GPU가 필요하거나 Mock을 구성해야 한다.
- EXAONE 모델과 vLLM 간 호환성을 위해 `vllm_stabilizer.py`에서 런타임 패치를 관리해야 한다. vLLM 또는 transformers 버전 업데이트 시 패치 코드 점검이 필수적이다.
- `enforce_eager=True` 설정으로 CUDA graph 최적화를 비활성화하여, 최대 처리량이 다소 감소한다. 모델 안정성이 확보되면 이 설정을 제거하여 성능을 개선할 수 있다.
- vLLM 자체가 상대적으로 빠르게 발전하는 프로젝트로, API 변경이 잦을 수 있어 버전 고정 및 호환성 테스트가 필요하다.

### 향후 고려사항

- vLLM의 EXAONE 공식 지원이 추가되면, `vllm_stabilizer.py`의 런타임 패치를 단계적으로 제거한다.
- 모델 안정성이 충분히 검증되면 `enforce_eager=False`로 전환하여 CUDA graph 최적화를 활성화하고 처리량을 개선한다.
- 요청량이 단일 GPU 처리 한계를 초과할 경우, vLLM의 텐서 병렬(tensor parallelism) 기능으로 다중 GPU 확장을 검토한다.
- vLLM 메이저 버전 업그레이드 시, `AsyncLLMEngine` API 변경 여부를 확인하고 `api_server.py`의 호환성을 점검한다.
