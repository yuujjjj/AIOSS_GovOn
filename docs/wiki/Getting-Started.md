# Getting Started

GovOn 추론 서버를 로컬 환경에서 기동하고 API를 호출하는 과정을 안내한다.

---

## 목차

- [사전 요구사항](#사전-요구사항)
- [저장소 클론](#저장소-클론)
- [의존성 설치](#의존성-설치)
- [환경변수 설정](#환경변수-설정)
- [추론 서버 기동](#추론-서버-기동)
- [API 테스트](#api-테스트)
- [관련 문서](#관련-문서)

---

## 사전 요구사항

| 항목 | 최소 요구 | 권장 |
|------|-----------|------|
| Python | 3.10+ | 3.10.x |
| CUDA | 12.x | 12.1+ |
| GPU VRAM | 16 GB (AWQ INT4 모델 기준) | 24 GB 이상 |
| RAM | 16 GB | 32 GB |
| 디스크 | 30 GB (모델 + 인덱스) | 50 GB |

NVIDIA 드라이버와 CUDA Toolkit이 설치되어 있는지 확인한다.

```bash
nvidia-smi          # GPU 인식 및 드라이버 버전 확인
nvcc --version      # CUDA 컴파일러 버전 확인
python --version    # Python 3.10 이상인지 확인
```

---

## 저장소 클론

```bash
git clone https://github.com/um-yunsang/GovOn.git
cd GovOn
```

---

## 의존성 설치

가상환경을 생성한 뒤 의존성을 설치한다.

```bash
python -m venv .venv
source .venv/bin/activate
```

### 기본 의존성

```bash
pip install -r requirements.txt
```

`requirements.txt`에는 다음 주요 패키지가 포함되어 있다.

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `torch` | >= 2.1.0 | PyTorch 딥러닝 프레임워크 |
| `transformers` | >= 4.40.0 | EXAONE 모델 로드 |
| `vllm` | >= 0.4.0 | 고성능 추론 엔진 |
| `fastapi` | >= 0.100.0 | API 서버 |
| `sentence-transformers` | >= 2.2.0 | multilingual-e5-large 임베딩 |
| `faiss-cpu` | >= 1.7.4 | FAISS 벡터 검색 |
| `loguru` | >= 0.7.0 | 구조화 로깅 |

### 추론 서버 전용 설치

추론 서버만 실행할 경우 다음 명령어를 사용한다.

```bash
pip install -e ".[inference]"
```

### 개발 도구 설치

테스트, 린터, 포매터를 포함한 개발 환경이 필요하면 다음을 실행한다.

```bash
pip install -e ".[dev]"
```

---

## 환경변수 설정

추론 서버는 환경변수로 동작을 제어한다. `.env` 파일을 프로젝트 루트에 생성하거나 셸에서 직접 export한다.

```bash
# .env 예시
MODEL_PATH=umyunsang/GovOn-EXAONE-LoRA-v2    # HuggingFace 모델 경로 또는 로컬 경로
DATA_PATH=data/processed/v2_train.jsonl        # RAG용 학습 데이터 (JSONL)
INDEX_PATH=models/faiss_index/complaints.index # FAISS 인덱스 파일 경로
GPU_UTILIZATION=0.8                            # GPU 메모리 사용 비율 (0.0~1.0)
MAX_MODEL_LEN=8192                             # 최대 시퀀스 길이
API_KEY=your-secret-api-key                    # API 인증 키 (미설정 시 인증 건너뜀)
CORS_ORIGINS=http://localhost:3000             # 허용할 CORS 출처 (쉼표 구분)
```

### 환경변수 상세 설명

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_PATH` | `umyunsang/GovOn-EXAONE-LoRA-v2` | EXAONE AWQ 양자화 모델 경로. HuggingFace Hub ID 또는 로컬 디렉토리 경로를 지정한다. |
| `DATA_PATH` | `data/processed/v2_train.jsonl` | RAG 인덱스 빌드에 사용할 JSONL 데이터 경로. `INDEX_PATH`에 인덱스가 없을 때 이 데이터로 인덱스를 자동 생성한다. |
| `INDEX_PATH` | `models/faiss_index/complaints.index` | FAISS 인덱스 파일 경로. 파일이 존재하면 로드하고, 없으면 `DATA_PATH`에서 빌드한다. |
| `GPU_UTILIZATION` | `0.8` | vLLM이 사용할 GPU 메모리 비율. OOM 발생 시 `0.7` 이하로 낮춘다. |
| `MAX_MODEL_LEN` | `8192` | 입력+출력 합산 최대 토큰 수. VRAM이 부족하면 `4096`으로 줄인다. |
| `API_KEY` | 미설정 | `X-API-Key` 헤더로 전달하는 인증 키. 미설정 시 인증 없이 접근 가능하다 (개발 환경용). |
| `CORS_ORIGINS` | 빈 문자열 | 허용할 CORS 출처 목록. 쉼표로 구분한다. 빈 문자열이면 CORS 미들웨어를 추가하지 않는다. |

---

## 추론 서버 기동

환경변수를 설정한 뒤 다음 명령어로 서버를 기동한다.

```bash
uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000 --reload
```

서버가 정상 기동되면 다음 과정이 순차적으로 실행된다.

1. EXAONE 호환 런타임 패치 적용 (`vllm_stabilizer.apply_transformers_patch`)
2. vLLM `AsyncLLMEngine` 초기화 (모델 로딩, GPU 메모리 할당)
3. RAG Retriever 초기화 (FAISS 인덱스 로드 또는 빌드, multilingual-e5-large 임베딩 모델 로드)

기동 완료 후 `http://localhost:8000/docs`에서 Swagger UI를 확인할 수 있다.

---

## API 테스트

### 헬스 체크

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

정상 응답 예시:

```json
{
    "status": "healthy",
    "rag_enabled": true,
    "indexes": null
}
```

### 텍스트 생성 (비스트리밍)

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "prompt": "[|system|]당신은 민원 처리 전문가입니다.[|endofturn|]\n[|user|]도로 포장이 파손되어 위험합니다. 보수 요청합니다.[|endofturn|]\n[|assistant|]",
    "max_tokens": 512,
    "temperature": 0.7,
    "use_rag": true
  }'
```

응답 예시:

```json
{
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "text": "안녕하세요. 도로 포장 파손 관련 민원을 접수해 주셔서 감사합니다...",
    "prompt_tokens": 45,
    "completion_tokens": 128,
    "retrieved_cases": [
        {
            "id": "case_001",
            "category": "도로/교통",
            "complaint": "도로 포장 파손 신고",
            "answer": "해당 지역 도로 보수 공사를 진행하겠습니다.",
            "score": 0.92
        }
    ]
}
```

### 텍스트 생성 (스트리밍)

```bash
curl -X POST http://localhost:8000/v1/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "prompt": "[|system|]당신은 민원 처리 전문가입니다.[|endofturn|]\n[|user|]소음 민원을 제기합니다.[|endofturn|]\n[|assistant|]",
    "max_tokens": 256,
    "stream": true,
    "use_rag": true
  }'
```

스트리밍 응답은 SSE(Server-Sent Events) 형식으로 전달된다.

### 유사 민원 검색

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "query": "도로 포장 파손",
    "doc_type": "case",
    "top_k": 5
  }'
```

---

## 관련 문서

- [[Development-Guide]] - 개발 환경 설정, 브랜치 전략, 파이프라인 실행법
- [[Troubleshooting]] - GPU OOM, vLLM 오류, FAISS 인덱스 문제 해결
