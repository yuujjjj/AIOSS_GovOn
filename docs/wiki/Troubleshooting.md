# Troubleshooting

GovOn 추론 서버 및 학습 파이프라인에서 자주 발생하는 문제와 해결 방법을 안내한다.

---

## 목차

- [GPU OOM (Out of Memory)](#gpu-oom-out-of-memory)
- [vLLM 서빙 오류](#vllm-서빙-오류)
- [AWQ 양자화 호환성 문제](#awq-양자화-호환성-문제)
- [FAISS 인덱스 로드 실패](#faiss-인덱스-로드-실패)
- [API Key 인증 오류](#api-key-인증-오류)
- [기타 문제](#기타-문제)
- [관련 문서](#관련-문서)

---

## GPU OOM (Out of Memory)

### 증상

서버 기동 또는 추론 요청 시 다음과 유사한 오류가 발생한다.

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate XX MiB (GPU X; XX GiB total capacity; XX GiB already allocated)
```

### 원인

vLLM 엔진이 GPU 메모리를 초과하여 할당하려고 시도한다. AWQ INT4 모델(EXAONE-Deep-7.8B)은 약 6~8 GB VRAM을 사용하지만, vLLM의 KV 캐시와 임베딩 모델이 추가 메모리를 소비한다.

### 해결 방법

**1단계: `GPU_UTILIZATION` 낮추기**

기본값은 `0.8`이다. VRAM 16 GB 환경에서 OOM이 발생하면 `0.7` 이하로 조정한다.

```bash
export GPU_UTILIZATION=0.7
```

**2단계: `MAX_MODEL_LEN` 줄이기**

기본값은 `8192`이다. KV 캐시 메모리를 줄이려면 `4096` 또는 `2048`로 낮춘다.

```bash
export MAX_MODEL_LEN=4096
```

**3단계: 다른 GPU 프로세스 종료**

```bash
# GPU 사용 프로세스 확인
nvidia-smi

# 불필요한 프로세스 종료
kill -9 <PID>
```

**4단계: PyTorch 캐시 정리 후 재기동**

```bash
python -c "import torch; torch.cuda.empty_cache()"
uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 환경별 권장 설정

| VRAM | `GPU_UTILIZATION` | `MAX_MODEL_LEN` |
|------|-------------------|------------------|
| 16 GB | `0.7` | `4096` |
| 24 GB | `0.8` | `8192` |
| 40 GB+ | `0.85` | `8192` |

---

## vLLM 서빙 오류

### EXAONE 런타임 패치 실패

#### 증상

서버 기동 시 다음과 유사한 오류가 발생한다.

```
AttributeError: module 'transformers.modeling_rope_utils' has no attribute 'RopeParameters'
AttributeError: module 'transformers.utils.generic' has no attribute 'check_model_inputs'
```

#### 원인

EXAONE-Deep-7.8B 모델은 최신 transformers API를 사용하는데, 설치된 transformers 버전에 해당 속성이 없는 경우 발생한다. GovOn은 `src/inference/vllm_stabilizer.py`에서 런타임 패치로 이 문제를 해결한다.

#### 해결 방법

1. `vllm_stabilizer.py`의 패치가 정상 적용되는지 확인한다.

```bash
python -c "from src.inference.vllm_stabilizer import apply_transformers_patch; apply_transformers_patch()"
```

정상이면 다음 메시지가 출력된다.

```
Applying runtime patches for EXAONE...
  [SUCCESS] Injected RopeParameters into transformers.modeling_rope_utils
  [SUCCESS] Injected check_model_inputs into transformers.utils.generic
  [SUCCESS] Injected ALL_ATTENTION_FUNCTIONS dummy
```

2. 패치가 이미 적용 완료된 경우(속성이 이미 존재) 성공 메시지 없이 종료된다. 이는 정상이다.

3. 여전히 오류가 발생하면 transformers 버전을 확인한다.

```bash
pip show transformers
```

`>= 4.40.0`인지 확인하고, 버전이 낮으면 업그레이드한다.

```bash
pip install --upgrade transformers>=4.40.0
```

### trust_remote_code 오류

#### 증상

```
ValueError: Loading ... requires you to execute the configuration file in that repo on your local machine.
```

#### 원인

EXAONE 모델은 커스텀 코드를 사용하므로 `trust_remote_code=True` 설정이 필요하다.

#### 해결 방법

GovOn의 `api_server.py`는 이 설정을 기본으로 활성화한다 (`TRUST_REMOTE_CODE = True`). 직접 vLLM을 사용하는 경우 다음을 확인한다.

```python
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    trust_remote_code=True,  # 반드시 True
    # ...
)
```

### enforce_eager 관련 오류

#### 증상

CUDA graph 관련 오류 또는 불안정한 추론 결과가 발생한다.

#### 해결 방법

GovOn은 EXAONE 패치 모델의 안정성을 위해 `enforce_eager=True`를 기본 사용한다. 이 설정은 `api_server.py`의 `vLLMEngineManager.initialize()`에 적용되어 있다.

---

## AWQ 양자화 호환성 문제

### AutoAWQ 설치 오류

#### 증상

```
ImportError: No module named 'awq'
```

또는 CUDA 버전 불일치로 빌드가 실패한다.

#### 해결 방법

1. AutoAWQ를 설치한다.

```bash
pip install autoawq>=0.2.0
```

2. CUDA 버전과 PyTorch CUDA 버전이 일치하는지 확인한다.

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

두 버전이 다르면 PyTorch를 CUDA 버전에 맞게 재설치한다.

```bash
# CUDA 12.1 예시
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### EXAONE 모델 양자화 시 monkey-patch 오류

#### 증상

`merge_lora.py` 또는 `quantize_awq.py` 실행 시 `AttributeError`가 발생한다.

#### 해결 방법

`src/quantization/merge_lora.py`에는 EXAONE 호환에 필요한 monkey-patch가 포함되어 있다. 다음 항목이 패치된다.

- `transformers.modeling_rope_utils.RopeParameters`
- `transformers.utils.generic.check_model_inputs`
- `transformers.utils.generic.maybe_autocast`
- `transformers.integrations.use_kernel_forward_from_hub`
- `transformers.masking_utils.create_causal_mask`

이 패치는 스크립트 실행 시 자동 적용된다. 별도 스크립트에서 모델을 로드하려면 `merge_lora.py` 상단의 패치 코드를 참고하여 적용한다.

### 캘리브레이션 데이터 형식 오류

#### 증상

양자화 시 캘리브레이션 샘플이 0개로 준비된다.

#### 해결 방법

캘리브레이션 데이터(`quantize_awq.py`의 `CALIB_DATA_PATH`)가 JSONL 형식이며 각 행에 `instruction`, `input`, `output` 필드가 포함되어 있는지 확인한다.

---

## FAISS 인덱스 로드 실패

### 인덱스 파일 미존재

#### 증상

```
ERROR - Data path not found: data/processed/v2_train.jsonl
```

또는 서버 기동 시 RAG 기능이 비활성화된다.

#### 해결 방법

1. `INDEX_PATH` 환경변수에 지정된 경로에 인덱스 파일이 있는지 확인한다.

```bash
ls -la models/faiss_index/complaints.index
ls -la models/faiss_index/complaints.index.meta.json
```

2. 인덱스가 없으면 `DATA_PATH`에 JSONL 데이터를 배치한다. 서버 기동 시 자동으로 인덱스를 빌드한다.

```bash
export DATA_PATH=data/processed/v2_train.jsonl
export INDEX_PATH=models/faiss_index/complaints.index
```

3. 인덱스를 수동으로 빌드하려면 Python에서 직접 실행한다.

```python
from src.inference.retriever import CivilComplaintRetriever

retriever = CivilComplaintRetriever(
    data_path="data/processed/v2_train.jsonl"
)
retriever.save_index("models/faiss_index/complaints.index")
```

### 메타데이터 파일 누락

#### 증상

```
FileNotFoundError: .../complaints.index.meta.json
```

#### 원인

FAISS 인덱스 파일(`complaints.index`)은 존재하지만, 함께 저장되는 메타데이터 파일(`complaints.index.meta.json`)이 누락되었다.

#### 해결 방법

인덱스와 메타데이터를 함께 재빌드한다. 위의 수동 빌드 방법을 사용한다.

### MultiIndexManager 인덱스 로드 실패

#### 증상

```
WARNING - 인덱스 자동 로드 실패 (case): ...
```

#### 해결 방법

`MultiIndexManager`는 `models/faiss_index/` 디렉토리 하위에 타입별 디렉토리 구조를 기대한다.

```
models/faiss_index/
+-- case/
|   +-- index.faiss
|   +-- metadata.json
+-- law/
+-- manual/
+-- notice/
+-- index_registry.json
```

해당 디렉토리와 파일이 존재하는지 확인한다. `index_registry.json`이 손상된 경우 삭제하면 자동으로 재생성된다.

---

## API Key 인증 오류

### 증상

```json
{"detail": "유효하지 않은 API 키입니다."}
```

HTTP 상태 코드 `401 Unauthorized`가 반환된다.

### 원인

요청의 `X-API-Key` 헤더 값이 서버의 `API_KEY` 환경변수와 일치하지 않는다.

### 해결 방법

**1. 환경변수 확인**

```bash
echo $API_KEY
```

**2. 요청 헤더 확인**

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{"prompt": "테스트", "max_tokens": 10}'
```

`X-API-Key` 헤더 값이 서버의 `API_KEY` 환경변수 값과 정확히 일치해야 한다.

**3. 개발 환경에서 인증 비활성화**

`API_KEY` 환경변수를 설정하지 않으면 인증을 건너뛴다. 개발 환경에서는 이 방법을 사용할 수 있다.

```bash
unset API_KEY
```

> **주의**: 프로덕션 환경에서는 반드시 `API_KEY`를 설정한다.

---

## 기타 문제

### Rate Limiting 관련

#### 증상

```json
{"error": "Rate limit exceeded"}
```

#### 원인

`slowapi` 기반 Rate Limiting이 활성화되어 있으며, 기본 제한을 초과했다.

| 엔드포인트 | 제한 |
|------------|------|
| `/v1/generate` | 30회/분 |
| `/v1/stream` | 30회/분 |
| `/search` | 60회/분 |

#### 해결 방법

- 요청 빈도를 줄인다.
- `slowapi`가 설치되어 있지 않으면 Rate Limiting이 자동으로 비활성화된다. 개발 환경에서 제한을 해제하려면 `slowapi`를 제거한다.

```bash
pip uninstall slowapi
```

### CORS 오류

#### 증상

브라우저에서 API 호출 시 CORS 관련 오류가 발생한다.

#### 해결 방법

`CORS_ORIGINS` 환경변수에 프론트엔드 출처를 추가한다. 여러 출처는 쉼표로 구분한다.

```bash
export CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### 임베딩 모델 다운로드 실패

#### 증상

```
OSError: Can't load tokenizer for 'intfloat/multilingual-e5-large'
```

#### 해결 방법

네트워크 연결을 확인하고 HuggingFace Hub에 접근 가능한지 확인한다. 오프라인 환경에서는 모델을 미리 다운로드하여 로컬 경로를 지정한다.

```python
# 온라인 환경에서 미리 다운로드
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-large")
model.save("models/multilingual-e5-large")
```

---

## 관련 문서

- [[Getting-Started]] - 사전 요구사항, 설치, 서버 기동, API 테스트
- [[Development-Guide]] - 개발 환경 설정, 브랜치 전략, 파이프라인 실행법
