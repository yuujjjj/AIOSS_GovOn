# Getting Started

처음 AIOSS_GovOn 저장소를 받아 로컬 환경에서 문서 확인, 테스트 실행, 기본 개발 준비를 끝내기 위한 시작 문서입니다.

빠른 이동: [Wiki Home](README.md) | [Development Guide](Development-Guide.md) | [Troubleshooting](Troubleshooting.md)

## 1. 준비 사항

- Python `3.10+`
- Git `2.30+`
- GPU 작업이 필요하면 CUDA `12.x`와 PyTorch `2.x` 권장
- 데이터 수집이나 모델 다운로드를 진행할 경우 API 키 또는 Hugging Face 토큰 준비

## 2. 저장소 클론 및 가상환경 생성

```bash
git clone https://github.com/yuujjjj/AIOSS_GovOn.git
cd AIOSS_GovOn

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

## 3. 의존성 설치

기본 환경만 빠르게 맞출 때:

```bash
pip install -r requirements.txt
```

개발 도구까지 함께 설치할 때:

```bash
pip install -e ".[dev]"
```

FastAPI + vLLM 추론 서버까지 실행할 때:

```bash
pip install -e ".[inference]"
```

## 4. 첫 확인

저장소 루트에서 아래 두 가지 중 하나를 먼저 실행하면 환경 점검이 쉽습니다.

```bash
pytest tests/test_data_collection_preprocessing/ -v
```

```bash
python -m src.data_collection_preprocessing.pipeline --mode full --mock
```

실데이터 수집까지 하려면 먼저 환경 변수를 준비합니다.

```bash
cp src/data_collection_preprocessing/.env.example .env
```

`.env`에는 `AIHUB_API_KEY`, `SEOUL_API_KEY`, `DATA_GO_KR_API_KEY` 등을 채우고, `aihubshell` 실행 권한도 확인합니다.

## 5. 저장소 핵심 디렉터리

- `src/data_collection_preprocessing/`: 수집, PII 마스킹, 전처리 파이프라인
- `src/training/`: QLoRA 학습 스크립트와 설정
- `src/quantization/`: AWQ 양자화 스크립트
- `src/evaluation/`: 평가 스크립트
- `src/inference/`: FastAPI + vLLM 추론 서버
- `tests/`: 전처리 모듈 중심 테스트
- `docs/`: 연구, 산출물, 위키 문서

## 6. 다음 단계

- 개발 규칙과 일반 작업 흐름은 [Development Guide](Development-Guide.md)에서 확인합니다.
- 설치 오류, EXAONE 버전 문제, GPU 메모리 이슈는 [Troubleshooting](Troubleshooting.md)을 확인합니다.
- 프로젝트 전반 개요는 [README](../../README.md)를 먼저 읽어도 됩니다.
