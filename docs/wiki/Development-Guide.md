# Development Guide

AIOSS_GovOn 저장소에서 문서, 데이터 파이프라인, 학습/평가 스크립트를 안전하게 다루기 위한 기본 개발 흐름을 정리한 문서입니다.

빠른 이동: [Wiki Home](README.md) | [Getting Started](Getting-Started.md) | [Troubleshooting](Troubleshooting.md)

## 1. 브랜치 및 커밋 규칙

기본 작업 흐름은 `main` 브랜치 기준입니다.

```bash
git checkout main
git pull upstream main
git checkout -b docs/이슈번호-설명
```

브랜치 네이밍 규칙:

- `feat/*`: 기능 개발
- `fix/*`: 버그 수정
- `docs/*`: 문서 작업
- `chore/*`: 설정 및 인프라 작업

커밋 메시지는 Conventional Commits 형식을 따르며, 저장소 가이드 기준으로 한글 설명을 사용합니다.

```text
docs: 위키 시작 문서 추가
fix: EXAONE 평가 스크립트 경로 수정
```

세부 규칙은 [CONTRIBUTING](../../CONTRIBUTING.md)를 참고합니다.

## 2. 일반 개발 루프

1. 관련 이슈를 확인하거나 생성합니다.
2. 브랜치를 만든 뒤 필요한 파일만 수정합니다.
3. 저장소 루트에서 테스트나 스모크 체크를 실행합니다.
4. 문서 변경이 있으면 관련 가이드를 함께 갱신합니다.
5. `main` 대상으로 PR을 열고 이슈를 연결합니다.

## 3. 자주 쓰는 작업

### 데이터 수집 및 전처리

```bash
cp src/data_collection_preprocessing/.env.example .env
python -m src.data_collection_preprocessing.pipeline --mode full --mock
```

실데이터를 사용할 때는 `.env`의 API 키와 `AIHUB_SHELL_PATH=./aihubshell` 설정을 먼저 점검합니다.

### 테스트

```bash
pytest
```

전처리 모듈만 빠르게 확인할 때:

```bash
pytest tests/test_data_collection_preprocessing/ -v
```

### 학습

현재 저장소의 대표 QLoRA 학습 진입점은 `src/training/train_qlora.py`입니다.

```bash
python src/training/train_qlora.py \
  --train_path data/processed/civil_complaint_train.jsonl \
  --val_path data/processed/civil_complaint_val.jsonl \
  --output_dir ./models/checkpoints/exaone-civil-qlora \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 2e-4
```

학습 환경 이슈는 [Troubleshooting](Troubleshooting.md)과 [학습 환경 메모](../../src/training/ENVIRONMENT_NOTES.md)를 함께 봅니다.

### 추론 서버

FastAPI + vLLM 서버는 optional dependency가 필요합니다.

```bash
pip install -e ".[inference]"
uvicorn src.inference.api_server:app --reload
```

## 4. 작업 전 확인할 점

- `src/evaluation/`과 `src/quantization/`의 일부 스크립트는 `/content/...` 형태의 Colab 경로를 그대로 사용합니다.
- 로컬 실행 전에는 모델 경로, 데이터 경로, 출력 디렉터리를 현재 환경에 맞게 바꿔야 합니다.
- EXAONE 계열 작업은 `transformers` 버전과 Hugging Face `revision` 불일치 시 조용히 잘못된 결과를 낼 수 있습니다.

이 세 가지는 실제 개발 중 가장 자주 시간을 잃는 지점이므로, 문제 조짐이 보이면 바로 [Troubleshooting](Troubleshooting.md)으로 이동하는 편이 빠릅니다.

## 5. 관련 문서

- 시작 순서가 필요하면 [Getting Started](Getting-Started.md)
- 문제 해결이 필요하면 [Troubleshooting](Troubleshooting.md)
- 저장소 정책 전체는 [CONTRIBUTING](../../CONTRIBUTING.md)
