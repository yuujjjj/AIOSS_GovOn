# Development Guide

GovOn 프로젝트의 개발 워크플로우, 브랜치 전략, 데이터 파이프라인, 학습, 양자화, 평가, 테스트 실행법을 안내한다.

---

## 목차

- [브랜치 전략](#브랜치-전략)
- [커밋 컨벤션](#커밋-컨벤션)
- [PR 규칙](#pr-규칙)
- [코드 스타일](#코드-스타일)
- [데이터 수집 및 전처리 파이프라인](#데이터-수집-및-전처리-파이프라인)
- [QLoRA 파인튜닝](#qlora-파인튜닝)
- [AWQ 양자화](#awq-양자화)
- [모델 평가](#모델-평가)
- [테스트 실행](#테스트-실행)
- [관련 문서](#관련-문서)

---

## 브랜치 전략

GovOn은 `main`과 `develop` 두 개의 장기 브랜치를 운영한다.

| 브랜치 | 용도 | 직접 push |
|--------|------|-----------|
| `main` | 프로덕션 릴리스 | **금지** |
| `develop` | 개발 통합 브랜치 | PR을 통해서만 병합 |

### 기능 브랜치 네이밍

이슈 번호를 포함하여 브랜치를 생성한다.

```
feat/이슈번호-설명      # 신규 기능
fix/이슈번호-설명       # 버그 수정
docs/이슈번호-설명      # 문서 작업
chore/이슈번호-설명     # 설정, 빌드, CI 등
```

예시:

```bash
git checkout develop
git pull origin develop
git checkout -b feat/42-add-law-index
```

---

## 커밋 컨벤션

커밋 메시지는 **한글**로 작성하며, Conventional Commits 형식을 따른다.

```
<type>: <설명>
```

### 사용 가능한 type

| type | 용도 |
|------|------|
| `feat` | 신규 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 추가 또는 수정 |
| `style` | 코드 포매팅, 세미콜론 등 (로직 변경 없음) |
| `refactor` | 리팩토링 (기능 변경 없음) |
| `test` | 테스트 추가 또는 수정 |
| `chore` | 빌드, CI, 의존성 관리 |
| `perf` | 성능 개선 |

예시:

```bash
git commit -m "feat: 법령 인덱스 검색 엔드포인트 추가"
git commit -m "fix: FAISS 인덱스 메타데이터 경로 오류 수정"
```

---

## PR 규칙

- PR 대상 브랜치는 항상 `develop`이다.
- `main`에 직접 push하지 않는다.
- PR 제목도 Conventional Commits 형식으로 작성한다.

---

## 코드 스타일

| 항목 | 규칙 |
|------|------|
| 포매터 | `black` (line-length=100) |
| import 정렬 | `isort` (black profile) |
| 린터 | `flake8` |
| 타입 힌트 | 필수 (Python 3.10+ 문법) |
| 로깅 | `loguru.logger` 사용 (`print()` 금지) |
| API 에러 | 내부 정보 노출 금지 (스택 트레이스를 클라이언트에 반환하지 않음) |

### 포매팅 및 린트 실행

```bash
black --line-length 100 src/
isort --profile black src/
flake8 src/
```

---

## 데이터 수집 및 전처리 파이프라인

`src/data_collection_preprocessing/` 모듈은 AI Hub에서 민원 데이터를 수집하고 다음 단계를 처리한다.

1. AI Hub 원본 데이터 수집
2. PII(개인정보) 마스킹
3. EXAONE 채팅 템플릿 형식으로 변환
4. AWQ 캘리브레이션 데이터 생성

### 전체 파이프라인 실행

```bash
python -m src.data_collection_preprocessing.pipeline --mode full
```

파이프라인이 완료되면 `data/processed/` 디렉토리에 JSONL 형식의 학습 데이터가 생성된다.

### EXAONE 채팅 템플릿 형식

변환된 데이터는 다음 형식을 따른다.

```
[|system|]시스템 프롬프트[|endofturn|]
[|user|]민원 내용[|endofturn|]
[|assistant|]답변 내용[|endofturn|]
```

---

## QLoRA 파인튜닝

`src/training/train_qlora.py`를 사용하여 EXAONE-Deep-7.8B 모델을 QLoRA 방식으로 파인튜닝한다.

### 주요 학습 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--model_id` | `LGAI-EXAONE/EXAONE-Deep-7.8B` | 베이스 모델 ID |
| `--train_path` | (필수) | 학습 데이터 JSONL 경로 |
| `--val_path` | (필수) | 검증 데이터 JSONL 경로 |
| `--output_dir` | `./models/checkpoints/exaone-civil-qlora` | 체크포인트 저장 디렉토리 |
| `--epochs` | `3` | 학습 에폭 수 |
| `--batch_size` | `4` | 배치 크기 |
| `--grad_accum` | `4` | 그래디언트 누적 스텝 |
| `--lr` | `2e-4` | 학습률 |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--max_seq_length` | `2048` | 최대 시퀀스 길이 |
| `--wandb_project` | `exaone-civil-complaint` | WandB 프로젝트 이름 |

### 실행 예시

```bash
python -m src.training.train_qlora \
  --train_path data/processed/v2_train.jsonl \
  --val_path data/processed/v2_val.jsonl \
  --output_dir models/checkpoints/exaone-civil-qlora \
  --epochs 3 \
  --batch_size 4 \
  --lr 2e-4
```

학습에는 4-bit 양자화(BitsAndBytes)와 SFTTrainer(trl)를 사용한다. WandB를 통해 학습 메트릭을 실시간으로 모니터링할 수 있다.

### 실험 관리

여러 하이퍼파라미터 조합으로 실험을 실행하려면 `src/training/run_experiments.py`를 사용한다.

```bash
python -m src.training.run_experiments
```

---

## AWQ 양자화

QLoRA 파인튜닝이 완료되면 LoRA 어댑터를 베이스 모델에 병합한 뒤 AWQ 양자화를 수행한다.

### 단계 1: LoRA 병합

`src/quantization/merge_lora.py`를 실행하여 LoRA 어댑터를 베이스 모델에 병합한다. BF16 정밀도의 전체 모델이 출력된다.

```bash
python -m src.quantization.merge_lora
```

병합 전 EXAONE 호환 monkey-patch가 자동 적용된다 (`RopeParameters`, `check_model_inputs`, `ALL_ATTENTION_FUNCTIONS` 등).

### 단계 2: AWQ 양자화

`src/quantization/quantize_awq.py`를 실행하여 W4A16g128 양자화를 수행한다.

```bash
python -m src.quantization.quantize_awq
```

양자화에는 AutoAWQ(>= 0.2.0)를 사용하며, 도메인 특화 캘리브레이션 데이터(학습 데이터 512 샘플)를 활용한다.

### 양자화 설정

| 항목 | 값 |
|------|-----|
| 양자화 방식 | AWQ (W4A16g128) |
| Weight bit | 4-bit |
| Activation bit | 16-bit |
| Group size | 128 |
| 캘리브레이션 샘플 | 512개 |

---

## 모델 평가

`src/evaluation/` 디렉토리에는 다양한 평가 스크립트가 포함되어 있다.

| 스크립트 | 용도 |
|----------|------|
| `evaluate_model.py` | 기본 모델 평가 |
| `evaluate_m3_vllm.py` | vLLM 기반 M3 평가 |
| `evaluate_m3_autoawq.py` | AWQ 양자화 모델 평가 |
| `evaluate_m3_final.py` | 최종 M3 통합 평가 |
| `evaluate_m3_stable.py` | 안정화된 M3 평가 |
| `evaluate_exaone_m3.py` | EXAONE M3 특화 평가 |

### 실행 예시

```bash
# vLLM 기반 평가
python -m src.evaluation.evaluate_m3_vllm

# AWQ 양자화 모델 평가
python -m src.evaluation.evaluate_m3_autoawq

# 최종 통합 평가
python -m src.evaluation.evaluate_m3_final
```

---

## 테스트 실행

GovOn은 `pytest`를 사용하여 테스트를 실행한다.

### 전체 테스트 실행 (커버리지 포함)

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 모듈별 테스트

```bash
# 추론 모듈 테스트
pytest tests/test_inference/ -v

# 데이터 파이프라인 테스트
pytest tests/test_data_collection_preprocessing/ -v
```

### 단일 파일 테스트

```bash
pytest tests/test_inference/test_schemas.py -v
```

### 테스트 관련 의존성

테스트 실행에는 개발 도구 설치가 필요하다.

```bash
pip install -e ".[dev]"
```

이 명령으로 `pytest`, `pytest-cov`, `pytest-asyncio`, `black`, `isort`, `flake8`, `mypy`가 설치된다.

---

## 관련 문서

- [[Getting-Started]] - 사전 요구사항, 설치, 서버 기동, API 테스트
- [[Troubleshooting]] - GPU OOM, vLLM 오류, FAISS 인덱스 문제 해결
