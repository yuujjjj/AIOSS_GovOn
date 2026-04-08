# Troubleshooting

AIOSS_GovOn 저장소에서 반복적으로 발생하는 환경, 경로, 모델 호환성 문제를 빠르게 진단하기 위한 문서입니다.

빠른 이동: [Wiki Home](README.md) | [Getting Started](Getting-Started.md) | [Development Guide](Development-Guide.md)

## 1. `pytest`가 없거나 import 오류가 발생한다

증상:

- `pytest: command not found`
- `ModuleNotFoundError`
- 프로젝트 모듈을 찾지 못해 테스트가 실패

해결:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
python -m pytest tests/test_data_collection_preprocessing/ -v
```

항상 저장소 루트에서 실행하고, 가상환경이 활성화되어 있는지 먼저 확인합니다.

## 2. 데이터 수집 파이프라인에서 API 키 또는 다운로드가 실패한다

확인 순서:

1. `.env`를 `src/data_collection_preprocessing/.env.example` 기준으로 만들었는지 확인
2. `AIHUB_API_KEY`, `SEOUL_API_KEY`, `DATA_GO_KR_API_KEY` 값이 실제 키인지 확인
3. `AIHUB_SHELL_PATH=./aihubshell` 경로가 맞는지 확인
4. `aihubshell` 실행 권한이 있는지 확인

```bash
chmod +x aihubshell
```

빠른 점검용으로는 실데이터 대신 mock 모드부터 실행하는 편이 안전합니다.

```bash
python -m src.data_collection_preprocessing.pipeline --mode full --mock
```

## 3. EXAONE 모델이 Colab 또는 최신 환경에서 이상 동작한다

대표 증상:

- `rope_parameters` 관련 `TypeError`
- `get_input_embeddings` 관련 `NotImplementedError`
- `AttentionInterface` 관련 `AttributeError`
- 한국어처럼 보이지만 의미 없는 쓰레기 출력

핵심 원인:

- 학습 시점과 다른 `transformers` 버전 또는 다른 Hugging Face `revision`으로 EXAONE 코드를 로드한 경우

권장 대응:

```bash
pip uninstall -y transformers accelerate
pip install "transformers>=4.44,<4.50" "accelerate>=1.3.0,<2.0" peft
```

모델 로딩 시에는 `trust_remote_code=True`와 `revision="17b70148e344"`를 함께 고정합니다.

세부 배경과 예제 코드는 [docs/colab-version-compatibility.md](../colab-version-compatibility.md)에 정리되어 있습니다.

## 4. 학습 또는 양자화 중 GPU 메모리가 부족하다

대응 방법:

- `train_qlora.py` 실행 시 `--batch_size`를 줄이고 `--grad_accum`을 늘립니다.
- 메모리 파편화가 의심되면 `PYTORCH_ALLOC_CONF=expandable_segments:True`를 사용합니다.
- 현재 환경 메모는 A100 `40GB` 기준으로 기록된 내용이 있어, L4 `24GB`에서는 더 보수적으로 잡아야 합니다.
- AWQ 양자화는 시간이 오래 걸리므로 디스크 여유 공간과 GPU 메모리를 함께 확인합니다.

환경 메모는 [src/training/ENVIRONMENT_NOTES.md](../../src/training/ENVIRONMENT_NOTES.md)를 참고합니다.

## 5. 평가/양자화 스크립트가 로컬에서 바로 실행되지 않는다

원인:

- 일부 스크립트가 `/content/ondevice-ai-civil-complaint/...` 같은 Colab 절대 경로를 하드코딩하고 있습니다.

해결:

```bash
rg -n "/content/ondevice-ai-civil-complaint" src
```

검색 결과에 나온 모델, 데이터, 출력 경로를 현재 로컬 경로로 바꾼 뒤 실행합니다. 특히 `src/evaluation/`과 `src/quantization/` 작업 전에는 이 확인이 거의 필수입니다.

## 6. 생성이 중간에 끊기지 않거나 응답이 이상하게 이어진다

EXAONE의 EOS 토큰은 `[|endofturn|]` 계열을 함께 고려해야 합니다. 생성 품질이나 종료 시점이 이상하면 `eos_token_id` 설정을 다시 확인합니다.

이 이슈도 버전 불일치와 함께 나타나는 경우가 많으므로, 먼저 [Getting Started](Getting-Started.md)의 기본 설치를 다시 맞추고, 이후 [Development Guide](Development-Guide.md)의 실행 경로 점검 순서로 돌아가는 편이 빠릅니다.
