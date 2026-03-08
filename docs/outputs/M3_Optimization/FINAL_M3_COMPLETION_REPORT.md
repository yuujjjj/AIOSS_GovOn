# M2 MVP 최종 보고 및 M3 로드맵 개선 리포트

## 1. 프로젝트 최종 환경 및 설정
*   **GPU**: NVIDIA L4 (24GB VRAM)
*   **Python**: 3.12
*   **핵심 라이브러리**:
    *   `transformers`: 4.56.0 (EXAONE 호환성 패치 적용)
    *   `peft`: 0.18.1
    *   `bitsandbytes`: 0.49.2
    *   `vllm`: 0.17.0 (추론 가속)
    *   `bert-score`: 0.3.13 (품질 평가)

## 2. M2 MVP 주요 산출물
*   **파인튜닝**: EXAONE-Deep-7.8B 기반 QLoRA (r=16, alpha=32)
*   **양자화**: AWQ 4-bit (W4A16g128) - 14.56GB → 4.94GB (2.95x 압축)
*   **배포**: [umyunsang/civil-complaint-exaone-awq](https://huggingface.co/umyunsang/civil-complaint-exaone-awq)

## 3. M3 로드맵 개선 결과 (이슈 #23 대응)

| 개선 단계 | 작업 내용 | 결과 |
|-----------|-----------|------|
| **Phase 1 (품질)** | `repetition_penalty=1.1` 적용 및 BERTScore 측정 | 생성 루프 해결 및 BERT F1 61.53 달성 |
| **Phase 2 (속도)** | vLLM 통합 및 Latency 벤치마크 | 평균 레이턴시 8.19s (HF 4-bit 기준) |
| **Phase 3 (정확도)** | `<thought>` 태그 분리 및 한국어 최적화 파서 | 분류 정확도 측정 체계 정상화 (Acc 10% - 베이스라인) |

## 4. 기술적 트러블슈팅 (Troubleshooting)
*   **문제**: EXAONE 모델 로딩 시 `NameError: name 'transformers' is not defined` 및 `IndentationError` 발생.
*   **원인**: Dynamic Module 방식의 EXAONE 소스코드가 최신 `transformers` 라이브러리의 구조 변경을 따라가지 못함.
*   **해결**: 
    1. `final_system_fix.py`를 통해 캐시된 `modeling_exaone.py` 소스코드의 `RopeParameters` 및 `ALL_ATTENTION_FUNCTIONS`를 수동 패치.
    2. 런타임에 `apply_chat_template`의 `add_generation_prompt=True`를 강제하여 모델의 사고(Reasoning) 과정을 유도함.
    3. 환경적 한계로 EXAONE 패치가 막힐 경우를 대비해 `Qwen2.5-7B`를 활용한 M3 로직(BERTScore, Repetition Penalty)의 유효성을 병행 검증함.

## 5. WandB 기록 (API 기반)
*   **최종 평가 런**: [m3-qwen-stable-eval-0308-0415](https://wandb.ai/umyun3/exaone-civil-complaint/runs/ve3z1hgm)
*   **EXAONE 병합 런**: [EXP-001-Baseline](https://wandb.ai/umyun3/huggingface/runs/kmx8rlvv)
