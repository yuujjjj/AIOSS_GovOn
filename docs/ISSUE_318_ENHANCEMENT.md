# [TASK] EXAONE-Deep-7.8B 파인튜닝 고도화 및 행안부 AI 학습데이터 연동

## 1. 개요
기존 v2 파인튜닝 과정에서 발생한 기술적 한계를 극복하고, 행정안전부 '정부 공문서 AI 학습데이터'를 신규 확보하여 모델의 공문서 이해도 및 생성 품질을 고도화함. GPU 자원 효율성을 극대화하기 위해 A100(40GB) 기반의 QLoRA 학습 체계를 표준화함.

## 2. v2 학습 과정의 문제점 및 해결 방안 (W&B 분석 결과)

### 🚀 기술적 도전 과제 (Problems)
- **VRAM 부족 (OOM)**: 7.8B 모델 학습 시 L4(24GB) 환경에서 평가(Evaluation) 시 VRAM 임계치 도달.
- **라이브러리 호환성**: `transformers 5.3.0` 버전에서 `check_model_inputs` 삭제로 인한 EXAONE 모델 로딩 에러.
- **모델 구조적 결함**: PEFT 연동 시 `get_input_embeddings` 메서드 부재로 인한 `NotImplementedError` 발생.
- **평가 지표의 부재**: 단순 Loss 감소 외에 실제 공문서 문체(개조식)에 대한 정량적 검증 부족.

### ✅ 해결 전략 (Solutions)
- **런타임 업그레이드**: Google Colab A100(40GB) 환경으로 전환하여 `per_device_eval_batch_size=4` 확보.
- **Compatibility Patches**:
    - `transformers` 라이브러리 내 `generic.py` 수동 패치 적용 (`check_model_inputs` 더미 함수 추가).
    - `train_qlora.py` 내 `get_input_embeddings` 및 `get_output_embeddings` 동적 몽키 패치 적용.
- **메모리 최적화**: `paged_adamw_8bit` 옵티마이저 및 `bf16/tf32` 혼합 정밀도 학습 적용.
- **평가 표준화**: `SacreBLEU`, `ROUGE-L`, `BERTScore`를 도입하여 공문서 특유의 문체 변환 성능 검증.

## 3. 행안부 정부 공문서 AI 학습데이터 연동 계획

### 📊 API 테스트 결과 및 수집 전략
- **엔드포인트**: `apis.data.go.kr/1741000/publicDoc/getDocAll`
- **데이터 형식**: 전체 데이터셋이 ZIP 압축 파일 (`publicDocAll.zip`) 형태로 제공됨 확인.
- **데이터 구성**: 보도자료, 연설문, 발간사, 정책보고서 등 5종 공문서 (말뭉치 + Q&A + 요약/재구성 태스크).

### 🛠 데이터 전처리 파이프라인
1. **Collector**: API 호출을 통해 최신 `zipdownload.do` 링크 확보 및 자동 다운로드.
2. **Parser**: ZIP 내부의 `html(표)`, `그림(경로)` 데이터를 구조화된 JSONL로 변환.
3. **Task Construction**: LLM 파인튜닝을 위한 `[|system|]`, `[|user|]`, `[|assistant|]` 형식의 Chat Template 적용.

## 4. 핵심 변경 파일 (Core Files)
- `src/training/train_qlora.py`: EXAONE-Deep 호환성 패치 및 최적화 하이퍼파라미터 반영.
- `scripts/collect_data/collect_public_docs.py`: ZIP 기반 대량 수집 및 GovOn 표준 포맷 변환 로직 고도화.
- `notebooks/M3_issue70_retrain/retrain_v2_exaone.ipynb`: GPU 환경(Colab) 최적화 학습 프로세스 기록.

## 5. 향후 일정
- [ ] 행안부 데이터 ZIP 파싱 로직 구현 및 데이터셋 통합
- [ ] A100 환경에서의 1-Epoch 벤치마크 수행
- [ ] W&B 리포트를 통한 v2 대비 성능 비교 (Loss/Metrics)
