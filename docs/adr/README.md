# 기술결정기록 (Architecture Decision Records)

GovOn 프로젝트의 주요 기술 의사결정을 체계적으로 기록하고 추적하기 위한 디렉토리입니다.

## ADR 인덱스

| ADR | 제목 | 상태 | 작성일 |
|-----|------|------|--------|
| [ADR-001](ADR-001-exaone-model-selection.md) | EXAONE-Deep-7.8B 모델 선정 | Accepted | 2026-03-25 |
| [ADR-002](ADR-002-awq-quantization.md) | AWQ W4A16g128 양자화 방식 선정 | Accepted | 2026-03-25 |
| [ADR-003](ADR-003-vllm-serving.md) | vLLM 추론 서빙 엔진 선정 | Accepted | 2026-03-25 |
| [ADR-004](ADR-004-faiss-vector-search.md) | FAISS 기반 벡터 검색 엔진 선정 | Accepted | 2026-03-25 |
| [ADR-004 (확장)](../architecture/ADR-004-enhanced-rag-architecture.md) | 확장된 RAG 아키텍처 설계 | Proposed | 2026-03-22 |

## ADR 작성 템플릿

새로운 기술결정기록을 작성할 때 아래 템플릿을 사용합니다. 파일명은 `ADR-NNN-간략한-설명.md` 형식을 따릅니다.

```markdown
# ADR-NNN: [결정 제목]

## Status

Proposed | Accepted | Deprecated | Superseded by ADR-XXX

## Context

이 결정을 내리게 된 배경과 문제 상황을 기술합니다.
- 해결해야 할 기술적 과제는 무엇인가?
- 어떤 제약 조건이 존재하는가? (성능, 보안, 인프라, 라이선스 등)
- 이해관계자의 요구사항은 무엇인가?

## 검토 후보

| 후보 | 장점 | 단점 |
|------|------|------|
| 후보 A | ... | ... |
| 후보 B | ... | ... |
| 후보 C | ... | ... |

## Decision

최종 결정 사항과 그 근거를 기술합니다.

## Consequences

### 긍정적 영향
- 이 결정으로 인해 개선되는 점

### 부정적 영향
- 이 결정으로 인해 감수해야 하는 점

### 향후 고려사항
- 상황 변화 시 재검토가 필요한 조건
```

## ADR 상태 정의

- **Proposed**: 검토 중인 결정. 팀 논의가 진행 중이거나 구현 전 단계.
- **Accepted**: 합의되어 적용 중인 결정.
- **Deprecated**: 더 이상 유효하지 않은 결정. 사유를 본문에 기록.
- **Superseded by ADR-XXX**: 후속 결정으로 대체된 경우. 대체 ADR 번호를 명시.

## 작성 원칙

1. **결정의 이유(WHY)를 중심으로** -- 무엇을 결정했는지보다 왜 그렇게 결정했는지가 더 중요합니다.
2. **트레이드오프를 명시** -- 선택하지 않은 대안과 그 이유도 함께 기록합니다.
3. **되돌릴 수 있는 결정을 선호** -- 변경 비용이 낮은 선택지를 우선 고려합니다.
4. **간결하게 작성** -- 한 문서에 하나의 결정만 다룹니다.
