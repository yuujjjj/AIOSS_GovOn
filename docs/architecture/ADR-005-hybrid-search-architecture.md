# ADR-005: 하이브리드 검색 아키텍처

**문서 ID**: ADR-005
**작성일**: 2026-03-25
**상태**: Accepted
**선행 문서**: ADR-004 (확장된 RAG 아키텍처 설계)
**마일스톤**: M3

---

## 목차

1. [Status](#status)
2. [Context](#context)
3. [Decision](#decision)
4. [Alternatives Considered](#alternatives-considered)
5. [RRF 융합 설계](#rrf-융합-설계)
6. [한국어 토크나이저 폴백 체인](#한국어-토크나이저-폴백-체인)
7. [아키텍처 다이어그램](#아키텍처-다이어그램)
8. [구현 파일 매핑](#구현-파일-매핑)
9. [Graceful Degradation 전략](#graceful-degradation-전략)
10. [Consequences](#consequences)

---

## Status

Accepted

---

## Context

GovOn의 RAG 검색은 FAISS 기반 Dense 검색(코사인 유사도)만 사용한다. 의미적으로 유사한 민원을 찾는 데는 효과적이지만, 다음과 같은 키워드 정확 매칭이 필요한 경우에는 한계가 있다:

- **법령 조항 번호**: "개인정보보호법 제17조", "도로교통법 시행령 제2조"
- **부서/기관명**: "도시계획과", "건축허가팀"
- **고유명사/약어**: "4대 보험", "LH 공사"

ADR-004에서 Dense + Sparse 하이브리드 검색 도입을 결정했다. 이 문서는 구체적인 구현 아키텍처와 설계 결정을 기록한다.

---

## Decision

Dense(FAISS) + Sparse(BM25) 하이브리드 검색을 **Weighted Reciprocal Rank Fusion(RRF)** 으로 융합한다.

---

## Alternatives Considered

| 대안 | 설명 | 기각 사유 |
|------|------|----------|
| Dense-only | 현재 상태 유지 (FAISS만) | 키워드 정확 매칭 불가 |
| BM25-only | 키워드 검색만 사용 | 의미 유사도 검색 불가 |
| Learned Fusion | 학습 기반 점수 융합 (cross-encoder 등) | 학습 데이터 부족, 복잡도 높음 |
| **Weighted RRF** | **채택** | **학습 불필요, 해석 가능, 검증된 방법** |

Weighted RRF는 학습 데이터 없이 즉시 적용할 수 있고, 가중치 조절만으로 도메인 특성을 반영할 수 있다. 폐쇄망 환경에서 추가 학습 파이프라인 없이 운용 가능하다는 점이 결정적이었다.

---

## RRF 융합 설계

### 공식

```
score(d) = Σ (w_i / (k + rank_i(d)))
```

- `k = 60` (학술 표준값, smoothing parameter; Cormack et al., 2009)
- `rank`는 1-based
- 최종 점수는 최대 RRF 점수로 나누어 **0~1 정규화**

### 데이터 타입별 가중치

| 데이터 타입 | Dense 가중치 | Sparse 가중치 | 근거 |
|------------|-------------|-------------|------|
| CASE (유사사례) | 1.0 | 0.7 | 의미 유사도 우선 |
| LAW (법령) | 0.9 | 1.2 | 법령 조항 번호 정확 매칭 중요 |
| MANUAL (매뉴얼) | 0.8 | 0.8 | 균형 |
| NOTICE (공시정보) | 0.6 | 0.6 | 보조 정보 |

가중치는 `DEFAULT_RRF_WEIGHTS` 딕셔너리로 정의하며, 초기화 시 커스텀 가중치를 주입할 수 있다.

### 검색 모드

`SearchMode` Enum으로 3가지 모드를 지원한다:

| 모드 | 설명 | 용도 |
|------|------|------|
| `dense` | FAISS 의미 검색만 | 의미 유사도만 필요한 경우 |
| `sparse` | BM25 키워드 검색만 | 키워드 정확 매칭만 필요한 경우 |
| `hybrid` | Dense + Sparse 병렬 실행 후 RRF 융합 | **기본값**. 대부분의 검색에 사용 |

---

## 한국어 토크나이저 폴백 체인

BM25 검색의 품질은 형태소 분석에 크게 의존한다. 폐쇄망 등 다양한 설치 환경을 지원하기 위해 **초기화**와 **런타임** 두 단계로 폴백 체인을 구현한다:

**초기화 폴백 체인:**

```
Mecab (선호) → Okt (필수 최소) → RuntimeError (둘 다 실패 시)
```

**런타임 폴백:**

```
형태소 분석 실패 시 → 공백 분리 (최후 수단, 개별 호출 레벨)
```

| 토크나이저 | 장점 | 단점 |
|-----------|------|------|
| Mecab | 빠르고 정확한 형태소 분석 | C 라이브러리(mecab-ko) 설치 필요 |
| Okt (konlpy) | Java 기반, 설치 용이 | Mecab보다 느림 |
| 공백 분리 | 의존성 없음, 런타임 에러 시 방어 로직 | 초기화 폴백이 아닌 런타임 전용. 형태소 분석 없이 단순 분리 |

- `tokenizer_type="auto"` 설정 시 Mecab을 시도하고 실패하면 Okt로 자동 폴백한다.
- 한국어 불용어(`_STOPWORDS`) 필터링을 적용하여 검색 품질을 유지한다.
- 환경변수 `BM25_INDEX_HMAC_KEY` 설정 시 인덱스 파일의 HMAC 서명 검증을 수행한다 (pickle deserialization 보안).

> **주의**: Mecab과 Okt(konlpy) 모두 설치되지 않은 환경에서는 BM25Indexer 초기화가 실패합니다. 폐쇄망 환경에서는 최소 konlpy(Okt)를 사전 설치해야 합니다.

---

## 아키텍처 다이어그램

```
Query
  ├─ embed_query() ──→ FAISS Dense Search ──→ Dense Results (ranked) ──┐
  │                                                                     │
  └─ tokenize()   ──→ BM25 Sparse Search ──→ Sparse Results (ranked) ─┤
                                                                       │
                      [asyncio.gather: 병렬 실행]                       │
                                                                       ▼
                                         Weighted RRF Fusion ◄─────────┘
                                               │
                                         Normalized Results (score 0~1)
                                               │
                                         Top-K Response
```

- Dense/Sparse 검색은 `asyncio.gather()`로 **병렬 실행**한다.
- `doc_id`를 기준으로 결과를 병합하여 동일 문서의 점수를 합산한다.
- 임베딩은 `multilingual-e5-large` 모델을 사용하며, 쿼리에 `query:` prefix를 부착한다.

---

## 구현 파일 매핑

| 구현 요소 | 파일 | 비고 |
|----------|------|------|
| HybridSearchEngine | `src/inference/hybrid_search.py` | 핵심 모듈 (Issue #154) |
| BM25Indexer | `src/inference/bm25_indexer.py` | Issue #153 (KoreanTokenizer 포함) |
| MultiIndexManager | `src/inference/index_manager.py` | Issue #150 |
| 스키마 확장 | `src/inference/schemas.py` | SearchMode 추가 |
| API 통합 | `src/inference/api_server.py` | `/v1/search` + `/search` (하위 호환) |
| 파이프라인 통합 | `src/data_collection_preprocessing/pipeline.py` | `--mode bm25` |

---

## Graceful Degradation 전략

시스템 안정성을 위해 3단계 폴백을 적용한다:

| 장애 상황 | 동작 | 로그 레벨 |
|----------|------|----------|
| BM25 인덱스 미로드 | `hybrid` → `dense` 자동 폴백 | WARNING |
| HybridSearchEngine 미초기화 | 레거시 `CivilComplaintRetriever` 폴백 | WARNING |
| 검색 중 예외 발생 | 빈 결과 반환 (서비스 중단 방지) | ERROR |

핵심 원칙: **검색 실패가 서비스 전체 장애로 전파되지 않는다.**

---

## Consequences

### 이점

- 키워드 정확 매칭과 의미 유사도 검색을 **동시에 지원**한다
- 학습 데이터 없이 즉시 적용 가능하여 폐쇄망 운용에 적합하다
- 데이터 타입별 가중치로 도메인 특성(법령은 키워드 우선 등)을 반영한다
- `SearchMode` 선택으로 상황에 맞는 유연한 검색이 가능하다
- Graceful Degradation으로 BM25 미구축 환경에서도 서비스가 정상 동작한다

### 비용

- BM25 인덱스 추가 저장 비용 (pickle 직렬화)
- 데이터 파이프라인에 `--mode bm25` 빌드 단계 추가 필요
- 디버깅 복잡도 증가: 검색 결과가 Dense/Sparse 중 어디서 기여했는지 추적 필요
- 한국어 형태소 분석기 설치 의존성 (Mecab 또는 konlpy)
