# ADR-004: FAISS 기반 벡터 검색 엔진 선정

| 항목       | 내용                     |
| ---------- | ------------------------ |
| **상태**   | Accepted                 |
| **일자**   | 2026-03-25               |
| **작성자** | Backend Architect        |
| **관련**   | GovOn RAG 파이프라인     |

---

## 1. 컨텍스트

GovOn 시스템은 민원인의 질의에 대해 유사 민원 사례를 검색(RAG)하여 EXAONE-Deep-7.8B 모델의 응답 품질을 높인다. 벡터 검색 엔진을 선정할 때 다음 제약 조건이 존재한다.

- **폐쇄망 운용 필수**: 공공기관 내부망에서 동작해야 하므로 외부 SaaS 또는 별도 서버 프로세스에 대한 의존을 최소화해야 한다.
- **오프라인 배포**: 인터넷 접근 없이 인덱스 파일만으로 검색이 가능해야 한다.
- **데이터 타입 다중 관리**: 유사사례(CASE), 법령(LAW), 매뉴얼(MANUAL), 공시정보(NOTICE) 등 4개 도메인의 인덱스를 독립적으로 운영해야 한다.
- **GPU 자원 제한**: 추론 서버가 16GB VRAM을 vLLM에 할당(gpu_memory_utilization=0.8)하므로 벡터 검색은 CPU에서 동작해야 한다.
- **임베딩 차원**: multilingual-e5-large 모델의 출력 차원이 1024이다.

---

## 2. 검토 후보

| 후보        | 유형              | 외부 서버 필요 | 폐쇄망 적합성 | 대규모 확장성 | 비고                                      |
| ----------- | ----------------- | -------------- | -------------- | ------------- | ----------------------------------------- |
| **FAISS**   | 라이브러리        | 없음           | 최적           | 중~대         | Meta 개발, pip 설치만으로 사용 가능       |
| **Chroma**  | 임베디드 DB       | 선택적         | 양호           | 중            | SQLite 기반, 소규모에 적합                |
| **Qdrant**  | 클라이언트-서버   | 필요           | 보통           | 대            | gRPC/REST 서버 별도 운영 필요             |
| **Milvus**  | 분산 벡터 DB      | 필요           | 낮음           | 초대          | etcd, MinIO 등 의존성 다수                |
| **pgvector**| PostgreSQL 확장   | DB 서버 필요   | 양호           | 중            | PostgreSQL 운영 필수, 전용 인덱스 한계    |

### 2.1 평가 기준별 비교

**폐쇄망 배포 복잡도**

- FAISS: `pip install faiss-cpu` 단일 패키지. 별도 프로세스 없음.
- Chroma: pip 설치 가능하나 내부적으로 SQLite + hnswlib 의존성 존재.
- Qdrant/Milvus: 별도 서버 프로세스 필수. Docker 또는 바이너리 배포가 필요하며, 폐쇄망에서 운영 부담 증가.
- pgvector: PostgreSQL 서버에 확장 설치 필요. GovOn은 이미 SQLAlchemy + PostgreSQL을 사용하지만, 벡터 인덱스 성능이 전용 솔루션 대비 열위.

**검색 성능 (dim=1024, CPU)**

- FAISS IndexFlatIP: 정확도 100%(brute-force), 10만건 이하에서 수 ms 수준 응답.
- FAISS IndexIVFFlat: 10만건 이상 시 근사 검색으로 전환하여 성능 유지 가능.
- Chroma HNSW: 메모리 사용량이 FAISS 대비 높고, 1024차원에서의 벡터 인덱스 빌드 시간이 길다.
- pgvector ivfflat: PostgreSQL의 shared_buffers 설정에 의존하며, 전용 벡터 엔진 대비 2~5배 느림.

**운영 단순성**

- FAISS: 인덱스 파일(.faiss) + 메타데이터 JSON으로 관리. 파일 복사만으로 배포 완료.
- Qdrant/Milvus: 스냅샷/백업 관리, 클러스터 모니터링 등 추가 운영 비용.

---

## 3. 결정

**FAISS IndexFlatIP + multilingual-e5-large (dim=1024)** 를 벡터 검색 엔진으로 선정한다.

### 3.1 핵심 구현 구조

#### CivilComplaintRetriever (`src/inference/retriever.py`)

단일 FAISS 인덱스를 관리하는 기본 검색기이다.

- **임베딩 모델**: `intfloat/multilingual-e5-large` (SentenceTransformer)
- **인덱스 타입**: `faiss.IndexFlatIP` (Inner Product)
- **코사인 유사도 구현**: 임베딩 시 `normalize_embeddings=True`로 L2 정규화 후, Inner Product를 코사인 유사도로 사용
- **쿼리 prefix 규칙**: E5 모델 사양에 따라 검색 쿼리에 `"query: "` prefix, 문서에 `"passage: "` prefix 적용
- **인덱스 직렬화**: `faiss.write_index()` / `faiss.read_index()`로 파일 기반 저장/로드
- **메타데이터 관리**: 인덱스 파일 경로에 `.meta.json` 접미사를 붙여 JSON으로 별도 저장
- **검색 메서드**: `search(query, top_k=5)` -- 쿼리 임베딩 후 FAISS 검색, 메타데이터와 score를 병합하여 반환

```
# 인덱스 파일 구조 (CivilComplaintRetriever)
models/faiss_index/
  +-- complaints.index          # FAISS 바이너리 인덱스
  +-- complaints.index.meta.json  # 메타데이터 (id, category, complaint, answer)
```

#### MultiIndexManager (`src/inference/index_manager.py`)

4개 데이터 타입별 독립 인덱스를 관리하는 확장 매니저이다.

- **인덱스 타입 열거형**: `IndexType(str, Enum)` -- CASE, LAW, MANUAL, NOTICE
- **기본 임베딩 차원**: 1024 (생성자 `embedding_dim` 파라미터)
- **인덱스 생성**: `_create_index()` 메서드로 `faiss.IndexFlatIP(embedding_dim)` 생성
- **IVFFlat 자동 전환**: 문서 수가 10만건(`_IVF_THRESHOLD = 100_000`) 이상이면 `IndexFlatIP` -> `IndexIVFFlat` 자동 전환
  - `_IVF_NLIST = 256` (클러스터 수)
  - `_IVF_NPROBE = 16` (검색 시 탐색 클러스터 수)
  - `_maybe_upgrade_to_ivf()` 메서드에서 기존 벡터를 추출하여 IVFFlat 학습 후 재추가
- **메타데이터 스키마**: `DocumentMetadata` 데이터클래스 -- doc_id, doc_type, source, title, category, reliability_score, created_at, updated_at, valid_from, valid_until, chunk_index, chunk_total, extras
- **레지스트리**: `index_registry.json`으로 각 인덱스의 doc_count, index_class, embedding_dim, last_updated 추적
- **검색**: `search(index_type, query_vector, top_k=5)` -- IVFFlat 인덱스일 경우 nprobe 자동 설정

```
# 디렉토리 구조 (MultiIndexManager)
models/faiss_index/
  +-- case/
  |   +-- index.faiss
  |   +-- metadata.json
  +-- law/
  |   +-- index.faiss
  |   +-- metadata.json
  +-- manual/
  |   +-- index.faiss
  |   +-- metadata.json
  +-- notice/
  |   +-- index.faiss
  |   +-- metadata.json
  +-- index_registry.json
```

#### vLLMEngineManager (`src/inference/api_server.py`)

RAG 파이프라인에서 검색기를 통합 관리한다.

- `retriever` (CivilComplaintRetriever)와 `index_manager` (MultiIndexManager) 두 인스턴스를 보유
- `_augment_prompt()`: 검색 결과를 EXAONE 채팅 템플릿에 삽입하여 RAG 프롬프트 생성
- `_escape_special_tokens()`: `[|user|]`, `[|assistant|]` 등 EXAONE 특수 토큰을 이스케이프하여 Prompt Injection 방어
- `_extract_query()`: 정규식으로 `[|user|]...[|endofturn|]` 블록에서 "민원 내용:" 이후 텍스트를 추출
- 기본 설정: `INDEX_PATH = "models/faiss_index/complaints.index"`, RAG 검색 시 `top_k=3`

### 3.2 의존성

```
# requirements.txt
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
```

---

## 4. 결정 근거 요약

| 기준               | FAISS 선정 이유                                                              |
| ------------------ | ---------------------------------------------------------------------------- |
| 폐쇄망 적합성     | pip 패키지 하나로 설치 완료. 외부 서버/프로세스 의존 없음                    |
| 오프라인 운용      | `.faiss` 파일 + JSON 메타데이터만으로 완전한 오프라인 검색 가능              |
| CPU 전용 동작      | `faiss-cpu` 패키지로 GPU 없이 동작. vLLM에 GPU 자원을 온전히 할당           |
| 검색 정확도        | IndexFlatIP(brute-force)로 정확도 100% 보장. 정규화 벡터로 코사인 유사도 구현|
| 확장성             | 10만건 이상 시 IndexIVFFlat 자동 전환으로 검색 성능 유지                     |
| 운영 단순성        | 파일 복사만으로 인덱스 배포. 별도 데몬/서비스 불필요                         |
| 생태계 성숙도      | Meta 개발, 학계/산업계에서 가장 널리 사용되는 벡터 검색 라이브러리           |

---

## 5. 결과 (Consequences)

### 5.1 긍정적 결과

- 폐쇄망 환경에서 추가 인프라 없이 RAG 검색 파이프라인을 즉시 운용할 수 있다.
- 인덱스 파일 기반이므로 버전 관리(Git LFS) 및 배포가 단순하다.
- MultiIndexManager를 통해 데이터 도메인별 독립 인덱스를 운영하여 관심사를 분리할 수 있다.
- IndexFlatIP -> IndexIVFFlat 자동 전환으로 데이터 규모 증가에 유연하게 대응한다.

### 5.2 부정적 결과 및 한계

- **실시간 인덱스 업데이트 불가**: FAISS는 인덱스에 벡터를 추가할 수는 있으나, 개별 벡터의 삭제/수정을 지원하지 않는다. 문서 변경 시 인덱스를 전체 재빌드해야 한다.
- **인덱스 파일 별도 관리 필요**: 인덱스 바이너리 파일과 메타데이터 JSON을 코드와 별도로 관리해야 한다. 배포 파이프라인에 인덱스 빌드/배포 단계를 포함해야 한다.
- **메타데이터 검색 제한**: FAISS는 벡터 유사도 검색만 지원한다. 카테고리 필터링, 날짜 범위 검색 등 메타데이터 기반 필터링은 애플리케이션 레벨에서 후처리해야 한다.
- **분산 검색 미지원**: 단일 프로세스 내에서만 동작하므로, 추후 다중 서버 구성 시 인덱스 동기화 전략이 필요하다.
- **IVFFlat 전환 시 일시적 지연**: 10만건 도달 시 `_maybe_upgrade_to_ivf()`에서 전체 벡터를 추출하여 재학습하므로 일시적인 처리 지연이 발생한다.

### 5.3 향후 고려 사항

- 문서 수가 100만건을 초과할 경우 IndexIVFPQ 또는 HNSW 인덱스 타입 도입을 검토한다.
- 메타데이터 필터링 수요가 증가하면 FAISS 검색 후 애플리케이션 레벨 필터링 파이프라인을 고도화하거나, Qdrant 등 필터링 내장 엔진으로의 마이그레이션을 검토한다.
- 다중 서버 배포 시 인덱스 파일을 공유 스토리지(NFS 등)에 배치하거나, 인덱스 빌드 서버를 별도로 운영하는 방안을 검토한다.
