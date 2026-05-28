# Inner Source 도입 로드맵

## 목적

이 문서는 조직 내부에 `Inner Source` 협업 방식을 도입하기 위한 단계형 실행 계획입니다.  
핵심 목표는 “코드를 공유한다”가 아니라, 여러 팀이 동일한 저장소를 오픈소스처럼 문서화하고, 이슈 기반으로 논의하고, PR과 리뷰로 변경을 축적하는 운영 체계를 만드는 것입니다.

## 적용 범위

- 공용 라이브러리 및 자동화 스크립트
- 실험 코드와 모델 운영 파이프라인
- 문서, 템플릿, 위키, 운영 가이드
- GitHub Actions 및 저장소 자동화

## 도입 원칙

- 구두 요청보다 이슈 등록을 우선합니다.
- 변경은 작은 PR 단위로 분리합니다.
- 중요한 결정은 회의록보다 저장소 기록에 남깁니다.
- 관리 부담을 줄이기 위해 처음에는 “전사 확산”이 아니라 “시간 제한 파일럿”으로 시작합니다.
- 팀 기여를 확장할 수 있도록 Maintainer 외에 `Trusted Committer` 경로를 둡니다.

## 로드맵 개요

| 단계 | 기간 | 목표 | 핵심 산출물 |
|------|------|------|-------------|
| Phase 0 | 1-2주 | 파일럿 준비 | 기본 문서, 코드 소유권, 운영 원칙 |
| Phase 1 | 3-6주 | 저장소 단위 시범 운영 | 이슈 기반 협업, PR 리뷰 정착 |
| Phase 2 | 7-10주 | 의사결정 표준화 | RFC 절차, 리뷰 SLA, 지표 관리 |
| Phase 3 | 11-16주 | 조직 확산 | Trusted Committer 운영, 재사용 저장소 확대 |

## Phase 0. 파일럿 준비

### 목표

- 저장소를 누구나 이해할 수 있는 self-service 상태로 만듭니다.
- 책임자, 리뷰 흐름, 라이선스 정책을 먼저 명확히 합니다.

### 실행 항목

- `README`, `CONTRIBUTING`, `CODE_OF_CONDUCT`, `LICENSE`, `SECURITY` 정비
- `GOVERNANCE.md`와 `.github/CODEOWNERS` 추가
- 라이선스 선택 기준 문서화
- 이슈 템플릿, PR 템플릿, 기본 라벨 구조 점검

### 완료 기준

- 신규 참여자가 문서만 읽고 로컬 실행과 기여 경로를 이해할 수 있음
- Maintainer와 리뷰 책임 경로가 공개되어 있음
- 저장소 기본 라이선스와 외부 모델/데이터 라이선스 구분 원칙이 정리되어 있음

## Phase 1. 저장소 단위 시범 운영

### 목표

- 한 저장소에서 Inner Source 운영 방식을 실제로 돌려 봅니다.
- 구두 협업을 이슈/PR 중심 흐름으로 전환합니다.

### 실행 항목

- 기능 요청, 버그, 문서 작업을 모두 GitHub Issue로 등록
- 작업 단위를 작게 나눠 PR 생성
- PR 본문에 변경 배경, 영향 범위, 검증 결과를 필수화
- `CODEOWNERS` 기반으로 리뷰 요청
- 문서와 코드 변경을 함께 반영

### 권장 운영 규칙

- `main` 직접 push 금지
- 모든 변경은 PR 병합으로만 반영
- 회의 후 결정 사항은 이슈 또는 PR 코멘트로 문서화

### KPI

- 주요 작업의 이슈 추적률
- PR당 평균 리뷰 대기 시간
- 문서 반영 누락 비율
- 외부 팀 또는 비주력 팀 기여 수

## Phase 2. 의사결정 표준화

### 목표

- 협업이 늘어나도 혼란이 커지지 않도록 의사결정 절차를 문서화합니다.
- 큰 변경은 RFC 기반으로 검토합니다.

### 실행 항목

- RFC 템플릿 도입
- 아키텍처 변경, 브랜치 전략 변경, 배포 흐름 변경 시 RFC 필수화
- 이슈 라벨 체계 정비
- 리뷰 시작 목표 시간과 병합 기준 수립

### RFC 대상 예시

- 모델 교체
- 데이터 처리 파이프라인 구조 변경
- GitHub Actions 구조 변경
- 공용 모듈 분리 또는 통합

### KPI

- RFC가 필요한 변경의 문서화율
- 설계 변경 후 재작업 감소율
- 장기 미결 PR 수 감소

## Phase 3. 조직 확산

### 목표

- 파일럿 저장소에서 검증한 방식을 다른 저장소와 팀으로 확장합니다.
- 반복 기여자를 `Trusted Committer`로 육성해 유지보수 부담을 분산합니다.

### 실행 항목

- 반복 기여자 선발 기준 정의
- Trusted Committer 후보 공개 및 권한 위임 절차 정립
- 공통 템플릿 저장소 또는 운영 체크리스트 배포
- 재사용 가치가 높은 저장소부터 우선 확산

### Trusted Committer 기준 예시

- 3회 이상 의미 있는 PR 기여
- 리뷰 피드백 반영 능력 입증
- 문서/이슈/리뷰 채널에서 지속적 참여
- Maintainer 1인 이상 추천

### KPI

- 팀 외 기여자 수
- Trusted Committer 수
- 공용 저장소 재사용 팀 수
- 릴리스 또는 산출물 재사용 횟수

## 운영 리스크와 대응

### 리스크 1. 관리자는 효과를 확신하지 못함

- 대응: 전사 확대 대신 4개월 파일럿으로 시작
- 대응: 리뷰 시간, 기여 수, 재사용 수 같은 지표를 미리 정의

### 리스크 2. 기록보다 구두 협업이 계속 우세함

- 대응: 병합 전 이슈 링크와 PR 설명을 필수화
- 대응: 회의 결론을 반드시 이슈/PR에 남기도록 규칙화

### 리스크 3. Maintainer 병목

- 대응: `CODEOWNERS`로 영역별 책임을 분산
- 대응: 반복 기여자를 Trusted Committer로 승격

### 리스크 4. 문서와 실제 운영이 분리됨

- 대응: PR 체크리스트에 문서 업데이트 포함
- 대응: 분기마다 거버넌스와 기여 문서 점검

## AIOSS_GovOn 즉시 실행안

현재 저장소에서 바로 적용할 우선순위는 아래와 같습니다.

1. 기본 OSS/Inner Source 문서 유지
2. 이슈 기반 작업 비율 확대
3. `CODEOWNERS` 리뷰 흐름 정착
4. 큰 구조 변경에 RFC 문서 도입
5. 반복 기여자에 대한 Trusted Committer 후보군 정의

## 관련 문서

- [README.md](../README.md)
- [GOVERNANCE.md](../GOVERNANCE.md)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)
- [OSS 라이선스 비교 및 선택 기준](./oss-license-comparison.md)

## 공식 참고 자료

- InnerSource Commons, Standard Base Documentation: https://patterns.innersourcecommons.org/p/base-documentation
- InnerSource Commons, Start as an Experiment: https://patterns.innersourcecommons.org/p/start-as-experiment
- InnerSource Commons, Issue Tracker Use Cases: https://patterns.innersourcecommons.org/p/issue-tracker
- InnerSource Commons, Transparent Cross-Team Decision Making using RFCs: https://patterns.innersourcecommons.org/p/transparent-cross-team-decision-making-using-rfcs
- InnerSource Commons, Trusted Committer: https://patterns.innersourcecommons.org/p/trusted-committer
