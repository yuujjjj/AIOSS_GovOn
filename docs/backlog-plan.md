# Sprint Backlog Plan

AIOSS_GovOn 저장소의 스프린트 운영을 위한 백로그 초안이다.
완료한 작업과 앞으로 진행할 작업을 함께 관리할 수 있도록 구성했다.

## Milestones

### Sprint 1 - 기반 정비 및 DORA 자동화

- 목표: GitHub 운영 체계, 이슈 템플릿, DORA 자동 수집 워크플로우를 정리한다.
- 기간 예시: 2026-03-19 ~ 2026-03-26

### Sprint 2 - 대시보드 및 결과 정리

- 목표: DORA 결과 시각화, README 제출물, 발표 자료 정리를 마무리한다.
- 기간 예시: 2026-03-27 ~ 2026-04-03

## Label Taxonomy

### Type

- `bug`
- `documentation`
- `question`
- `enhancement`
- `maintenance`

### Priority

- `critical`
- `high`
- `medium`
- `low`

### Status

- `status:new`
- `status:in-progress`
- `status:blocked`
- `status:ready`
- `status:investigating`
- `status:done`

### Size

- `XS`: 1시간 미만
- `S`: 반나절
- `M`: 2일
- `L`: 3~5일
- `XL`: 1주 이상

## Backlog

| No | Milestone | Title | Labels |
|----|-----------|-------|--------|
| 1 | Sprint 1 - 기반 정비 및 DORA 자동화 | GitHub 이슈·라벨·마일스톤 운영 세팅 정리 | `maintenance`, `high`, `status:ready`, `S` |
| 2 | Sprint 1 - 기반 정비 및 DORA 자동화 | DORA Lead Time 계산 로직 구현 | `enhancement`, `high`, `status:ready`, `M` |
| 3 | Sprint 1 - 기반 정비 및 DORA 자동화 | GitHub Actions 자동 수집 워크플로우 작성 | `enhancement`, `critical`, `status:ready`, `L` |
| 4 | Sprint 1 - 기반 정비 및 DORA 자동화 | DORA 메트릭 JSON 저장 구조 정리 | `maintenance`, `medium`, `status:new`, `S` |
| 5 | Sprint 1 - 기반 정비 및 DORA 자동화 | Bug 이슈 템플릿 정비 | `documentation`, `medium`, `status:ready`, `XS` |
| 6 | Sprint 1 - 기반 정비 및 DORA 자동화 | Feature 이슈 템플릿 정비 | `documentation`, `medium`, `status:ready`, `XS` |
| 7 | Sprint 1 - 기반 정비 및 DORA 자동화 | 라벨 체계 정리 및 자동 라벨링 점검 | `maintenance`, `high`, `status:new`, `S` |
| 8 | Sprint 1 - 기반 정비 및 DORA 자동화 | DORA 수집 워크플로우 수동 실행 옵션 정리 | `enhancement`, `medium`, `status:ready`, `S` |
| 9 | Sprint 2 - 대시보드 및 결과 정리 | README에 DORA 사용법 문서화 | `documentation`, `high`, `status:ready`, `S` |
| 10 | Sprint 2 - 대시보드 및 결과 정리 | README에 DORA 대시보드 이미지 첨부 | `documentation`, `high`, `status:new`, `XS` |
| 11 | Sprint 1 - 기반 정비 및 DORA 자동화 | Sprint 1 마일스톤 생성 및 목표 정의 | `maintenance`, `medium`, `status:ready`, `XS` |
| 12 | Sprint 2 - 대시보드 및 결과 정리 | DORA 수집 결과 발표용 정리 | `documentation`, `medium`, `status:new`, `S` |
| 13 | Sprint 2 - 대시보드 및 결과 정리 | Sprint 2 마일스톤 생성 및 목표 정의 | `maintenance`, `medium`, `status:ready`, `XS` |

## Notes

- 완료한 작업도 이슈로 남기고 닫으면 스프린트 진행 흔적을 보여주기 쉽다.
- `Run workflow`로 DORA를 수동 실행한 뒤 결과를 해당 이슈에 링크하면 과제 제출 자료로 활용하기 좋다.
