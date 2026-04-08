# Branch Strategy

이 저장소는 기능 단위의 feature 브랜치 전략을 기본으로 사용한다.

## 기본 원칙

- `main`: 배포 및 제출 기준 브랜치
- `develop`: 통합 개발 브랜치
- `feature/*`: 기능 개발 브랜치
- `fix/*`: 버그 수정 브랜치

## 작업 흐름

1. `main`을 기준으로 작업 브랜치를 생성한다.
2. 브랜치 이름은 `feature/*` 또는 `fix/*` 패턴으로 작업 목적이 드러나도록 작성한다.
3. 커밋은 Conventional Commits 형식을 따른다.
4. 작업이 끝나면 Pull Request를 생성하고 리뷰를 반영한다.
5. 리뷰가 끝난 변경만 기준 브랜치로 병합한다.

## 자동 검사

- PR 브랜치명은 `Contribution Guardrails` 워크플로우에서 검사한다.
- `feature/<작업설명>` 또는 `fix/<작업설명>` 형식이 아니면 PR 검사가 실패한다.

## 브랜치 이름 예시

- `feature/dora-dashboard-docs`
- `feature/github-project-automation`
- `fix/workflow-grafana-auth`
