# 프로젝트 거버넌스

이 문서는 `AIOSS_GovOn` 저장소의 오픈소스 운영 원칙과 팀 내부 `Inner Source` 협업 규칙을 정의합니다.

## 운영 원칙

- 저장소의 변경 이력, 의사결정, 작업 상태는 GitHub 이슈와 Pull Request를 중심으로 공개적으로 관리합니다.
- 기능 추가, 버그 수정, 문서 개선은 가능한 한 이슈 기반으로 추적합니다.
- 중요한 변경은 리뷰를 거쳐 병합하며, 문서와 코드의 정합성을 함께 유지합니다.
- 팀 내부 협업은 `Inner Source` 방식으로 운영하되, 외부 기여자도 동일한 문서와 프로세스를 통해 참여할 수 있어야 합니다.

## 역할

### Maintainers

- 저장소 방향성과 우선순위를 관리합니다.
- PR 병합, 릴리스 판단, 보안 및 커뮤니티 이슈 대응을 담당합니다.
- 행동 강령과 기여 규칙 준수 여부를 최종 판단합니다.

현재 Maintainers:

- [@yuujjjj](https://github.com/yuujjjj)
- [@umyunsang](https://github.com/umyunsang)

### Reviewers

- 담당 영역의 변경 사항을 검토합니다.
- 설계, 테스트, 문서 반영 여부를 확인합니다.
- 필요 시 추가 이슈 분할 또는 후속 작업을 제안합니다.

### Contributors

- 이슈를 생성하거나 할당받아 브랜치에서 작업합니다.
- 변경 의도와 테스트 결과를 PR 본문에 명확히 남깁니다.
- 리뷰 피드백을 반영하고 관련 문서를 업데이트합니다.

## 의사결정 규칙

- 문서, 예제, 경미한 설정 변경: 최소 1명의 승인 후 병합
- 애플리케이션 로직, 모델 파이프라인, 워크플로우 변경: 최소 1명의 Maintainer 또는 해당 `CODEOWNERS` 승인 후 병합
- 구조 변경이나 운영 정책 변경: 이슈 또는 PR 본문에 변경 배경과 영향 범위를 명시

## 브랜치 및 병합 정책

- 기본 브랜치는 `main`입니다.
- 직접 push 대신 기능 브랜치와 PR 기반 병합을 기본으로 합니다.
- 브랜치명은 `feat/*`, `fix/*`, `docs/*`, `chore/*` 형식을 권장합니다.
- 병합 전 체크 항목:
  - 관련 이슈 연결
  - 변경 이유 설명
  - 테스트 또는 검증 결과 기록
  - 필요한 문서 업데이트 반영

## 코드 소유권

- 코드 소유권은 [.github/CODEOWNERS](.github/CODEOWNERS) 에서 관리합니다.
- 핵심 디렉터리 변경 시 지정된 소유자가 우선적으로 리뷰합니다.

## 커뮤니티 운영

- 행동 기준은 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)를 따릅니다.
- 기여 절차는 [CONTRIBUTING.md](CONTRIBUTING.md)를 따릅니다.
- 보안 이슈는 [SECURITY.md](SECURITY.md)를 따릅니다.
- 라이선스 선택 기준은 [docs/oss-license-comparison.md](docs/oss-license-comparison.md)를 참고합니다.
- Inner Source 확산 계획은 [docs/innersource-plan.md](docs/innersource-plan.md)를 참고합니다.

## 응답 목표

- 신규 이슈 1차 응답: 3영업일 이내
- PR 리뷰 시작: 5영업일 이내
- 문서/설정 변경 PR: 가능한 빠른 병합

## 변경 관리

- 이 문서는 프로젝트 운영 상황에 따라 갱신할 수 있습니다.
- 거버넌스 변경은 PR로 제안하고, Maintainer 검토 후 반영합니다.
