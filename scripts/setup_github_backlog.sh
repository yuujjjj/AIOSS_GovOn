#!/usr/bin/env bash

set -euo pipefail

REPO="${1:-yuujjjj/AIOSS_GovOn}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_cmd gh

gh auth status >/dev/null 2>&1 || {
  echo "GitHub CLI is not authenticated. Run: gh auth login" >&2
  exit 1
}

ensure_label() {
  local name="$1"
  local color="$2"
  local description="$3"

  if gh label list --repo "$REPO" --limit 200 --json name --jq '.[].name' | grep -Fxq "$name"; then
    gh label edit "$name" --repo "$REPO" --color "$color" --description "$description" >/dev/null
  else
    gh label create "$name" --repo "$REPO" --color "$color" --description "$description" >/dev/null
  fi
}

ensure_milestone() {
  local title="$1"
  local description="$2"
  local due_on="$3"

  if gh api "repos/${REPO}/milestones" --paginate --jq '.[].title' | grep -Fxq "$title"; then
    return
  fi

  gh api "repos/${REPO}/milestones" \
    --method POST \
    -f title="$title" \
    -f description="$description" \
    -f due_on="$due_on" >/dev/null
}

get_milestone_number() {
  local title="$1"
  gh api "repos/${REPO}/milestones" --paginate --jq ".[] | select(.title == \"${title}\") | .number" | head -n 1
}

ensure_issue() {
  local title="$1"
  local milestone_title="$2"
  local labels_csv="$3"
  local body="$4"

  local existing_number
  existing_number="$(gh issue list --repo "$REPO" --state all --search "in:title \"$title\"" --json number,title --jq ".[] | select(.title == \"$title\") | .number" | head -n 1)"

  if [ -n "${existing_number:-}" ]; then
    echo "Issue exists: #${existing_number} ${title}"
    return
  fi

  local milestone_number
  milestone_number="$(get_milestone_number "$milestone_title")"

  gh issue create \
    --repo "$REPO" \
    --title "$title" \
    --body "$body" \
    --label "$labels_csv" \
    --milestone "$milestone_number" >/dev/null

  echo "Created issue: ${title}"
}

echo "Configuring labels..."
ensure_label "bug" "d73a4a" "버그 리포트"
ensure_label "documentation" "0075ca" "문서 작업"
ensure_label "question" "d876e3" "질문 및 확인 필요"
ensure_label "enhancement" "1d76db" "기능 개선 및 신규 구현"
ensure_label "maintenance" "5319e7" "운영 및 유지보수"
ensure_label "critical" "b60205" "가장 높은 우선순위"
ensure_label "high" "d93f0b" "높은 우선순위"
ensure_label "medium" "fbca04" "보통 우선순위"
ensure_label "low" "0e8a16" "낮은 우선순위"
ensure_label "status:new" "ededed" "새로 등록됨"
ensure_label "status:in-progress" "1d76db" "진행 중"
ensure_label "status:blocked" "000000" "외부 의존성으로 보류"
ensure_label "status:ready" "0e8a16" "바로 작업 가능"
ensure_label "status:investigating" "c5def5" "조사 중"
ensure_label "XS" "f9d0c4" "1시간 미만"
ensure_label "S" "fef2c0" "반나절"
ensure_label "M" "c2e0c6" "2일"
ensure_label "L" "bfdadc" "3~5일"
ensure_label "XL" "7057ff" "1주 이상"

echo "Configuring milestones..."
ensure_milestone \
  "Sprint 1 - 기반 정비 및 DORA 자동화" \
  "GitHub 운영 체계, 이슈 템플릿, DORA 자동 수집 워크플로우를 정리한다." \
  "2026-03-26T23:59:59Z"

ensure_milestone \
  "Sprint 2 - 대시보드 및 결과 정리" \
  "DORA 결과 시각화, README 제출물, 발표 자료 정리를 마무리한다." \
  "2026-04-03T23:59:59Z"

echo "Creating backlog issues..."
ensure_issue \
  "GitHub 이슈·라벨·마일스톤 운영 세팅 정리" \
  "Sprint 1 - 기반 정비 및 DORA 자동화" \
  "maintenance,high,status:ready,S" \
  $'## 목표\n- GitHub 저장소 운영에 필요한 이슈, 라벨, 마일스톤 세팅을 정리한다.\n\n## 작업 내용\n- [ ] 라벨 체계 반영\n- [ ] 마일스톤 2개 구성\n- [ ] 이슈 10개 이상 등록\n\n## 산출물\n- GitHub Issues 운영 체계'

ensure_issue \
  "DORA Lead Time 계산 로직 구현" \
  "Sprint 1 - 기반 정비 및 DORA 자동화" \
  "enhancement,high,status:ready,M" \
  $'## 목표\n- PR의 첫 커밋 시점부터 merge 시점까지의 Lead Time을 계산한다.\n\n## 작업 내용\n- [ ] PR 커밋 히스토리 조회\n- [ ] 첫 커밋 기준 산식 반영\n- [ ] 수집 결과 JSON에 반영\n\n## 산출물\n- .github/workflows/dora-metrics.yml'

ensure_issue \
  "GitHub Actions 자동 수집 워크플로우 작성" \
  "Sprint 1 - 기반 정비 및 DORA 자동화" \
  "enhancement,critical,status:ready,L" \
  $'## 목표\n- DORA 4대 지표를 GitHub Actions로 자동 수집한다.\n\n## 작업 내용\n- [ ] schedule 트리거 구성\n- [ ] push 트리거 구성\n- [ ] workflow_dispatch 구성\n- [ ] 결과 파일 저장\n\n## 산출물\n- .github/workflows/dora-metrics.yml'

ensure_issue \
  "DORA 메트릭 JSON 저장 구조 정리" \
  "Sprint 1 - 기반 정비 및 DORA 자동화" \
  "maintenance,medium,status:new,S" \
  $'## 목표\n- DORA 수집 결과를 날짜별 JSON 파일로 정리한다.\n\n## 작업 내용\n- [ ] 파일명 규칙 정리\n- [ ] 메타데이터 구조 정리\n- [ ] 예시 결과 확인\n\n## 산출물\n- metrics/dora/dora-YYYYMMDD.json'

ensure_issue \
  "Bug 이슈 템플릿 정비" \
  "Sprint 1 - 기반 정비 및 DORA 자동화" \
  "documentation,medium,status:ready,XS" \
  $'## 목표\n- 버그 리포트 이슈를 한글 기준으로 일관되게 작성할 수 있게 한다.\n\n## 작업 내용\n- [ ] 항목 구성 점검\n- [ ] 재현 절차 문구 점검\n- [ ] 환경 정보 입력 항목 점검\n\n## 산출물\n- .github/ISSUE_TEMPLATE/bug_report.yml'

ensure_issue \
  "Feature 이슈 템플릿 정비" \
  "Sprint 1 - 기반 정비 및 DORA 자동화" \
  "documentation,medium,status:ready,XS" \
  $'## 목표\n- 기능 구현 이슈를 일관된 형식으로 작성할 수 있게 한다.\n\n## 작업 내용\n- [ ] 제목 규칙 정리\n- [ ] 목표/작업/완료 기준 점검\n- [ ] 기본 라벨 점검\n\n## 산출물\n- .github/ISSUE_TEMPLATE/feature_task.yml'

ensure_issue \
  "라벨 체계 정리 및 자동 라벨링 점검" \
  "Sprint 1 - 기반 정비 및 DORA 자동화" \
  "maintenance,high,status:new,S" \
  $'## 목표\n- 저장소 라벨 체계를 Type, Priority, Status, Size 기준으로 정리한다.\n\n## 작업 내용\n- [ ] 라벨 생성\n- [ ] 색상 및 설명 정리\n- [ ] PR 자동 라벨링 규칙 점검\n\n## 산출물\n- 저장소 라벨 세트\n- .github/labeler.yml'

ensure_issue \
  "DORA 수집 워크플로우 수동 실행 옵션 정리" \
  "Sprint 1 - 기반 정비 및 DORA 자동화" \
  "enhancement,medium,status:ready,S" \
  $'## 목표\n- Actions에서 DORA 수집을 수동 실행할 때 필요한 입력 옵션을 정리한다.\n\n## 작업 내용\n- [ ] collect_enabled 입력 확인\n- [ ] publish_to_grafana 입력 확인\n- [ ] window_days 입력 확인\n\n## 산출물\n- .github/workflows/dora-metrics.yml'

ensure_issue \
  "README에 DORA 사용법 문서화" \
  "Sprint 2 - 대시보드 및 결과 정리" \
  "documentation,high,status:ready,S" \
  $'## 목표\n- 저장소 사용자와 평가자가 DORA 실행 방법을 바로 이해할 수 있게 문서화한다.\n\n## 작업 내용\n- [ ] 자동 실행 방식 설명\n- [ ] 수동 실행 방식 설명\n- [ ] 결과 확인 위치 설명\n\n## 산출물\n- README.md'

ensure_issue \
  "README에 DORA 대시보드 이미지 첨부" \
  "Sprint 2 - 대시보드 및 결과 정리" \
  "documentation,high,status:new,XS" \
  $'## 목표\n- README에서 DORA 대시보드 결과를 이미지로 바로 보여준다.\n\n## 작업 내용\n- [ ] 대시보드 화면 캡처\n- [ ] docs/images 경로 저장\n- [ ] README 이미지 링크 추가\n\n## 산출물\n- docs/images/dora-dashboard.png\n- README.md'

ensure_issue \
  "Sprint 1 마일스톤 생성 및 목표 정의" \
  "Sprint 1 - 기반 정비 및 DORA 자동화" \
  "maintenance,medium,status:ready,XS" \
  $'## 목표\n- Sprint 1 마일스톤을 생성하고 목표를 명확히 정의한다.\n\n## 작업 내용\n- [ ] 마일스톤 제목 설정\n- [ ] 설명 작성\n- [ ] 관련 이슈 연결\n\n## 산출물\n- Sprint 1 milestone'

ensure_issue \
  "DORA 수집 결과 발표용 정리" \
  "Sprint 2 - 대시보드 및 결과 정리" \
  "documentation,medium,status:new,S" \
  $'## 목표\n- DORA 수집 결과를 발표 및 보고 자료에 활용할 수 있게 정리한다.\n\n## 작업 내용\n- [ ] 핵심 수치 요약\n- [ ] 대시보드 캡처 정리\n- [ ] 제출용 문구 정리\n\n## 산출물\n- 발표/보고용 요약 자료'

ensure_issue \
  "Sprint 2 마일스톤 생성 및 목표 정의" \
  "Sprint 2 - 대시보드 및 결과 정리" \
  "maintenance,medium,status:ready,XS" \
  $'## 목표\n- Sprint 2 마일스톤을 생성하고 결과 정리 범위를 확정한다.\n\n## 작업 내용\n- [ ] 마일스톤 제목 설정\n- [ ] 설명 작성\n- [ ] 관련 이슈 연결\n\n## 산출물\n- Sprint 2 milestone'

echo "Backlog setup complete for ${REPO}"
