#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/check_contribution_rules.sh branch <branch-name>
  scripts/check_contribution_rules.sh conventional <label> <subject>
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

MODE="$1"
shift

case "$MODE" in
  branch)
    BRANCH_NAME="$1"
    if [[ ! "$BRANCH_NAME" =~ ^(feature|fix)/[a-z0-9._-]+$ ]]; then
      fail "Branch '$BRANCH_NAME' must match 'feature/<name>' or 'fix/<name>'."
    fi
    echo "Branch name ok: $BRANCH_NAME"
    ;;
  conventional)
    if [ "$#" -lt 2 ]; then
      usage
      exit 1
    fi
    LABEL="$1"
    SUBJECT="$2"
    if [[ ! "$SUBJECT" =~ ^(feat|fix|docs|style|refactor|test|chore|perf)(\([a-z0-9._/-]+\))?!?:\ .+$ ]]; then
      fail "$LABEL '$SUBJECT' is not a valid Conventional Commit subject."
    fi
    echo "$LABEL ok: $SUBJECT"
    ;;
  *)
    usage
    exit 1
    ;;
esac
