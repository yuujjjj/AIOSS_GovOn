# Conventional Commits Guide

이 저장소의 커밋 메시지는 Conventional Commits 형식을 따른다.

## 형식

`type: summary`

예시:

- `feat: add issue template validation`
- `fix: handle grafana auth errors`
- `docs: update dashboard submission guide`
- `chore: refresh dora metrics snapshot`

## 주요 타입

- `feat`: 사용자 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 수정
- `refactor`: 동작 변경 없는 구조 개선
- `test`: 테스트 추가 또는 수정
- `chore`: 설정, 도구, 데이터 갱신

## 작성 기준

- summary는 영어 동사 원형 기준으로 짧게 작성한다.
- 한 커밋에는 하나의 목적만 담는다.
- PR 제목도 가능하면 같은 형식을 유지한다.
