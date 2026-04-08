# 기여 가이드 (Contributing Guide)

GovOn 프로젝트에 관심을 가져주셔서 감사합니다!
이 문서는 프로젝트에 기여하는 방법을 안내합니다.

## 목차

- [행동 강령](#행동-강령)
- [기여 방법](#기여-방법)
- [개발 환경 설정](#개발-환경-설정)
- [브랜치 전략](#브랜치-전략)
- [커밋 컨벤션](#커밋-컨벤션)
- [Pull Request 가이드](#pull-request-가이드)
- [이슈 작성 가이드](#이슈-작성-가이드)
- [코드 리뷰 가이드](#코드-리뷰-가이드)
- [코드 스타일](#코드-스타일)

## 행동 강령

이 프로젝트는 [행동 강령](CODE_OF_CONDUCT.md)을 따릅니다.
프로젝트에 참여함으로써 이 강령을 준수하는 것에 동의하게 됩니다.

## 기여 방법

### 1. 이슈 확인 또는 생성

- 기존 [이슈 목록](https://github.com/GovOn-Org/GovOn/issues)에서 작업할 이슈를 확인합니다.
- 새로운 버그나 기능 제안이 있다면 이슈를 먼저 생성합니다.
- 이슈 템플릿을 활용하여 작성해 주세요:
  - **기능 요청**: 새로운 기능 제안
  - **버그 리포트**: 버그 신고
  - **문서 작업**: 문서 개선 및 추가
  - **AIOSS 과제**: 수업 관련 과제

### 2. Fork & Clone

```bash
# 레포지토리 Fork 후 클론
git clone https://github.com/<your-username>/AIOSS_GovOn.git
cd AIOSS_GovOn

# upstream 설정
git remote add upstream https://github.com/yuujjjj/AIOSS_GovOn.git
```

> **팀 내부 기여자**는 Fork 없이 직접 브랜치를 생성하여 작업할 수 있습니다.

### 3. 브랜치 생성 및 작업

```bash
git checkout main
git pull upstream main
git checkout -b feature/이슈번호-작업설명
```

### 4. 커밋 및 Push

```bash
git add <변경된 파일>
git commit -m "feat: 작업 내용 설명"
git push origin feature/이슈번호-작업설명
```

### 5. Pull Request 생성

- `main` 브랜치를 대상으로 PR을 생성합니다.
- PR 템플릿에 따라 내용을 작성합니다.
- 관련 이슈를 연결합니다 (`Closes #이슈번호`).

## 개발 환경 설정

### 필수 요구사항

- Python 3.10 이상
- Git 2.30 이상
- (GPU 작업 시) CUDA 12.x, PyTorch 2.x

### 환경 구축

```bash
# 저장소 클론
git clone https://github.com/GovOn-Org/GovOn.git
cd GovOn

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 브랜치 전략

이 저장소는 `main` 기준의 feature 브랜치 전략을 사용합니다.

| 브랜치 | 용도 | 규칙 |
|--------|------|------|
| `main` | 프로덕션 (안정 버전) | 직접 push 금지, PR 머지만 허용 |
| `feature/*` | 기능 및 문서 작업 | `feature/<작업설명>` 형식 |
| `fix/*` | 버그 수정 | `fix/<작업설명>` 형식 |

### 브랜치 이름 규칙

- 모든 PR 브랜치는 `feature/*` 또는 `fix/*` 패턴을 따라야 합니다.
- 예시:
  - `feature/dora-dashboard-docs`
  - `feature/branch-commit-guardrails`
  - `fix/workflow-grafana-auth`
- 이 규칙은 GitHub Actions `Contribution Guardrails` 워크플로우에서 자동 검사합니다.

## 커밋 컨벤션

[Conventional Commits](https://www.conventionalcommits.org/) 형식을 따릅니다.
커밋 제목(subject)은 아래 형식으로 작성합니다.

### 형식

```
<type>: <description>

[optional body]

[optional footer(s)]
```

### 타입

| 타입 | 설명 | 예시 |
|------|------|------|
| `feat` | 새 기능 추가 | `feat: QLoRA 학습 스크립트 구현` |
| `fix` | 버그 수정 | `fix: AWQ 양자화 OOM 해결` |
| `docs` | 문서 수정 | `docs: API 명세서 업데이트` |
| `style` | 코드 포맷팅 | `style: black 포맷터 적용` |
| `refactor` | 코드 리팩토링 | `refactor: 추론 엔진 구조 개선` |
| `test` | 테스트 추가/수정 | `test: 민원 분류 단위 테스트 추가` |
| `chore` | 빌드/설정 변경 | `chore: GitHub Actions 워크플로우 추가` |
| `perf` | 성능 개선 | `perf: vLLM 배치 추론 속도 최적화` |

### 자동 검사 대상

- PR 제목도 동일한 Conventional Commits 형식을 사용합니다.
- PR에 포함된 모든 커밋 subject도 동일한 형식을 사용합니다.
- 허용 타입: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`
- 예시:
  - `docs: add branch strategy guardrails`
  - `chore: enforce branch strategy and conventional commits`

## Pull Request 가이드

### PR 작성 규칙

1. **제목**: 커밋 컨벤션과 동일한 형식 사용
2. **본문**: PR 템플릿에 따라 작성
   - 작업 배경 설명
   - 주요 변경 사항 요약
   - 테스트 결과 기록
3. **연결**: 관련 이슈를 `Closes #이슈번호`로 연결
4. **리뷰어**: 최소 1명의 리뷰어를 지정
5. **브랜치명**: `feature/*` 또는 `fix/*` 패턴 사용

### PR 체크리스트

- [ ] 커밋 메시지가 컨벤션을 따르는가?
- [ ] PR 제목이 Conventional Commits 형식을 따르는가?
- [ ] 브랜치 이름이 `feature/*` 또는 `fix/*` 패턴을 따르는가?
- [ ] 관련 이슈가 연결되어 있는가?
- [ ] 테스트가 통과하는가?
- [ ] 문서가 업데이트되었는가? (필요한 경우)

## 이슈 작성 가이드

### 버그 리포트

- 재현 가능한 단계를 명확히 기술
- 예상 동작과 실제 동작을 구분하여 기록
- 환경 정보 포함 (OS, Python 버전, GPU 등)

### 기능 요청

- 해결하려는 문제를 먼저 설명
- 제안하는 해결 방법을 기술
- 대안이 있다면 함께 기록

## 코드 리뷰 가이드

### 리뷰어 역할

- `[MUST]`: 반드시 수정해야 할 사항 (보안, 버그, 성능 이슈)
- `[SHOULD]`: 수정을 강하게 권장하는 사항
- `[NITS]`: 사소한 개선 사항 (코드 스타일 등)
- `[QUESTION]`: 이해를 위한 질문

### 리뷰 기준

1. **정확성**: 코드가 의도한 대로 동작하는가?
2. **가독성**: 코드가 이해하기 쉬운가?
3. **테스트**: 적절한 테스트가 포함되어 있는가?
4. **보안**: 민감 정보(API 키, 모델 가중치 경로)가 노출되지 않는가?

## 코드 스타일

### Python

- [PEP 8](https://peps.python.org/pep-0008/) 스타일 가이드를 따릅니다.
- 함수/클래스에 docstring을 작성합니다.
- 타입 힌트를 적극 활용합니다.

```python
def classify_complaint(text: str, model_name: str = "exaone") -> dict:
    """민원 텍스트를 분류합니다.

    Args:
        text: 분류할 민원 텍스트
        model_name: 사용할 모델 이름

    Returns:
        분류 결과를 담은 딕셔너리
    """
    ...
```

### 파일 구조

- 모듈별로 디렉토리를 분리합니다 (`src/training/`, `src/inference/` 등).
- 설정 파일은 `configs/` 디렉토리에 관리합니다.
- 노트북은 `notebooks/` 디렉토리에 관리합니다.

## 질문이 있다면

- [GitHub Issues](https://github.com/GovOn-Org/GovOn/issues)에서 `question` 라벨로 이슈를 생성해 주세요.
- [GitHub Discussions](https://github.com/GovOn-Org/GovOn/discussions) (활성화 시)에서 자유롭게 질문할 수 있습니다.

감사합니다!
