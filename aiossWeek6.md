# AIOSS Week 6: Matrix 빌드 과제 요약

## 과제 소개
- 과제 2: Matrix 빌드
- Mission 02: Test Across Environments
- 하나의 설정으로 여러 환경을 동시에 테스트
- Matrix 전략을 사용해 다양한 Node.js 버전 및 운영체제 호환성을 검증하는 파이프라인 구축

## 구현 요구 사항
1. Node 버전 매트릭스
   - `node-version: [16, 18, 20]`
   - 여러 버전에서 동시에 테스트 실행
2. OS 매트릭스 확장
   - 실행 환경(Runner)을 `ubuntu`, `windows`, `macOS` 등으로 확장
   - `os: [ubuntu-latest, windows-latest]` 추가하여 크로스 플랫폼 검증
3. 전략(Strategy) 구성
   - Job 레벨에서 `strategy` 키워드 사용
   - 매트릭스 변수를 정의하고, `steps`에서 `{{ matrix.node-version }}` 형태로 참조
4. 결과 비교 분석
   - Actions 탭에서 조합(N x M)만큼 생성된 하위 작업들이 병렬 실행되는 것을 확인
   - 특정 환경에서 성공 여부를 비교 분석

## 기대효과
- Node.js 버전 이슈 조기 발견
- Windows/Ubuntu 등 플랫폼 간 차이 조기 감지
- CI 파이프라인 품질 및 안정성 향상

## 추가 메모
- macOS를 포함하고 싶다면 `os: [ubuntu-latest, windows-latest, macos-latest]`로 확장
- 필요 시 `strategy.fail-fast: false`로 구성하여 모든 조합 결과를 종합적으로 확인
