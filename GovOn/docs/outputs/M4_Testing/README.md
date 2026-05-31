# M4: 테스트 및 문서화 (Week 13-16)

**기간**: 2026-05-25 ~ 2026-06-19
**상태**: 진행 중
**최종 수정**: 2026-05-31

---

## 진행 현황 요약

M4 단계는 전체 시스템의 통합 테스트, 문서화, 사용자 수용 테스트(UAT), 최종 발표를 수행합니다.
LLM 페르소나 UAT 및 RAG Feature Flag A/B 실험 경로를 먼저 구현했으며, 실제 14일 운영 결과는
운영 종료 후 별도 리포트로 기록합니다.

---

## 산출물 체크리스트

### Week 13: 통합 테스트
- [ ] integration_test_report.md - 통합 테스트 결과
- [ ] performance_benchmark.md - 성능 벤치마크 (전체 KPI 검증)
- [ ] bug_fix_log.md - 버그 수정 로그

### Week 14: 문서화
- [ ] user_manual.md - 사용자 매뉴얼
- [ ] technical_docs.md - 기술 문서 (API, 아키텍처)
- [ ] installation_guide.md - 설치 가이드 (폐쇄망 배포)
- [ ] README.md 최종 업데이트

### Week 15: 사용자 수용 테스트 (UAT)
- [x] uat_plan.md - LLM 페르소나 UAT 및 RAG Feature Flag A/B 계획서
- [x] ab-test-dry-run-report.md - 저장·집계 경로 검증용 합성 dry-run 리포트
- [ ] uat_results.md - UAT 결과 리포트
- [ ] feedback_summary.md - 피드백 요약

### Week 16: 최종 발표
- [ ] final_presentation.pptx - 최종 발표 자료
- [ ] demo_video.mp4 - 데모 영상
- [ ] retrospective.md - 프로젝트 회고록

---

## 완료 기준

- [ ] 모든 KPI 목표 달성
- [ ] UAT 통과 (사용자 만족도 >= 3.5/5.0)
- [ ] 문서화 완료 (README, 매뉴얼, 기술 문서)
- [ ] 최종 발표 성공

---

## 의존성

- M3의 백엔드/프론트엔드(Figma MCP 기반)/Docker 구현 완료 필수
- M3의 RAG 파이프라인 구현 완료 필수

---

**작성일**: 2026-03-05
**최종 수정일**: 2026-05-31
