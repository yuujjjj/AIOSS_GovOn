# Generated User Scenario Test Code

Generated from the `govon-ux-panel-v1` LLM user panel. Each scenario maps to one
feedback row in `docs/experiments/llm-user-feedback.jsonl` through
`generated_code_id`.

```ts
type GeneratedScenario = {
  id: string;
  personaPattern: string;
  userId: string;
  variant: "control" | "guided";
  prompt: string;
  expectedEvidence: string[];
};

export const generatedGovOnScenarios: GeneratedScenario[] = [
  {
    id: "SCN-001",
    personaPattern: "throughput_first_clerk",
    userId: "llm-clerk-001",
    variant: "guided",
    prompt: "40건의 생활 불편 민원을 분류하고 반복 민원을 먼저 처리한다.",
    expectedEvidence: ["category", "urgency", "next_action", "due_date"],
  },
  {
    id: "SCN-002",
    personaPattern: "accessibility_first_elder",
    userId: "llm-elder-002",
    variant: "guided",
    prompt: "고령 사용자가 소음 민원 답변 초안을 이해하고 다음 행동을 찾는다.",
    expectedEvidence: ["plain_summary", "large_step_labels", "due_date"],
  },
  {
    id: "SCN-003",
    personaPattern: "accessibility_audit_advocate",
    userId: "llm-advocate-003",
    variant: "guided",
    prompt: "장애인 이동권 민원에서 접근성 기준과 담당 부서를 확인한다.",
    expectedEvidence: ["policy_basis", "department_route", "checklist"],
  },
  {
    id: "SCN-004",
    personaPattern: "plain_language_multilingual_parent",
    userId: "llm-parent-004",
    variant: "control",
    prompt: "다문화 가정 보호자가 보육 지원 답변을 쉬운 표현으로 이해한다.",
    expectedEvidence: ["plain_language", "follow_up_channel"],
  },
  {
    id: "SCN-005",
    personaPattern: "privacy_skeptic_reviewer",
    userId: "llm-privacy-005",
    variant: "guided",
    prompt: "개인정보 보호 담당자가 전화번호와 주민등록번호 마스킹을 점검한다.",
    expectedEvidence: ["redaction", "audit_log", "masked_preview"],
  },
  {
    id: "SCN-006",
    personaPattern: "mobile_field_inspector",
    userId: "llm-field-006",
    variant: "control",
    prompt: "현장 점검 공무원이 모바일에서 도로 파손 신고 체크리스트를 확인한다.",
    expectedEvidence: ["location", "field_checklist", "photo_request"],
  },
  {
    id: "SCN-007",
    personaPattern: "api_integrator_developer",
    userId: "llm-dev-007",
    variant: "guided",
    prompt: "기관 연동 개발자가 동일 user_id의 실험 할당 일관성을 검증한다.",
    expectedEvidence: ["stable_assignment", "event_schema", "request_id"],
  },
  {
    id: "SCN-008",
    personaPattern: "evidence_seeking_policy_analyst",
    userId: "llm-policy-008",
    variant: "guided",
    prompt: "정책 분석가가 반복 민원 유형과 답변 근거를 비교한다.",
    expectedEvidence: ["trend_summary", "evidence_link", "exportable_metrics"],
  },
  {
    id: "SCN-009",
    personaPattern: "empathy_first_social_worker",
    userId: "llm-social-009",
    variant: "control",
    prompt: "복지 상담사가 긴급 복지 민원 답변의 공감 톤과 기한을 확인한다.",
    expectedEvidence: ["tone", "deadline", "department_route"],
  },
  {
    id: "SCN-010",
    personaPattern: "time_constrained_business_owner",
    userId: "llm-owner-010",
    variant: "control",
    prompt: "소상공인이 불법 주정차 민원을 빠르게 제출하고 처리 예상일을 확인한다.",
    expectedEvidence: ["submission_status", "due_date", "contact_channel"],
  },
];

test.describe("generated LLM user scenarios", () => {
  for (const scenario of generatedGovOnScenarios) {
    test(`${scenario.id} ${scenario.personaPattern}`, async ({ request }) => {
      const response = await request.post("/v1/complaints/draft", {
        headers: {
          "X-User-Id": scenario.userId,
          "X-Feature-Flag": `EXPERIMENT_COMPLAINT_RESPONSE_LAYOUT_FORCE_VARIANT=${scenario.variant}`,
        },
        data: {
          prompt: scenario.prompt,
          expectedEvidence: scenario.expectedEvidence,
        },
      });

      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      for (const evidence of scenario.expectedEvidence) {
        expect(JSON.stringify(body)).toContain(evidence);
      }
    });
  }
});
```

Evaluation results are recorded in `llm-user-feedback.jsonl`; the generated
code IDs are treated as scenario fixtures for experiment review and weekly
reporting automation.
