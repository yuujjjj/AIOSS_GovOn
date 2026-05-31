import { describe, expect, it } from "vitest";
import {
  buildComplaintWorkItem,
  buildResponseChecklist,
  calculateResponseDueDate,
  classifyComplaintUrgency,
  detectComplaintCategory,
  maskSensitiveContactInfo,
  normalizeComplaintText,
} from "./complaint-workflow";

describe("complaint workflow core functions", () => {
  it("normalizes whitespace and strips control characters", () => {
    expect(normalizeComplaintText("  road\n\nnoise\t complaint\u0000 ")).toBe(
      "road noise complaint",
    );
  });

  it("masks email, phone, and resident registration identifiers", () => {
    const masked = maskSensitiveContactInfo(
      "Contact user@example.com or 010-1234-5678. ID 900101-1234567.",
    );

    expect(masked).toBe("Contact [email] or [phone]. ID [resident-id].");
  });

  it("detects complaint category from domain keywords", () => {
    expect(detectComplaintCategory("The crosswalk and road are unsafe")).toBe("traffic");
    expect(detectComplaintCategory("Trash smell near the school")).toBe("environment");
    expect(detectComplaintCategory("Childcare benefit question")).toBe("welfare");
    expect(detectComplaintCategory("General city hall question")).toBe("general");
  });

  it("classifies critical urgency before high urgency", () => {
    expect(classifyComplaintUrgency("Repeated unsafe road collapse risk")).toBe("critical");
    expect(classifyComplaintUrgency("Repeated unresolved request")).toBe("high");
    expect(classifyComplaintUrgency("Regular complaint")).toBe("standard");
  });

  it("calculates business-day due dates by urgency", () => {
    const friday = new Date("2026-05-29T09:00:00Z");

    expect(calculateResponseDueDate(friday, "critical").toISOString().slice(0, 10)).toBe(
      "2026-06-01",
    );
    expect(calculateResponseDueDate(friday, "high").toISOString().slice(0, 10)).toBe(
      "2026-06-03",
    );
    expect(calculateResponseDueDate(friday, "standard").toISOString().slice(0, 10)).toBe(
      "2026-06-09",
    );
  });

  it("builds checklist with routing and escalation steps", () => {
    expect(buildResponseChecklist("traffic", "high")).toEqual([
      "Escalate for same-day triage",
      "Confirm complaint scope",
      "Mask personal information",
      "Attach policy or case evidence",
      "Route to traffic desk",
    ]);
  });

  it("builds a complete work item from raw complaint input", () => {
    const workItem = buildComplaintWorkItem({
      id: "CASE-1",
      text: "  Repeat parking complaint from user@example.com  ",
      receivedAt: new Date("2026-05-29T09:00:00Z"),
    });

    expect(workItem.id).toBe("CASE-1");
    expect(workItem.maskedText).toContain("[email]");
    expect(workItem.category).toBe("traffic");
    expect(workItem.urgency).toBe("high");
    expect(workItem.draft).toContain("Complaint CASE-1 is classified as traffic.");
  });
});
