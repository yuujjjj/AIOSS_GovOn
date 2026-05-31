export type ComplaintCategory = "traffic" | "environment" | "welfare" | "general";
export type ComplaintUrgency = "standard" | "high" | "critical";

export interface ComplaintInput {
  id: string;
  text: string;
  receivedAt: Date;
}

export interface ComplaintWorkItem {
  id: string;
  normalizedText: string;
  maskedText: string;
  category: ComplaintCategory;
  urgency: ComplaintUrgency;
  dueDate: Date;
  checklist: string[];
  draft: string;
}

const categoryKeywords: Record<Exclude<ComplaintCategory, "general">, string[]> = {
  traffic: ["road", "traffic", "parking", "bus", "subway", "crosswalk"],
  environment: ["trash", "noise", "smell", "dust", "pollution", "waste"],
  welfare: ["welfare", "disability", "elderly", "benefit", "childcare", "health"],
};

const urgencyKeywords: Record<ComplaintUrgency, string[]> = {
  critical: ["danger", "injury", "fire", "collapse", "emergency", "unsafe"],
  high: ["repeat", "delayed", "urgent", "deadline", "again", "unresolved"],
  standard: [],
};

const businessDayOffsets: Record<ComplaintUrgency, number> = {
  critical: 1,
  high: 3,
  standard: 7,
};

export function normalizeComplaintText(text: string): string {
  return text
    .replace(/[\u0000-\u001F\u007F]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function maskSensitiveContactInfo(text: string): string {
  return text
    .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, "[email]")
    .replace(/\b\d{3}-\d{3,4}-\d{4}\b/g, "[phone]")
    .replace(/\b\d{6}-\d{7}\b/g, "[resident-id]");
}

export function detectComplaintCategory(text: string): ComplaintCategory {
  const normalized = text.toLowerCase();

  for (const [category, keywords] of Object.entries(categoryKeywords)) {
    if (keywords.some((keyword) => normalized.includes(keyword))) {
      return category as ComplaintCategory;
    }
  }

  return "general";
}

export function classifyComplaintUrgency(text: string): ComplaintUrgency {
  const normalized = text.toLowerCase();

  if (urgencyKeywords.critical.some((keyword) => normalized.includes(keyword))) {
    return "critical";
  }

  if (urgencyKeywords.high.some((keyword) => normalized.includes(keyword))) {
    return "high";
  }

  return "standard";
}

export function calculateResponseDueDate(
  receivedAt: Date,
  urgency: ComplaintUrgency,
): Date {
  const dueDate = new Date(receivedAt);
  let remainingBusinessDays = businessDayOffsets[urgency];

  while (remainingBusinessDays > 0) {
    dueDate.setDate(dueDate.getDate() + 1);
    const day = dueDate.getDay();
    if (day !== 0 && day !== 6) {
      remainingBusinessDays -= 1;
    }
  }

  return dueDate;
}

export function buildResponseChecklist(
  category: ComplaintCategory,
  urgency: ComplaintUrgency,
): string[] {
  const checklist = [
    "Confirm complaint scope",
    "Mask personal information",
    "Attach policy or case evidence",
  ];

  if (category !== "general") {
    checklist.push(`Route to ${category} desk`);
  }

  if (urgency !== "standard") {
    checklist.unshift("Escalate for same-day triage");
  }

  return checklist;
}

export function buildComplaintWorkItem(input: ComplaintInput): ComplaintWorkItem {
  const normalizedText = normalizeComplaintText(input.text);
  const maskedText = maskSensitiveContactInfo(normalizedText);
  const category = detectComplaintCategory(maskedText);
  const urgency = classifyComplaintUrgency(maskedText);
  const dueDate = calculateResponseDueDate(input.receivedAt, urgency);
  const checklist = buildResponseChecklist(category, urgency);
  const draft = [
    `Complaint ${input.id} is classified as ${category}.`,
    `Urgency is ${urgency}.`,
    `Target response date is ${dueDate.toISOString().slice(0, 10)}.`,
  ].join(" ");

  return {
    id: input.id,
    normalizedText,
    maskedText,
    category,
    urgency,
    dueDate,
    checklist,
    draft,
  };
}
