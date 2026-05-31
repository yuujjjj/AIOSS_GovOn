import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import MainPage from "./page";

describe("Main page refactor safety", () => {
  it("renders the complaint workflow summary from core functions", () => {
    render(<MainPage />);

    expect(screen.getByRole("heading", { name: "민원 처리 대시보드" })).toBeTruthy();
    expect(screen.getByText("DEMO-001")).toBeTruthy();
    expect(screen.getByText("traffic")).toBeTruthy();
    expect(screen.getByText("high")).toBeTruthy();
    expect(screen.getByText("2026-06-03")).toBeTruthy();
    expect(screen.getByText("Escalate for same-day triage")).toBeTruthy();
  });
});
