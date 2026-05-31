import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Home from "./page";

describe("Home page legacy behavior", () => {
  it("renders the GovOn entry points", () => {
    render(<Home />);

    expect(screen.getByRole("heading", { name: "GovOn" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "새 분석 생성" }).getAttribute("href")).toBe(
      "/main",
    );
    expect(screen.getByRole("link", { name: "기록 보기" }).getAttribute("href")).toBe(
      "/main",
    );
  });
});
