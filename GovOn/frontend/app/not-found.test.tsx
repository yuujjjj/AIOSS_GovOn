import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import NotFound from "./not-found";

describe("Not found page", () => {
  it("keeps a recovery link to the main route", () => {
    render(<NotFound />);

    expect(screen.getByRole("heading", { name: "404" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "메인으로 돌아가기" }).getAttribute("href")).toBe(
      "/",
    );
  });
});
