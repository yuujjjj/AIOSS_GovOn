import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import DeploymentEvidencePage from "./page";

describe("Deployment evidence page", () => {
  it("exposes testing and rollout evidence links", () => {
    render(<DeploymentEvidencePage />);

    expect(screen.getByRole("heading", { name: "Single public URL for grading" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Feature Flag Evidence" })).toBeTruthy();
    expect(screen.getByRole("link", { name: /Feature flag code/ }).getAttribute("href")).toContain(
      "feature_flags.py",
    );
    expect(screen.getByRole("link", { name: /Rollout setting/ }).getAttribute("href")).toContain(
      "canary-rollout.yml",
    );
  });
});
