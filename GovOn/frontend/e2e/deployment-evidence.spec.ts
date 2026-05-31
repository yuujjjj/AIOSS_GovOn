import { expect, test } from "@playwright/test";

test("deployment evidence page exposes testing evidence", async ({ page }) => {
  await page.goto("/deployment-evidence");

  await expect(page.getByRole("heading", { name: "Single public URL for grading" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Feature Flag Evidence" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Testing Evidence" })).toBeVisible();
  await expect(page.getByRole("link", { name: /Feature flag code/ })).toHaveAttribute(
    "href",
    /feature_flags\.py/,
  );
  await expect(page.getByRole("link", { name: /Rollout setting/ })).toHaveAttribute(
    "href",
    /canary-rollout\.yml/,
  );
  await expect(page.getByRole("link", { name: /Core unit tests/ })).toHaveAttribute(
    "href",
    /complaint-workflow\.test\.ts/,
  );
  await expect(page.getByRole("link", { name: /Playwright E2E/ })).toHaveAttribute(
    "href",
    /deployment-evidence\.spec\.ts/,
  );
});
