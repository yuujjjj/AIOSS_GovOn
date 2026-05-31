import { describe, expect, it } from "vitest";
import { GET } from "./route";

describe("Vercel health route", () => {
  it("returns healthy service metadata", async () => {
    const response = GET();
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body).toEqual({
      status: "healthy",
      service: "govon-vercel-serverless",
      runtime: "vercel",
    });
    expect(response.headers.get("cache-control")).toBe("no-store");
  });
});
