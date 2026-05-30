import { NextResponse } from "next/server";

export const dynamic = "force-static";

export function GET() {
  return NextResponse.json(
    {
      status: "healthy",
      service: "govon-vercel-serverless",
      runtime: "vercel",
    },
    {
      headers: {
        "Cache-Control": "no-store",
      },
    },
  );
}
