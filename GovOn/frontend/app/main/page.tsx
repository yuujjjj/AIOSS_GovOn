import Link from "next/link";
import { buildComplaintWorkItem } from "@/lib/complaint-workflow";

export default function MainPage() {
  const workItem = buildComplaintWorkItem({
    id: "DEMO-001",
    text: "Repeated road noise and unresolved crosswalk report from 010-1234-5678.",
    receivedAt: new Date("2026-05-29T09:00:00Z"),
  });

  return (
    <div className="min-h-screen bg-slate-50 px-6 py-10 dark:bg-zinc-950">
      <main className="mx-auto max-w-4xl">
        <h1 className="text-3xl font-bold text-slate-950 dark:text-zinc-50">민원 처리 대시보드</h1>
        <p className="mt-3 text-base text-slate-700 dark:text-zinc-300">
          GovOn에 오신 것을 환영합니다.
        </p>

        <section className="mt-8 border border-slate-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-lg font-semibold text-slate-950 dark:text-zinc-50">
            자동 분류 결과
          </h2>
          <dl className="mt-4 grid gap-4 text-sm sm:grid-cols-2">
            <div>
              <dt className="font-medium text-slate-500 dark:text-zinc-400">민원 ID</dt>
              <dd className="mt-1 text-slate-950 dark:text-zinc-50">{workItem.id}</dd>
            </div>
            <div>
              <dt className="font-medium text-slate-500 dark:text-zinc-400">카테고리</dt>
              <dd className="mt-1 text-slate-950 dark:text-zinc-50">{workItem.category}</dd>
            </div>
            <div>
              <dt className="font-medium text-slate-500 dark:text-zinc-400">긴급도</dt>
              <dd className="mt-1 text-slate-950 dark:text-zinc-50">{workItem.urgency}</dd>
            </div>
            <div>
              <dt className="font-medium text-slate-500 dark:text-zinc-400">응답 목표일</dt>
              <dd className="mt-1 text-slate-950 dark:text-zinc-50">
                {workItem.dueDate.toISOString().slice(0, 10)}
              </dd>
            </div>
          </dl>
          <p className="mt-5 text-sm text-slate-700 dark:text-zinc-300">{workItem.draft}</p>
          <ul className="mt-4 list-inside list-disc text-sm text-slate-700 dark:text-zinc-300">
            {workItem.checklist.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>

        <Link
          href="/"
          className="mt-8 inline-flex bg-blue-600 px-5 py-2 font-medium text-white hover:bg-blue-700"
        >
          홈으로 돌아가기
        </Link>
      </main>
    </div>
  );
}
