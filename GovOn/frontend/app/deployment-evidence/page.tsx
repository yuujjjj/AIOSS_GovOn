const primaryLinks = [
  {
    label: "Frontend live URL",
    href: "https://yuujjjj.github.io/AIOSS_GovOn/",
    value: "GitHub Pages",
  },
  {
    label: "PR preview URL",
    href: "https://govon-frontend-aqrca5xyy-uz-s-projects2.vercel.app/",
    value: "Vercel Preview",
  },
  {
    label: "Serverless health URL",
    href: "https://govon-frontend.vercel.app/api/health",
    value: "HTTP 200 health check",
  },
  {
    label: "Repository evidence document",
    href: "https://github.com/yuujjjj/AIOSS_GovOn/blob/main/docs/deployment-evidence.md",
    value: "Full checklist",
  },
];

const workflowLinks = [
  {
    label: "CI/CD and GitHub Pages",
    href: "https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26690051678",
    value: "success",
  },
  {
    label: "Vercel Serverless Deploy",
    href: "https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26690051682",
    value: "success",
  },
  {
    label: "Vercel Monitoring",
    href: "https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26690595815",
    value: "success",
  },
  {
    label: "Docker Image Publish",
    href: "https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26686568590",
    value: "success",
  },
  {
    label: "npm Package Publish",
    href: "https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26553776392",
    value: "success",
  },
  {
    label: "npm Security Scan",
    href: "https://github.com/yuujjjj/AIOSS_GovOn/actions/runs/26689735642",
    value: "success",
  },
];

const requirementRows = [
  {
    requirement: "Frontend auto deployment",
    evidence: "GitHub Pages deployment through .github/workflows/ci-cd.yml",
    status: "Satisfied",
  },
  {
    requirement: "PR preview environment",
    evidence: "Vercel Preview deployment returns HTTP 200 without login",
    status: "Satisfied",
  },
  {
    requirement: "Docker build, push, and local verification",
    evidence: "GHCR image build/push plus local container /health smoke test",
    status: "Satisfied",
  },
  {
    requirement: "External cloud serverless deployment",
    evidence: "Vercel production deployment with /api/health",
    status: "Satisfied",
  },
  {
    requirement: "Health check and monitoring",
    evidence: "Deploy health-check job and scheduled monitoring workflow",
    status: "Satisfied",
  },
  {
    requirement: "npm package publish and version update",
    evidence: "@yuujjjj/govon-frontend@1.0.1 published to GitHub Packages",
    status: "Satisfied",
  },
  {
    requirement: "Dependabot policy and security automation",
    evidence: "Dependabot groups, auto-merge policy, npm audit/Snyk report workflow",
    status: "Satisfied",
  },
];

function EvidenceLink({
  href,
  label,
  value,
}: {
  href: string;
  label: string;
  value: string;
}) {
  return (
    <a
      className="block border border-slate-200 bg-white p-4 text-slate-950 shadow-sm transition hover:border-emerald-500 hover:bg-emerald-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50 dark:hover:border-emerald-400 dark:hover:bg-emerald-950/30"
      href={href}
      rel="noreferrer"
      target="_blank"
    >
      <span className="block text-sm font-semibold">{label}</span>
      <span className="mt-1 block break-all text-sm text-slate-600 dark:text-zinc-400">{value}</span>
    </a>
  );
}

export default function DeploymentEvidencePage() {
  return (
    <main className="min-h-screen bg-slate-50 px-5 py-8 text-slate-950 dark:bg-zinc-950 dark:text-zinc-50 sm:px-8">
      <div className="mx-auto max-w-6xl">
        <header className="border-b border-slate-200 pb-8 dark:border-zinc-800">
          <p className="text-sm font-semibold uppercase tracking-normal text-emerald-700 dark:text-emerald-400">
            GovOn Deployment Evidence
          </p>
          <h1 className="mt-3 text-3xl font-bold sm:text-4xl">Single public URL for grading</h1>
          <p className="mt-4 max-w-3xl text-base leading-7 text-slate-700 dark:text-zinc-300">
            This page collects the unauthenticated live URLs and workflow evidence for frontend deployment,
            PR preview, Docker publishing, Vercel serverless deployment, monitoring, npm package publishing,
            Dependabot policy, and security scanning.
          </p>
          <p className="mt-3 text-sm text-slate-600 dark:text-zinc-400">Last checked: 2026-05-31 KST.</p>
        </header>

        <section className="py-8">
          <h2 className="text-xl font-semibold">Submit This Page</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700 dark:text-zinc-300">
            Use this GitHub Pages URL as the single submitted website URL. The links below show the live
            frontend, public PR preview, serverless health check, and repository evidence.
          </p>
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {primaryLinks.map((link) => (
              <EvidenceLink key={link.href} {...link} />
            ))}
          </div>
        </section>

        <section className="border-t border-slate-200 py-8 dark:border-zinc-800">
          <h2 className="text-xl font-semibold">Requirement Status</h2>
          <div className="mt-5 overflow-x-auto border border-slate-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
            <table className="w-full min-w-[720px] border-collapse text-left text-sm">
              <thead className="bg-slate-100 text-slate-700 dark:bg-zinc-900 dark:text-zinc-300">
                <tr>
                  <th className="border-b border-slate-200 px-4 py-3 dark:border-zinc-800">Requirement</th>
                  <th className="border-b border-slate-200 px-4 py-3 dark:border-zinc-800">Evidence</th>
                  <th className="border-b border-slate-200 px-4 py-3 dark:border-zinc-800">Status</th>
                </tr>
              </thead>
              <tbody>
                {requirementRows.map((row) => (
                  <tr key={row.requirement}>
                    <td className="border-b border-slate-100 px-4 py-3 font-medium dark:border-zinc-900">
                      {row.requirement}
                    </td>
                    <td className="border-b border-slate-100 px-4 py-3 text-slate-700 dark:border-zinc-900 dark:text-zinc-300">
                      {row.evidence}
                    </td>
                    <td className="border-b border-slate-100 px-4 py-3 font-semibold text-emerald-700 dark:border-zinc-900 dark:text-emerald-400">
                      {row.status}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="border-t border-slate-200 py-8 dark:border-zinc-800">
          <h2 className="text-xl font-semibold">Workflow Evidence</h2>
          <div className="mt-5 grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {workflowLinks.map((link) => (
              <EvidenceLink key={link.href} {...link} />
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
