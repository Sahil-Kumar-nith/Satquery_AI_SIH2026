const DATASETS: {
  name: string;
  role: string;
  detail: string;
  link?: string;
}[] = [
  {
    name: "BigEarthNet.txt",
    role: "Training / adaptation",
    detail: "Primary dataset for remote-sensing adaptation — co-registered Sentinel-1 SAR + Sentinel-2 multispectral imagery with text annotations.",
    link: "https://arxiv.org/abs/2603.29630",
  },
  {
    name: "RSVQA",
    role: "Public evaluation",
    detail: "Single-image visual question answering.",
  },
  {
    name: "VRSBench",
    role: "Public evaluation",
    detail: "Single-image captioning and grounding.",
  },
  {
    name: "CDVQA",
    role: "Public evaluation",
    detail: "Multitemporal change-based visual question answering.",
  },
  {
    name: "ISRO / SAC hidden set",
    role: "Final judged evaluation",
    detail: "Cartosat-2S optical + RISAT SAR pairs. Not accessible to participants — annotations are never disclosed.",
  },
];

const METRICS = [
  { label: "VQA Accuracy", suite: "RSVQA" },
  { label: "Grounding IoU", suite: "VRSBench" },
  { label: "Change F1", suite: "CDVQA" },
  { label: "Optical+SAR agreement", suite: "internal" },
] as const;

export default function BenchmarksView() {
  return (
    <div className="mx-auto max-w-4xl space-y-6 px-6 py-8">
      <div>
        <h2 className="text-lg font-semibold text-[var(--text)]">Datasets &amp; Benchmarks</h2>
        <p className="mt-1 text-sm text-[var(--text-dim)]">
          Exactly what the problem statement names — training data kept separate from evaluation data, and the
          hidden ISRO/SAC set is never assumed to be available locally.
        </p>
      </div>

      <div className="overflow-hidden rounded-lg border border-[var(--border)]">
        <table className="w-full text-left text-sm">
          <thead className="bg-[var(--bg-panel)] text-[11px] uppercase tracking-wide text-[var(--text-faint)]">
            <tr>
              <th className="px-4 py-2.5 font-medium">Dataset</th>
              <th className="px-4 py-2.5 font-medium">Role</th>
              <th className="px-4 py-2.5 font-medium">Detail</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border)]">
            {DATASETS.map((d) => (
              <tr key={d.name} className="align-top">
                <td className="px-4 py-3 font-medium text-[var(--text)]">
                  {d.link ? (
                    <a href={d.link} target="_blank" rel="noreferrer" className="text-[var(--accent-text)] underline">
                      {d.name}
                    </a>
                  ) : (
                    d.name
                  )}
                </td>
                <td className="px-4 py-3 text-xs text-[var(--text-dim)]">{d.role}</td>
                <td className="px-4 py-3 text-xs text-[var(--text-dim)]">{d.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-panel)] p-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-[var(--text-faint)]">
          Evaluation Metrics
        </h3>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {METRICS.map((m) => (
            <div key={m.label} className="rounded-md border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-3">
              <div className="text-xs text-[var(--text-faint)]">{m.label}</div>
              <div className="mt-1 text-sm font-medium text-[var(--text-dim)]">Not yet measured</div>
              <div className="mt-0.5 text-[10px] text-[var(--text-faint)]">{m.suite}</div>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs italic text-[var(--text-faint)]">
          No synthetic or demo numbers are shown as benchmark results. These populate once a real run against the
          public test splits above has been executed.
        </p>
      </div>
    </div>
  );
}
