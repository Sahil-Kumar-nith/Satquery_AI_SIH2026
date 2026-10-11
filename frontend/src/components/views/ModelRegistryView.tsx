const REGISTRY_ENTRIES = [
  {
    task: "VQA",
    psRequirement: "Single-image VQA (mandatory baseline)",
    tool: "BLIP-VQA-base",
    detail: "+ BigEarthNet LoRA adapter, when trained (adaptation/train_lora_vqa.py) — otherwise base weights are used and reported as such.",
    status: "ready",
  },
  {
    task: "Captioning",
    psRequirement: "Additional single-image task (captioning)",
    tool: "captioning_model",
    detail: "Not yet integrated — Grounding is currently the implemented \u201cadditional single-image task\u201d instead.",
    status: "planned",
  },
  {
    task: "Grounding",
    psRequirement: "Additional single-image task (text-guided region grounding)",
    tool: "OWL-ViT",
    detail: "Open-vocabulary detector, base weights. No remote-sensing grounding adapter exists yet.",
    status: "ready",
  },
  {
    task: "Change Detection",
    psRequirement: "Multi-image change analysis (mandatory)",
    tool: "change_model",
    detail: "Pixel-diff z-score change map + rule-based description. Deterministic, not a trained model.",
    status: "ready",
  },
  {
    task: "Change VQA",
    psRequirement: "Multi-image change analysis (mandatory)",
    tool: "change_vqa_model",
    detail: "Composite before/after image passed through BLIP-VQA, combined with the change map.",
    status: "ready",
  },
  {
    task: "Optical + SAR",
    psRequirement: "Cross-modal pair analysis (mandatory)",
    tool: "optical_sar_fusion_model",
    detail: "Percentile-threshold agreement fusion over both bands. Deterministic, not a trained fusion model.",
    status: "ready",
  },
] as const;

const STATUS_STYLE: Record<string, string> = {
  ready: "bg-emerald-900/60 text-emerald-300",
  planned: "bg-amber-900/60 text-amber-300",
};

const STATUS_LABEL: Record<string, string> = {
  ready: "● READY",
  planned: "○ PLANNED",
};

export default function ModelRegistryView() {
  return (
    <div className="mx-auto max-w-4xl space-y-6 px-6 py-8">
      <div>
        <h2 className="text-lg font-semibold text-[var(--text)]">Specialist Model Registry</h2>
        <p className="mt-1 text-sm text-[var(--text-dim)]">
          What the agent actually selects from — pulled from{" "}
          <code className="rounded bg-[var(--bg-elevated)] px-1 py-0.5 text-xs">backend/agent/registry.py</code>.
          No fabricated benchmark scores here — accuracy for the adapted model will be added once BigEarthNet
          fine-tuning and evaluation are complete.
        </p>
      </div>

      <div className="overflow-hidden rounded-lg border border-[var(--border)]">
        <table className="w-full text-left text-sm">
          <thead className="bg-[var(--bg-panel)] text-[11px] uppercase tracking-wide text-[var(--text-faint)]">
            <tr>
              <th className="px-4 py-2.5 font-medium">Task</th>
              <th className="px-4 py-2.5 font-medium">Tool</th>
              <th className="px-4 py-2.5 font-medium">Notes</th>
              <th className="px-4 py-2.5 font-medium">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border)]">
            {REGISTRY_ENTRIES.map((row) => (
              <tr key={row.task} className="align-top">
                <td className="px-4 py-3">
                  <div className="font-medium text-[var(--text)]">{row.task}</div>
                  <div className="text-xs text-[var(--text-faint)]">{row.psRequirement}</div>
                </td>
                <td className="px-4 py-3 font-mono text-xs text-[var(--accent-text)]">{row.tool}</td>
                <td className="px-4 py-3 text-xs text-[var(--text-dim)]">{row.detail}</td>
                <td className="px-4 py-3">
                  <span className={`rounded px-2 py-0.5 text-[11px] font-medium ${STATUS_STYLE[row.status]}`}>
                    {STATUS_LABEL[row.status]}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-panel)] p-4 text-xs text-[var(--text-dim)]">
        <span className="font-medium text-[var(--text)]">Adaptive Evidence Acquisition</span> sits above this
        registry, not inside it — the Evidence Sufficiency Engine decides which of these tools to invoke (and
        whether to swap Optical for SAR) per query; it does not add a new model of its own.
      </div>
    </div>
  );
}
