"use client";

import { QueryResponse } from "@/lib/api";

interface ReportViewProps {
  latestResult: QueryResponse | null;
}

function buildReportText(result: QueryResponse): string {
  const lines: string[] = [];
  lines.push("SATQUERY AI — ANALYSIS REPORT");
  lines.push("=".repeat(40));
  lines.push(`Generated: ${new Date().toLocaleString()}`);
  lines.push("");
  lines.push(`Query: ${result.query}`);
  lines.push(`Task: ${result.task_label} (${result.task})`);
  lines.push(`PS requirement: ${result.ps_requirement}`);
  lines.push(`Status: ${result.status}`);
  lines.push(`Selected tool: ${result.selected_tool}`);
  lines.push("");
  lines.push("EXECUTION TRACE");
  lines.push("-".repeat(40));
  result.execution_trace.forEach((line, i) => lines.push(`${String(i + 1).padStart(2, "0")}. ${line}`));
  lines.push("");
  if (result.input_check.issues.length > 0) {
    lines.push("INPUT WARNINGS");
    lines.push("-".repeat(40));
    result.input_check.issues.forEach((issue) => lines.push(`- ${issue}`));
    lines.push("");
  }
  lines.push("RESULT");
  lines.push("-".repeat(40));
  lines.push(JSON.stringify(result.tool_output, null, 2));
  lines.push("");
  lines.push(`Note: ${result.note}`);
  return lines.join("\n");
}

export default function ReportView({ latestResult }: ReportViewProps) {
  function download() {
    if (!latestResult) return;
    const text = buildReportText(latestResult);
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `satquery-report-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-8">
      <div>
        <h2 className="text-lg font-semibold text-[var(--text)]">Analysis Report</h2>
        <p className="mt-1 text-sm text-[var(--text-dim)]">
          Builds a plain-text export of the query, execution trace, and result — the auditable summary the PS asks
          for. PDF/formatted export is a later-phase upgrade; this is the P2 preview.
        </p>
      </div>

      {!latestResult ? (
        <div className="rounded-lg border border-dashed border-[var(--border)] bg-[var(--bg-panel)] p-8 text-center text-sm text-[var(--text-faint)]">
          Run a query on the Dashboard first — the most recent result will be reportable here.
        </div>
      ) : (
        <div className="space-y-4 rounded-lg border border-[var(--border)] bg-[var(--bg-panel)] p-5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded bg-[var(--accent-bg)] px-2 py-0.5 text-xs font-medium text-[var(--accent-text)]">
              {latestResult.task_label}
            </span>
            <span className="text-xs text-[var(--text-faint)]">{latestResult.ps_requirement}</span>
          </div>
          <p className="text-sm text-[var(--text)]">&ldquo;{latestResult.query}&rdquo;</p>
          <pre className="max-h-64 overflow-auto rounded bg-[var(--bg-app)] p-3 font-mono text-[11px] leading-relaxed text-[var(--text-dim)]">
            {buildReportText(latestResult)}
          </pre>
          <button
            onClick={download}
            className="rounded-md bg-[var(--accent-strong)] px-4 py-2 text-sm font-medium text-white hover:bg-[var(--accent-strong-hover)]"
          >
            Download report (.txt)
          </button>
        </div>
      )}
    </div>
  );
}
