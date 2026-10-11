"use client";

import { useRef, useState } from "react";
import { ApiError, ImageRole, Modality, UploadResponse, uploadImage } from "@/lib/api";

type SlotState =
  | { kind: "idle" }
  | { kind: "uploading" }
  | { kind: "done"; meta: UploadResponse }
  | { kind: "error"; message: string };

interface UploadSlotProps {
  label: string;
  modality: Modality;
  role: ImageRole;
  onUploaded: (meta: UploadResponse) => void;
  onCleared: () => void;
}

export default function UploadSlot({ label, modality, role, onUploaded, onCleared }: UploadSlotProps) {
  const [state, setState] = useState<SlotState>({ kind: "idle" });
  const [reportOpen, setReportOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File | null) {
    if (!file) return;
    setState({ kind: "uploading" });
    try {
      const meta = await uploadImage(file, modality, role);
      setState({ kind: "done", meta });
      onUploaded(meta);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Upload failed.";
      setState({ kind: "error", message });
    }
  }

  function reset() {
    setState({ kind: "idle" });
    setReportOpen(false);
    if (inputRef.current) inputRef.current.value = "";
    onCleared();
  }

  const dotClass =
    state.kind === "done"
      ? "bg-emerald-400"
      : state.kind === "error"
        ? "bg-red-500"
        : state.kind === "uploading"
          ? "bg-yellow-400"
          : "bg-[var(--text-faint)]";

  return (
    <div className="relative flex items-center gap-1">
      <input
        ref={inputRef}
        type="file"
        accept=".tif,.tiff,.png,.jpg,.jpeg"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
      />

      <button
        type="button"
        onClick={() => (state.kind === "done" ? reset() : inputRef.current?.click())}
        disabled={state.kind === "uploading"}
        title={
          state.kind === "done"
            ? `${state.meta.filename} — ${state.meta.width}×${state.meta.height}, ${state.meta.band_count} band(s)${
                state.meta.crs ? `, ${state.meta.crs}` : " (no CRS)"
              }. Click to clear.`
            : state.kind === "error"
              ? state.message
              : `Upload ${label}`
        }
        className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs transition-colors ${
          state.kind === "done"
            ? "border-emerald-800 bg-emerald-950/40 text-emerald-300 hover:bg-emerald-950/70"
            : state.kind === "error"
              ? "border-red-900 bg-red-950/30 text-red-300 hover:bg-red-950/60"
              : "border-[var(--border)] bg-[var(--bg-panel)] text-[var(--text-dim)] hover:border-[var(--accent)] hover:text-[var(--text)]"
        }`}
      >
        <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
        {label}
        {state.kind === "uploading" && "…"}
      </button>

      {/* Phase 14 (UI): surfaces Phase 13's preprocessing_report — was
          computed by the backend on every upload but never shown. */}
      {state.kind === "done" && state.meta.preprocessing_report && (
        <>
          <button
            type="button"
            onClick={() => setReportOpen((o) => !o)}
            title="Preprocessing diagnostics"
            aria-label="Preprocessing diagnostics"
            className={`flex h-4 w-4 items-center justify-center rounded-full border text-[9px] leading-none transition-colors ${
              reportOpen
                ? "border-[var(--accent)] text-[var(--accent)]"
                : "border-[var(--border)] text-[var(--text-faint)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
            }`}
          >
            i
          </button>
          {reportOpen && (
            <PreprocessingPopover
              report={state.meta.preprocessing_report}
              onClose={() => setReportOpen(false)}
            />
          )}
        </>
      )}
    </div>
  );
}

function CheckDot({ ok }: { ok: boolean }) {
  return <span className={`inline-block h-1.5 w-1.5 rounded-full ${ok ? "bg-emerald-400" : "bg-red-500"}`} />;
}

/** Phase 14 (UI): read-only diagnostics popover for Phase 13's
 * preprocessing_report (integrity / georeferencing / spectral bands /
 * Dark-Object-Subtraction atmospheric correction). Purely display —
 * never re-computes or overrides what the backend reported, and keeps
 * the same honesty-contract qualifiers (DOS is not a physically-modeled
 * correction; georeferencing is a check, not a re-registration). */
function PreprocessingPopover({
  report,
  onClose,
}: {
  report: NonNullable<import("@/lib/api").UploadResponse["preprocessing_report"]>;
  onClose: () => void;
}) {
  return (
    <div
      className="absolute left-0 top-[calc(100%+6px)] z-20 w-72 space-y-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg-panel)] p-3 text-xs shadow-xl"
      onMouseLeave={onClose}
    >
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold tracking-wide text-[var(--text-faint)]">
          PREPROCESSING DIAGNOSTICS
        </span>
        <button onClick={onClose} className="text-[var(--text-faint)] hover:text-[var(--text)]" aria-label="Close">
          ✕
        </button>
      </div>

      <div className="flex items-start gap-1.5">
        <CheckDot ok={report.integrity_ok} />
        <div>
          <p className="font-medium text-[var(--text)]">Integrity</p>
          <p className="text-[var(--text-faint)]">{report.integrity_note}</p>
        </div>
      </div>

      {report.geometric && (
        <div className="flex items-start gap-1.5">
          <CheckDot ok={report.geometric.is_georeferenced} />
          <div>
            <p className="font-medium text-[var(--text)]">
              Georeferencing — {report.geometric.is_georeferenced ? "present" : "missing"}
            </p>
            {report.geometric.is_georeferenced && (
              <p className="text-[var(--text-dim)]">
                {report.geometric.crs ?? "CRS n/a"}
                {report.geometric.pixel_size_x != null &&
                  ` · pixel ${report.geometric.pixel_size_x.toFixed(2)}×${report.geometric.pixel_size_y?.toFixed(2)}`}
              </p>
            )}
            <p className="text-[var(--text-faint)]">{report.geometric.note}</p>
          </div>
        </div>
      )}

      {report.spectral && (
        <div className="flex items-start gap-1.5">
          <CheckDot ok={report.spectral.supports_true_color} />
          <div>
            <p className="font-medium text-[var(--text)]">
              Spectral — {report.spectral.band_count} band(s), {report.spectral.dtype}
            </p>
            <p className="text-[var(--text-dim)]">
              {report.spectral.supports_true_color ? "True color ✓" : "True color ✗"}
              {" · "}
              {report.spectral.supports_nir_index ? "NIR index ✓" : "NIR index ✗"}
            </p>
            <p className="text-[var(--text-faint)]">{report.spectral.note}</p>
          </div>
        </div>
      )}

      {report.atmospheric && (
        <div className="flex items-start gap-1.5">
          <CheckDot ok={report.atmospheric.applied} />
          <div>
            <p className="font-medium text-[var(--text)]">
              Atmospheric — {report.atmospheric.applied ? report.atmospheric.method : "not applied"}
            </p>
            <p className="text-[var(--text-faint)]">{report.atmospheric.note}</p>
          </div>
        </div>
      )}
    </div>
  );
}
