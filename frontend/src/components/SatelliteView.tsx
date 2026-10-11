"use client";

import { useMemo, useState } from "react";
import BeforeAfterSlider from "./BeforeAfterSlider";
import {
  GroundingOutput,
  QueryResponse,
  UploadResponse,
  getConfidenceMap,
  isChangeDetectionOutput,
  isChangeVQAOutput,
  isGroundingOutput,
  isOpticalSarOutput,
  previewUrl,
} from "@/lib/api";

type SlotMap = Record<
  "optical" | "sar" | "before" | "after" | "sar_before" | "sar_after",
  UploadResponse | undefined
>;

interface SatelliteViewProps {
  slots: SlotMap;
  latestResult: QueryResponse | null;
}

type LayerKey = "optical" | "sar" | "change" | "grounding" | "confidence";

const LAYER_LABELS: Record<LayerKey, string> = {
  optical: "Optical",
  sar: "SAR",
  change: "Change",
  grounding: "Grounding",
  // Phase 14 (UI): the "Confidence" toggle from uisuggestions.docx's
  // "Analysis Layers" panel — previously a label with nothing wired to
  // it. Now backed by Phase 13's real per-pixel confidence_map_png.
  confidence: "Confidence",
};

/** Heuristic only (see Phase 10 note): the backend's evidence engine
 * decides internally which image it used; the structured response
 * doesn't currently return that as a field, so we infer it from the
 * execution trace text, which literally says "SAR activation" when it
 * happens (Phase 9, evidence/sufficiency.py). Good enough for the
 * Satellite View to highlight the right thumbnail — not used for
 * anything that affects the actual answer. */
function inferActiveModality(result: QueryResponse | null): "optical" | "sar" | null {
  if (!result) return null;
  const traceText = result.execution_trace.join(" ").toLowerCase();
  if (traceText.includes("sar activation")) return "sar";
  if (result.task === "vqa" || result.task === "grounding" || result.task === "captioning") {
    return "optical";
  }
  return null;
}

export default function SatelliteView({ slots, latestResult }: SatelliteViewProps) {
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    optical: true,
    sar: true,
    change: true,
    grounding: true,
    confidence: false,
  });

  const anyUploaded = Object.values(slots).some(Boolean);
  const activeModality = useMemo(() => inferActiveModality(latestResult), [latestResult]);

  function toggleLayer(key: LayerKey) {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  const output = latestResult?.tool_output ?? null;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-2">
        <span className="text-xs font-medium tracking-wide text-[var(--text-dim)]">SATELLITE VIEW</span>
        <div className="flex items-center gap-1">
          {(Object.keys(LAYER_LABELS) as LayerKey[]).map((key) => (
            <button
              key={key}
              onClick={() => toggleLayer(key)}
              className={`rounded px-2 py-1 text-[11px] font-medium transition-colors ${
                layers[key]
                  ? "bg-[var(--accent-bg)] text-[var(--accent-text)]"
                  : "bg-[var(--bg-panel)] text-[var(--text-faint)] hover:text-[var(--text-dim)]"
              }`}
            >
              {LAYER_LABELS[key]}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {!anyUploaded && (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-[var(--text-faint)]">Upload an image below to see it here.</p>
          </div>
        )}

        {anyUploaded && !latestResult && (
          <DefaultThumbnailGrid slots={slots} />
        )}

        {latestResult && output && isGroundingOutput(output) && layers.grounding && (
          <GroundingView slots={slots} activeModality={activeModality} output={output} />
        )}

        {latestResult &&
          output &&
          (isChangeDetectionOutput(output) || isChangeVQAOutput(output)) &&
          layers.change &&
          (() => {
            // Phase 12: if the Evidence Sufficiency Engine activated the SAR
            // before/after pair (execution trace says "SAR activation"),
            // show THAT pair in the slider — otherwise fall back to the
            // optical Before/After, same inference approach as
            // inferActiveModality() above (the API response doesn't
            // currently return which pair was used as a structured field).
            const usedSar = activeModality === "sar" && slots.sar_before && slots.sar_after;
            const beforeMeta = usedSar ? slots.sar_before : slots.before;
            const afterMeta = usedSar ? slots.sar_after : slots.after;
            if (!beforeMeta || !afterMeta) return null;
            return (
              <div className="space-y-2">
                <p className="text-xs text-[var(--text-faint)]">
                  Drag the slider — {isChangeDetectionOutput(output) ? output.description : output.answer}
                </p>
                {usedSar && (
                  <p className="text-[11px] font-medium text-[var(--accent)]">
                    ☁→🛰 Showing the SAR Before/After pair — optical was flagged insufficient (cloud cover).
                  </p>
                )}
                <BeforeAfterSlider
                  beforeSrc={previewUrl(beforeMeta.image_id)}
                  afterSrc={previewUrl(afterMeta.image_id)}
                  changeBox={output.change_map.largest_change_box}
                  changeBoxImageSize={{ width: output.change_map.width, height: output.change_map.height }}
                />
                <ChangeStatBar pct={output.change_map.changed_pixel_pct} level={output.change_map.change_level} />
              </div>
            );
          })()}

        {latestResult && output && isOpticalSarOutput(output) && slots.optical && slots.sar && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              {layers.optical && (
                <LabeledImage src={previewUrl(slots.optical.image_id)} label="OPTICAL" />
              )}
              {layers.sar && <LabeledImage src={previewUrl(slots.sar.image_id)} label="SAR" />}
            </div>
            <AgreementBar
              agreement={output.agreement_pct}
              opticalOnly={output.optical_only_pct}
              sarOnly={output.sar_only_pct}
              target={output.target}
            />
          </div>
        )}

        {/* Phase 14 (UI): Phase 13's real confidence_map_png, previously
            computed by the backend on every result but never rendered.
            Shown for every task type when the layer is on — generic
            because the field lives on every tool_output shape. */}
        {latestResult && layers.confidence && <ConfidenceMapCard output={output} />}

        {/* Fallback: VQA / captioning / blocked / no-render-path results
            still show whichever image was in play. */}
        {latestResult &&
          (!output ||
            (!isGroundingOutput(output) &&
              !isChangeDetectionOutput(output) &&
              !isChangeVQAOutput(output) &&
              !isOpticalSarOutput(output))) && (
            <DefaultThumbnailGrid slots={slots} activeModality={activeModality} />
          )}
      </div>
    </div>
  );
}

function DefaultThumbnailGrid({
  slots,
  activeModality,
}: {
  slots: SlotMap;
  activeModality?: "optical" | "sar" | null;
}) {
  const entries = (["optical", "sar", "before", "after", "sar_before", "sar_after"] as const).filter(
    (k) => slots[k],
  );
  if (entries.length === 0) {
    return <p className="text-sm text-[var(--text-faint)]">Upload an image below to see it here.</p>;
  }
  return (
    <div className="grid grid-cols-2 gap-3">
      {entries.map((key) => (
        <LabeledImage
          key={key}
          src={previewUrl(slots[key]!.image_id)}
          label={key.toUpperCase().replace("_", " ")}
          active={activeModality === key}
        />
      ))}
    </div>
  );
}

function LabeledImage({ src, label, active }: { src: string; label: string; active?: boolean }) {
  return (
    <div
      className={`relative overflow-hidden rounded-lg border bg-[var(--bg-app)] ${
        active ? "border-[var(--accent)] ring-1 ring-[var(--accent)]/50" : "border-[var(--border)]"
      }`}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={label} className="aspect-video w-full object-cover" />
      <span className="absolute left-2 top-2 rounded bg-[var(--bg-panel-soft)] px-2 py-0.5 text-[10px] font-medium tracking-wide text-[var(--text-dim)]">
        {label}
      </span>
      {active && (
        <span className="absolute right-2 top-2 rounded bg-[var(--accent-bg)] px-2 py-0.5 text-[10px] font-medium text-[var(--accent-text)]">
          ACTIVE
        </span>
      )}
    </div>
  );
}

function GroundingView({
  slots,
  activeModality,
  output,
}: {
  slots: SlotMap;
  activeModality: "optical" | "sar" | null;
  output: GroundingOutput;
}) {
  const meta = activeModality === "sar" ? slots.sar : slots.optical;
  const { box, found, image_width: imgW, image_height: imgH, phrase } = output;

  if (!meta) return <p className="text-sm text-[var(--text-faint)]">Source image not available for overlay.</p>;

  return (
    <div className="space-y-2">
      <div className="relative overflow-hidden rounded-lg border border-[var(--accent)]/60 bg-[var(--bg-app)]">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={previewUrl(meta.image_id)} alt={phrase} className="w-full" />
        {found && box && (
          <div
            className="absolute border-2 border-[var(--accent)] bg-[var(--accent)]/10"
            style={{
              left: `${(box.x_min / imgW) * 100}%`,
              top: `${(box.y_min / imgH) * 100}%`,
              width: `${((box.x_max - box.x_min) / imgW) * 100}%`,
              height: `${((box.y_max - box.y_min) / imgH) * 100}%`,
            }}
          >
            <span className="absolute -top-5 left-0 rounded bg-[var(--accent-bg)] px-1.5 py-0.5 text-[10px] text-[var(--accent-text)]">
              {phrase}
            </span>
          </div>
        )}
      </div>
      {!found && (
        <p className="text-xs text-amber-300">No matching region found for &ldquo;{phrase}&rdquo;.</p>
      )}
    </div>
  );
}

/** Phase 14 (UI): renders Phase 13's real `confidence_map_png`. This is
 * the "Confidence" entry in uisuggestions.docx's Analysis Layers panel —
 * previously a checkbox with no data behind it.
 *
 * Honesty contract (matches evidence/confidence_map.py — never blur this
 * distinction in the UI):
 *   - "change_diff" / "optical_sar_agreement": the map IS the real
 *     per-pixel evidence array the scalar stats were computed from.
 *   - "grounding_synthetic": a synthetic Gaussian heatmap centered on the
 *     returned box, scaled by match score — NOT a learned saliency map.
 *   - "flat_tint": a flat color tint for the answer-level confidence
 *     bucket — NOT spatial at all (VQA has no spatial output).
 */
function ConfidenceMapCard({ output }: { output: QueryResponse["tool_output"] }) {
  const cm = getConfidenceMap(output);

  if (!cm) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border)] bg-[var(--bg-panel-soft)] px-3 py-3 text-xs text-[var(--text-faint)]">
        No confidence map on this result.
      </div>
    );
  }

  const kindLabel =
    cm.confidence_map_kind === "change_diff"
      ? "Real per-pixel evidence — change diff magnitude"
      : cm.confidence_map_kind === "optical_sar_agreement"
        ? "Real per-pixel evidence — optical/SAR agreement mask"
        : cm.confidence_map_kind === "grounding_synthetic"
          ? "Synthetic heatmap — centered on detected box, not learned saliency"
          : cm.confidence_map_kind === "flat_tint"
            ? "Flat tint — answer-level confidence only, not spatial"
            : cm.confidence_map_kind;

  const isReal = cm.confidence_map_kind === "change_diff" || cm.confidence_map_kind === "optical_sar_agreement";
  const isFlat = cm.confidence_map_kind === "flat_tint";

  return (
    <div className="space-y-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-panel-soft)] p-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold tracking-wide text-[var(--text-faint)]">CONFIDENCE MAP</span>
        <span
          className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
            isReal
              ? "bg-emerald-900/60 text-emerald-300"
              : isFlat
                ? "bg-[var(--bg-elevated)] text-[var(--text-faint)]"
                : "bg-amber-900/60 text-amber-300"
          }`}
        >
          {isReal ? "MODEL EVIDENCE" : isFlat ? "NOT SPATIAL" : "SYNTHETIC AID"}
        </span>
      </div>

      {isFlat ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={`data:image/png;base64,${cm.confidence_map_png}`}
          alt="Confidence tint"
          className="h-8 w-full rounded object-cover"
        />
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={`data:image/png;base64,${cm.confidence_map_png}`}
          alt="Confidence map"
          className="aspect-video w-full rounded border border-[var(--border)] object-contain bg-[var(--bg-app)]"
        />
      )}

      <p className="text-[11px] text-[var(--text-dim)]">{kindLabel}</p>
      {cm.confidence_map_note && <p className="text-[11px] italic text-[var(--text-faint)]">{cm.confidence_map_note}</p>}
    </div>
  );
}

function ChangeStatBar({ pct, level }: { pct: number; level: "high" | "medium" | "low" }) {
  const color = level === "high" ? "bg-red-500" : level === "medium" ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-[var(--text-dim)]">
        <span>Changed area</span>
        <span>{pct}% — {level.toUpperCase()}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--bg-elevated)]">
        <div className={`h-full ${color}`} style={{ width: `${Math.min(100, pct)}%` }} />
      </div>
    </div>
  );
}

function AgreementBar({
  agreement,
  opticalOnly,
  sarOnly,
  target,
}: {
  agreement: number;
  opticalOnly: number;
  sarOnly: number;
  target: string;
}) {
  return (
    <div className="space-y-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-panel-soft)] p-3">
      <p className="text-xs text-[var(--text-dim)]">
        Evidence agreement — target: <span className="text-[var(--text)]">{target.replace("_", " ")}</span>
      </p>
      <div className="flex h-2 w-full overflow-hidden rounded-full bg-[var(--bg-elevated)]">
        <div className="h-full bg-[var(--accent)]" style={{ width: `${agreement}%` }} title={`Agreement ${agreement}%`} />
        <div
          className="h-full bg-emerald-600"
          style={{ width: `${opticalOnly}%` }}
          title={`Optical-only ${opticalOnly}%`}
        />
        <div className="h-full bg-amber-600" style={{ width: `${sarOnly}%` }} title={`SAR-only ${sarOnly}%`} />
      </div>
      <div className="flex flex-wrap gap-3 text-[11px] text-[var(--text-faint)]">
        <span><span className="text-[var(--accent)]">■</span> Both sensors {agreement}%</span>
        <span><span className="text-emerald-500">■</span> Optical only {opticalOnly}%</span>
        <span><span className="text-amber-500">■</span> SAR only {sarOnly}%</span>
      </div>
    </div>
  );
}
