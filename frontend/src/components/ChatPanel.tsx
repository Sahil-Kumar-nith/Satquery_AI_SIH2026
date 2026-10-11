"use client";

import { useEffect, useState } from "react";
import {
  ApiError,
  BoundingBox,
  ChangeMap,
  ConfidenceLevel,
  NlpLocationSource,
  QueryResponse,
  SourceRef,
  UploadResponse,
  getConfidenceMap,
  inferLocationSource,
  isAreaEstimateOutput,
  isChangeDetectionOutput,
  isChangeVQAOutput,
  isGroundingOutput,
  isOpticalSarOutput,
  isVQAOutput,
  runQuery,
  selectImagesForQuery,
} from "@/lib/api";

const CONFIDENCE_STYLES: Record<ConfidenceLevel, string> = {
  high: "bg-emerald-900/60 text-emerald-300",
  medium: "bg-amber-900/60 text-amber-300",
  insufficient: "bg-red-900/60 text-red-300",
};

const CHANGE_LEVEL_STYLES: Record<"high" | "medium" | "low", string> = {
  high: "bg-red-900/60 text-red-300",
  medium: "bg-amber-900/60 text-amber-300",
  low: "bg-emerald-900/60 text-emerald-300",
};

interface ChatEntry {
  id: string;
  query: string;
  usedImageIds?: string[];
  result?: QueryResponse;
  error?: string;
}

/** Phase 15: pull a plain-text "answer" out of whichever tool_output shape
 * this result actually has, for use as follow-up context on the NEXT
 * query (see lib/api.ts runQuery's `followup` param / agent/followup.py).
 * Falls back to the free-text `note` field so even a blocked/insufficient
 * result still gives the next turn something to reference. */
function extractAnswerText(result: QueryResponse): string | undefined {
  const output = result.tool_output as Record<string, unknown> | null;
  if (output && typeof output === "object") {
    if (typeof output.answer === "string") return output.answer;
    if (typeof output.description === "string") return output.description;
  }
  return result.note || undefined;
}

/** Phase 16: pull the previous turn's grounding box + which image it was
 * computed on, so follow-up hops 2 ("How much area?") and 3 ("Was this
 * visible in SAR?") have more than just free text to work from — see
 * backend/agent/followup.py and api/query.py's followup_intent() dispatch.
 * Returns nulls when the previous turn wasn't a (found) grounding result;
 * the backend handlers degrade gracefully and explain what's missing. */
function extractPreviousGrounding(result: QueryResponse): { box: BoundingBox | null; imageId: string | null } {
  if (isGroundingOutput(result.tool_output) && result.tool_output.found && result.tool_output.box) {
    return { box: result.tool_output.box, imageId: result.tool_output.image_id ?? null };
  }
  return { box: null, imageId: null };
}

const TASK_LABELS: Record<string, string> = {
  vqa: "VQA",
  captioning: "Captioning",
  grounding: "Grounding",
  change_detection: "Change Detection",
  change_vqa: "Change VQA",
  optical_sar: "Optical + SAR",
  area_estimate: "Follow-up: Area",
  sar_cross_check: "Follow-up: SAR Check",
};

const STATUS_STYLES: Record<QueryResponse["status"], string> = {
  routed: "bg-emerald-900/60 text-emerald-300",
  blocked: "bg-red-900/60 text-red-300",
  insufficient_evidence: "bg-amber-900/60 text-amber-300",
  conflicting_evidence: "bg-orange-900/60 text-orange-300",
};

const STATUS_LABELS: Record<QueryResponse["status"], string> = {
  routed: "ROUTED",
  blocked: "BLOCKED",
  insufficient_evidence: "INSUFFICIENT EVIDENCE",
  conflicting_evidence: "CONFLICTING EVIDENCE",
};

type UploadedSlots = Partial<
  Record<"optical" | "sar" | "before" | "after" | "sar_before" | "sar_after", UploadResponse>
>;

interface ChatPanelProps {
  uploaded: UploadedSlots;
  onResult: (result: QueryResponse) => void;
  /** Phase 12: a region description queued from the Map tab's draw tools
   * (click-point / rectangle / polygon selection). When set, it's appended
   * to the chat input once, then onInsertConsumed() clears it — lets the
   * Map tab hand off a selection without owning ChatPanel's input state. */
  pendingInsert?: string | null;
  onInsertConsumed?: () => void;
}

export default function ChatPanel({ uploaded, onResult, pendingInsert, onInsertConsumed }: ChatPanelProps) {
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!pendingInsert) return;
    setInput((prev) => (prev.trim() ? `${prev.trim()} ${pendingInsert}` : pendingInsert));
    onInsertConsumed?.();
    // Only re-run when pendingInsert itself changes — onInsertConsumed is
    // expected to be a stable-enough closure from the parent per render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingInsert]);

  const allImages = Object.values(uploaded).filter((m): m is UploadResponse => !!m);
  const hasAnyImage = allImages.length > 0;
  const canSubmit = input.trim().length > 0 && hasAnyImage && !busy;

  // Preview of which images this query will actually use — fixes the
  // known Phase 9 bug where every uploaded image was sent regardless of
  // relevance (see lib/api.ts selectImagesForQuery for details).
  const preview = input.trim() ? selectImagesForQuery(input, allImages) : null;

  async function handleSubmit() {
    if (!canSubmit) return;
    const query = input.trim();
    const { imageIds } = selectImagesForQuery(query, allImages);
    const id = `${Date.now()}`;
    setInput("");
    setBusy(true);
    setEntries((prev) => [...prev, { id, query, usedImageIds: imageIds }]);

    if (imageIds.length === 0) {
      setEntries((prev) =>
        prev.map((e) =>
          e.id === id
            ? {
                ...e,
                error:
                  "No uploaded image matched this question's task. Upload the relevant image(s) — see the hint below the upload bar.",
              }
            : e,
        ),
      );
      setBusy(false);
      return;
    }

    // Phase 15/16: follow-up context — hand the backend the previous turn's
    // query + answer text so a short referential query ("Where exactly?")
    // can lean on it instead of requiring the user to restate everything.
    // Phase 16 also carries the previous grounding box + which image it
    // came from, so hops 2 ("How much area?") and 3 ("Was this visible in
    // SAR?") have more than just text to work from. See agent/followup.py;
    // this is per-request, nothing is stored here beyond the current
    // render's entries list.
    const lastCompleted = [...entries].reverse().find((e) => e.result);
    const previousGrounding = lastCompleted?.result ? extractPreviousGrounding(lastCompleted.result) : null;
    const followup = lastCompleted?.result
      ? {
          previousQuery: lastCompleted.query,
          previousAnswer: extractAnswerText(lastCompleted.result),
          previousBox: previousGrounding?.box ?? null,
          previousImageId: previousGrounding?.imageId ?? null,
        }
      : undefined;

    try {
      const result = await runQuery(query, imageIds, followup);
      setEntries((prev) => prev.map((e) => (e.id === id ? { ...e, result } : e)));
      onResult(result);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Request failed — is the backend running?";
      setEntries((prev) => prev.map((e) => (e.id === id ? { ...e, error: message } : e)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
        {entries.length === 0 && (
          <p className="text-sm text-[var(--text-faint)]">
            {!hasAnyImage
              ? "Upload at least one image below, then ask a question."
              : 'Ask something, e.g. "What changed between these images?"'}
          </p>
        )}

        {entries.map((entry) => (
          <div key={entry.id} className="space-y-2">
            <div className="rounded-lg bg-[var(--bg-elevated)] px-3 py-2 text-sm text-[var(--text)]">{entry.query}</div>

            {entry.error && (
              <div className="rounded-lg border border-red-800 bg-red-950/40 px-3 py-2 text-sm text-red-300">
                {entry.error}
              </div>
            )}

            {entry.result && (
              <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-panel-soft)] px-3 py-2 text-sm space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded bg-[var(--accent-bg)] px-2 py-0.5 text-xs font-medium text-[var(--accent-text)]">
                    {TASK_LABELS[entry.result.task] ?? entry.result.task}
                  </span>
                  <span
                    className={`rounded px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[entry.result.status]} ${
                      entry.result.status === "insufficient_evidence" || entry.result.status === "conflicting_evidence"
                        ? "pulse-dot"
                        : ""
                    }`}
                  >
                    {STATUS_LABELS[entry.result.status]}
                  </span>
                  <span className="text-xs text-[var(--text-faint)]">{entry.result.ps_requirement}</span>
                </div>

                {/* Phase 14 (UI): surfaces Phase 13's NLP location/time
                    extraction + structured source attribution — both were
                    computed by the backend on every query but never
                    rendered by the frontend. */}
                <NlpExtractionRow result={entry.result} />
                <SourcesRow sources={entry.result.sources} />

                <ResultBody result={entry.result} />
                <ConfidenceMapHint output={entry.result.tool_output} />

                {entry.result.status === "conflicting_evidence" && (
                  <div className="rounded-lg border border-orange-800 bg-orange-950/30 px-3 py-2 text-xs text-orange-300">
                    ⚠ Two independently-computed signals disagree on this result — see the Verification
                    Trace line below. Treat the answer as unresolved, not confidently wrong or right.
                  </div>
                )}

                <ExecutionTrace lines={entry.result.execution_trace} />

                {entry.result.input_check.issues.length > 0 && (
                  <ul className="list-disc space-y-0.5 pl-4 text-xs text-amber-300">
                    {entry.result.input_check.issues.map((issue, i) => (
                      <li key={i}>{issue}</li>
                    ))}
                  </ul>
                )}

                <p className="text-xs italic text-[var(--text-faint)]">{entry.result.note}</p>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="border-t border-[var(--border)] p-3 space-y-2">
        {preview && preview.imageIds.length > 0 && (
          <p className="text-[11px] text-[var(--text-faint)]">
            Will use:{" "}
            <span className="text-[var(--accent)]">
              {TASK_LABELS[preview.task === "change" ? "change_detection" : preview.task] ?? preview.task}
            </span>
            {" · "}
            {preview.imageIds.length} image(s)
          </p>
        )}
        <div className="flex gap-2">
          <input
            className="flex-1 rounded-md border border-[var(--border)] bg-[var(--bg-panel)] px-3 py-2 text-sm text-[var(--text)] placeholder:text-[var(--text-faint)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            placeholder={!hasAnyImage ? "Upload an image first…" : "Ask a question about the imagery…"}
            value={input}
            disabled={!hasAnyImage || busy}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSubmit();
            }}
          />
          <button
            className="rounded-md bg-[var(--accent-strong)] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[var(--accent-strong-hover)] disabled:bg-[var(--bg-elevated)] disabled:text-[var(--text-faint)]"
            disabled={!canSubmit}
            onClick={handleSubmit}
          >
            {busy ? "Routing…" : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
}

/** Evidence-first result rendering — answer + confidence + spatial
 * evidence lead, per the Mission Control UI direction (uisuggestions):
 * this is an analysis tool, not a chatbot transcript. */
function ResultBody({ result }: { result: QueryResponse }) {
  const output = result.tool_output;

  if (isVQAOutput(output)) {
    return (
      <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-app)] px-3 py-2.5 space-y-1.5">
        <p className="text-base text-[var(--text)]">{output.answer}</p>
        <div className="flex flex-wrap items-center gap-2">
          <ConfidencePill level={output.confidence_level} pct={output.confidence * 100} />
          {!output.adapted && <BasePill />}
          <span className="text-xs text-[var(--text-faint)]">{output.latency_ms} ms</span>
        </div>
        <p className="text-xs italic text-[var(--text-faint)]">{output.confidence_note}</p>
      </div>
    );
  }

  if (isGroundingOutput(output)) {
    return (
      <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-app)] px-3 py-2.5 space-y-1.5">
        <p className="text-sm text-[var(--text)]">
          Phrase: <span className="font-medium">&ldquo;{output.phrase}&rdquo;</span>
        </p>
        {output.found && output.box ? (
          <p className="text-xs text-emerald-300">Region located — see Satellite View for the overlay.</p>
        ) : (
          <p className="text-xs text-amber-300">No matching region found for this phrase.</p>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <ConfidencePill level={output.confidence_level} pct={output.score * 100} />
          {!output.adapted && <BasePill />}
          <span className="text-xs text-[var(--text-faint)]">{output.latency_ms} ms</span>
        </div>
        <p className="text-xs italic text-[var(--text-faint)]">{output.confidence_note}</p>
      </div>
    );
  }

  if (isAreaEstimateOutput(output)) {
    return (
      <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-app)] px-3 py-2.5 space-y-1.5">
        <p className="text-sm text-[var(--text)]">
          Area of <span className="font-medium">&ldquo;{output.target_phrase}&rdquo;</span>
        </p>
        {output.area_km2 !== null ? (
          <p className="text-base text-[var(--text)]">
            ~<span className="font-semibold">{output.area_km2.toFixed(4)}</span> km&sup2;
          </p>
        ) : (
          <p className="text-sm text-amber-300">Area unavailable — see note below.</p>
        )}
        <p className="text-xs italic text-[var(--text-faint)]">{output.area_note}</p>
      </div>
    );
  }

  if (isChangeDetectionOutput(output)) {
    return (
      <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-app)] px-3 py-2.5 space-y-1.5">
        <p className="text-base text-[var(--text)]">{output.description}</p>
        <ChangeMapPills changeMap={output.change_map} />
        <VerificationBadge verification={output.verification} />
      </div>
    );
  }

  if (isChangeVQAOutput(output)) {
    return (
      <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-app)] px-3 py-2.5 space-y-1.5">
        <p className="text-base text-[var(--text)]">{output.answer}</p>
        <div className="flex flex-wrap items-center gap-2">
          <ConfidencePill level={output.vqa_confidence_level} pct={output.vqa_confidence * 100} />
          {!output.adapted && <BasePill />}
        </div>
        <ChangeMapPills changeMap={output.change_map} />
        <VerificationBadge verification={output.verification} />
      </div>
    );
  }

  if (isOpticalSarOutput(output)) {
    const confStyle =
      output.confidence_level === "high"
        ? "bg-emerald-900/60 text-emerald-300"
        : output.confidence_level === "medium"
          ? "bg-amber-900/60 text-amber-300"
          : "bg-red-900/60 text-red-300";
    return (
      <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-app)] px-3 py-2.5 space-y-1.5">
        <p className="text-base text-[var(--text)]">{output.answer}</p>
        <div className="flex flex-wrap items-center gap-2">
          <span className={`rounded px-2 py-0.5 text-xs font-medium ${confStyle}`}>
            {output.confidence_level.toUpperCase()}
          </span>
          <span className="text-xs text-[var(--text-faint)]">
            {output.agreement_pct}% sensor agreement · target: {output.target.replace("_", " ")}
          </span>
        </div>
        <VerificationBadge verification={output.verification} />
      </div>
    );
  }

  return null;
}

/** Phase 15: surfaces evidence/verification.py's second, independently-
 * computed check. Deliberately labeled "2 independent methods", never
 * "2 models" or "AI verified" — see that module's honesty contract. */
function VerificationBadge({
  verification,
}: {
  verification?: { agree: boolean; method_a: string; method_b: string; note: string };
}) {
  if (!verification) return null;
  const style = verification.agree ? "bg-emerald-900/60 text-emerald-300" : "bg-amber-900/60 text-amber-300";
  return (
    <div className="space-y-1">
      <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${style}`}>
        {verification.agree ? "2/2 independent methods agree" : "1/2 independent methods — methods disagree"}
      </span>
      <p className="text-xs italic text-[var(--text-faint)]">{verification.note}</p>
    </div>
  );
}

function ChangeMapPills({ changeMap }: { changeMap: ChangeMap }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className={`rounded px-2 py-0.5 text-xs font-medium ${CHANGE_LEVEL_STYLES[changeMap.change_level]}`}>
        {changeMap.changed_pixel_pct}% changed · {changeMap.change_level.toUpperCase()}
      </span>
      <span className="text-xs italic text-[var(--text-faint)]">{changeMap.change_note}</span>
    </div>
  );
}

function ConfidencePill({ level, pct }: { level: ConfidenceLevel; pct: number }) {
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-medium ${CONFIDENCE_STYLES[level]}`}>
      {level.toUpperCase()} · {pct.toFixed(0)}%
    </span>
  );
}

const LOCATION_TRUST_LABEL: Record<NlpLocationSource, string> = {
  gazetteer: "gazetteer match",
  heuristic: "heuristic guess",
  "gazetteer+heuristic": "gazetteer + heuristic",
  none: "no location text",
  unknown: "",
};

const LOCATION_TRUST_STYLE: Record<NlpLocationSource, string> = {
  gazetteer: "bg-emerald-900/60 text-emerald-300",
  heuristic: "bg-amber-900/60 text-amber-300",
  "gazetteer+heuristic": "bg-emerald-900/60 text-emerald-300",
  none: "bg-[var(--bg-elevated)] text-[var(--text-faint)]",
  unknown: "bg-[var(--bg-elevated)] text-[var(--text-faint)]",
};

/** Phase 14 (UI): renders agent/nlp_extract.py's output (Phase 13
 * backend). Rule-based pattern matching, NOT a trained NER/temporal
 * model — the trust badge (gazetteer vs heuristic) and the tooltip note
 * make that explicit rather than presenting every chip as equally
 * reliable. Renders nothing if the query had no detectable location/time
 * text (the map selection remains the authoritative source of scope). */
function NlpExtractionRow({ result }: { result: QueryResponse }) {
  const { extracted_locations, extracted_time_expressions, nlp_extraction_note, execution_trace } = result;
  if (extracted_locations.length === 0 && extracted_time_expressions.length === 0) return null;

  const trust = inferLocationSource(execution_trace);

  return (
    <div className="flex flex-wrap items-center gap-1.5" title={nlp_extraction_note}>
      {extracted_locations.map((loc) => (
        <span key={loc} className="rounded bg-[var(--bg-elevated)] px-2 py-0.5 text-[11px] text-[var(--text-dim)]">
          📍 {loc}
        </span>
      ))}
      {extracted_time_expressions.map((t) => (
        <span key={t} className="rounded bg-[var(--bg-elevated)] px-2 py-0.5 text-[11px] text-[var(--text-dim)]">
          🗓 {t}
        </span>
      ))}
      {extracted_locations.length > 0 && LOCATION_TRUST_LABEL[trust] && (
        <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${LOCATION_TRUST_STYLE[trust]}`}>
          {LOCATION_TRUST_LABEL[trust]}
        </span>
      )}
    </div>
  );
}

/** Phase 14 (UI): renders api/query.py's `sources` field (Phase 13
 * backend) — machine-readable evidence attribution, separate from the
 * free-text `note`. Answers "which uploaded file(s) is this evidence
 * based on?" at a glance. */
function SourcesRow({ sources }: { sources: SourceRef[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-[var(--text-faint)]">
      <span className="font-medium tracking-wide">SOURCES</span>
      {sources.map((s) => (
        <span
          key={s.image_id}
          className="rounded border border-[var(--border)] px-1.5 py-0.5 text-[var(--text-dim)]"
          title={s.image_id}
        >
          {s.filename} · {s.modality}/{s.role}
        </span>
      ))}
    </div>
  );
}

/** Phase 14 (UI): points to the Satellite View's new Confidence layer
 * (see SatelliteView.tsx ConfidenceMapCard) instead of duplicating the
 * image render here — the chat panel stays text/evidence-summary only. */
function ConfidenceMapHint({ output }: { output: QueryResponse["tool_output"] }) {
  const cm = getConfidenceMap(output);
  if (!cm) return null;
  const isSpatial = cm.confidence_map_kind !== "flat_tint";
  return (
    <p className="text-[11px] text-[var(--text-faint)]">
      {isSpatial ? "🗺" : "🎨"} Confidence {isSpatial ? "map" : "tint"} available — toggle{" "}
      <span className="text-[var(--accent)]">Confidence</span> in Satellite View.
    </p>
  );
}

function BasePill() {
  return (
    <span className="rounded bg-[var(--bg-elevated)] px-2 py-0.5 text-xs text-[var(--text-dim)]">base model — not yet RS-adapted</span>
  );
}

/** Numbered agent-step trace — matches the "AGENT EXECUTION 01 02 03…"
 * treatment from uisuggestions.docx rather than a raw scrolling log. */
function ExecutionTrace({ lines }: { lines: string[] }) {
  return (
    <div className="rounded bg-[var(--bg-app)] px-2 py-2 font-mono text-xs text-[var(--text-dim)]">
      <div className="mb-1 text-[10px] font-semibold tracking-wide text-[var(--text-faint)]">EXECUTION TRACE</div>
      <div className="space-y-0.5">
        {lines.map((line, i) => (
          <div key={i} className="flex gap-2">
            <span className="w-5 shrink-0 text-right text-[var(--text-faint)]">{String(i + 1).padStart(2, "0")}</span>
            <span
              className={
                line.startsWith("  ✗")
                  ? "text-red-400"
                  : line.toLowerCase().includes("conflicting")
                    ? "text-orange-400"
                    : line.toLowerCase().includes("insufficient") || line.toLowerCase().includes("warning")
                      ? "text-amber-400"
                      : ""
              }
            >
              {line}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
