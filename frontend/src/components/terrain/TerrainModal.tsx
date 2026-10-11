"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { previewUrl, UploadResponse } from "@/lib/api";
import { processDepthImage, DepthResult } from "@/lib/terrain/depthClient";
import { buildTerrainMath, type Pick, type TerrainMath } from "@/lib/terrain/measure";

// Terrain3D pulls in react-three-fiber/three, which touches WebGL/canvas
// APIs that don't exist during SSR — load it client-side only.
const Terrain3D = dynamic(() => import("./Terrain3D"), { ssr: false });

type SlotKey = "optical" | "sar" | "before" | "after" | "sar_before" | "sar_after";

const SLOT_LABELS: Record<SlotKey, string> = {
  optical: "Optical",
  sar: "SAR",
  before: "Before",
  after: "After",
  sar_before: "SAR Before",
  sar_after: "SAR After",
};

interface TerrainModalProps {
  open: boolean;
  onClose: () => void;
  slots: Partial<Record<SlotKey, UploadResponse>>;
  initialSlot?: SlotKey;
  /** From the Map tab's "Snip satellite image" -> "Open in 3D Terrain"
   * hand-off: an image that was never uploaded to a slot at all. When set,
   * this takes priority over slot selection. */
  externalImage?: { file: File; label: string } | null;
}

export default function TerrainModal({ open, onClose, slots, initialSlot, externalImage }: TerrainModalProps) {
  const availableSlots = useMemo(
    () => (Object.keys(slots) as SlotKey[]).filter((k) => slots[k]),
    [slots],
  );

  const [slotKey, setSlotKey] = useState<SlotKey | null>(initialSlot ?? availableSlots[0] ?? null);
  const [hologram, setHologram] = useState(true);
  const [heightScale, setHeightScale] = useState(0.5);
  const [resolution, setResolution] = useState(150);
  const [tool, setTool] = useState<"probe" | "distance" | "profile" | "slope">("probe");
  const [picks, setPicks] = useState<Pick[]>([]);
  const [profile, setProfile] = useState<number[] | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [depthResult, setDepthResult] = useState<DepthResult | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);

  useEffect(() => {
    if (!open) return;
    if (!slotKey && availableSlots.length > 0) setSlotKey(availableSlots[0]);
  }, [open, availableSlots, slotKey]);

  // Fetch the selected slot's raster preview (or use the externally-passed
  // snip file) and run monocular depth on it.
  useEffect(() => {
    if (!open) return;
    if (!externalImage && !slotKey) return;

    let cancelled = false;
    setLoading(true);
    setError(null);
    setDepthResult(null);
    setPicks([]);
    setProfile(null);

    (async () => {
      try {
        let file: File;
        if (externalImage) {
          file = externalImage.file;
        } else {
          const meta = slots[slotKey as SlotKey];
          if (!meta) return;
          const res = await fetch(previewUrl(meta.image_id));
          if (!res.ok) throw new Error(`Could not load preview image (${res.status})`);
          const blob = await res.blob();
          file = new File([blob], `${meta.image_id}.png`, { type: blob.type || "image/png" });
        }
        if (cancelled) return;
        setImageFile(file);

        const result = await processDepthImage(file);
        if (cancelled) return;
        setDepthResult(result);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "3D terrain generation failed");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, slotKey, externalImage]);

  const tm: TerrainMath | null = useMemo(() => {
    if (!depthResult) return null;
    return buildTerrainMath(depthResult.heightmap, depthResult.width, depthResult.height, depthResult.mode, depthResult.geo);
  }, [depthResult]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-black/80 backdrop-blur-sm">
      <div className="flex items-center gap-3 border-b border-[var(--border)] bg-[var(--bg-panel)] px-4 py-2.5">
        <span className="text-sm font-semibold text-[var(--text)]">🧊 3D Terrain View</span>
        <span className="rounded bg-[var(--accent-bg)] px-2 py-0.5 text-[11px] font-medium text-[var(--accent-text)]">
          Prototype visualization — not a PS-scored model
        </span>

        <div className="ml-6 flex items-center gap-1">
          {externalImage ? (
            <span className="rounded bg-[var(--accent-bg)] px-2.5 py-1 text-xs font-medium text-[var(--accent-text)]">
              📎 {externalImage.label}
            </span>
          ) : (
            <>
              {availableSlots.length === 0 && (
                <span className="text-xs text-[var(--text-faint)]">Upload an image first</span>
              )}
              {availableSlots.map((k) => (
                <button
                  key={k}
                  onClick={() => setSlotKey(k)}
                  className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                    slotKey === k
                      ? "bg-[var(--accent-bg)] text-[var(--accent-text)]"
                      : "text-[var(--text-dim)] hover:bg-[var(--bg-elevated)]"
                  }`}
                >
                  {SLOT_LABELS[k]}
                </button>
              ))}
            </>
          )}
        </div>

        <div className="flex-1" />

        <label className="flex items-center gap-1.5 text-xs text-[var(--text-dim)]">
          <input type="checkbox" checked={hologram} onChange={(e) => setHologram(e.target.checked)} />
          Hologram mode
        </label>

        <label className="flex items-center gap-1.5 text-xs text-[var(--text-dim)]">
          Height
          <input
            type="range"
            min={0.1}
            max={1.5}
            step={0.05}
            value={heightScale}
            onChange={(e) => setHeightScale(parseFloat(e.target.value))}
          />
        </label>

        <div className="flex items-center gap-1">
          {(["probe", "distance", "profile", "slope"] as const).map((t) => (
            <button
              key={t}
              onClick={() => {
                setTool(t);
                setPicks([]);
                setProfile(null);
              }}
              className={`rounded px-2 py-1 text-[11px] capitalize ${
                tool === t ? "bg-[var(--accent-bg)] text-[var(--accent-text)]" : "text-[var(--text-faint)] hover:bg-[var(--bg-elevated)]"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <button
          onClick={onClose}
          className="rounded border border-[var(--border)] px-2.5 py-1 text-xs text-[var(--text-dim)] hover:bg-[var(--bg-elevated)]"
        >
          ✕ Close
        </button>
      </div>

      <div className="relative flex-1">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-[var(--text-dim)]">
            Running monocular depth estimation…
          </div>
        )}

        {error && !loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-6 text-center">
            <span className="text-sm text-red-400">⚠ {error}</span>
            <span className="text-xs text-[var(--text-faint)]">
              Terrain worker must be running separately: <code>cd terrain-worker &amp;&amp; uvicorn main:app --port 8010</code>
            </span>
          </div>
        )}

        {!loading && !error && depthResult && imageFile && tm && (
          <Terrain3D
            depthResult={depthResult}
            imageFile={imageFile}
            hologram={hologram}
            heightScale={heightScale}
            resolution={resolution}
            tm={tm}
            picks={picks}
            setPicks={setPicks}
            setProfile={setProfile}
            tool={tool}
            objects={[]}
          />
        )}
      </div>

      <div className="border-t border-[var(--border)] bg-[var(--bg-panel)] px-4 py-2 text-[11px] text-[var(--text-faint)]">
        {depthResult ? (
          <>
            Mode: <span className="text-[var(--text-dim)]">{depthResult.mode === "absolute" ? "absolute (GCP-calibrated)" : "relative depth — unitless, illustrative only"}</span>
            {" · "}Inference: {depthResult.inference_ms.toFixed(0)} ms
            {" · "}Model: Depth-Anything V2 (generic monocular depth, not RS-adapted)
          </>
        ) : (
          "Elevation shown here is a generic monocular depth estimate, not calibrated real-world elevation unless ground-control points are supplied. This view is a visualization aid only — it does not feed into VQA, change detection, or optical–SAR evidence."
        )}
      </div>
    </div>
  );
}
