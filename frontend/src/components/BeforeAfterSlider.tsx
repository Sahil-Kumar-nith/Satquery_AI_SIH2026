"use client";

import { useRef, useState } from "react";

interface BeforeAfterSliderProps {
  beforeSrc: string;
  afterSrc: string;
  /** Optional overlay box, in 0-1 fractional coordinates of the AFTER
   * image, drawn on top so the largest detected change region is visible
   * regardless of slider position. */
  changeBox?: { x_min: number; y_min: number; x_max: number; y_max: number } | null;
  changeBoxImageSize?: { width: number; height: number } | null;
}

/** Drag (or click) anywhere across the image to reveal BEFORE vs AFTER.
 * Pure CSS clip-path — no extra dependency. */
export default function BeforeAfterSlider({
  beforeSrc,
  afterSrc,
  changeBox,
  changeBoxImageSize,
}: BeforeAfterSliderProps) {
  const [pct, setPct] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  function updateFromClientX(clientX: number) {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const next = ((clientX - rect.left) / rect.width) * 100;
    setPct(Math.min(100, Math.max(0, next)));
  }

  const boxStyle =
    changeBox && changeBoxImageSize
      ? {
          left: `${(changeBox.x_min / changeBoxImageSize.width) * 100}%`,
          top: `${(changeBox.y_min / changeBoxImageSize.height) * 100}%`,
          width: `${((changeBox.x_max - changeBox.x_min) / changeBoxImageSize.width) * 100}%`,
          height: `${((changeBox.y_max - changeBox.y_min) / changeBoxImageSize.height) * 100}%`,
        }
      : null;

  return (
    <div className="space-y-1.5">
      <div
        ref={containerRef}
        className="relative aspect-video w-full select-none overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--bg-app)]"
        onMouseDown={(e) => {
          dragging.current = true;
          updateFromClientX(e.clientX);
        }}
        onMouseMove={(e) => {
          if (dragging.current) updateFromClientX(e.clientX);
        }}
        onMouseUp={() => (dragging.current = false)}
        onMouseLeave={() => (dragging.current = false)}
        onTouchStart={(e) => updateFromClientX(e.touches[0].clientX)}
        onTouchMove={(e) => updateFromClientX(e.touches[0].clientX)}
      >
        {/* AFTER — full size, base layer */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={afterSrc} alt="After" className="absolute inset-0 h-full w-full object-cover" draggable={false} />

        {/* BEFORE — same full size, revealed left-of-slider via clip-path
            (robust to layout timing, unlike a width-based clipped div). */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={beforeSrc}
          alt="Before"
          className="absolute inset-0 h-full w-full object-cover"
          style={{ clipPath: `inset(0 ${100 - pct}% 0 0)` }}
          draggable={false}
        />

        {/* Change region highlight (drawn relative to AFTER image) */}
        {boxStyle && (
          <div
            className="absolute border-2 border-red-500/90 bg-red-500/10"
            style={boxStyle}
            title="Largest detected change region"
          />
        )}

        {/* Divider handle */}
        <div className="absolute inset-y-0 w-0.5 bg-[var(--accent)]/90" style={{ left: `${pct}%` }}>
          <div className="absolute top-1/2 left-1/2 flex h-7 w-7 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-[var(--accent)] bg-[var(--bg-app)] text-[var(--accent)] shadow">
            <span className="text-[10px]">↔</span>
          </div>
        </div>

        <span className="absolute left-2 top-2 rounded bg-[var(--bg-panel-soft)] px-2 py-0.5 text-[10px] font-medium tracking-wide text-[var(--text-dim)]">
          BEFORE
        </span>
        <span className="absolute right-2 top-2 rounded bg-[var(--bg-panel-soft)] px-2 py-0.5 text-[10px] font-medium tracking-wide text-[var(--text-dim)]">
          AFTER
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={pct}
        onChange={(e) => setPct(Number(e.target.value))}
        className="w-full accent-[var(--accent)]"
        aria-label="Before/after slider"
      />
    </div>
  );
}
