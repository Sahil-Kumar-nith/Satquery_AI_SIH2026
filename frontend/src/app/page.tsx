"use client";

import { useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import GeoMap from "@/components/GeoMap";
import MenuBar, { TabId } from "@/components/MenuBar";
import SatelliteView from "@/components/SatelliteView";
import UploadSlot from "@/components/UploadSlot";
import BenchmarksView from "@/components/views/BenchmarksView";
import ModelRegistryView from "@/components/views/ModelRegistryView";
import ReportView from "@/components/views/ReportView";
import TerrainModal from "@/components/terrain/TerrainModal";
import { ImageRole, Modality, QueryResponse, UploadResponse } from "@/lib/api";

type SlotKey = "optical" | "sar" | "before" | "after" | "sar_before" | "sar_after";

const SLOTS: { key: SlotKey; label: string; modality: Modality; role: ImageRole }[] = [
  { key: "optical", label: "Optical", modality: "optical", role: "single" },
  { key: "sar", label: "SAR", modality: "sar", role: "single" },
  { key: "before", label: "Before", modality: "optical", role: "before" },
  { key: "after", label: "After", modality: "optical", role: "after" },
  // Phase 12: SAR-before/SAR-after — closes the gap PHASE11.md flagged
  // ("change tasks have no complementary SAR pair to fail over to").
  // Optional: change detection works fine with just Optical Before/After;
  // these two are only needed to demo the cloud→SAR interception on a
  // change query, mirroring the Optical/SAR slots' role for single-image
  // queries.
  { key: "sar_before", label: "SAR Before", modality: "sar", role: "before" },
  { key: "sar_after", label: "SAR After", modality: "sar", role: "after" },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabId>("dashboard");
  const [uploaded, setUploaded] = useState<Partial<Record<SlotKey, UploadResponse>>>({});
  const [latestResult, setLatestResult] = useState<QueryResponse | null>(null);
  // Phase 12: a region description from the Map tab's draw tools, queued
  // up to be inserted into the chat input the next time ChatPanel mounts
  // it. Lifted up here (rather than living inside GeoMap or ChatPanel
  // alone) because the two live in different tabs.
  const [pendingInsert, setPendingInsert] = useState<string | null>(null);
  // Phase: 3D Terrain / Hologram — optional visualization aid, not part of
  // the PS-mandated pipeline. See TerrainModal for the judge-facing
  // disclosure (generic monocular depth, not an RS-adapted model).
  const [terrainOpen, setTerrainOpen] = useState(false);
  // An image handed straight from the Map tab's "Snip" tool to the 3D
  // Terrain viewer, bypassing the upload/slot flow entirely.
  const [terrainExternalImage, setTerrainExternalImage] = useState<{ file: File; label: string } | null>(null);

  const hasAnyImage = Object.keys(uploaded).length > 0;

  const readiness = [
    { label: "OPTICAL", ready: !!uploaded.optical },
    { label: "SAR", ready: !!uploaded.sar },
    { label: "GEO", ready: Object.values(uploaded).some((m) => m?.is_geospatial) },
    { label: "AI", ready: hasAnyImage },
  ];

  function handleRegionSelected(description: string) {
    setPendingInsert(description);
    setActiveTab("dashboard");
  }

  function handleOpenTerrainWithFile(file: File, label: string) {
    setTerrainExternalImage({ file, label });
    setTerrainOpen(true);
  }

  return (
    <div className="flex h-screen flex-col bg-[var(--bg-app)] text-[var(--text)]">
      <MenuBar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        readiness={readiness}
        onOpenTerrain={() => setTerrainOpen(true)}
        terrainAvailable={hasAnyImage}
      />

      <TerrainModal
        open={terrainOpen}
        onClose={() => {
          setTerrainOpen(false);
          setTerrainExternalImage(null);
        }}
        slots={uploaded}
        externalImage={terrainExternalImage}
      />

      {activeTab === "dashboard" && (
        <>
          {/* Main split layout: Satellite View (left) + Chat (right) */}
          <main className="grid flex-1 grid-cols-1 overflow-hidden md:grid-cols-[1.3fr_1fr]">
            <section className="flex flex-col overflow-hidden border-r border-[var(--border)] bg-[var(--bg-panel-soft)]">
              <SatelliteView
                slots={{
                  optical: uploaded.optical,
                  sar: uploaded.sar,
                  before: uploaded.before,
                  after: uploaded.after,
                  sar_before: uploaded.sar_before,
                  sar_after: uploaded.sar_after,
                }}
                latestResult={latestResult}
              />
            </section>

            <section className="flex flex-col overflow-hidden">
              <ChatPanel
                uploaded={uploaded}
                onResult={setLatestResult}
                pendingInsert={pendingInsert}
                onInsertConsumed={() => setPendingInsert(null)}
              />
            </section>
          </main>

          {/* Upload toolbar — POST /api/upload */}
          <footer className="flex flex-col gap-1 border-t border-[var(--border)] px-6 py-3 text-sm">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-[var(--text-faint)]">Upload</span>
              {SLOTS.map((slot) => (
                <UploadSlot
                  key={slot.key}
                  label={slot.label}
                  modality={slot.modality}
                  role={slot.role}
                  onUploaded={(meta) => setUploaded((prev) => ({ ...prev, [slot.key]: meta }))}
                  onCleared={() =>
                    setUploaded((prev) => {
                      const next = { ...prev };
                      delete next[slot.key];
                      return next;
                    })
                  }
                />
              ))}
              <span className="ml-auto text-xs text-[var(--text-faint)]">
                {hasAnyImage
                  ? `${Object.keys(uploaded).length} image(s) ready — ask a question →`
                  : "Upload an image to begin"}
              </span>
            </div>
            <p className="text-xs text-[var(--text-faint)]">
              <span className="text-[var(--text-dim)]">Optical / SAR</span> — for single-image questions (VQA,
              grounding, captioning). <span className="text-[var(--text-dim)]">Before / After</span> — for change
              questions (&ldquo;what changed&rdquo;, &ldquo;did it increase&rdquo;). The chat panel sends only
              the images relevant to your question automatically — no need to clear unrelated slots first.
              <br />
              <span className="text-[var(--text-dim)]">Cloud → SAR demo (single-image):</span> upload Optical + SAR
              of the same scene, then ask a plain question — a cloudy optical image is automatically swapped for
              the SAR one.{" "}
              <span className="text-[var(--text-dim)]">Cloud → SAR demo (change):</span> additionally upload
              SAR&nbsp;Before + SAR&nbsp;After — a cloudy Before/After optical pair fails over to the SAR pair the
              same way.
            </p>
          </footer>
        </>
      )}

      {activeTab === "map" && (
        <main className="grid flex-1 grid-cols-1 overflow-hidden md:grid-cols-[1.3fr_1fr]">
          <section className="flex flex-col overflow-hidden border-r border-[var(--border)] bg-[var(--bg-panel-soft)]">
            <GeoMap
              slots={{
                optical: uploaded.optical,
                sar: uploaded.sar,
                before: uploaded.before,
                after: uploaded.after,
              }}
              onRegionSelected={handleRegionSelected}
              onImageUploaded={(meta, slot) => setUploaded((prev) => ({ ...prev, [slot]: meta }))}
              onOpenTerrainWithFile={handleOpenTerrainWithFile}
            />
          </section>
          <section className="flex flex-col overflow-hidden">
            <ChatPanel
              uploaded={uploaded}
              onResult={setLatestResult}
              pendingInsert={pendingInsert}
              onInsertConsumed={() => setPendingInsert(null)}
            />
          </section>
        </main>
      )}

      {activeTab === "registry" && (
        <main className="flex-1 overflow-y-auto">
          <ModelRegistryView />
        </main>
      )}

      {activeTab === "benchmarks" && (
        <main className="flex-1 overflow-y-auto">
          <BenchmarksView />
        </main>
      )}

      {activeTab === "report" && (
        <main className="flex-1 overflow-y-auto">
          <ReportView latestResult={latestResult} />
        </main>
      )}
    </div>
  );
}
