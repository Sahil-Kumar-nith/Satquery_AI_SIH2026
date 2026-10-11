# 3D Terrain / Hologram — Integration Notes

## What this is
A "🧊 3D Terrain" button in the top bar that takes whichever image is
currently loaded (Optical / SAR / Before / After) and renders it as an
interactive 3D mesh, with an optional glowing wireframe "hologram" mode.
It runs Depth-Anything V2 (a generic monocular depth model, from the
uploaded DepthWizard project) as a **third, isolated service**.

## Classification against the ISRO PS — read this before the demo

| Check | Answer |
|---|---|
| Satisfies an official PS requirement? | **No.** VQA, captioning/grounding, change analysis, optical+SAR fusion, and agentic orchestration are the mandatory tasks. Monocular 3D terrain visualization is not one of them. |
| Improves evaluation score? | No — it isn't in the benchmarked task set. |
| Strengthens the Adaptive Evidence Acquisition story? | No. Don't fold it into the evidence/cloud-interception narrative — it's a separate, unrelated capability. |
| Improves the live demo? | **Possibly**, purely as a "wow" visual aside, if it doesn't compete for stage time with the hero SAR-interception demo. |

**Priority: P2 (optional).** Keep it fully decoupled from `agent/`,
`evidence/`, and the execution trace. If a judge asks "is this part of
your agentic pipeline?", the honest answer is no — it's a standalone
visualization layered on top of an already-uploaded image, and it should
be presented that way. Do not claim the heights are calibrated real-world
elevation unless you've actually supplied ground-control points (GCPs);
by default this is Depth-Anything V2's *relative* depth, which is
illustrative only.

## Architecture

```
Next.js (3000)
  ├─ existing SatQuery UI  → FastAPI backend (8000)   [PS-mandated pipeline — untouched]
  └─ TerrainModal          → /api/process-depth (Next.js proxy)
                                   │
                                   ▼
                        terrain-worker (FastAPI, 8010)
                                   │
                                   ▼
                     Depth-Anything V2 (Hugging Face)
```

Kept on its own port (8010) and its own process deliberately — a 3D/ML
dependency that fails to load (missing GPU, cold-start weight download,
`rasterio` not installed) must never be able to take down the PS-mandated
backend on 8000. If `terrain-worker` isn't running, the button still
works — the modal opens and shows a clear "worker unreachable" message
with the exact command to start it, instead of erroring the whole app.

## Files added / changed

**New:**
- `terrain-worker/` — copied from DepthWizard's `ml-worker/`, port changed 8000 → 8010.
- `frontend/src/lib/terrain/depthClient.ts` — trimmed client (just `processDepthImage`).
- `frontend/src/lib/terrain/measure.ts` — terrain math (heights, slope, profile).
- `frontend/src/components/terrain/Terrain3D.tsx`, `TerrainMesh.tsx` — 3D viewer (React Three Fiber).
- `frontend/src/components/terrain/TerrainModal.tsx` — glue: picks the active
  uploaded slot, fetches its preview PNG, calls the depth model, renders `Terrain3D`.
- `frontend/src/app/api/process-depth/route.ts` — Next.js proxy to `terrain-worker`.

**Changed:**
- `frontend/src/components/MenuBar.tsx` — added the "🧊 3D Terrain" button (desktop + mobile nav).
- `frontend/src/app/page.tsx` — added `terrainOpen` state, mounts `<TerrainModal>`.
- `frontend/package.json` — added `three`, `@react-three/fiber`, `@react-three/drei`.

**Deliberately not carried over** (cut for scope — add back later only if
there's spare time after all P0/P1 PS items are solid):
- `validate-dsm` (DSM accuracy validation against a reference raster) — no reference DSMs in this project.
- `detect-objects` / `ObjectFinder.tsx` — a second, separate object-detection model; would duplicate the PS's grounding requirement without adding score.
- `spectral.py` router — spectral-index endpoints, unrelated to this feature.
- GCP calibration UI — without it every result is relative depth (fine, it's disclosed in the footer).

## Running it

```bash
# Terminal 1 — existing PS backend (unchanged)
cd backend && uvicorn main:app --reload --port 8000

# Terminal 2 — terrain worker (new, isolated)
cd terrain-worker
pip install -r requirements.txt --break-system-packages
# torch: install the correct build for your machine first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
uvicorn main:app --reload --port 8010

# Terminal 3 — frontend
cd frontend
npm install
npm run dev
```

First click of "🧊 3D Terrain" after starting `terrain-worker` will be slow
(~seconds) while the Depth-Anything V2 weights download/load once; after
that it's cached in the worker process.

## Known limitations to disclose, not hide (3D Terrain)
- Depth-Anything V2 is a **general-purpose** depth model, not remote-sensing
  fine-tuned. It has no concept of GSD (ground sample distance), so absolute
  scale is meaningless without GCPs.
- Height values are visually exaggerated on the mesh (`heightScale` slider)
  for legibility — this is explicitly a *display* transform; `measure.ts`
  re-samples the raw (non-exaggerated) heightmap for any on-click readouts.
- GeoTIFF inputs are decoded to a plain RGB preview before depth estimation
  (bands 1-3), the same way the rest of SatQuery previews rasters — no new
  geospatial handling was added here.

## Addendum — live satellite basemap + Snip tool (Map tab)

The Map tab's basemap now has a **🛰 Satellite** mode alongside the existing
Street (OSM) view, plus a **Snip** tool. Read this before you present it —
it looks similar to zoom.earth but is a genuinely different, and in one way
better, data source.

### Why not just embed zoom.earth?
- Its imagery layers (EUMETSAT/Meteosat, INSAT, etc.) are a **third-party
  product** under its own terms — scraping or iframing it for pixel-level
  reuse (the "snip into an image file" requirement) isn't something we
  control or can license cleanly in a hackathon.
- It has no server-side endpoint we can call to get "one flat PNG for this
  exact bbox" — you'd be stuck screen-scraping a live map, which is fragile
  and likely against its ToS.

### What we used instead: NASA GIBS
[NASA's Global Imagery Browse Services](https://nasa-gibs.github.io/gibs-api-docs/) —
public domain, no API key, explicitly built for this kind of reuse.

- **Basemap tiles** (`GeoMap.tsx`): `VIIRS_SNPP_CorrectedReflectance_TrueColor`
  via the standard GIBS WMTS Web-Mercator tile template
  (`.../wmts/epsg3857/best/{layer}/default/{time}/{tileMatrixSet}/{z}/{y}/{x}.jpg`),
  same pattern as NASA's own published Leaflet examples.
- **Snip** (`/api/gibs-snapshot`): a server-side call to the GIBS **WMS**
  `GetMap` endpoint for the drawn rectangle's exact bbox + selected date,
  returning one PNG. Proxied server-side (not called from the browser) so
  there's no CORS dependency on GIBS's WMS headers.

### The honest trade-off — say this to judges, don't let them assume otherwise
GIBS true-color is a **daily composite** (one "best" cloud-free-ish pass
per calendar day), not zoom.earth's continuously updating geostationary
loop. You cannot scrub through *hours* of a single day. What you get
instead, and what zoom.earth can't give you: **any historical date**,
which means you can snip a real satellite pass from an actual past flood
event for the hero demo, not just whatever the sky looks like right now.

### The full loop this enables
1. Map tab → 🛰 Satellite → pick a date → draw a rectangle over the region of interest.
2. "✂ Snip satellite image" → fetches that exact bbox as a PNG.
3. Either:
   - **"Use in Chat"** → uploads it into any slot (Optical/SAR/Before/After) exactly like a manual file upload, then go ask a question in Dashboard, or
   - **"🧊 Open in 3D Terrain"** → sends it straight to the terrain viewer without uploading it anywhere first.

### Files touched
- `frontend/src/app/api/gibs-snapshot/route.ts` — new, server-side GIBS WMS proxy.
- `frontend/src/components/GeoMap.tsx` — rewritten: basemap switcher, date scrubber, snip flow.
- `frontend/src/components/terrain/TerrainModal.tsx` — added `externalImage` prop so a snipped file can skip the upload step entirely.
- `frontend/src/app/page.tsx` — wires `onImageUploaded` / `onOpenTerrainWithFile` from the Map tab.

### Known limitations
- GIBS's coarsest global tile level here is zoom 9 (`GoogleMapsCompatible_Level9`) — fine for country/state-scale framing (matches the reference screenshot's zoom level), but don't expect building-level street-view zoom on the satellite layer.
- No time-of-day slider — only a date. If you need the zoom.earth-style hour scrubber for the pitch, that requires a geostationary source (e.g. INSAT/Meteosat) with its own access terms — flag that as a separate, later integration, not something to promise for this hackathon.
- The snipped PNG carries no embedded geo-metadata (by design — it's a flat export), so it uploads as `is_geospatial: false`, same as any other plain image. The real bbox is known to the browser at snip time (shown in the "Selected: …" line) even though it isn't embedded in the file itself.

