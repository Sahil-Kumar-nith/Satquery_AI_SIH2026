"use client";

import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";

import { useEffect, useRef, useState } from "react";
import type { Map as LeafletMap, FeatureGroup as LeafletFeatureGroup, TileLayer as LeafletTileLayer } from "leaflet";
import { UploadResponse, Modality, previewUrl, uploadImage } from "@/lib/api";

/**
 * Phase 12 — real Leaflet map with region-selection draw tools.
 *
 * Phase 14 — adds a second, date-selectable basemap: NASA GIBS true-color
 * satellite imagery (VIIRS/Suomi NPP), as a free/legit stand-in for the
 * "live satellite map" reference (zoom.earth). Read this before demoing:
 *
 * - This is a DAILY composite per date, not a continuously-updating
 *   geostationary loop like zoom.earth's Meteosat/INSAT layers. Say "daily
 *   satellite pass, pick any date" in the pitch — not "live like zoom.earth".
 * - It's public-domain NASA data (no key, no scraping, no ToS risk),
 *   unlike embedding a third-party product's proprietary tiles would be.
 * - Precisely because it's date-addressable, you can pick a REAL past
 *   flood date for the Assam hero demo — something zoom.earth (near-real-
 *   time only) can't give you.
 *
 * Snip tool: draw a rectangle -> "Snip satellite image" fetches ONE flat
 * PNG for exactly that bbox+date from /api/gibs-snapshot (server-side
 * proxy, see that route for why) -> user can upload it into a slot
 * (Optical/SAR/Before/After) or send it straight to the 3D Terrain viewer.
 */

const ASSAM_DEMO_CENTER: [number, number] = [26.2006, 92.9376];
const ASSAM_DEMO_ZOOM = 8;

const GIBS_LAYER = "VIIRS_SNPP_CorrectedReflectance_TrueColor";
const GIBS_TILE_MATRIX_SET = "GoogleMapsCompatible_Level9";
const GIBS_MAX_ZOOM = 9;

type SlotMap = Partial<Record<"optical" | "sar" | "before" | "after", UploadResponse>>;
type BasemapMode = "street" | "satellite";

interface GeoMapProps {
  slots: SlotMap;
  onRegionSelected: (description: string) => void;
  /** Slot the snipped image gets uploaded into. */
  onImageUploaded?: (meta: UploadResponse, slot: "optical" | "sar" | "before" | "after") => void;
  /** Hand a snipped image straight to the 3D Terrain viewer without uploading it first. */
  onOpenTerrainWithFile?: (file: File, label: string) => void;
}

function isPlainLatLon(meta: UploadResponse | undefined): meta is UploadResponse & { bounds: [number, number, number, number] } {
  if (!meta || !meta.is_geospatial || !meta.bounds || !meta.crs) return false;
  const crs = meta.crs.toUpperCase();
  const isFourNineteenTwo = crs.includes("4326") || crs.includes("WGS 84") || crs.includes("WGS84");
  const [minx, miny, maxx, maxy] = meta.bounds;
  const looksLikeLatLon = minx >= -180 && maxx <= 180 && miny >= -90 && maxy <= 90;
  return isFourNineteenTwo && looksLikeLatLon;
}

function fmt(n: number): string {
  return n.toFixed(4);
}

/** GIBS "best" composite typically lags ~1 day for full global coverage. */
function defaultGibsDate(): string {
  const d = new Date(Date.now() - 24 * 60 * 60 * 1000);
  return d.toISOString().slice(0, 10);
}

function shiftDate(dateStr: string, days: number): string {
  const d = new Date(dateStr + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

interface RectBounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

export default function GeoMap({ slots, onRegionSelected, onImageUploaded, onOpenTerrainWithFile }: GeoMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const drawnRef = useRef<LeafletFeatureGroup | null>(null);
  const streetLayerRef = useRef<LeafletTileLayer | null>(null);
  const satelliteLayerRef = useRef<LeafletTileLayer | null>(null);

  const [ready, setReady] = useState(false);
  const [lastSelection, setLastSelection] = useState<string | null>(null);
  const [lastBounds, setLastBounds] = useState<RectBounds | null>(null);
  const [overlayCount, setOverlayCount] = useState(0);

  const [basemap, setBasemap] = useState<BasemapMode>("street");
  const [gibsDate, setGibsDate] = useState(defaultGibsDate);

  const [snipping, setSnipping] = useState(false);
  const [snipError, setSnipError] = useState<string | null>(null);
  const [snippedFile, setSnippedFile] = useState<File | null>(null);
  const [snippedPreviewUrl, setSnippedPreviewUrl] = useState<string | null>(null);
  const [uploadTarget, setUploadTarget] = useState<"optical" | "sar" | "before" | "after">("optical");
  const [uploadingSnip, setUploadingSnip] = useState(false);

  // Mount the map exactly once.
  useEffect(() => {
    let cancelled = false;

    (async () => {
      const L = (await import("leaflet")).default;
      await import("leaflet-draw");

      if (cancelled || !containerRef.current || mapRef.current) return;

      delete (L.Icon.Default.prototype as unknown as { _getIconUrl?: unknown })._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://unpkg.com/[email protected]/dist/images/marker-icon-2x.png",
        iconUrl: "https://unpkg.com/[email protected]/dist/images/marker-icon.png",
        shadowUrl: "https://unpkg.com/[email protected]/dist/images/marker-shadow.png",
      });

      const map = L.map(containerRef.current, {
        center: ASSAM_DEMO_CENTER,
        zoom: ASSAM_DEMO_ZOOM,
      });

      const street = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
      });
      street.addTo(map);
      streetLayerRef.current = street;

      const drawnItems = new L.FeatureGroup();
      map.addLayer(drawnItems);
      drawnRef.current = drawnItems;

      const drawControl = new (L.Control as unknown as { Draw: new (opts: unknown) => L.Control }).Draw({
        edit: { featureGroup: drawnItems, remove: true },
        draw: {
          marker: true,
          polygon: { allowIntersection: false, showArea: true },
          rectangle: true,
          circle: false,
          circlemarker: false,
          polyline: false,
        },
      });
      map.addControl(drawControl);

      map.on((L as unknown as { Draw: { Event: { CREATED: string } } }).Draw.Event.CREATED, (e: unknown) => {
        const event = e as { layer: L.Layer; layerType: string };
        drawnItems.clearLayers();
        drawnItems.addLayer(event.layer);
        setSnippedFile(null);
        setSnippedPreviewUrl(null);
        setSnipError(null);

        let description = "";
        if (event.layerType === "marker") {
          const { lat, lng } = (event.layer as L.Marker).getLatLng();
          description = `the point at ${fmt(lat)}°N, ${fmt(lng)}°E`;
          setLastBounds(null);
        } else if (event.layerType === "rectangle" || event.layerType === "polygon") {
          const bounds = (event.layer as L.Polygon).getBounds();
          description =
            `the ${event.layerType === "rectangle" ? "rectangular region" : "polygon region"} spanning ` +
            `${fmt(bounds.getSouth())}°N–${fmt(bounds.getNorth())}°N, ` +
            `${fmt(bounds.getWest())}°E–${fmt(bounds.getEast())}°E`;
          setLastBounds({
            west: bounds.getWest(),
            south: bounds.getSouth(),
            east: bounds.getEast(),
            north: bounds.getNorth(),
          });
        }
        if (description) setLastSelection(description);
      });

      mapRef.current = map;
      setReady(true);
    })();

    return () => {
      cancelled = true;
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, []);

  // Swap/refresh the satellite basemap layer whenever mode or date changes.
  useEffect(() => {
    if (!ready || !mapRef.current) return;

    (async () => {
      const L = (await import("leaflet")).default;
      const map = mapRef.current;
      if (!map) return;

      if (basemap === "satellite") {
        if (satelliteLayerRef.current) {
          map.removeLayer(satelliteLayerRef.current);
          satelliteLayerRef.current = null;
        }
        const sat = L.tileLayer(
          `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/${GIBS_LAYER}/default/${gibsDate}/${GIBS_TILE_MATRIX_SET}/{z}/{y}/{x}.jpg`,
          {
            maxZoom: GIBS_MAX_ZOOM,
            tileSize: 256,
            noWrap: true,
            bounds: [
              [-85.0511287776, -179.999999975],
              [85.0511287776, 179.999999975],
            ],
            attribution:
              '<a href="https://wiki.earthdata.nasa.gov/display/GIBS">NASA EOSDIS GIBS</a> — VIIRS/Suomi NPP true color, daily composite',
          },
        );
        sat.addTo(map);
        satelliteLayerRef.current = sat;
        if (streetLayerRef.current) map.removeLayer(streetLayerRef.current);
      } else {
        if (satelliteLayerRef.current) {
          map.removeLayer(satelliteLayerRef.current);
          satelliteLayerRef.current = null;
        }
        if (streetLayerRef.current) streetLayerRef.current.addTo(map);
      }
    })();
  }, [ready, basemap, gibsDate]);

  // Overlay any uploaded image that is honestly plain-lat/lon georeferenced.
  useEffect(() => {
    if (!ready || !mapRef.current) return;

    let cancelled = false;
    (async () => {
      const L = (await import("leaflet")).default;
      if (cancelled || !mapRef.current) return;

      const overlayGroupKey = "__satquery_overlays";
      const map = mapRef.current as unknown as Record<string, unknown>;
      const existing = map[overlayGroupKey] as import("leaflet").LayerGroup | undefined;
      existing?.clearLayers();
      const group = existing ?? L.layerGroup().addTo(mapRef.current);
      map[overlayGroupKey] = group;

      let count = 0;
      (["optical", "sar", "before", "after"] as const).forEach((key) => {
        const meta = slots[key];
        if (isPlainLatLon(meta)) {
          const [minx, miny, maxx, maxy] = meta.bounds;
          const bounds: [[number, number], [number, number]] = [
            [miny, minx],
            [maxy, maxx],
          ];
          L.imageOverlay(previewUrl(meta.image_id), bounds, { opacity: 0.85 }).addTo(group);
          count += 1;
        }
      });
      setOverlayCount(count);
      if (count > 0) {
        try {
          const layers = (group as unknown as { getLayers: () => import("leaflet").Layer[] }).getLayers();
          if (layers.length) {
            const fg = L.featureGroup(layers);
            mapRef.current!.fitBounds(fg.getBounds(), { maxZoom: 14, padding: [20, 20] });
          }
        } catch {
          /* non-fatal */
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [ready, slots]);

  function handleClear() {
    drawnRef.current?.clearLayers();
    setLastSelection(null);
    setLastBounds(null);
    setSnippedFile(null);
    setSnippedPreviewUrl(null);
    setSnipError(null);
  }

  function handleAsk() {
    if (!lastSelection) return;
    onRegionSelected(`Regarding ${lastSelection}:`);
  }

  async function handleSnip() {
    if (!lastBounds) return;
    setSnipping(true);
    setSnipError(null);
    setSnippedFile(null);
    setSnippedPreviewUrl(null);
    try {
      const qs = new URLSearchParams({
        west: String(lastBounds.west),
        south: String(lastBounds.south),
        east: String(lastBounds.east),
        north: String(lastBounds.north),
        date: gibsDate,
      });
      const res = await fetch(`/api/gibs-snapshot?${qs.toString()}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? `Snip failed (${res.status})`);
      }
      const blob = await res.blob();
      const file = new File([blob], `gibs_snip_${gibsDate}.png`, { type: "image/png" });
      setSnippedFile(file);
      setSnippedPreviewUrl(URL.createObjectURL(blob));
    } catch (err) {
      setSnipError(err instanceof Error ? err.message : "Snip failed");
    } finally {
      setSnipping(false);
    }
  }

  async function handleUploadSnip() {
    if (!snippedFile) return;
    setUploadingSnip(true);
    setSnipError(null);
    try {
      const modality: Modality = uploadTarget === "sar" ? "sar" : "optical";
      const role = uploadTarget === "before" || uploadTarget === "after" ? uploadTarget : "single";
      const meta = await uploadImage(snippedFile, modality, role);
      onImageUploaded?.(meta, uploadTarget);
    } catch (err) {
      setSnipError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploadingSnip(false);
    }
  }

  function handleOpenInTerrain() {
    if (!snippedFile) return;
    onOpenTerrainWithFile?.(snippedFile, `GIBS snip — ${gibsDate}`);
  }

  const anyGeoreferenced = (["optical", "sar", "before", "after"] as const).some((k) => isPlainLatLon(slots[k]));
  const canSnip = basemap === "satellite" && !!lastBounds;

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border)] px-4 py-2">
        <span className="text-xs font-medium tracking-wide text-[var(--text-dim)]">
          GEOSPATIAL MAP — click a point, or draw a rectangle/polygon
        </span>

        <div className="flex items-center gap-1">
          <button
            onClick={() => setBasemap("street")}
            className={`rounded px-2.5 py-1 text-[11px] font-medium ${
              basemap === "street" ? "bg-[var(--accent-bg)] text-[var(--accent-text)]" : "text-[var(--text-dim)] hover:bg-[var(--bg-elevated)]"
            }`}
          >
            Street
          </button>
          <button
            onClick={() => setBasemap("satellite")}
            className={`rounded px-2.5 py-1 text-[11px] font-medium ${
              basemap === "satellite" ? "bg-[var(--accent-bg)] text-[var(--accent-text)]" : "text-[var(--text-dim)] hover:bg-[var(--bg-elevated)]"
            }`}
          >
            🛰 Satellite
          </button>
        </div>

        {basemap === "satellite" && (
          <div className="flex items-center gap-1 text-[11px] text-[var(--text-dim)]">
            <button
              onClick={() => setGibsDate((d) => shiftDate(d, -1))}
              className="rounded border border-[var(--border)] px-1.5 hover:bg-[var(--bg-elevated)]"
              title="Previous day"
            >
              ◀
            </button>
            <input
              type="date"
              value={gibsDate}
              max={defaultGibsDate()}
              onChange={(e) => e.target.value && setGibsDate(e.target.value)}
              className="rounded border border-[var(--border)] bg-transparent px-1.5 py-0.5 font-mono text-[11px]"
            />
            <button
              onClick={() => setGibsDate((d) => shiftDate(d, 1))}
              disabled={gibsDate >= defaultGibsDate()}
              className="rounded border border-[var(--border)] px-1.5 hover:bg-[var(--bg-elevated)] disabled:opacity-30"
              title="Next day"
            >
              ▶
            </button>
          </div>
        )}

        <div className="flex items-center gap-2">
          {canSnip && (
            <button
              onClick={handleSnip}
              disabled={snipping}
              className="rounded bg-emerald-700 px-2.5 py-1 text-[11px] font-medium text-white transition-colors hover:bg-emerald-600 disabled:opacity-50"
            >
              {snipping ? "Snipping…" : "✂ Snip satellite image"}
            </button>
          )}
          <button
            onClick={handleAsk}
            disabled={!lastSelection}
            className="rounded bg-[var(--accent-strong)] px-2.5 py-1 text-[11px] font-medium text-white transition-colors hover:bg-[var(--accent-strong-hover)] disabled:bg-[var(--bg-elevated)] disabled:text-[var(--text-faint)]"
          >
            Ask about this region →
          </button>
          <button
            onClick={handleClear}
            disabled={!lastSelection}
            className="rounded border border-[var(--border)] px-2.5 py-1 text-[11px] text-[var(--text-dim)] transition-colors hover:text-[var(--text)] disabled:opacity-40"
          >
            Clear
          </button>
        </div>
      </div>

      <div ref={containerRef} className="min-h-0 flex-1" style={{ background: "#0a0a0a" }} />

      {(snippedPreviewUrl || snipError) && (
        <div className="flex items-center gap-3 border-t border-[var(--border)] bg-[var(--bg-panel)] px-4 py-2">
          {snipError && <span className="text-[11px] text-red-400">⚠ {snipError}</span>}
          {snippedPreviewUrl && (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={snippedPreviewUrl} alt="Snipped satellite region" className="h-14 w-20 rounded border border-[var(--border)] object-cover" />
              <select
                value={uploadTarget}
                onChange={(e) => setUploadTarget(e.target.value as typeof uploadTarget)}
                className="rounded border border-[var(--border)] bg-transparent px-2 py-1 text-[11px] text-[var(--text-dim)]"
              >
                <option value="optical">Upload as Optical</option>
                <option value="sar">Upload as SAR</option>
                <option value="before">Upload as Before</option>
                <option value="after">Upload as After</option>
              </select>
              <button
                onClick={handleUploadSnip}
                disabled={uploadingSnip}
                className="rounded bg-[var(--accent-strong)] px-2.5 py-1 text-[11px] font-medium text-white hover:bg-[var(--accent-strong-hover)] disabled:opacity-50"
              >
                {uploadingSnip ? "Uploading…" : "Use in Chat"}
              </button>
              <button
                onClick={handleOpenInTerrain}
                className="rounded border border-[var(--accent-strong)] px-2.5 py-1 text-[11px] text-[var(--accent-text)] hover:bg-[var(--accent-bg)]"
              >
                🧊 Open in 3D Terrain
              </button>
            </>
          )}
        </div>
      )}

      <div className="border-t border-[var(--border)] px-4 py-2 text-[11px] text-[var(--text-faint)]">
        {lastSelection ? (
          <span>
            Selected: <span className="text-[var(--accent)]">{lastSelection}</span> —{" "}
            <button onClick={handleAsk} className="underline hover:text-[var(--text)]">
              insert into chat
            </button>
          </span>
        ) : (
          <span>Use the draw tools (top-right of the map) to select a point, rectangle, or polygon.</span>
        )}
        <br />
        {basemap === "satellite" ? (
          <span>
            Satellite basemap: NASA GIBS VIIRS true-color, {gibsDate} (daily composite — not a live geostationary
            feed). Draw a rectangle to enable “Snip satellite image” and pull that exact region into SatQuery.
          </span>
        ) : anyGeoreferenced ? (
          <span>{overlayCount} uploaded image(s) placed on the map using their real geographic bounds (EPSG:4326).</span>
        ) : (
          <span>
            No uploaded image is plain-lat/lon (EPSG:4326) georeferenced, so none is overlaid on the base map — this
            is a real limitation, not a bug. Region selection and the base map still work regardless.
          </span>
        )}
      </div>
    </div>
  );
}
