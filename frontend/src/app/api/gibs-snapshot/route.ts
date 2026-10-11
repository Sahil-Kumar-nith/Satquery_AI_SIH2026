// src/app/api/gibs-snapshot/route.ts
//
// "Snip" tool support for the Map tab's live-satellite basemap. Given a
// bounding box + date, fetches ONE flat PNG covering exactly that area from
// NASA GIBS's public WMS (no API key, no auth) — this is what actually
// lets a drawn rectangle become a real image file the rest of SatQuery can
// use (upload as Optical/SAR, or open in the 3D Terrain viewer), instead of
// scraping/compositing individual map tiles client-side.
//
// Source: NASA GIBS WMS, EPSG:4326, VIIRS_SNPP_CorrectedReflectance_TrueColor.
// This is a DAILY composite (best available same-day pass), not a
// continuous geostationary loop — say so in the UI. It is a legitimate,
// freely-licensed public-domain NASA product; it is NOT the same data
// source as zoom.earth (which layers in EUMETSAT/Meteosat geostationary
// imagery under its own terms) and should not be presented as such.
//
// Docs: https://nasa-gibs.github.io/gibs-api-docs/access-basics/

import { NextRequest, NextResponse } from "next/server";

const GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi";
const LAYER = "VIIRS_SNPP_CorrectedReflectance_TrueColor";
const MAX_PIXELS_PER_SIDE = 2048;

function isFiniteNum(n: number) {
  return Number.isFinite(n);
}

export async function GET(req: NextRequest) {
  const params = req.nextUrl.searchParams;

  const west = parseFloat(params.get("west") ?? "");
  const south = parseFloat(params.get("south") ?? "");
  const east = parseFloat(params.get("east") ?? "");
  const north = parseFloat(params.get("north") ?? "");
  const date = params.get("date") ?? ""; // YYYY-MM-DD

  if (![west, south, east, north].every(isFiniteNum)) {
    return NextResponse.json({ error: "west/south/east/north must be numbers" }, { status: 400 });
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    return NextResponse.json({ error: "date must be YYYY-MM-DD" }, { status: 400 });
  }
  if (west >= east || south >= north) {
    return NextResponse.json({ error: "Invalid bbox: require west<east and south<north" }, { status: 400 });
  }
  if (west < -180 || east > 180 || south < -90 || north > 90) {
    return NextResponse.json({ error: "bbox out of [-180,180]/[-90,90] range" }, { status: 400 });
  }

  // Keep the exported image's aspect ratio matched to the drawn region,
  // capped so a huge rectangle can't trigger a multi-thousand-pixel fetch.
  const aspect = (east - west) / (north - south);
  let width: number;
  let height: number;
  if (aspect >= 1) {
    width = MAX_PIXELS_PER_SIDE;
    height = Math.max(64, Math.round(MAX_PIXELS_PER_SIDE / aspect));
  } else {
    height = MAX_PIXELS_PER_SIDE;
    width = Math.max(64, Math.round(MAX_PIXELS_PER_SIDE * aspect));
  }

  const wmsUrl =
    `${GIBS_WMS_URL}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap` +
    `&LAYERS=${LAYER}&SRS=EPSG:4326` +
    `&BBOX=${west},${south},${east},${north}` +
    `&WIDTH=${width}&HEIGHT=${height}` +
    `&FORMAT=image/png&TIME=${date}`;

  let upstream: Response;
  try {
    upstream = await fetch(wmsUrl, { signal: AbortSignal.timeout(30_000) });
  } catch (err) {
    console.error("GIBS WMS unreachable:", err);
    return NextResponse.json({ error: "NASA GIBS is unreachable right now." }, { status: 502 });
  }

  if (!upstream.ok) {
    const detail = await upstream.text().catch(() => "");
    return NextResponse.json({ error: "GIBS returned an error", detail }, { status: upstream.status });
  }

  const buf = await upstream.arrayBuffer();
  return new NextResponse(buf, {
    status: 200,
    headers: {
      "Content-Type": "image/png",
      "X-Gibs-Layer": LAYER,
      "X-Gibs-Date": date,
      "Cache-Control": "no-store",
    },
  });
}
