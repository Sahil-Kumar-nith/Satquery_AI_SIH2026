// src/app/api/process-depth/route.ts
//
// Thin proxy for the 3D Terrain / Hologram feature. All model logic lives
// in terrain-worker/ (a separate FastAPI process, port 8010 by default —
// kept isolated from the main SatQuery backend on 8000 so a heavy depth
// model never blocks or crashes the PS-mandated agent pipeline).

import { NextRequest, NextResponse } from "next/server";

const TERRAIN_WORKER_URL = process.env.TERRAIN_WORKER_URL ?? "http://localhost:8010";
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024; // keep in sync with terrain-worker MAX_UPLOAD_MB

const ACCEPTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"];

export async function POST(req: NextRequest) {
  let formData: FormData;
  try {
    formData = await req.formData();
  } catch {
    return NextResponse.json({ error: "Expected multipart/form-data" }, { status: 400 });
  }

  const file = formData.get("file");
  const gcpsField = formData.get("gcps");

  if (!(file instanceof File)) {
    return NextResponse.json({ error: "No file provided under field 'file'" }, { status: 400 });
  }

  const ext = "." + file.name.split(".").pop()?.toLowerCase();
  if (!ACCEPTED_EXTENSIONS.includes(ext)) {
    return NextResponse.json(
      { error: `Unsupported file type "${ext}". Accepted: ${ACCEPTED_EXTENSIONS.join(", ")}` },
      { status: 400 },
    );
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    return NextResponse.json(
      { error: `File exceeds ${MAX_UPLOAD_BYTES / 1024 / 1024}MB limit` },
      { status: 413 },
    );
  }

  const forward = new FormData();
  forward.append("file", file, file.name);
  if (typeof gcpsField === "string" && gcpsField.length > 0) {
    forward.append("gcps", gcpsField);
  }

  let workerRes: Response;
  try {
    workerRes = await fetch(`${TERRAIN_WORKER_URL}/process-depth`, {
      method: "POST",
      body: forward,
      // First request after a cold start loads the depth model (~seconds).
      signal: AbortSignal.timeout(120_000),
    });
  } catch (err) {
    console.error("Terrain worker unreachable:", err);
    return NextResponse.json(
      {
        error:
          "3D Terrain worker is unreachable. Start it with: cd terrain-worker && uvicorn main:app --port 8010",
      },
      { status: 502 },
    );
  }

  if (!workerRes.ok) {
    const detail = await workerRes.text().catch(() => "");
    return NextResponse.json({ error: "Terrain worker returned an error", detail }, { status: workerRes.status });
  }

  const data = await workerRes.json();
  return NextResponse.json(data, { status: 200 });
}
