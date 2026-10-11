"use client";
import { Suspense, useMemo } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid, Html, Line } from "@react-three/drei";
import TerrainMesh from "./TerrainMesh";
import { heightAt, profileBetween, formatHeight, PLANE_SIZE, type Pick, type TerrainMath } from "@/lib/terrain/measure";
import type { DepthResult, DetectedObject } from "@/lib/terrain/depthClient";

export interface Terrain3DProps {
  depthResult: DepthResult;
  imageFile: File;
  hologram: boolean;
  heightScale: number;
  resolution: number;
  tm: TerrainMath;
  picks: Pick[];
  setPicks: (picks: Pick[]) => void;
  setProfile: (profile: number[] | null) => void;
  tool: "probe" | "distance" | "profile" | "slope";
  objects: DetectedObject[];
}

export default function Terrain3D({ 
  depthResult, imageFile, hologram, heightScale, resolution, tm, picks, setPicks, setProfile, tool, objects 
}: Terrain3DProps) {
  const textureUrl = useMemo(() => URL.createObjectURL(imageFile), [imageFile]);

  const handlePick = (point: THREE.Vector3) => {
    const pick: Pick = { x: point.x, y: point.y, z: point.z, h: heightAt(tm, point.x, point.z) };
    if (tool === "probe" || tool === "slope") {
      setPicks([pick]);
      setProfile(null);
      return;
    }
    setPicks((() => {
      const next = picks.length >= 2 ? [pick] : [...picks, pick];
      if (next.length === 2) {
        setProfile(tool === "profile" ? profileBetween(tm, next[0], next[1]) : null);
      }
      return next;
    })());
  };

  const getMarkerPosition = (obj: DetectedObject): [number, number, number] => {
    const x = (obj.center_x - 0.5) * PLANE_SIZE;
    const z = (obj.center_y - 0.5) * PLANE_SIZE;
    const y = heightAt(tm, x, z) + 0.15;
    return [x, y, z];
  };

  return (
    <div className="relative w-full h-full">
      <Canvas shadows camera={{ position: [0, 12, 18], fov: 50 }}>
        <Suspense fallback={null}>
          <ambientLight intensity={hologram ? 0.4 : 0.6} />
          <directionalLight position={[15, 20, 10]} intensity={hologram ? 0.5 : 1.5} castShadow />
          <pointLight position={[-10, 10, -10]} intensity={hologram ? 0.8 : 0.2} color={hologram ? "#00ffcc" : "#ffffff"} />

          <TerrainMesh
            heightmap={depthResult.heightmap}
            sourceWidth={depthResult.width}
            sourceHeight={depthResult.height}
            textureUrl={textureUrl}
            resolution={resolution}
            heightScale={heightScale}
            hologram={hologram}
            onPick={handlePick}
          />

          <Grid infiniteGrid fadeDistance={40} cellColor={hologram ? "#1e1b4b" : "#1e293b"} sectionColor={hologram ? "#8b5cf6" : "#475569"} position={[0, -0.05, 0]} />

          {picks.map((p, i) => (
            <group key={i} position={[p.x, p.y + 0.12, p.z]}>
              <mesh>
                <sphereGeometry args={[0.12, 16, 16]} />
                <meshStandardMaterial color="#8b5cf6" emissive="#8b5cf6" emissiveIntensity={0.7} />
              </mesh>
              <Html center distanceFactor={18} className="pointer-events-none">
                <div className="px-2 py-1 rounded bg-black/80 text-xs text-primary font-mono">{formatHeight(tm, p.h)}</div>
              </Html>
            </group>
          ))}

          {objects.map((obj, i) => {
            const pos = getMarkerPosition(obj);
            return (
              <group key={i} position={pos}>
                <mesh>
                  <sphereGeometry args={[0.15, 16, 16]} />
                  <meshStandardMaterial color="#10b981" emissive="#10b981" emissiveIntensity={0.8} />
                </mesh>
                <Html center distanceFactor={18} className="pointer-events-none">
                  <div className="px-2 py-1 rounded bg-black/80 text-xs font-mono whitespace-nowrap border border-emerald-500/50">
                    <span className="text-emerald-400 capitalize">{obj.label}</span> • {(obj.score * 100).toFixed(0)}%
                  </div>
                </Html>
              </group>
            );
          })}

          {picks.length === 2 && (tool === "distance" || tool === "profile") && (
            <Line points={[[picks[0].x, picks[0].y, picks[0].z], [picks[1].x, picks[1].y, picks[1].z]]} color="#06b6d4" lineWidth={2} dashed />
          )}

          <OrbitControls enableDamping dampingFactor={0.08} minDistance={3} maxDistance={60} />
        </Suspense>
      </Canvas>
    </div>
  );
}