"use client";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import type { ThreeEvent } from "@react-three/fiber";
import { useTexture } from "@react-three/drei";
import { PLANE_SIZE } from "@/lib/terrain/measure";

export interface TerrainMeshProps {
  heightmap: Float32Array;
  sourceWidth: number;
  sourceHeight: number;
  textureUrl: string;
  resolution?: number;
  heightScale?: number;
  planeSize?: number;
  hologram?: boolean;
  onPick?: (point: THREE.Vector3) => void;
}

function sampleHeight(
  heightmap: Float32Array,
  srcW: number,
  srcH: number,
  u: number,
  v: number
): number {
  const x = u * (srcW - 1);
  const y = v * (srcH - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, srcW - 1);
  const y1 = Math.min(y0 + 1, srcH - 1);
  const fx = x - x0;
  const fy = y - y0;
  const h00 = heightmap[y0 * srcW + x0];
  const h10 = heightmap[y0 * srcW + x1];
  const h01 = heightmap[y1 * srcW + x0];
  const h11 = heightmap[y1 * srcW + x1];
  const top = h00 * (1 - fx) + h10 * fx;
  const bottom = h01 * (1 - fx) + h11 * fx;
  return top * (1 - fy) + bottom * fy;
}

export default function TerrainMesh({
  heightmap,
  sourceWidth,
  sourceHeight,
  textureUrl,
  resolution = 200,
  heightScale = 0.5, // Lower default to prevent extreme spikes
  planeSize = PLANE_SIZE,
  hologram = false,
  onPick,
}: TerrainMeshProps) {
  const texture = useTexture(textureUrl);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;

  const geometry = useMemo(() => {
    // 1. Fix the Aspect Ratio (Prevents stretching)
    const aspectRatio = sourceWidth / sourceHeight;
    let planeWidth = planeSize;
    let planeHeight = planeSize;
    
    if (aspectRatio > 1) {
      planeHeight = planeSize / aspectRatio;
    } else {
      planeWidth = planeSize * aspectRatio;
    }

    const geo = new THREE.PlaneGeometry(planeWidth, planeHeight, resolution - 1, resolution - 1);
    geo.rotateX(-Math.PI / 2);
    const posAttr = geo.attributes.position;
    
    // 2. Calculate min/max and clamp outliers (Fixes pink/white spikes)
    let min = Infinity;
    let max = -Infinity;
    const sorted = Array.from(heightmap).sort((a, b) => a - b);
    // Use the 2nd percentile and 98th percentile to remove wild AI noise
    const lowerBound = sorted[Math.floor(sorted.length * 0.02)];
    const upperBound = sorted[Math.floor(sorted.length * 0.98)];

    const range = Math.max(upperBound - lowerBound, 1e-6);

    for (let i = 0; i < posAttr.count; i++) {
      const gx = i % resolution;
      const gz = Math.floor(i / resolution);
      const u = gx / (resolution - 1);
      const v = gz / (resolution - 1);
      
      let rawHeight = sampleHeight(heightmap, sourceWidth, sourceHeight, u, v);
      
      // 3. Smooth & Clamp
      rawHeight = Math.max(lowerBound, Math.min(upperBound, rawHeight));
      
      const normalized = (rawHeight - lowerBound) / range;
      // Reduce height scale by half to prevent mountain-like spikes on buildings
      const y = normalized * heightScale * (planeSize * 0.5); 
      posAttr.setY(i, y);
    }
    geo.computeVertexNormals();
    return geo;
  }, [heightmap, sourceWidth, sourceHeight, resolution, heightScale, planeSize]);

  // Extract edges for hologram
  const edges = useMemo(() => new THREE.EdgesGeometry(geometry), [geometry]);
  const wireframeRef = useRef<THREE.MeshBasicMaterial>(null);

  useFrame(({ clock }) => {
    if (hologram && wireframeRef.current) {
      const t = clock.getElapsedTime();
      wireframeRef.current.opacity = 0.25 + Math.sin(t * 3) * 0.1;
    }
  });

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    onPick?.(e.point);
  };

  if (hologram) {
    return (
      <group onClick={handleClick}>
        <mesh geometry={geometry}>
          <meshBasicMaterial color="#00ffcc" transparent opacity={0.15} blending={THREE.AdditiveBlending} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
        <mesh geometry={geometry} raycast={() => null}>
          <meshBasicMaterial ref={wireframeRef} color="#00ffcc" wireframe transparent opacity={0.25} blending={THREE.AdditiveBlending} depthWrite={false} />
        </mesh>
        <lineSegments geometry={edges} raycast={() => null}>
          <lineBasicMaterial color="#00ffff" transparent opacity={0.9} />
        </lineSegments>
      </group>
    );
  }

  return (
    <mesh geometry={geometry} receiveShadow castShadow onClick={handleClick}>
      <meshStandardMaterial map={texture} side={THREE.DoubleSide} roughness={0.85} metalness={0} />
    </mesh>
  );
}