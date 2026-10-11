"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

type Status = "checking" | "online" | "offline";

export default function SystemStatus() {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        await getHealth();
        if (!cancelled) setStatus("online");
      } catch {
        if (!cancelled) setStatus("offline");
      }
    }

    poll();
    const interval = setInterval(poll, 10_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const config: Record<Status, { label: string; dot: string }> = {
    checking: { label: "Checking...", dot: "bg-yellow-400" },
    online: { label: "System OK", dot: "bg-emerald-400" },
    offline: { label: "Backend unreachable", dot: "bg-red-500" },
  };

  const { label, dot } = config[status];

  return (
    <div className="flex items-center gap-2 text-sm text-[var(--text-dim)]">
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      <span className="hidden sm:inline">{label}</span>
    </div>
  );
}
