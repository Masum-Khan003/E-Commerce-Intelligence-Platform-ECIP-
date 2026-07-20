"use client";

import { useBffData } from "@/lib/api";
import { KpiCard } from "@/components/KpiCard";
import { LoadingSkeleton, ErrorBanner } from "@/components/StatusStates";

interface ModuleVolume {
  module: string;
  count: number;
  avg_latency_ms: number;
}

interface OverviewData {
  total_predictions: number;
  by_module: ModuleVolume[];
  model_versions: Record<string, string>;
}

export default function OverviewPage() {
  const state = useBffData<OverviewData>("v1/analytics/overview");

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold mb-1">Overview</h1>
        <p className="text-sm text-neutral-500">
          Live prediction volume and model status across all three modules.
        </p>
      </div>

      {state.status === "loading" && <LoadingSkeleton label="Loading overview…" />}
      {state.status === "error" && (
        <ErrorBanner message="Dashboard temporarily unavailable — the E-CIP API may be down." />
      )}
      {state.status === "success" && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KpiCard label="Total predictions logged" value={state.data.total_predictions} />
            <KpiCard
              label="Modules with real traffic"
              value={state.data.by_module.length}
              sub="of 3 modules"
            />
            <KpiCard
              label="Retention avg latency"
              value={`${
                state.data.by_module.find((m) => m.module === "retention")?.avg_latency_ms.toFixed(1) ?? "—"
              } ms`}
              sub="target < 12ms p95"
            />
          </div>

          <div>
            <h2 className="text-sm font-medium text-neutral-300 mb-3">Prediction volume by module</h2>
            <div className="rounded-lg border border-neutral-800 divide-y divide-neutral-800">
              {state.data.by_module.length === 0 && (
                <div className="px-4 py-6 text-sm text-neutral-500 text-center">
                  No predictions logged yet — call POST /v1/retention/score to generate traffic.
                </div>
              )}
              {state.data.by_module.map((m) => (
                <div key={m.module} className="flex items-center justify-between px-4 py-3">
                  <span className="capitalize text-neutral-200">{m.module}</span>
                  <span className="text-sm text-neutral-400">
                    {m.count} predictions · {m.avg_latency_ms.toFixed(1)}ms avg
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h2 className="text-sm font-medium text-neutral-300 mb-3">Model version strip</h2>
            <div className="flex flex-wrap gap-2">
              {Object.entries(state.data.model_versions).map(([module, version]) => (
                <span
                  key={module}
                  className="text-xs font-mono px-3 py-1.5 rounded-full border border-neutral-800 bg-neutral-900 text-neutral-400"
                >
                  <span className="capitalize text-neutral-200">{module}</span>: {version}
                </span>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
