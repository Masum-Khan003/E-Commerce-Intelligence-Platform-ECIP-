"use client";

import { useBffData } from "@/lib/api";
import { LoadingSkeleton, ErrorBanner, EmptyState } from "@/components/StatusStates";

interface DriftEvent {
  module: string;
  feature_name: string | null;
  metric_value: number;
  threshold: number;
  alert_triggered: boolean;
  created_at: string;
}

interface FeatureDriftSummary {
  feature_name: string;
  latest_psi: number;
  alert_triggered: boolean;
}

interface DriftEventsData {
  events: DriftEvent[];
  by_feature: FeatureDriftSummary[];
}

export default function DriftMonitorPage() {
  const state = useBffData<DriftEventsData>("v1/drift-events");

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold mb-1">Drift Monitor</h1>
        <p className="text-sm text-neutral-500">
          Per-feature PSI gauges and the drift event timeline (mlops/drift_detector.py).
        </p>
      </div>

      {state.status === "loading" && <LoadingSkeleton label="Loading drift data…" />}
      {state.status === "error" && (
        <ErrorBanner message="Drift data unavailable." />
      )}
      {state.status === "success" && state.data.by_feature.length === 0 && (
        <EmptyState message="No drift checks recorded yet — run python mlops/drift_detector.py --write-db." />
      )}
      {state.status === "success" && state.data.by_feature.length > 0 && (
        <>
          <div>
            <h2 className="text-sm font-medium text-neutral-300 mb-3">Per-feature PSI (latest)</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {state.data.by_feature.map((f) => (
                <div
                  key={f.feature_name}
                  className={`rounded border px-3 py-2 flex items-center justify-between ${
                    f.alert_triggered ? "border-rose-900 bg-rose-950/30" : "border-neutral-800"
                  }`}
                >
                  <span className="text-sm text-neutral-300">{f.feature_name}</span>
                  <span
                    className={`text-sm font-mono ${
                      f.alert_triggered ? "text-rose-400" : "text-neutral-400"
                    }`}
                  >
                    PSI {f.latest_psi.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h2 className="text-sm font-medium text-neutral-300 mb-3">Event timeline</h2>
            <div className="rounded-lg border border-neutral-800 divide-y divide-neutral-800 max-h-96 overflow-y-auto">
              {state.data.events.map((event, i) => (
                <div key={i} className="flex items-center justify-between px-4 py-2 text-sm">
                  <span className="text-neutral-300">
                    {event.feature_name} ({event.module})
                  </span>
                  <span className="flex items-center gap-3">
                    <span className={event.alert_triggered ? "text-rose-400" : "text-neutral-500"}>
                      PSI {event.metric_value.toFixed(3)}
                    </span>
                    <span className="text-xs text-neutral-600">
                      {new Date(event.created_at).toLocaleString()}
                    </span>
                  </span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
