"use client";

import { useBffData } from "@/lib/api";
import { LoadingSkeleton, ErrorBanner, EmptyState } from "@/components/StatusStates";
import { RoiEstimator } from "@/components/RoiEstimator";

interface RiskBandCount {
  risk_band: string;
  count: number;
}

interface RetentionAnalytics {
  risk_band_distribution: RiskBandCount[];
  total_scored: number;
}

const BAND_COLOR: Record<string, string> = {
  LOW: "bg-emerald-500",
  MEDIUM: "bg-amber-500",
  HIGH: "bg-rose-500",
};

export default function RetentionAnalyticsPage() {
  const state = useBffData<RetentionAnalytics>("v1/analytics/retention");

  const nHighRisk =
    state.status === "success"
      ? state.data.risk_band_distribution.find((b) => b.risk_band === "HIGH")?.count ?? 0
      : 0;
  const maxCount =
    state.status === "success"
      ? Math.max(1, ...state.data.risk_band_distribution.map((b) => b.count))
      : 1;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold mb-1">Retention Analytics</h1>
        <p className="text-sm text-neutral-500">
          Risk band distribution and the retention-offer ROI estimator.
        </p>
      </div>

      {state.status === "loading" && <LoadingSkeleton label="Loading retention analytics…" />}
      {state.status === "error" && (
        <ErrorBanner message="Retention data unavailable. The E-CIP API may be down." />
      )}

      {state.status === "success" && state.data.total_scored === 0 && (
        <EmptyState message="No customers scored yet — call POST /v1/retention/score to generate data." />
      )}

      {state.status === "success" && state.data.total_scored > 0 && (
        <>
          <div>
            <h2 className="text-sm font-medium text-neutral-300 mb-3">
              Risk band distribution ({state.data.total_scored} scored)
            </h2>
            <div className="space-y-2">
              {state.data.risk_band_distribution.map((band) => (
                <div key={band.risk_band} className="flex items-center gap-3">
                  <span className="w-16 text-xs text-neutral-400">{band.risk_band}</span>
                  <div className="flex-1 h-4 bg-neutral-900 rounded overflow-hidden">
                    <div
                      className={`h-full ${BAND_COLOR[band.risk_band] ?? "bg-neutral-600"}`}
                      style={{ width: `${(band.count / maxCount) * 100}%` }}
                    />
                  </div>
                  <span className="w-10 text-right text-xs text-neutral-400">{band.count}</span>
                </div>
              ))}
            </div>
          </div>

          <RoiEstimator nHighRisk={nHighRisk} />
        </>
      )}
    </div>
  );
}
