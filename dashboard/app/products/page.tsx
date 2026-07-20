"use client";

import { useBffData } from "@/lib/api";
import { LoadingSkeleton, ErrorBanner, EmptyState } from "@/components/StatusStates";

interface HealthReady {
  status: string;
  models: Record<string, string>;
}

export default function ProductAnalyticsPage() {
  const state = useBffData<HealthReady>("health/ready");
  const productStatus = state.status === "success" ? state.data.models["efficientnet"] : null;
  const isTrained = productStatus === "loaded";

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold mb-1">Product Analytics</h1>
        <p className="text-sm text-neutral-500">
          Category distribution, confidence histogram, OOD counter, Grad-CAM viewer.
        </p>
      </div>

      {state.status === "loading" && <LoadingSkeleton label="Checking model status…" />}
      {state.status === "error" && (
        <ErrorBanner message="Dashboard temporarily unavailable — the E-CIP API may be down." />
      )}
      {state.status === "success" && !isTrained && (
        <EmptyState
          message={`EfficientNet-B3 not yet trained — ${
            productStatus ?? "status unknown"
          }. Module 1 requires GPU training on Colab/Kaggle (see HANDOFF.md). This page will populate automatically once models/product/weights/efficientnet_b3_best.pt exists.`}
        />
      )}
      {state.status === "success" && isTrained && (
        <EmptyState message="Model loaded, but no classification history has been recorded yet." />
      )}
    </div>
  );
}
