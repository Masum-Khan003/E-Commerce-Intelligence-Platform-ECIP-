"use client";

import { useBffData } from "@/lib/api";
import { LoadingSkeleton, ErrorBanner, EmptyState } from "@/components/StatusStates";

interface HealthReady {
  status: string;
  models: Record<string, string>;
}

export default function SentimentAnalyticsPage() {
  const state = useBffData<HealthReady>("health/ready");
  const sentimentStatus = state.status === "success" ? state.data.models["distilbert"] : null;
  const isTrained = sentimentStatus === "loaded";

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold mb-1">Sentiment Analytics</h1>
        <p className="text-sm text-neutral-500">
          Sentiment trend, aspect radar, review volume by rating.
        </p>
      </div>

      {state.status === "loading" && <LoadingSkeleton label="Checking model status…" />}
      {state.status === "error" && (
        <ErrorBanner message="Sentiment data unavailable. The E-CIP API may be down." />
      )}
      {state.status === "success" && !isTrained && (
        <EmptyState
          message={`DistilBERT not yet trained — ${
            sentimentStatus ?? "status unknown"
          }. Module 2 requires GPU training on Colab/Kaggle (see HANDOFF.md). This page will populate automatically once models/sentiment/weights/distilbert_sentiment_best.pt exists.`}
        />
      )}
      {state.status === "success" && isTrained && (
        <EmptyState message="Model loaded, but no analysis history has been recorded yet." />
      )}
    </div>
  );
}
