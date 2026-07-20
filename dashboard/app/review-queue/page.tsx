"use client";

import { useState } from "react";
import { useBffData } from "@/lib/api";
import { LoadingSkeleton, ErrorBanner, EmptyState } from "@/components/StatusStates";

interface ReviewQueueItem {
  id: string;
  request_id: string;
  module: string;
  trigger: string;
  payload: Record<string, unknown>;
  status: string;
  created_at: string;
}

interface ReviewQueueData {
  items: ReviewQueueItem[];
  total: number;
  depth: number;
}

export default function ReviewQueuePage() {
  const [moduleFilter, setModuleFilter] = useState<string>("");
  const [refreshKey, setRefreshKey] = useState(0);
  const query = moduleFilter ? `v1/review-queue?module=${moduleFilter}` : "v1/review-queue";
  const state = useBffData<ReviewQueueData>(`${query}${query.includes("?") ? "&" : "?"}_r=${refreshKey}`);

  async function resolve(id: string, resolution: "resolved" | "dismissed") {
    await fetch(`/api/bff/v1/review-queue/${id}/resolve`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ resolution }),
    });
    setRefreshKey((k) => k + 1);
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold mb-1">Review Queue</h1>
          <p className="text-sm text-neutral-500">
            Items flagged by drift alerts, low confidence, or OOD detection.
          </p>
        </div>
        <select
          value={moduleFilter}
          onChange={(e) => setModuleFilter(e.target.value)}
          className="bg-neutral-900 border border-neutral-800 rounded px-3 py-1.5 text-sm text-neutral-300"
        >
          <option value="">All modules</option>
          <option value="retention">Retention</option>
          <option value="product">Product</option>
          <option value="sentiment">Sentiment</option>
        </select>
      </div>

      {state.status === "loading" && <LoadingSkeleton label="Loading review queue…" />}
      {state.status === "error" && (
        <ErrorBanner message="Review queue temporarily unavailable." />
      )}
      {state.status === "success" && state.data.items.length === 0 && (
        <EmptyState message="No pending review items." />
      )}
      {state.status === "success" && state.data.items.length > 0 && (
        <div className="rounded-lg border border-neutral-800 divide-y divide-neutral-800">
          {state.data.items.map((item) => (
            <div key={item.id} className="flex items-center justify-between px-4 py-3">
              <div>
                <div className="text-sm text-neutral-200">
                  <span className="capitalize">{item.module}</span> — {item.trigger}
                </div>
                <div className="text-xs text-neutral-500 mt-0.5">
                  {item.request_id} · {new Date(item.created_at).toLocaleString()}
                </div>
              </div>
              <div className="flex gap-2">
                {item.status === "pending" ? (
                  <>
                    <button
                      onClick={() => resolve(item.id, "resolved")}
                      className="text-xs px-3 py-1 rounded border border-teal-800 text-teal-400 hover:bg-teal-950"
                    >
                      Resolve
                    </button>
                    <button
                      onClick={() => resolve(item.id, "dismissed")}
                      className="text-xs px-3 py-1 rounded border border-neutral-700 text-neutral-400 hover:bg-neutral-800"
                    >
                      Dismiss
                    </button>
                  </>
                ) : (
                  <span className="text-xs text-neutral-500 capitalize">{item.status}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
