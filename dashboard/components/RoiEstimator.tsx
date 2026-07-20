// dashboard/components/RoiEstimator.tsx
// Blueprint Section 16 — ROI estimator. Purely client-side: no API call
// on input change, matching the spec's "fully client-side" requirement.

"use client";

import { useState } from "react";

const CONVERSION_LEVELS = [0.10, 0.25, 0.45];
const LTV_LEVELS = [200, 480, 800];

function computeRoi(
  nHighRisk: number,
  reachPct: number,
  conversionRate: number,
  avgLtv: number,
  costPerOffer: number
) {
  const nTargeted = nHighRisk * reachPct;
  const nSaved = nTargeted * conversionRate;
  const revenueSaved = nSaved * avgLtv;
  const campaignCost = nTargeted * costPerOffer;
  const netRoi = revenueSaved - campaignCost;
  const roiMultiple = campaignCost > 0 ? revenueSaved / campaignCost : 0;
  return { nTargeted, nSaved, revenueSaved, campaignCost, netRoi, roiMultiple };
}

function formatGbp(value: number): string {
  return `£${Math.round(value).toLocaleString()}`;
}

export function RoiEstimator({ nHighRisk }: { nHighRisk: number }) {
  const [reachPct, setReachPct] = useState(0.6);
  const [conversionRate, setConversionRate] = useState(0.25);
  const [avgLtv, setAvgLtv] = useState(480);
  const [costPerOffer, setCostPerOffer] = useState(15);

  const result = computeRoi(nHighRisk, reachPct, conversionRate, avgLtv, costPerOffer);

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-5 space-y-5">
      <div>
        <h3 className="text-sm font-medium text-neutral-200">Retention ROI Estimator</h3>
        <p className="text-xs text-neutral-500 mt-1">
          {nHighRisk.toLocaleString()} high-risk customers currently on record. All inputs
          recalculate instantly — no API call.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <label className="space-y-1">
          <span className="text-neutral-400 text-xs">Reach (% of high-risk targeted)</span>
          <input
            type="number"
            min={0}
            max={100}
            value={Math.round(reachPct * 100)}
            onChange={(e) => setReachPct(Number(e.target.value) / 100)}
            className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 py-1 text-neutral-100"
          />
        </label>
        <label className="space-y-1">
          <span className="text-neutral-400 text-xs">Conversion rate (%)</span>
          <input
            type="number"
            min={0}
            max={100}
            value={Math.round(conversionRate * 100)}
            onChange={(e) => setConversionRate(Number(e.target.value) / 100)}
            className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 py-1 text-neutral-100"
          />
        </label>
        <label className="space-y-1">
          <span className="text-neutral-400 text-xs">Avg customer LTV (£)</span>
          <input
            type="number"
            min={0}
            value={avgLtv}
            onChange={(e) => setAvgLtv(Number(e.target.value))}
            className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 py-1 text-neutral-100"
          />
        </label>
        <label className="space-y-1">
          <span className="text-neutral-400 text-xs">Cost per offer (£)</span>
          <input
            type="number"
            min={0}
            value={costPerOffer}
            onChange={(e) => setCostPerOffer(Number(e.target.value))}
            className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 py-1 text-neutral-100"
          />
        </label>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded border border-neutral-800 px-3 py-2">
          <div className="text-xs text-neutral-500">Targeted</div>
          <div className="text-lg font-semibold text-neutral-100">
            {Math.round(result.nTargeted).toLocaleString()}
          </div>
        </div>
        <div className="rounded border border-neutral-800 px-3 py-2">
          <div className="text-xs text-neutral-500">Revenue saved</div>
          <div className="text-lg font-semibold text-teal-400">{formatGbp(result.revenueSaved)}</div>
        </div>
        <div className="rounded border border-neutral-800 px-3 py-2">
          <div className="text-xs text-neutral-500">Campaign cost</div>
          <div className="text-lg font-semibold text-neutral-100">{formatGbp(result.campaignCost)}</div>
        </div>
        <div className="rounded border border-neutral-800 px-3 py-2">
          <div className="text-xs text-neutral-500">Net ROI</div>
          <div className={`text-lg font-semibold ${result.netRoi >= 0 ? "text-teal-400" : "text-rose-400"}`}>
            {formatGbp(result.netRoi)} ({result.roiMultiple.toFixed(1)}x)
          </div>
        </div>
      </div>

      <div>
        <h4 className="text-xs uppercase tracking-wide text-neutral-500 mb-2">
          Sensitivity grid — net ROI (conversion × LTV)
        </h4>
        <div className="overflow-x-auto">
          <table className="text-xs w-full border-collapse">
            <thead>
              <tr>
                <th className="text-left text-neutral-500 font-normal pb-1">Conv. ↓ / LTV →</th>
                {LTV_LEVELS.map((ltv) => (
                  <th key={ltv} className="text-right text-neutral-500 font-normal pb-1 px-2">
                    £{ltv}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {CONVERSION_LEVELS.map((conv) => (
                <tr key={conv} className="border-t border-neutral-800">
                  <td className="py-1.5 text-neutral-400">{Math.round(conv * 100)}%</td>
                  {LTV_LEVELS.map((ltv) => {
                    const cell = computeRoi(nHighRisk, reachPct, conv, ltv, costPerOffer);
                    const isCurrent = conv === conversionRate && ltv === avgLtv;
                    return (
                      <td
                        key={ltv}
                        className={`text-right py-1.5 px-2 ${
                          isCurrent ? "bg-teal-950 text-teal-300 rounded" : "text-neutral-300"
                        }`}
                      >
                        {formatGbp(cell.netRoi)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
