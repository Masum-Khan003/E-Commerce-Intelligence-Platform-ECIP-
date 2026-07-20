export function KpiCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 px-5 py-4">
      <div className="text-xs uppercase tracking-wide text-neutral-500 mb-1">{label}</div>
      <div className="text-2xl font-semibold text-teal-400">{value}</div>
      {sub && <div className="text-xs text-neutral-500 mt-1">{sub}</div>}
    </div>
  );
}
