// dashboard/components/StatusStates.tsx
// Shared loading skeleton / error banner — blueprint §16 Fix #41: every
// data component needs a loading, success, and error state.

export function LoadingSkeleton({ label }: { label?: string }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="h-24 rounded-lg bg-neutral-800/60 animate-pulse border border-neutral-800"
        />
      ))}
      {label && <p className="col-span-full text-xs text-neutral-500">{label}</p>}
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-rose-900 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
      <strong className="text-rose-200">Data unavailable.</strong> {message}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-6 text-sm text-neutral-400 text-center">
      {message}
    </div>
  );
}
