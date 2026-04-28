import { useState } from 'react'
import { Loader2, CheckCircle2, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react'
import type { PipelineStatus } from '../types/api'

interface Props {
  status: PipelineStatus | null
  justReady: boolean
}

export default function StatusBanner({ status, justReady }: Props) {
  const [expanded, setExpanded] = useState(false)

  // Nothing to show once ready and the flash has faded
  if (!status) return null
  if (status.ready && !status.building && !justReady) return null

  // Error state
  if (status.error) {
    return (
      <div className="bg-red-50 border-b border-red-200 px-5 py-3 flex items-start gap-2.5">
        <AlertCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-medium text-red-800">Pipeline build failed</p>
          <p className="text-xs text-red-600 mt-0.5">{status.error}</p>
        </div>
      </div>
    )
  }

  // Just became ready — brief success flash
  if (justReady) {
    return (
      <div className="bg-emerald-50 border-b border-emerald-200 px-5 py-2.5 flex items-center gap-2.5">
        <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
        <p className="text-sm font-medium text-emerald-800">
          Pipeline ready — {status.logs.length} steps completed
        </p>
      </div>
    )
  }

  // Building
  if (status.building) {
    const latest = status.logs[status.logs.length - 1] ?? 'Initialising…'
    return (
      <div className="bg-amber-50 border-b border-amber-200 px-5 py-2.5">
        <div className="flex items-center gap-3">
          <Loader2 className="w-4 h-4 text-amber-500 animate-spin shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="flex items-baseline gap-2">
              <span className="text-sm font-semibold text-amber-800">Building pipeline</span>
              <span className="text-xs text-amber-500">
                {status.logs.length} / ~9 steps
              </span>
            </div>
            <p className="text-xs text-amber-600 truncate">{latest}</p>
          </div>
          <button
            onClick={() => setExpanded((e) => !e)}
            className="flex items-center gap-1 text-xs text-amber-500 hover:text-amber-700 transition-colors shrink-0"
          >
            {expanded ? 'hide' : 'show'} log
            {expanded
              ? <ChevronUp className="w-3 h-3" />
              : <ChevronDown className="w-3 h-3" />}
          </button>
        </div>

        {expanded && (
          <div className="mt-2 rounded-lg bg-amber-100/70 px-3 py-2 max-h-40 overflow-y-auto space-y-0.5">
            {status.logs.map((line, i) => (
              <p key={i} className="text-xs font-mono text-amber-800 leading-relaxed">
                <span className="text-amber-400 select-none mr-1">{i + 1}.</span>
                {line}
              </p>
            ))}
          </div>
        )}
      </div>
    )
  }

  // Server connecting (status fetched but neither building nor ready yet)
  return (
    <div className="bg-slate-50 border-b border-slate-200 px-5 py-2 flex items-center gap-2">
      <Loader2 className="w-3.5 h-3.5 text-slate-400 animate-spin" />
      <p className="text-xs text-slate-500">Connecting to server…</p>
    </div>
  )
}
