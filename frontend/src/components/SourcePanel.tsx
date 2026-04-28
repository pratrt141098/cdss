import type { SourceChunk } from '../types/api'

interface SourcePanelProps {
  sources: SourceChunk[]
}

export default function SourcePanel({ sources }: SourcePanelProps) {
  return (
    <div className="w-full mt-1 space-y-2">
      {sources.map((src, i) => (
        <div
          key={i}
          className="rounded-lg bg-slate-50 border border-slate-200 p-3 text-xs"
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-semibold text-slate-700">{src.section}</span>
            <span className="font-mono text-slate-400 bg-slate-100 rounded px-1.5 py-0.5">
              {src.score.toFixed(3)}
            </span>
          </div>
          <p className="text-slate-600 leading-relaxed font-mono whitespace-pre-wrap break-words">
            {src.text}
          </p>
        </div>
      ))}
    </div>
  )
}
