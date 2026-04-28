import { MessageSquare } from 'lucide-react'

const SUGGESTIONS = [
  'What medications is this patient on?',
  'What are the primary diagnoses?',
  'Are there any allergies or contraindications noted?',
  'What procedures were performed during this admission?',
]

interface EmptyStateProps {
  patientId: string
  onSuggest: (query: string) => void
}

export default function EmptyState({ patientId, onSuggest }: EmptyStateProps) {
  if (!patientId) {
    return (
      <div className="flex flex-col items-center justify-center h-full py-20 text-center">
        <div className="w-14 h-14 rounded-full bg-slate-100 flex items-center justify-center mb-4">
          <MessageSquare className="w-7 h-7 text-slate-400" />
        </div>
        <h2 className="text-base font-semibold text-slate-700 mb-1">No patient selected</h2>
        <p className="text-sm text-slate-500 max-w-xs">
          Select a patient from the sidebar to begin asking questions.
        </p>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center justify-center h-full py-16 text-center">
      <div className="w-14 h-14 rounded-full bg-blue-50 flex items-center justify-center mb-4">
        <MessageSquare className="w-7 h-7 text-blue-400" />
      </div>
      <h2 className="text-base font-semibold text-slate-800 mb-1">
        Ask about patient {patientId}
      </h2>
      <p className="text-sm text-slate-500 mb-6 max-w-sm">
        Answers are grounded in this patient's discharge notes only.
      </p>
      <div className="flex flex-col gap-2 w-full max-w-sm">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onSuggest(s)}
            className="text-left text-sm text-slate-600 bg-white border border-slate-200 rounded-lg px-4 py-2.5 hover:border-blue-300 hover:text-blue-700 hover:bg-blue-50 transition-colors"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  )
}
