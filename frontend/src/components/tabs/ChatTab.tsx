import { useRef, useEffect } from 'react'
import type { Message } from '../../types/api'
import ChatMessage from '../ChatMessage'
import ChatInput from '../ChatInput'
import EmptyState from '../EmptyState'
import { AlertCircle } from 'lucide-react'

interface Props {
  patientId: string
  messages: Message[]
  isQuerying: boolean
  error: string | null
  nResults: number
  prefill: string | undefined
  onSend: (query: string) => void
  onPrefillConsumed: () => void
}

export default function ChatTab({
  patientId,
  messages,
  isQuerying,
  error,
  prefill,
  onSend,
  onPrefillConsumed,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {error && (
          <div className="flex items-start gap-2 rounded-lg bg-red-50 border border-red-200 p-4 text-sm text-red-700">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {messages.length === 0 && !error && (
          <EmptyState patientId={patientId} onSuggest={(q) => onSend(q)} />
        )}

        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        <div ref={bottomRef} />
      </div>

      <ChatInput
        onSend={onSend}
        disabled={isQuerying || !patientId}
        placeholder={patientId ? `Ask about patient ${patientId}…` : 'Select a patient to begin'}
        prefill={prefill}
        onPrefillConsumed={onPrefillConsumed}
      />
    </div>
  )
}
