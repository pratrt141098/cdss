import { useState } from 'react'
import { ChevronDown, ChevronUp, FileText, Loader2, User, Bot } from 'lucide-react'
import type { Message } from '../types/api'
import SourcePanel from './SourcePanel'

interface ChatMessageProps {
  message: Message
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shrink-0 mt-0.5">
          <Bot className="w-4 h-4 text-blue-600" />
        </div>
      )}

      <div className={`max-w-[80%] flex flex-col gap-1 ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-blue-600 text-white rounded-tr-sm'
              : 'bg-white border border-slate-200 text-slate-800 rounded-tl-sm shadow-sm'
          }`}
        >
          {message.loading ? (
            <div className="flex items-center gap-2 text-slate-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Thinking…</span>
            </div>
          ) : (
            <p className="whitespace-pre-wrap">{message.content}</p>
          )}
        </div>

        {!isUser && !message.loading && message.sources && (
          <div className="flex items-center gap-3 px-1">
            <span className="text-xs text-slate-400">
              {message.elapsed_s?.toFixed(1)}s &middot; {message.n_chunks} chunks
            </span>
            <button
              onClick={() => setSourcesOpen((o) => !o)}
              className="flex items-center gap-1 text-xs text-blue-500 hover:text-blue-700 transition-colors"
            >
              <FileText className="w-3 h-3" />
              {sourcesOpen ? 'Hide' : 'Show'} sources
              {sourcesOpen ? (
                <ChevronUp className="w-3 h-3" />
              ) : (
                <ChevronDown className="w-3 h-3" />
              )}
            </button>
          </div>
        )}

        {sourcesOpen && message.sources && <SourcePanel sources={message.sources} />}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center shrink-0 mt-0.5">
          <User className="w-4 h-4 text-slate-600" />
        </div>
      )}
    </div>
  )
}
