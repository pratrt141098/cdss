import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import { Send } from 'lucide-react'

interface ChatInputProps {
  onSend: (query: string) => void
  disabled?: boolean
  placeholder?: string
  prefill?: string
  onPrefillConsumed?: () => void
}

export default function ChatInput({
  onSend,
  disabled,
  placeholder,
  prefill,
  onPrefillConsumed,
}: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Accept pre-filled queries from suggestion clicks
  useEffect(() => {
    if (prefill) {
      setValue(prefill)
      textareaRef.current?.focus()
      onPrefillConsumed?.()
    }
  }, [prefill, onPrefillConsumed])

  // Auto-resize to content
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [value])

  function submit() {
    const q = value.trim()
    if (!q || disabled) return
    onSend(q)
    setValue('')
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="px-6 py-4 bg-white border-t border-slate-200">
      <div className="flex items-end gap-3 rounded-xl border border-slate-300 bg-white px-4 py-3 focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500 transition-all">
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={placeholder ?? 'Ask about this patient…'}
          className="flex-1 resize-none bg-transparent text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none disabled:opacity-50 max-h-40 leading-relaxed"
        />
        <button
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="p-1.5 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
          aria-label="Send"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
      <p className="mt-2 text-xs text-slate-400 text-center">
        Shift+Enter for new line &middot; Enter to send
      </p>
    </div>
  )
}
