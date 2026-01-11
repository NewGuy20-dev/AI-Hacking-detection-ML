'use client'

import { useState } from 'react'
import { ResultCard } from './ResultCard'
import { usePayloadScan } from '@/hooks/usePredict'
import { Search, Sparkles, Code, AlertTriangle, Terminal } from 'lucide-react'

const EXAMPLES = [
  { label: "SQL Injection", value: "' OR '1'='1", icon: AlertTriangle, color: "text-red-400" },
  { label: "XSS Attack", value: "<script>alert('XSS')</script>", icon: Code, color: "text-orange-400" },
  { label: "Command Injection", value: "; cat /etc/passwd", icon: Terminal, color: "text-amber-400" },
  { label: "Safe Input", value: "Hello, this is a normal message", icon: Sparkles, color: "text-emerald-400" },
]

export function PayloadScanner() {
  const [payload, setPayload] = useState('')
  const { mutate, data, isPending, reset } = usePayloadScan()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (payload.trim()) mutate(payload)
  }

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="rounded-2xl border border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden">
        <form onSubmit={handleSubmit}>
          <div className="p-4 border-b border-border/50">
            <label className="text-sm font-medium text-muted-foreground">Payload Input</label>
          </div>
          <div className="p-4">
            <textarea
              value={payload}
              onChange={(e) => setPayload(e.target.value)}
              placeholder="Enter payload to analyze (e.g., ' OR 1=1--)"
              rows={4}
              className="w-full bg-transparent resize-none text-foreground placeholder:text-muted-foreground focus:outline-none font-mono text-sm"
            />
          </div>
          <div className="p-4 border-t border-border/50 bg-muted/30 flex items-center justify-between gap-4">
            <div className="text-xs text-muted-foreground">
              {payload.length} characters
            </div>
            <div className="flex gap-2">
              {data && (
                <button
                  type="button"
                  onClick={() => { reset(); setPayload('') }}
                  className="px-4 py-2 text-sm font-medium rounded-lg border border-border hover:bg-muted transition-colors"
                >
                  Clear
                </button>
              )}
              <button
                type="submit"
                disabled={isPending || !payload.trim()}
                className="flex items-center gap-2 px-5 py-2 text-sm font-medium rounded-lg bg-gradient-to-r from-primary to-purple-600 text-white hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isPending ? (
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
                {isPending ? 'Analyzing...' : 'Analyze Payload'}
              </button>
            </div>
          </div>
        </form>
      </div>

      {/* Example Payloads */}
      <div className="space-y-3">
        <p className="text-sm font-medium text-muted-foreground">Quick Examples</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.value}
              onClick={() => setPayload(ex.value)}
              className="flex items-center gap-3 p-3 rounded-xl border border-border/50 bg-card/30 hover:bg-card/60 hover:border-border transition-all text-left group"
            >
              <div className={`p-2 rounded-lg bg-muted/50 ${ex.color}`}>
                <ex.icon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium">{ex.label}</p>
                <p className="text-xs text-muted-foreground truncate font-mono">{ex.value}</p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Result */}
      <ResultCard result={data || null} input={data ? payload : undefined} type="payload" />
    </div>
  )
}
