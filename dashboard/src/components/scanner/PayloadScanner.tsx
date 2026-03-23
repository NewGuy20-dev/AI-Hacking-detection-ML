'use client'

import { useState } from 'react'
import { ResultCard } from './ResultCard'
import { usePayloadScan } from '@/hooks/usePredict'
import { Search, Sparkles, Code, AlertTriangle, Terminal, Cpu } from 'lucide-react'

const EXAMPLES = [
  { label: "SQL_INJECTION", value: "' OR '1'='1", icon: AlertTriangle, color: "text-danger border-danger/50 bg-danger/10" },
  { label: "XSS_ATTACK", value: "<script>alert('XSS')</script>", icon: Code, color: "text-warning border-warning/50 bg-warning/10" },
  { label: "CMD_INJECTION", value: "; cat /etc/passwd", icon: Terminal, color: "text-secondary border-secondary/50 bg-secondary/10" },
  { label: "SAFE_INPUT", value: "Hello, this is a normal message", icon: Sparkles, color: "text-success border-success/50 bg-success/10" },
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
      <div className="border border-cyber-border/80 bg-black/60 backdrop-blur-sm relative overflow-hidden group">
        <div className="absolute top-0 left-0 w-1 h-full bg-primary opacity-50 shadow-neon-primary" />
        <form onSubmit={handleSubmit}>
          <div className="p-4 border-b border-cyber-border/80 bg-white/5 flex items-center justify-between">
            <label className="text-xs font-mono tracking-widest font-bold text-primary uppercase flex items-center gap-2">
              <Cpu className="w-4 h-4" /> PAYLOAD_INPUT_STREAM
            </label>
            <div className="text-[10px] text-cyber-muted font-mono tracking-widest uppercase border border-cyber-border/50 px-2 py-1 bg-black">
              CHARS: {payload.length}
            </div>
          </div>
          <div className="p-4 bg-black/40 relative">
            <div className="absolute inset-0 bg-cyber-grid opacity-10 pointer-events-none" />
            <textarea
              value={payload}
              onChange={(e) => setPayload(e.target.value)}
              placeholder="ENTER_PAYLOAD_DATA (E.G., ' OR 1=1--)"
              rows={4}
              className="w-full bg-transparent resize-none focus:outline-none font-mono text-sm placeholder:text-cyber-muted/40 text-cyber-text focus:text-white transition-colors relative z-10"
              spellCheck="false"
            />
          </div>
          <div className="p-4 border-t border-cyber-border/80 bg-black flex items-center justify-end gap-3">
            {data && (
              <button
                type="button"
                onClick={() => { reset(); setPayload('') }}
                className="px-6 py-2 text-[10px] uppercase font-mono font-bold tracking-widest border border-cyber-border hover:bg-white/5 transition-colors text-cyber-text hover:text-white"
              >
                [ CLEAR_BUFFER ]
              </button>
            )}
            <button
              type="submit"
              disabled={isPending || !payload.trim()}
              className="cyber-button flex items-center gap-2 px-6 py-2 disabled:opacity-50 disabled:cursor-not-allowed group/btn overflow-hidden relative"
            >
              <div className="absolute inset-0 bg-primary/20 scale-x-0 group-hover/btn:scale-x-100 origin-left transition-transform duration-300" />
              {isPending ? (
                <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin relative z-10" />
              ) : (
                <Search className="w-4 h-4 relative z-10" />
              )}
              <span className="text-[10px] tracking-widest relative z-10">
                {isPending ? '[ ANALYZING... ]' : '[ EXECUTE_SCAN ]'}
              </span>
            </button>
          </div>
        </form>
      </div>

      {/* Example Payloads */}
      <div className="space-y-3">
        <p className="text-[10px] font-mono font-bold tracking-widest text-cyber-muted uppercase">
          // QUICK_INJECTION_VECTORS
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.value}
              onClick={() => setPayload(ex.value)}
              className="flex items-center gap-3 p-3 border border-cyber-border/50 bg-black/40 hover:bg-white/5 hover:border-cyber-border transition-all text-left font-mono group"
            >
              <div className={`p-2 border ${ex.color}`}>
                <ex.icon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[10px] font-bold tracking-widest uppercase text-white group-hover:text-primary transition-colors">
                  {ex.label}
                </p>
                <p className="text-xs text-cyber-muted truncate mt-1">
                  <span className="text-primary/50 mr-1 select-none">{'>'}</span>{ex.value}
                </p>
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
