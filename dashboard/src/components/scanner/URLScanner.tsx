'use client'

import { useState } from 'react'
import { ResultCard } from './ResultCard'
import { useURLScan } from '@/hooks/usePredict'
import { Globe, Search, Shield, AlertTriangle, ExternalLink, Network } from 'lucide-react'

const EXAMPLES = [
  { label: "GOOGLE_SAFE_NODE", value: "https://google.com", safe: true },
  { label: "PHISHING_HONEYPOT", value: "http://paypa1-secure.tk/login", safe: false },
  { label: "GITHUB_REPLICA", value: "https://github.com", safe: true },
]

export function URLScanner() {
  const [url, setUrl] = useState('')
  const { mutate, data, isPending, reset } = useURLScan()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (url.trim()) mutate(url)
  }

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="border border-cyber-border/80 bg-black/60 backdrop-blur-sm relative overflow-hidden group">
        <div className="absolute top-0 left-0 w-1 h-full bg-secondary opacity-50 shadow-[0_0_15px_rgba(191,0,255,0.8)]" />
        <form onSubmit={handleSubmit}>
          <div className="p-4 border-b border-cyber-border/80 bg-white/5">
            <label className="text-xs font-mono tracking-widest font-bold text-secondary uppercase flex items-center gap-2">
              <Network className="w-4 h-4" /> URL_ENDPOINT_TARGET
            </label>
          </div>
          <div className="p-4 bg-black/40 relative">
            <div className="absolute inset-0 bg-cyber-grid opacity-10 pointer-events-none" />
            <div className="flex items-center gap-3 relative z-10">
              <div className="p-2 border border-cyber-border/50 bg-black">
                <Globe className="w-5 h-5 text-secondary animate-pulse" />
              </div>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="HTTPS://EXAMPLE.COM"
                className="flex-1 bg-transparent text-cyber-text focus:text-white placeholder:text-cyber-muted/40 focus:outline-none font-mono text-sm tracking-widest transition-colors uppercase"
                spellCheck="false"
              />
              {url && (
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 border border-cyber-border/50 bg-black hover:bg-white/10 hover:border-white/30 transition-colors text-cyber-muted hover:text-white"
                >
                  <ExternalLink className="w-4 h-4" />
                </a>
              )}
            </div>
          </div>
          <div className="p-4 border-t border-cyber-border/80 bg-black flex items-center justify-end gap-3">
            {data && (
              <button
                type="button"
                onClick={() => { reset(); setUrl('') }}
                className="px-6 py-2 text-[10px] uppercase font-mono font-bold tracking-widest border border-cyber-border hover:bg-white/5 transition-colors text-cyber-text hover:text-white"
              >
                [ CLEAR_TARGET ]
              </button>
            )}
            <button
              type="submit"
              disabled={isPending || !url.trim()}
              className="cyber-button flex items-center gap-2 px-6 py-2 disabled:opacity-50 disabled:cursor-not-allowed group/btn overflow-hidden relative !text-secondary hover:!border-secondary hover:!shadow-[0_0_15px_rgba(191,0,255,0.5)]"
              style={{ background: 'linear-gradient(180deg, rgba(191,0,255,0.05) 0%, rgba(191,0,255,0) 100%)'}}
            >
              <div className="absolute inset-0 bg-secondary/20 scale-x-0 group-hover/btn:scale-x-100 origin-left transition-transform duration-300" />
              {isPending ? (
                <div className="w-4 h-4 border-2 border-secondary border-t-transparent rounded-full animate-spin relative z-10" />
              ) : (
                <Search className="w-4 h-4 relative z-10" />
              )}
              <span className="text-[10px] tracking-widest relative z-10 font-bold">
                {isPending ? '[ SCANNING_NODE... ]' : '[ EXECUTE_SCAN ]'}
              </span>
            </button>
          </div>
        </form>
      </div>

      {/* Example URLs */}
      <div className="space-y-3">
        <p className="text-[10px] font-mono font-bold tracking-widest text-cyber-muted uppercase">
          // KNOWN_URL_TARGETS
        </p>
        <div className="flex flex-wrap gap-3">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.value}
              onClick={() => setUrl(ex.value)}
              className={`flex items-center gap-3 px-4 py-3 border transition-all font-mono text-left group ${
                ex.safe 
                  ? 'border-success/30 bg-success/5 hover:bg-success/10 hover:border-success/50'
                  : 'border-danger/30 bg-danger/5 hover:bg-danger/10 hover:border-danger/50'
              }`}
            >
              <div className={`p-1.5 border ${ex.safe ? 'border-success/50 text-success' : 'border-danger/50 text-danger'}`}>
                {ex.safe ? <Shield className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4 group-hover:animate-pulse" />}
              </div>
              <span className="text-[10px] font-bold tracking-widest uppercase text-white group-hover:text-primary transition-colors">
                {ex.label}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Result */}
      <ResultCard result={data || null} input={data ? url : undefined} type="url" />
    </div>
  )
}
