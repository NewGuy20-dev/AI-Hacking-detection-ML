'use client'

import { useState } from 'react'
import { ResultCard } from './ResultCard'
import { useURLScan } from '@/hooks/usePredict'
import { Globe, Search, Shield, AlertTriangle, ExternalLink } from 'lucide-react'

const EXAMPLES = [
  { label: "Google (Safe)", value: "https://google.com", safe: true },
  { label: "Phishing Example", value: "http://paypa1-secure.tk/login", safe: false },
  { label: "GitHub (Safe)", value: "https://github.com", safe: true },
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
      <div className="rounded-2xl border border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden">
        <form onSubmit={handleSubmit}>
          <div className="p-4 border-b border-border/50">
            <label className="text-sm font-medium text-muted-foreground">URL Input</label>
          </div>
          <div className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-muted/50">
                <Globe className="w-5 h-5 text-muted-foreground" />
              </div>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com"
                className="flex-1 bg-transparent text-foreground placeholder:text-muted-foreground focus:outline-none text-sm"
              />
              {url && (
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg hover:bg-muted/50 transition-colors text-muted-foreground hover:text-foreground"
                >
                  <ExternalLink className="w-4 h-4" />
                </a>
              )}
            </div>
          </div>
          <div className="p-4 border-t border-border/50 bg-muted/30 flex items-center justify-end gap-2">
            {data && (
              <button
                type="button"
                onClick={() => { reset(); setUrl('') }}
                className="px-4 py-2 text-sm font-medium rounded-lg border border-border hover:bg-muted transition-colors"
              >
                Clear
              </button>
            )}
            <button
              type="submit"
              disabled={isPending || !url.trim()}
              className="flex items-center gap-2 px-5 py-2 text-sm font-medium rounded-lg bg-gradient-to-r from-primary to-purple-600 text-white hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isPending ? (
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              {isPending ? 'Analyzing...' : 'Analyze URL'}
            </button>
          </div>
        </form>
      </div>

      {/* Example URLs */}
      <div className="space-y-3">
        <p className="text-sm font-medium text-muted-foreground">Quick Examples</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.value}
              onClick={() => setUrl(ex.value)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl border transition-all ${
                ex.safe 
                  ? 'border-emerald-500/30 bg-emerald-500/5 hover:bg-emerald-500/10 text-emerald-400'
                  : 'border-red-500/30 bg-red-500/5 hover:bg-red-500/10 text-red-400'
              }`}
            >
              {ex.safe ? <Shield className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
              <span className="text-sm font-medium">{ex.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Result */}
      <ResultCard result={data || null} input={data ? url : undefined} type="url" />
    </div>
  )
}
