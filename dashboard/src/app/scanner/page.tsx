'use client'

import { useState } from 'react'
import { PayloadScanner, URLScanner } from '@/components/scanner'
import { Shield, FileCode, Globe, Zap } from 'lucide-react'

export default function ScannerPage() {
  const [activeTab, setActiveTab] = useState<'payload' | 'url'>('payload')

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-primary/10 via-purple-500/5 to-transparent border border-border/50 p-6">
        <div className="absolute top-0 right-0 w-64 h-64 bg-primary/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
        <div className="relative flex items-center gap-4">
          <div className="p-3 bg-gradient-to-br from-primary to-purple-600 rounded-xl shadow-lg">
            <Shield className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Security Scanner</h1>
            <p className="text-muted-foreground">Analyze payloads and URLs for potential threats</p>
          </div>
        </div>
        
        {/* Quick stats */}
        <div className="flex items-center gap-6 mt-4 pt-4 border-t border-border/50">
          <div className="flex items-center gap-2 text-sm">
            <Zap className="w-4 h-4 text-amber-500" />
            <span className="text-muted-foreground">Real-time analysis</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Shield className="w-4 h-4 text-emerald-500" />
            <span className="text-muted-foreground">ML-powered detection</span>
          </div>
        </div>
      </div>

      {/* Tab Switcher */}
      <div className="flex p-1 bg-muted/50 rounded-xl w-fit">
        <button
          onClick={() => setActiveTab('payload')}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium transition-all duration-200 ${
            activeTab === 'payload'
              ? 'bg-background shadow-sm text-foreground'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          <FileCode className="w-4 h-4" />
          Payload Scanner
        </button>
        <button
          onClick={() => setActiveTab('url')}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium transition-all duration-200 ${
            activeTab === 'url'
              ? 'bg-background shadow-sm text-foreground'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          <Globe className="w-4 h-4" />
          URL Scanner
        </button>
      </div>

      {/* Scanner Content */}
      <div className="animate-in">
        {activeTab === 'payload' ? <PayloadScanner /> : <URLScanner />}
      </div>
    </div>
  )
}
