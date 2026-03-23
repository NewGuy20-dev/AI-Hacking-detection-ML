'use client'

import { useState, useEffect, useMemo } from 'react'
import { useHistoryStore } from '@/stores/historyStore'
import { Trash2, Download, Clock, FileCode, Globe, History, Shield, ShieldAlert, AlertTriangle, Search, X, Terminal } from 'lucide-react'

type FilterType = 'all' | 'threats' | 'safe'
type ScanType = 'all' | 'payload' | 'url'

export default function HistoryPage() {
  const [mounted, setMounted] = useState(false)
  const history = useHistoryStore((s) => s.history)
  const clearHistory = useHistoryStore((s) => s.clearHistory)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState<FilterType>('all')
  const [scanType, setScanType] = useState<ScanType>('all')

  useEffect(() => {
    useHistoryStore.persist.rehydrate()
    setMounted(true)
  }, [])

  const filteredHistory = useMemo(() => {
    return history.filter(item => {
      const matchesSearch = item.input.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesFilter = filterType === 'all' || 
        (filterType === 'threats' && item.result.is_attack) ||
        (filterType === 'safe' && !item.result.is_attack)
      const matchesScanType = scanType === 'all' || item.type === scanType
      return matchesSearch && matchesFilter && matchesScanType
    })
  }, [history, searchQuery, filterType, scanType])

  const exportJSON = () => {
    const blob = new Blob([JSON.stringify(filteredHistory, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'scan-history.json'
    a.click()
  }

  const threatCount = history.filter(h => h.result.is_attack).length
  const safeCount = history.filter(h => !h.result.is_attack).length

  if (!mounted) {
    return (
      <div className="space-y-6 animate-in">
        <div className="h-32 rounded-sm bg-cyber-border/20 border-cyber-border animate-pulse" />
        <div className="h-12 rounded-sm bg-cyber-border/20 border-cyber-border animate-pulse" />
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-24 rounded-sm bg-cyber-border/20 border-cyber-border animate-pulse" />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-in fade-in zoom-in-95 duration-500">
      {/* Header */}
      <div className="relative overflow-hidden bg-black/60 border border-cyber-border/80 p-6 cyber-card group backdrop-blur-md text-white">
        <div className="absolute top-0 right-0 w-64 h-64 bg-secondary/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
        <div className="absolute top-0 left-0 w-2 h-full bg-secondary shadow-[0_0_15px_rgba(191,0,255,0.8)]" />
        
        <div className="relative flex flex-col sm:flex-row sm:items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-black border border-secondary/30 relative">
              <div className="absolute inset-0 bg-secondary/10 animate-pulse" />
              <History className="w-8 h-8 text-secondary relative z-10" />
            </div>
            <div>
              <h1 className="text-2xl font-bold font-sans tracking-widest uppercase flex items-center gap-3">
                SYS_ARCHIVE <span className="text-[10px] bg-secondary/20 text-secondary border border-secondary/50 px-2 py-0.5 mt-1 font-mono tracking-widest font-bold shadow-[0_0_10px_rgba(191,0,255,0.3)]">{history.length} ENTRIES</span>
              </h1>
              <p className="text-cyber-muted font-mono text-sm mt-1 tracking-wider">// Recorded Threat Intelligence Log</p>
            </div>
          </div>
          
          <div className="flex gap-4">
            <button
              onClick={exportJSON}
              disabled={filteredHistory.length === 0}
              className="flex items-center gap-2 px-4 py-2 text-[10px] uppercase tracking-widest font-bold font-mono border border-cyber-border/50 bg-black hover:bg-white/5 hover:border-white/30 text-cyber-text transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
            >
              <Download className="w-4 h-4 group-hover:-translate-y-1 transition-transform" />
              [ EXPORT_LOG ]
            </button>
            <button
              onClick={clearHistory}
              disabled={history.length === 0}
              className="flex items-center gap-2 px-4 py-2 text-[10px] uppercase tracking-widest font-bold font-mono border border-danger/30 bg-danger/5 hover:bg-danger/10 hover:shadow-neon-danger hover:border-danger text-danger transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
            >
              <Trash2 className="w-4 h-4 group-hover:scale-110 transition-transform" />
              [ PURGE_ALL ]
            </button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="flex items-center gap-8 mt-6 pt-4 border-t border-cyber-border/50 font-mono text-sm">
          <div className="flex items-center gap-3 bg-danger/5 border border-danger/30 px-3 py-1.5 shadow-[inset_0_0_10px_rgba(255,0,60,0.1)]">
            <AlertTriangle className="w-4 h-4 text-danger animate-pulse shadow-neon-danger rounded-full" />
            <span className="text-danger tracking-widest font-bold uppercase">{threatCount} THREATS</span>
          </div>
          <div className="flex items-center gap-3 bg-success/5 border border-success/30 px-3 py-1.5 shadow-[inset_0_0_10px_rgba(0,255,102,0.1)]">
            <Shield className="w-4 h-4 text-success" />
            <span className="text-success tracking-widest font-bold uppercase">{safeCount} SAFE</span>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col md:flex-row gap-4">
        {/* Search */}
        <div className="relative flex-1 group">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <Search className="w-4 h-4 text-primary group-focus-within:animate-pulse" />
            <span className="text-primary font-mono ml-2 font-bold select-none">{'>'}</span>
          </div>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="SEARCH_QUERY..."
            className="w-full pl-12 pr-10 py-3 bg-black/60 border border-cyber-border focus:border-primary focus:shadow-[inset_0_0_15px_rgba(0,243,255,0.1)] outline-none text-white font-mono placeholder:text-cyber-muted/50 transition-all uppercase tracking-wider"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-2 hover:bg-danger/20 hover:text-danger text-cyber-muted transition-colors rounded-sm"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Filter Buttons */}
        <div className="flex gap-4">
          <div className="flex p-1 bg-black/40 border border-cyber-border">
            {(['all', 'threats', 'safe'] as FilterType[]).map((type) => (
              <button
                key={type}
                onClick={() => setFilterType(type)}
                className={`px-4 py-2 text-[10px] font-bold font-mono uppercase tracking-widest transition-all ${
                  filterType === type 
                    ? 'bg-primary/20 text-primary border border-primary/50 shadow-[inset_0_0_10px_rgba(0,243,255,0.2)]' 
                    : 'text-cyber-muted hover:text-white border border-transparent'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
          <div className="flex p-1 bg-black/40 border border-cyber-border">
            {(['all', 'payload', 'url'] as ScanType[]).map((type) => (
              <button
                key={type}
                onClick={() => setScanType(type)}
                className={`px-4 py-2 text-[10px] font-bold font-mono uppercase tracking-widest transition-all ${
                  scanType === type 
                    ? 'bg-secondary/20 text-secondary border border-secondary/50 shadow-[inset_0_0_10px_rgba(191,0,255,0.2)]' 
                    : 'text-cyber-muted hover:text-white border border-transparent'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* History List */}
      {filteredHistory.length === 0 ? (
        <div className="border border-cyber-border border-dashed bg-black/40 p-16 text-center">
          <div className="inline-flex p-4 bg-black border border-cyber-border mb-4">
            <Terminal className="w-10 h-10 text-cyber-muted opacity-50" />
          </div>
          <p className="text-primary font-mono tracking-widest uppercase font-bold text-lg">
            [ {history.length === 0 ? 'ARCHIVE_EMPTY' : 'NO_MATCHING_RECORDS'} ]
          </p>
          <p className="text-sm text-cyber-muted mt-2 font-mono uppercase tracking-wider">
            {history.length === 0 ? 'SYS awaits operation logging.' : 'Adjust search paramaters to retrieve data.'}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredHistory.map((item, index) => (
            <div
              key={item.id}
              className={`group flex flex-col md:flex-row md:items-center gap-4 p-4 border transition-all duration-300 font-mono bg-black/40 backdrop-blur-sm border-l-4 hover:bg-white/5 ${
                item.result.is_attack 
                  ? 'border-l-danger border-cyber-border/50 hover:border-danger/50' 
                  : 'border-l-success border-cyber-border/50 hover:border-success/50'
              }`}
              style={{ animationDelay: `${index * 50}ms`, animationFillMode: 'both' }}
            >
              {/* Icon */}
              <div className={`p-3 shrink-0 flex items-center justify-center border bg-black ${
                item.result.is_attack ? 'border-danger/30 shadow-[inset_0_0_10px_rgba(255,0,60,0.1)]' : 'border-success/30'
              }`}>
                {item.result.is_attack ? (
                  <ShieldAlert className="w-6 h-6 text-danger group-hover:animate-pulse" />
                ) : (
                  <Shield className="w-6 h-6 text-success" />
                )}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0 flex flex-col justify-center">
                <div className="flex items-center gap-3 mb-2">
                  <span className={`text-[10px] font-bold px-2 py-0.5 uppercase tracking-widest border border-dashed ${
                    item.type === 'payload' ? 'text-primary border-primary/50' : 'text-secondary border-secondary/50'
                  }`}>
                    {item.type === 'payload' ? <FileCode className="w-3 h-3 inline mr-1" /> : <Globe className="w-3 h-3 inline mr-1" />}
                    {item.type}
                  </span>
                  
                  {item.result.attack_type && (
                    <span className="text-[10px] text-danger/80 uppercase tracking-widest">
                      // {item.result.attack_type.replace(/_/g, ' ')}
                    </span>
                  )}
                </div>
                
                <p className="font-mono text-sm break-all text-cyber-text group-hover:text-white transition-colors">
                  <span className="text-primary/50 mr-2 select-none">{'>'}</span>{item.input}
                </p>
                
                <div className="flex items-center gap-6 mt-3 text-[10px] text-cyber-muted font-mono tracking-widest uppercase">
                  <span className="flex items-center gap-1.5 opacity-70">
                    <Clock className="w-3 h-3" />
                    {new Date(item.timestamp).toISOString().replace('T', ' ')}
                  </span>
                  <span className="opacity-70 border-l border-cyber-border pl-4">EXEC_TIME: {item.result.processing_time_ms.toFixed(1)}MS</span>
                </div>
              </div>

              {/* Status */}
              <div className="flex flex-col items-end justify-center gap-2 shrink-0 border-t md:border-t-0 md:border-l border-cyber-border/50 pt-4 md:pt-0 md:pl-6 min-w-[120px]">
                <span className={`text-2xl font-bold font-mono tracking-tighter ${item.result.is_attack ? 'text-danger' : 'text-success'}`}>
                  {(item.result.confidence * 100).toFixed(0)}%
                </span>
                
                <span className={`text-[10px] font-bold px-3 py-1 tracking-widest uppercase border bg-black ${
                  item.result.severity === 'CRITICAL' ? 'border-danger/50 text-danger' :
                  item.result.severity === 'HIGH' ? 'border-warning/50 text-warning' :
                  item.result.severity === 'MEDIUM' ? 'border-secondary/50 text-secondary' :
                  'border-success/50 text-success'
                }`}>
                  Lvl: {item.result.severity}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
