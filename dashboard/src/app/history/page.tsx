'use client'

import { useState, useEffect, useMemo } from 'react'
import { useHistoryStore } from '@/stores/historyStore'
import { Trash2, Download, Clock, FileCode, Globe, History, Shield, AlertTriangle, Search, Filter, X } from 'lucide-react'

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
        <div className="h-32 rounded-2xl bg-muted/50 animate-pulse" />
        <div className="h-12 rounded-xl bg-muted/50 animate-pulse" />
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-24 rounded-xl bg-muted/50 animate-pulse" />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-violet-500/10 via-purple-500/5 to-transparent border border-border/50 p-6">
        <div className="absolute top-0 right-0 w-64 h-64 bg-violet-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
        <div className="relative flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl shadow-lg">
              <History className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Scan History</h1>
              <p className="text-muted-foreground">{history.length} total scans recorded</p>
            </div>
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={exportJSON}
              disabled={filteredHistory.length === 0}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl border border-border hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
            <button
              onClick={clearHistory}
              disabled={history.length === 0}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl border border-border hover:bg-destructive/10 hover:border-destructive/50 hover:text-destructive transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Trash2 className="w-4 h-4" />
              Clear All
            </button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="flex items-center gap-6 mt-4 pt-4 border-t border-border/50">
          <div className="flex items-center gap-2 text-sm">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            <span className="text-muted-foreground">{threatCount} threats</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Shield className="w-4 h-4 text-emerald-400" />
            <span className="text-muted-foreground">{safeCount} safe</span>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search history..."
            className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-border/50 bg-card/50 text-sm focus:outline-none focus:border-primary transition-colors"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded-full hover:bg-muted"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>

        {/* Filter Buttons */}
        <div className="flex gap-2">
          <div className="flex p-1 bg-muted/50 rounded-lg">
            {(['all', 'threats', 'safe'] as FilterType[]).map((type) => (
              <button
                key={type}
                onClick={() => setFilterType(type)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors capitalize ${
                  filterType === type ? 'bg-background shadow-sm' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
          <div className="flex p-1 bg-muted/50 rounded-lg">
            {(['all', 'payload', 'url'] as ScanType[]).map((type) => (
              <button
                key={type}
                onClick={() => setScanType(type)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors capitalize ${
                  scanType === type ? 'bg-background shadow-sm' : 'text-muted-foreground hover:text-foreground'
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
        <div className="rounded-2xl border border-dashed border-border/50 bg-muted/20 p-12 text-center">
          <div className="inline-flex p-4 rounded-full bg-muted/50 mb-4">
            <Clock className="w-8 h-8 text-muted-foreground" />
          </div>
          <p className="text-muted-foreground font-medium">
            {history.length === 0 ? 'No scan history yet' : 'No results match your filters'}
          </p>
          <p className="text-sm text-muted-foreground/70 mt-1">
            {history.length === 0 ? 'Start scanning to build your history' : 'Try adjusting your search or filters'}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {filteredHistory.map((item, index) => (
            <div
              key={item.id}
              className={`group rounded-xl border overflow-hidden transition-all hover:shadow-md ${
                item.result.is_attack 
                  ? 'border-red-500/30 bg-red-500/5 hover:border-red-500/50' 
                  : 'border-emerald-500/30 bg-emerald-500/5 hover:border-emerald-500/50'
              }`}
              style={{ animationDelay: `${index * 30}ms` }}
            >
              <div className="p-4">
                <div className="flex items-start gap-4">
                  {/* Icon */}
                  <div className={`p-2 rounded-lg shrink-0 ${item.result.is_attack ? 'bg-red-500/20' : 'bg-emerald-500/20'}`}>
                    {item.result.is_attack ? (
                      <AlertTriangle className="w-5 h-5 text-red-400" />
                    ) : (
                      <Shield className="w-5 h-5 text-emerald-400" />
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                        item.type === 'payload' ? 'bg-blue-500/20 text-blue-400' : 'bg-purple-500/20 text-purple-400'
                      }`}>
                        {item.type === 'payload' ? <FileCode className="w-3 h-3 inline mr-1" /> : <Globe className="w-3 h-3 inline mr-1" />}
                        {item.type}
                      </span>
                      {item.result.attack_type && (
                        <span className="text-xs text-muted-foreground">
                          {item.result.attack_type.replace(/_/g, ' ')}
                        </span>
                      )}
                    </div>
                    <p className="font-mono text-sm break-all">{item.input}</p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {new Date(item.timestamp).toLocaleString()}
                      </span>
                      <span>{item.result.processing_time_ms.toFixed(1)}ms</span>
                    </div>
                  </div>

                  {/* Status */}
                  <div className="flex flex-col items-end gap-1 shrink-0">
                    <span className={`text-lg font-bold ${item.result.is_attack ? 'text-red-400' : 'text-emerald-400'}`}>
                      {(item.result.confidence * 100).toFixed(0)}%
                    </span>
                    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                      item.result.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400' :
                      item.result.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400' :
                      item.result.severity === 'MEDIUM' ? 'bg-amber-500/20 text-amber-400' :
                      'bg-emerald-500/20 text-emerald-400'
                    }`}>
                      {item.result.severity}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
