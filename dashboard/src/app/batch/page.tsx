'use client'

import { useState, useCallback } from 'react'
import { useBatchScan } from '@/hooks/usePredict'
import { Upload, Download, FileText, FolderUp, Shield, AlertTriangle, CheckCircle, Clock, Trash2, Sparkles } from 'lucide-react'

export default function BatchPage() {
  const [file, setFile] = useState<File | null>(null)
  const [lines, setLines] = useState<string[]>([])
  const [dragActive, setDragActive] = useState(false)
  const { mutate, data, isPending, reset } = useBatchScan()

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    const f = e.dataTransfer.files[0]
    if (f?.type === 'text/plain' || f?.name.endsWith('.txt')) processFile(f)
  }, [])

  const processFile = async (f: File) => {
    setFile(f)
    const text = await f.text()
    const parsed = text.split('\n').filter((l) => l.trim()).slice(0, 100)
    setLines(parsed)
  }

  const handleSubmit = () => {
    if (lines.length > 0) mutate(lines)
  }

  const handleClear = () => {
    setFile(null)
    setLines([])
    reset()
  }

  const exportCSV = () => {
    if (!data) return
    const csv = ['Input,Verdict,Confidence,Attack Type,Severity']
    data.results.forEach((r, i) => {
      csv.push(`"${lines[i]}",${r.is_attack ? 'Malicious' : 'Safe'},${(r.confidence * 100).toFixed(1)}%,${r.attack_type || 'N/A'},${r.severity}`)
    })
    const blob = new Blob([csv.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'scan-results.csv'
    a.click()
  }

  const threatCount = data?.results.filter(r => r.is_attack).length || 0
  const safeCount = data?.results.filter(r => !r.is_attack).length || 0

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-amber-500/10 via-orange-500/5 to-transparent border border-border/50 p-6">
        <div className="absolute top-0 right-0 w-64 h-64 bg-amber-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
        <div className="relative flex items-center gap-4">
          <div className="p-3 bg-gradient-to-br from-amber-500 to-orange-600 rounded-xl shadow-lg">
            <FolderUp className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Batch Analysis</h1>
            <p className="text-muted-foreground">Upload a file to analyze multiple payloads at once</p>
          </div>
        </div>
      </div>

      {/* Upload Area */}
      <div
        className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 ${
          dragActive 
            ? 'border-primary bg-primary/5' 
            : file 
              ? 'border-emerald-500/50 bg-emerald-500/5' 
              : 'border-border/50 hover:border-border bg-card/30'
        }`}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={() => setDragActive(false)}
      >
        <input
          type="file"
          accept=".txt"
          className="hidden"
          id="file-upload"
          onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])}
        />
        <label htmlFor="file-upload" className="block cursor-pointer p-12 text-center">
          <div className={`inline-flex p-4 rounded-full mb-4 ${file ? 'bg-emerald-500/20' : 'bg-muted/50'}`}>
            {file ? (
              <FileText className="w-10 h-10 text-emerald-400" />
            ) : (
              <Upload className="w-10 h-10 text-muted-foreground" />
            )}
          </div>
          
          {file ? (
            <>
              <p className="text-lg font-medium text-emerald-400">{file.name}</p>
              <p className="text-sm text-muted-foreground mt-1">{lines.length} lines loaded • Click to replace</p>
            </>
          ) : (
            <>
              <p className="text-lg font-medium">Drop your file here</p>
              <p className="text-sm text-muted-foreground mt-1">or click to browse • .txt files only • max 100 lines</p>
            </>
          )}
        </label>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleSubmit}
          disabled={isPending || lines.length === 0}
          className="flex items-center gap-2 px-5 py-2.5 text-sm font-medium rounded-xl bg-gradient-to-r from-amber-500 to-orange-600 text-white hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isPending ? (
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <Sparkles className="w-4 h-4" />
          )}
          {isPending ? 'Analyzing...' : `Analyze ${lines.length} Items`}
        </button>
        
        {data && (
          <button
            onClick={exportCSV}
            className="flex items-center gap-2 px-5 py-2.5 text-sm font-medium rounded-xl border border-border hover:bg-muted transition-colors"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        )}
        
        {(file || data) && (
          <button
            onClick={handleClear}
            className="flex items-center gap-2 px-5 py-2.5 text-sm font-medium rounded-xl border border-border hover:bg-destructive/10 hover:border-destructive/50 hover:text-destructive transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Clear
          </button>
        )}
      </div>

      {/* Results */}
      {data && (
        <div className="space-y-4">
          {/* Summary Stats */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="rounded-xl border border-border/50 bg-card/50 p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-muted/50">
                  <FileText className="w-5 h-5 text-muted-foreground" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{data.results.length}</p>
                  <p className="text-xs text-muted-foreground">Total Scanned</p>
                </div>
              </div>
            </div>
            <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-red-500/20">
                  <AlertTriangle className="w-5 h-5 text-red-400" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-red-400">{threatCount}</p>
                  <p className="text-xs text-muted-foreground">Threats Found</p>
                </div>
              </div>
            </div>
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-emerald-500/20">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-emerald-400">{safeCount}</p>
                  <p className="text-xs text-muted-foreground">Safe Items</p>
                </div>
              </div>
            </div>
          </div>

          {/* Results List */}
          <div className="rounded-2xl border border-border/50 bg-card/50 overflow-hidden">
            <div className="p-4 border-b border-border/50 flex items-center justify-between">
              <span className="font-medium">Scan Results</span>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Clock className="w-4 h-4" />
                <span>{data.total_processing_time_ms.toFixed(0)}ms total</span>
              </div>
            </div>
            <div className="max-h-96 overflow-y-auto divide-y divide-border/30">
              {data.results.map((r, i) => (
                <div
                  key={i}
                  className="flex items-center gap-4 p-4 hover:bg-muted/30 transition-colors"
                >
                  <div className={`p-2 rounded-lg ${r.is_attack ? 'bg-red-500/20' : 'bg-emerald-500/20'}`}>
                    {r.is_attack ? (
                      <AlertTriangle className="w-4 h-4 text-red-400" />
                    ) : (
                      <Shield className="w-4 h-4 text-emerald-400" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-mono truncate">{lines[i]}</p>
                    {r.attack_type && (
                      <p className="text-xs text-muted-foreground">{r.attack_type.replace(/_/g, ' ')}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                      r.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400' :
                      r.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400' :
                      r.severity === 'MEDIUM' ? 'bg-amber-500/20 text-amber-400' :
                      'bg-emerald-500/20 text-emerald-400'
                    }`}>
                      {r.severity}
                    </span>
                    <span className={`text-sm font-semibold ${r.is_attack ? 'text-red-400' : 'text-emerald-400'}`}>
                      {(r.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
