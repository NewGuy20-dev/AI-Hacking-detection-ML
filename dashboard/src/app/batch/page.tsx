'use client'

import { useState, useCallback } from 'react'
import { useBatchScan } from '@/hooks/usePredict'
import { Upload, Download, FileText, FolderUp, Shield, ShieldAlert, CheckCircle, Clock, Trash2, Crosshair, Terminal } from 'lucide-react'

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
    if (lines.length > 0) mutate({ payloads: lines })
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
    <div className="space-y-6 animate-in fade-in zoom-in-95 duration-500">
      {/* Header */}
      <div className="relative overflow-hidden bg-black/60 border border-cyber-border/80 backdrop-blur-md p-6 cyber-card group">
        <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />
        <div className="absolute top-0 left-0 w-2 h-full bg-primary shadow-neon-primary" />
        
        <div className="relative flex items-center gap-4 pl-4">
          <div className="p-3 bg-black border border-primary/30 flex items-center justify-center relative group-hover:shadow-[0_0_15px_rgba(0,243,255,0.3)] transition-all">
            <div className="absolute inset-0 bg-primary/10 animate-pulse" />
            <FolderUp className="w-8 h-8 text-primary relative z-10" />
          </div>
          <div>
            <h1 className="text-2xl font-bold font-sans tracking-widest text-white uppercase flex items-center gap-3">
              BATCH_ANALYSIS <span className="text-[10px] bg-primary/20 text-primary border border-primary/50 px-2 py-0.5 mt-1 animate-pulse">FILE_UPLOAD</span>
            </h1>
            <p className="text-cyber-muted font-mono text-sm mt-1 uppercase tracking-wider">// Analyze multiple payloads simultaneously</p>
          </div>
        </div>
      </div>

      {/* Upload Area */}
      <div
        className={`relative border-2 border-dashed transition-all duration-300 p-1 font-mono ${
          dragActive 
            ? 'border-primary bg-primary/5 shadow-neon-primary' 
            : file 
              ? 'border-success/50 bg-success/5 shadow-[0_0_15px_rgba(0,255,102,0.1)]' 
              : 'border-cyber-border/50 hover:border-primary/50 bg-black/40'
        }`}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={() => setDragActive(false)}
      >
        <div className="absolute inset-0 bg-cyber-grid opacity-10 pointer-events-none" />
        
        <input
          type="file"
          accept=".txt"
          className="hidden"
          id="file-upload"
          onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])}
        />
        <label htmlFor="file-upload" className="block cursor-pointer p-12 text-center relative z-10">
          <div className={`mx-auto w-16 h-16 flex items-center justify-center border transition-all duration-300 mb-6 ${
            file ? 'bg-success/10 border-success/50 shadow-[0_0_15px_rgba(0,255,102,0.3)]' : 'bg-black/50 border-cyber-border/50 group-hover:border-primary/50'
          }`}>
            {file ? (
              <FileText className="w-8 h-8 text-success animate-pulse" />
            ) : (
              <Upload className={`w-8 h-8 ${dragActive ? 'text-primary animate-pulse' : 'text-cyber-muted'}`} />
            )}
          </div>
          
          {file ? (
            <div className="space-y-2">
              <p className="text-lg font-bold text-success tracking-widest uppercase">[{file.name}]</p>
              <p className="text-sm text-success/70 font-bold uppercase tracking-widest">
                <span className="text-white mr-2">{lines.length}</span>LINES_LOADED
              </p>
              <p className="text-xs text-cyber-muted mt-4 tracking-widest">[ CLICK_TO_REPLACE ]</p>
            </div>
          ) : (
            <div className="space-y-2">
              <p className="text-lg font-bold text-white tracking-widest uppercase">DROP_FILE_HERE</p>
              <p className="text-sm text-cyber-muted uppercase tracking-widest">or click to browse</p>
              <div className="inline-block mt-4 text-[10px] text-cyber-muted/50 border border-cyber-border/30 px-3 py-1 bg-black/50">
                .TXT FORMAT ONLY // MAX_LINES: 100
              </div>
            </div>
          )}
        </label>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={handleSubmit}
          disabled={isPending || lines.length === 0}
          className="cyber-button flex-1 justify-center disabled:opacity-50 disabled:cursor-not-allowed group relative overflow-hidden bg-primary/10 border-primary shadow-neon-primary text-primary hover:bg-primary/20 hover:text-white"
        >
          <div className="absolute inset-0 bg-cyber-scanline opacity-20 group-hover:opacity-40 animate-scan pointer-events-none" />
          {isPending ? (
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          ) : (
            <Crosshair className="w-4 h-4 group-hover:scale-110 transition-transform" />
          )}
          <span className="tracking-widest font-bold">
            {isPending ? '[ ANALYZING_DATA... ]' : `[ INITIATE_SCAN // ${lines.length}_ITEMS ]`}
          </span>
        </button>
        
        {data && (
          <button
            onClick={exportCSV}
            className="px-6 py-2 flex items-center justify-center gap-2 border border-cyber-border bg-black/50 text-cyber-text hover:text-white hover:border-white/30 font-mono tracking-widest transition-colors text-sm uppercase group"
          >
            <Download className="w-4 h-4 group-hover:-translate-y-1 transition-transform" />
            [ EXPORT_CSV ]
          </button>
        )}
        
        {(file || data) && (
          <button
            onClick={handleClear}
            className="px-6 py-2 flex items-center justify-center gap-2 border border-danger/30 bg-danger/5 text-danger hover:bg-danger/10 hover:shadow-neon-danger hover:border-danger font-mono tracking-widest transition-all text-sm uppercase group"
          >
            <Trash2 className="w-4 h-4 group-hover:scale-110 transition-transform" />
            [ PURGE_DATA ]
          </button>
        )}
      </div>

      {/* Results */}
      {data && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          {/* Summary Stats */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            <div className="border border-primary/30 bg-black/40 p-5 relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                <FileText className="w-16 h-16 text-primary" />
              </div>
              <p className="text-3xl font-bold font-mono text-primary text-shadow-sm mb-1">{data.results.length}</p>
              <p className="text-[10px] text-primary/70 font-mono tracking-widest uppercase">[ TOTAL_SCANNED ]</p>
            </div>
            
            <div className="border border-danger/30 bg-black/40 p-5 relative overflow-hidden group shadow-[inset_0_0_15px_rgba(255,0,60,0.05)]">
              <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                <ShieldAlert className="w-16 h-16 text-danger" />
              </div>
              <p className="text-3xl font-bold font-mono text-danger shadow-neon-danger mb-1">{threatCount}</p>
              <p className="text-[10px] text-danger/70 font-mono tracking-widest uppercase">[ THREATS_DETECTED ]</p>
            </div>

            <div className="border border-success/30 bg-black/40 p-5 relative overflow-hidden group shadow-[inset_0_0_15px_rgba(0,255,102,0.05)]">
              <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                <CheckCircle className="w-16 h-16 text-success" />
              </div>
              <p className="text-3xl font-bold font-mono text-success shadow-[0_0_10px_rgba(0,255,102,0.5)] mb-1">{safeCount}</p>
              <p className="text-[10px] text-success/70 font-mono tracking-widest uppercase">[ SAFE_ENTITIES ]</p>
            </div>
          </div>

          {/* Results List */}
          <div className="border border-cyber-border/80 bg-black/60 backdrop-blur-md relative">
            <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
            
            <div className="p-4 border-b border-cyber-border/50 flex items-center justify-between bg-white/5">
              <span className="font-bold tracking-widest uppercase text-white font-mono flex items-center gap-2">
                <Terminal className="w-4 h-4 text-primary" /> 
                BATCH_RESULTS
              </span>
              <div className="flex items-center gap-2 text-[10px] text-cyber-muted font-mono tracking-widest border border-cyber-border/50 px-2 py-1 bg-black">
                <Clock className="w-3 h-3 text-primary" />
                <span>SYS_TIME: {data.total_processing_time_ms.toFixed(0)}MS</span>
              </div>
            </div>
            
            <div className="max-h-96 overflow-y-auto">
              {data.results.map((r, i) => (
                <div
                  key={i}
                  className={`flex flex-col sm:flex-row sm:items-center gap-4 p-4 border-b border-cyber-border/30 hover:bg-white/5 transition-colors font-mono group border-l-2 ${
                    r.is_attack ? 'border-l-danger bg-danger/5' : 'border-l-success'
                  }`}
                >
                  <div className={`p-2 border bg-black shrink-0 ${
                    r.is_attack ? 'border-danger/50' : 'border-success/50'
                  }`}>
                    {r.is_attack ? (
                      <ShieldAlert className="w-4 h-4 text-danger group-hover:animate-pulse" />
                    ) : (
                      <CheckCircle className="w-4 h-4 text-success" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm truncate text-cyber-text tracking-wider group-hover:text-white transition-colors">
                      <span className="text-primary/50 mr-2">{'>'}</span>{lines[i]}
                    </p>
                    {r.attack_type && (
                      <p className="text-[10px] text-danger/70 uppercase tracking-widest mt-1">
                        TYPE: [ {r.attack_type.replace(/_/g, ' ')} ]
                      </p>
                    )}
                  </div>
                  <div className="flex items-center gap-4 shrink-0 mt-2 sm:mt-0">
                    <span className={`text-[10px] font-bold px-2 py-1 tracking-widest border bg-black ${
                      r.severity === 'CRITICAL' ? 'border-danger/50 text-danger' :
                      r.severity === 'HIGH' ? 'border-warning/50 text-warning' :
                      r.severity === 'MEDIUM' ? 'border-secondary/50 text-secondary' :
                      'border-success/50 text-success'
                    }`}>
                      {r.severity}
                    </span>
                    <span className={`text-sm font-bold w-12 text-right ${r.is_attack ? 'text-danger' : 'text-success'}`}>
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
