'use client'

import { PredictResponse } from '@/types/api'
import { Shield, ShieldAlert, Clock, Zap, AlertTriangle, CheckCircle, Info, ScanLine, Terminal } from 'lucide-react'

interface ResultCardProps {
  result: PredictResponse | null
  input?: string
  type?: 'payload' | 'url'
}

function getSeverityConfig(severity: string) {
  switch (severity) {
    case 'CRITICAL':
      return { color: 'text-danger', bg: 'bg-danger/10', border: 'border-danger/50', icon: AlertTriangle, shadow: 'shadow-neon-danger' }
    case 'HIGH':
      return { color: 'text-warning', bg: 'bg-warning/10', border: 'border-warning/50', icon: AlertTriangle, shadow: 'shadow-[0_0_10px_rgba(255,176,0,0.5)]' }
    case 'MEDIUM':
      return { color: 'text-secondary', bg: 'bg-secondary/10', border: 'border-secondary/50', icon: Info, shadow: 'shadow-[0_0_10px_rgba(191,0,255,0.5)]' }
    default:
      return { color: 'text-success', bg: 'bg-success/10', border: 'border-success/50', icon: CheckCircle, shadow: 'shadow-[0_0_10px_rgba(0,255,102,0.5)]' }
  }
}

export function ResultCard({ result, input, type = 'payload' }: ResultCardProps) {
  if (!result) {
    return (
      <div className="border border-cyber-border border-dashed bg-black/40 p-12 text-center relative overflow-hidden group">
        <div className="absolute inset-0 bg-cyber-grid opacity-10 pointer-events-none" />
        <div className="inline-flex p-4 border border-cyber-border/50 bg-black mb-4 relative z-10">
          <ScanLine className="w-8 h-8 text-cyber-muted opacity-50" />
        </div>
        <p className="text-primary font-mono tracking-widest uppercase font-bold text-sm relative z-10">
          [ AWAITING_{type.toUpperCase()}_INPUT ]
        </p>
        <p className="text-xs text-cyber-muted mt-2 font-mono uppercase tracking-wider relative z-10">
          SYS STANDBY...
        </p>
      </div>
    )
  }

  const severityConfig = getSeverityConfig(result.severity)
  const confidencePercent = (result.confidence * 100).toFixed(1)

  return (
    <div className={`border bg-black/80 font-mono transition-all animate-in zoom-in-95 duration-300 relative overflow-hidden ${
      result.is_attack 
        ? 'border-danger/50 shadow-[0_0_20px_rgba(255,0,60,0.15)]' 
        : 'border-success/50 shadow-[0_0_20px_rgba(0,255,102,0.15)]'
    }`}>
      {/* Background Decorator */}
      <div className={`absolute top-0 right-0 w-64 h-64 rounded-full blur-3xl opacity-20 pointer-events-none -translate-y-1/2 translate-x-1/2 ${
        result.is_attack ? 'bg-danger' : 'bg-success'
      }`} />

      {/* Header */}
      <div className={`p-5 border-b relative z-10 ${result.is_attack ? 'border-danger/30 bg-danger/5' : 'border-success/30 bg-success/5'}`}>
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className={`p-3 border ${result.is_attack ? 'border-danger/50 bg-black' : 'border-success/50 bg-black'}`}>
              {result.is_attack ? (
                <ShieldAlert className="w-7 h-7 text-danger animate-pulse" />
              ) : (
                <Shield className="w-7 h-7 text-success" />
              )}
            </div>
            <div>
              <h3 className={`text-xl font-bold tracking-widest uppercase flex items-center gap-3 ${result.is_attack ? 'text-danger' : 'text-success'}`}>
                {result.is_attack ? 'THREAT_DETECTED' : 'SAFE_ENTITY'}
                {result.is_attack && <span className="w-2 h-2 bg-danger rounded-full animate-pulse shadow-neon-danger" />}
              </h3>
              {result.attack_type && (
                <p className="text-[10px] text-danger/80 uppercase tracking-widest mt-1">
                  // {result.attack_type.replace(/_/g, ' ')}
                </p>
              )}
            </div>
          </div>
          
          {/* Severity Badge */}
          <div className={`flex flex-col items-end gap-1 px-3 py-1.5 border bg-black ${severityConfig.border}`}>
            <span className={`text-[10px] font-bold tracking-widest uppercase ${severityConfig.color}`}>
              [ SEVERITY ]
            </span>
            <div className="flex items-center gap-1.5">
              <severityConfig.icon className={`w-3 h-3 ${severityConfig.color}`} />
              <span className={`text-sm font-bold tracking-widest uppercase ${severityConfig.color}`}>
                {result.severity}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Confidence Bar */}
      <div className="p-5 border-b border-cyber-border/50 relative z-10 bg-black/40">
        <div className="flex items-center justify-between mb-3 text-[10px] tracking-widest uppercase font-bold">
          <span className="text-cyber-muted flex items-center gap-2">
            <Terminal className="w-3 h-3 text-primary" /> CONFIDENCE_LEVEL
          </span>
          <span className={`text-sm ${result.is_attack ? 'text-danger' : 'text-success'}`}>
            {confidencePercent}%
          </span>
        </div>
        <div className="h-2 bg-cyber-border/30 overflow-hidden border border-cyber-border">
          <div 
            className={`h-full transition-all duration-1000 ease-out flex items-center justify-end pr-1 ${
              result.is_attack 
                ? 'bg-danger shadow-neon-danger' 
                : 'bg-success shadow-[0_0_10px_rgba(0,255,102,0.8)]'
            }`}
            style={{ width: `${confidencePercent}%` }}
          >
            <div className="w-1 h-full bg-white opacity-50" />
          </div>
        </div>
      </div>

      {/* Input Display */}
      {input && (
        <div className="p-5 border-b border-cyber-border/50 relative z-10">
          <p className="text-[10px] font-bold text-cyber-muted mb-2 uppercase tracking-widest">
            // ANALYZED_TARGET_DATA
          </p>
          <div className="p-3 bg-black border border-cyber-border/50 relative group">
            <div className="absolute top-0 left-0 w-1 h-full bg-cyber-border transition-colors group-hover:bg-primary" />
            <code className="text-sm break-all font-mono text-cyber-text tracking-wider pl-2 block">
              <span className="text-primary/50 select-none mr-2">{'>'}</span>{input}
            </code>
          </div>
        </div>
      )}

      {/* Footer Stats */}
      <div className="p-4 bg-black flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-6 border-t border-cyber-border/80 relative z-10">
        <div className="flex items-center gap-2 text-[10px] text-cyber-muted uppercase tracking-widest">
          <Clock className="w-3 h-3" />
          <span>SYS_TIME: <span className="text-white">{result.processing_time_ms.toFixed(1)}MS</span></span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-cyber-muted uppercase tracking-widest">
          <Zap className="w-3 h-3" />
          <span>ENGINE: <span className="text-primary">OMNI_ML_V4</span></span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-cyber-muted uppercase tracking-widest sm:ml-auto">
          <CheckCircle className="w-3 h-3 text-success" />
          <span>CHECKSUM_VERIFIED</span>
        </div>
      </div>
    </div>
  )
}
