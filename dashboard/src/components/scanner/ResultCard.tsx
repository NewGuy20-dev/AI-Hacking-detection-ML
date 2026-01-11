'use client'

import { PredictResponse } from '@/types/api'
import { Shield, ShieldAlert, Clock, Zap, AlertTriangle, CheckCircle, Info } from 'lucide-react'

interface ResultCardProps {
  result: PredictResponse | null
  input?: string
  type?: 'payload' | 'url'
}

function getSeverityConfig(severity: string) {
  switch (severity) {
    case 'CRITICAL':
      return { color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/30', icon: AlertTriangle }
    case 'HIGH':
      return { color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/30', icon: AlertTriangle }
    case 'MEDIUM':
      return { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30', icon: Info }
    default:
      return { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', icon: CheckCircle }
  }
}

export function ResultCard({ result, input, type = 'payload' }: ResultCardProps) {
  if (!result) {
    return (
      <div className="rounded-2xl border border-dashed border-border/50 bg-muted/20 p-12 text-center">
        <div className="inline-flex p-4 rounded-full bg-muted/50 mb-4">
          <Shield className="w-8 h-8 text-muted-foreground" />
        </div>
        <p className="text-muted-foreground font-medium">Enter a {type} to analyze</p>
        <p className="text-sm text-muted-foreground/70 mt-1">Results will appear here</p>
      </div>
    )
  }

  const severityConfig = getSeverityConfig(result.severity)
  const confidencePercent = (result.confidence * 100).toFixed(1)

  return (
    <div className={`rounded-2xl border overflow-hidden transition-all ${
      result.is_attack 
        ? 'border-red-500/30 bg-red-500/5' 
        : 'border-emerald-500/30 bg-emerald-500/5'
    }`}>
      {/* Header */}
      <div className={`p-5 border-b ${result.is_attack ? 'border-red-500/20' : 'border-emerald-500/20'}`}>
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-xl ${result.is_attack ? 'bg-red-500/20' : 'bg-emerald-500/20'}`}>
              {result.is_attack ? (
                <ShieldAlert className="w-7 h-7 text-red-400" />
              ) : (
                <Shield className="w-7 h-7 text-emerald-400" />
              )}
            </div>
            <div>
              <h3 className={`text-xl font-bold ${result.is_attack ? 'text-red-400' : 'text-emerald-400'}`}>
                {result.is_attack ? 'Threat Detected' : 'Safe'}
              </h3>
              {result.attack_type && (
                <p className="text-sm text-muted-foreground mt-0.5">
                  {result.attack_type.replace(/_/g, ' ')}
                </p>
              )}
            </div>
          </div>
          
          {/* Severity Badge */}
          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg ${severityConfig.bg} ${severityConfig.border} border`}>
            <severityConfig.icon className={`w-4 h-4 ${severityConfig.color}`} />
            <span className={`text-sm font-semibold ${severityConfig.color}`}>{result.severity}</span>
          </div>
        </div>
      </div>

      {/* Confidence Bar */}
      <div className="p-5 border-b border-border/30">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-muted-foreground">Confidence Level</span>
          <span className={`text-sm font-bold ${result.is_attack ? 'text-red-400' : 'text-emerald-400'}`}>
            {confidencePercent}%
          </span>
        </div>
        <div className="h-2 bg-muted/50 rounded-full overflow-hidden">
          <div 
            className={`h-full rounded-full transition-all duration-500 ${
              result.is_attack 
                ? 'bg-gradient-to-r from-red-500 to-orange-500' 
                : 'bg-gradient-to-r from-emerald-500 to-green-400'
            }`}
            style={{ width: `${confidencePercent}%` }}
          />
        </div>
      </div>

      {/* Input Display */}
      {input && (
        <div className="p-5 border-b border-border/30">
          <p className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">Analyzed Input</p>
          <div className="p-3 rounded-lg bg-background/50 border border-border/50">
            <code className="text-sm break-all font-mono text-foreground/80">{input}</code>
          </div>
        </div>
      )}

      {/* Footer Stats */}
      <div className="p-4 bg-muted/20 flex items-center gap-6">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Clock className="w-4 h-4" />
          <span>{result.processing_time_ms.toFixed(1)}ms</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Zap className="w-4 h-4" />
          <span>ML Analysis</span>
        </div>
      </div>
    </div>
  )
}
