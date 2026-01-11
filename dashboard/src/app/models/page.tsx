'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { Cpu, Activity, Zap, CheckCircle, XCircle, Clock, Brain, AlertTriangle, TrendingUp, Server, RefreshCw } from 'lucide-react'

const MODEL_PERFORMANCE = [
  { name: 'Payload CNN', accuracy: 99.89, type: 'PyTorch', description: 'Deep learning model for payload analysis', color: 'from-blue-500 to-cyan-400' },
  { name: 'URL CNN', accuracy: 97.47, type: 'PyTorch', description: 'Convolutional network for URL classification', color: 'from-purple-500 to-pink-400' },
  { name: 'TimeSeries LSTM', accuracy: 75.38, type: 'PyTorch', description: 'Sequential pattern detection', color: 'from-amber-500 to-orange-400' },
]

const KNOWN_LIMITATIONS = [
  { input: '<3 emoji patterns', issue: '~95% false positive rate', severity: 'high' },
  { input: 'SELECT FROM menu', issue: '~72% incorrectly flagged', severity: 'medium' },
  { input: 'Script tags in text', issue: 'Context-dependent accuracy', severity: 'low' },
]

export default function ModelsPage() {
  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ['models'],
    queryFn: api.ready,
    refetchInterval: 30000,
  })

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    if (hours > 0) return `${hours}h ${minutes}m`
    return `${minutes}m ${Math.floor(seconds % 60)}s`
  }

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-cyan-500/10 via-blue-500/5 to-transparent border border-border/50 p-6">
        <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
        <div className="relative flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl shadow-lg">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">ML Models</h1>
              <p className="text-muted-foreground">Model status and performance metrics</p>
            </div>
          </div>
          
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl border border-border hover:bg-muted transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isFetching ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* API Status Card */}
      <div className={`rounded-2xl border overflow-hidden ${
        error ? 'border-red-500/30 bg-red-500/5' : 'border-emerald-500/30 bg-emerald-500/5'
      }`}>
        <div className="p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-xl ${error ? 'bg-red-500/20' : 'bg-emerald-500/20'}`}>
                <Server className={`w-6 h-6 ${error ? 'text-red-400' : 'text-emerald-400'}`} />
              </div>
              <div>
                <h3 className="font-semibold">API Status</h3>
                <p className="text-sm text-muted-foreground">Backend service health</p>
              </div>
            </div>
            
            {isLoading ? (
              <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-muted/50">
                <div className="w-4 h-4 border-2 border-muted-foreground/30 border-t-muted-foreground rounded-full animate-spin" />
                <span className="text-sm">Checking...</span>
              </div>
            ) : error ? (
              <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/20">
                <XCircle className="w-4 h-4 text-red-400" />
                <span className="text-sm font-medium text-red-400">Offline</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/20">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-medium text-emerald-400">Online</span>
              </div>
            )}
          </div>
          
          {data && (
            <div className="flex items-center gap-6 mt-4 pt-4 border-t border-border/30">
              <div className="flex items-center gap-2 text-sm">
                <Clock className="w-4 h-4 text-muted-foreground" />
                <span className="text-muted-foreground">Uptime: {formatUptime(data.uptime_seconds)}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Activity className="w-4 h-4 text-muted-foreground" />
                <span className="text-muted-foreground">Status: {data.status}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Loaded Models */}
      <div className="rounded-2xl border border-border/50 bg-card/50 overflow-hidden">
        <div className="p-5 border-b border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-purple-500/20">
              <Cpu className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="font-semibold">Loaded Models</h3>
              <p className="text-sm text-muted-foreground">Currently active ML models</p>
            </div>
          </div>
        </div>
        <div className="p-5">
          {data?.models_loaded ? (
            <div className="flex flex-wrap gap-2">
              {data.models_loaded.pytorch.map((m) => (
                <span key={m} className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-500/10 border border-blue-500/30 text-sm font-medium text-blue-400">
                  <Zap className="w-3 h-3" />
                  {m}
                </span>
              ))}
              {data.models_loaded.sklearn.map((m) => (
                <span key={m} className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-sm font-medium text-amber-400">
                  <TrendingUp className="w-3 h-3" />
                  {m}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground text-sm">No models loaded or API unavailable</p>
          )}
        </div>
      </div>

      {/* Model Performance */}
      <div className="rounded-2xl border border-border/50 bg-card/50 overflow-hidden">
        <div className="p-5 border-b border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-500/20">
              <TrendingUp className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <h3 className="font-semibold">Model Performance</h3>
              <p className="text-sm text-muted-foreground">Accuracy metrics from training</p>
            </div>
          </div>
        </div>
        <div className="divide-y divide-border/30">
          {MODEL_PERFORMANCE.map((model) => (
            <div key={model.name} className="p-5 hover:bg-muted/30 transition-colors">
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <div className={`p-2.5 rounded-xl bg-gradient-to-br ${model.color}`}>
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <p className="font-medium">{model.name}</p>
                    <p className="text-sm text-muted-foreground">{model.description}</p>
                    <span className="text-xs text-muted-foreground/70">{model.type}</span>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`text-2xl font-bold bg-gradient-to-r ${model.color} bg-clip-text text-transparent`}>
                    {model.accuracy}%
                  </p>
                  <p className="text-xs text-muted-foreground">accuracy</p>
                </div>
              </div>
              {/* Accuracy bar */}
              <div className="mt-3 h-1.5 bg-muted/50 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full bg-gradient-to-r ${model.color}`}
                  style={{ width: `${model.accuracy}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Known Limitations */}
      <div className="rounded-2xl border border-amber-500/30 bg-amber-500/5 overflow-hidden">
        <div className="p-5 border-b border-amber-500/20">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-amber-500/20">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            </div>
            <div>
              <h3 className="font-semibold">Known Limitations</h3>
              <p className="text-sm text-muted-foreground">Edge cases with reduced accuracy</p>
            </div>
          </div>
        </div>
        <div className="divide-y divide-amber-500/10">
          {KNOWN_LIMITATIONS.map((item, i) => (
            <div key={i} className="p-4 flex items-center justify-between gap-4">
              <div>
                <p className="font-mono text-sm">{item.input}</p>
              </div>
              <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                item.severity === 'high' ? 'bg-red-500/20 text-red-400' :
                item.severity === 'medium' ? 'bg-amber-500/20 text-amber-400' :
                'bg-blue-500/20 text-blue-400'
              }`}>
                {item.issue}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
