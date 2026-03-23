'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent } from '@/components/ui/Card'
import { useHistoryStore } from '@/stores/historyStore'
import { ShieldAlert, ShieldCheck, Clock, Terminal } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { ActivitySkeleton } from '@/components/ui/Skeleton'

export function RecentActivity() {
  const [mounted, setMounted] = useState(false)
  const history = useHistoryStore((s) => s.history)
  
  useEffect(() => {
    useHistoryStore.persist.rehydrate()
    setMounted(true)
  }, [])

  if (!mounted) return <ActivitySkeleton />

  const recentHistory = history.slice(0, 5)

  return (
    <Card className="bg-black/60 border border-cyber-border/80 p-6 relative overflow-hidden group">
      {/* Target reticle decoration */}
      <div className="absolute top-6 right-6 opacity-20 group-hover:opacity-100 transition-opacity">
        <div className="w-8 h-8 border border-primary/30 rounded-full flex items-center justify-center animate-spin" style={{ animationDuration: '10s' }}>
          <div className="w-1 h-1 bg-primary rounded-full" />
        </div>
      </div>

      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-bold text-white font-mono tracking-widest flex items-center gap-3">
            <span className="text-primary animate-pulse select-none">■</span>
            SYS.LOGS
          </h3>
          <p className="text-xs text-cyber-muted font-mono mt-1">// LATEST_INTERCEPTIONS</p>
        </div>
        <button className="text-[10px] uppercase font-mono tracking-widest text-primary hover:text-white hover:bg-primary/20 transition-all border border-primary/20 px-3 py-1 bg-primary/5">
          [ VIEW_ALL ]
        </button>
      </div>

      <div className="space-y-3 relative z-10">
        {recentHistory.length === 0 ? (
          <div className="text-center py-10 border border-cyber-border/50 border-dashed bg-black/40">
            <Terminal className="w-8 h-8 mx-auto mb-3 text-cyber-muted opacity-50" />
            <p className="text-cyber-muted font-mono text-sm tracking-widest">[ EVENT_BUFFER_EMPTY ]</p>
          </div>
        ) : (
          recentHistory.map((item, i) => (
            <div 
              key={item.id || i} 
              className="flex flex-col sm:flex-row sm:items-center gap-3 p-3 bg-black/40 border-l-2 hover:bg-white/5 transition-all duration-300 font-mono text-sm group/item"
              style={{
                borderColor: item.result.is_attack ? 'var(--tw-colors-danger)' : 'var(--tw-colors-success)'
              }}
            >
              <div className="flex items-center justify-center w-8 h-8 bg-black border border-cyber-border text-center shrink-0">
                {item.result.is_attack ? (
                  <ShieldAlert className="w-4 h-4 text-danger group-hover/item:animate-pulse shadow-neon-danger" />
                ) : (
                  <ShieldCheck className="w-4 h-4 text-success group-hover/item:shadow-[0_0_10px_rgba(0,255,102,0.5)]" />
                )}
              </div>
              
              <div className="flex-1 min-w-0 flex flex-col mt-2 sm:mt-0">
                <p className="text-cyber-text truncate tracking-wide text-xs">
                  <span className="text-primary/70 select-none mr-2">{'>'}</span> 
                  {item.input}
                </p>
                <div className="flex items-center gap-2 mt-1 opacity-60">
                  <Clock className="w-3 h-3 text-cyber-muted" />
                  <span className="text-[10px] text-cyber-muted tracking-widest">
                    {new Date(item.timestamp).toISOString()}
                  </span>
                </div>
              </div>

              <div className="shrink-0 mt-2 sm:mt-0">
                <Badge variant={item.result.is_attack ? 'danger' : 'success'}>
                  AUTH: {(item.result.confidence * 100).toFixed(0)}%
                </Badge>
              </div>
            </div>
          ))
        )}
      </div>
    </Card>
  )
}
