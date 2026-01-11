'use client'

import { useEffect, useState } from 'react'
import { Card, CardHeader, CardContent } from '@/components/ui/Card'
import { useHistoryStore } from '@/stores/historyStore'
import { Shield, AlertTriangle, Clock, ArrowRight } from 'lucide-react'
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
    <Card className="hover-lift glass border-0 animate-in">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">Recent Activity</h3>
            <p className="text-sm text-muted-foreground">Latest security scans</p>
          </div>
          <button className="text-sm text-primary hover:underline flex items-center gap-1">
            View All
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {recentHistory.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Shield className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No recent activity</p>
              <p className="text-xs mt-1">Scan some payloads to see results here</p>
            </div>
          ) : (
            recentHistory.map((item, i) => (
              <div 
                key={item.id || i} 
                className="flex items-center gap-3 p-3 rounded-xl bg-card/50 hover:bg-card transition-all duration-200 border border-border/50 hover:border-border animate-in group cursor-default"
                style={{ animationDelay: `${i * 50}ms` }}
              >
                <div className={`p-2 rounded-lg transition-all duration-200 ${item.result.is_attack ? 'bg-destructive/10 group-hover:bg-destructive/20' : 'bg-success/10 group-hover:bg-success/20'}`}>
                  {item.result.is_attack ? (
                    <AlertTriangle className="w-4 h-4 text-destructive" />
                  ) : (
                    <Shield className="w-4 h-4 text-success" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{item.input}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <Clock className="w-3 h-3 text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
                <Badge variant={item.result.is_attack ? 'destructive' : 'success'}>
                  {(item.result.confidence * 100).toFixed(0)}%
                </Badge>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  )
}
