'use client'

import { useEffect, useState } from 'react'
import { StatsGrid, AttackChart, RecentActivity } from '@/components/dashboard'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { useStatsStore } from '@/stores/statsStore'
import { RotateCcw, Shield, Activity, TrendingUp } from 'lucide-react'

export default function DashboardPage() {
  const [mounted, setMounted] = useState(false)
  const reset = useStatsStore((s) => s.reset)
  const total = useStatsStore((s) => s.total)
  const malicious = useStatsStore((s) => s.malicious)

  useEffect(() => {
    useStatsStore.persist.rehydrate()
    setMounted(true)
  }, [])

  const detectionRate = mounted && total > 0 ? ((malicious / total) * 100).toFixed(1) : '0.0'

  return (
    <div className="space-y-6 animate-in">
      {/* Hero Section */}
      <div className="glass rounded-2xl p-6 md:p-8 border hover-lift">
        <div className="flex flex-col sm:flex-row justify-between items-start gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-xl">
                <Shield className="w-8 h-8 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
                  Security Dashboard
                </h1>
                <p className="text-muted-foreground text-sm md:text-base">Real-time threat detection & analysis</p>
              </div>
            </div>
          </div>
          <Button onClick={reset} variant="ghost" className="hover-lift">
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
        </div>
        
        {/* Quick Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
          <div className="flex items-center gap-3 p-4 bg-success/10 rounded-xl">
            <Activity className="w-5 h-5 text-success" />
            <div>
              <p className="text-xs text-muted-foreground">System Status</p>
              <p className="font-semibold text-success">Operational</p>
            </div>
          </div>
          <div className="flex items-center gap-3 p-4 bg-primary/10 rounded-xl">
            <TrendingUp className="w-5 h-5 text-primary" />
            <div>
              <p className="text-xs text-muted-foreground">Detection Rate</p>
              <p className="font-semibold">{detectionRate}%</p>
            </div>
          </div>
          <div className="flex items-center gap-3 p-4 bg-warning/10 rounded-xl">
            <Shield className="w-5 h-5 text-warning" />
            <div>
              <p className="text-xs text-muted-foreground">Active Models</p>
              <p className="font-semibold">6/6</p>
            </div>
          </div>
        </div>
      </div>

      <StatsGrid />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <AttackChart />
        <RecentActivity />
      </div>
    </div>
  )
}
