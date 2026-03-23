'use client'

import { StatsGrid, AttackChart, RecentActivity } from '@/components/dashboard'
import { Button } from '@/components/ui/Button'
import { Card, CardHeader } from '@/components/ui/Card'
import { useStatsStore } from '@/stores/statsStore'
import { RotateCcw, Crosshair } from 'lucide-react'

export default function DashboardPage() {
  const reset = useStatsStore((s) => s.reset)

  return (
    <div className="space-y-6 animate-in fade-in zoom-in-95 duration-500">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold font-sans tracking-wide text-white flex items-center gap-3">
            <span className="text-primary font-mono select-none animate-pulse">■</span> 
            COMMAND_CENTER
          </h1>
          <p className="text-cyber-muted font-mono text-sm mt-1">Real-time threat monitoring and intelligence gathering</p>
        </div>
        
        <div className="flex gap-4">
          <Button onClick={reset} variant="ghost" className="border border-danger/30 text-danger hover:bg-danger/10 hover:border-danger hover:shadow-neon-danger transition-all duration-300">
            <RotateCcw className="w-4 h-4 mr-2" />
            [ RESET_SYS ]
          </Button>
          <Button variant="primary" className="animate-pulse-slow">
            <Crosshair className="w-4 h-4 mr-2" />
            [ INIT_SCAN ]
          </Button>
        </div>
      </div>

      <StatsGrid />

      <div className="grid lg:grid-cols-2 gap-6 mt-6">
        <AttackChart />
        <RecentActivity />
      </div>
    </div>
  )
}
