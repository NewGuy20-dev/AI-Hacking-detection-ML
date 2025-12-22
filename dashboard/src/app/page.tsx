'use client'

import { StatsGrid, AttackChart, RecentActivity } from '@/components/dashboard'
import { Button } from '@/components/ui'
import { useStatsStore } from '@/stores/statsStore'
import { RotateCcw } from 'lucide-react'

export default function DashboardPage() {
  const reset = useStatsStore((s) => s.reset)

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
          📊 Dashboard
        </h1>
        <Button onClick={reset}>
          <RotateCcw className="w-4 h-4 mr-2" />
          Reset Stats
        </Button>
      </div>

      <StatsGrid />

      <div className="grid lg:grid-cols-2 gap-6">
        <AttackChart />
        <RecentActivity />
      </div>
    </div>
  )
}
