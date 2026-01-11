'use client'

import { StatsGrid, AttackChart, RecentActivity } from '@/components/dashboard'
import { Button } from '@/components/ui/Button'
import { Card, CardHeader } from '@/components/ui/Card'
import { useStatsStore } from '@/stores/statsStore'
import { RotateCcw } from 'lucide-react'

export default function DashboardPage() {
  const reset = useStatsStore((s) => s.reset)

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-primary">
              📊 Dashboard
            </h1>
            <Button onClick={reset} variant="ghost">
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset Stats
            </Button>
          </div>
        </CardHeader>
      </Card>

      <StatsGrid />

      <div className="grid lg:grid-cols-2 gap-6">
        <AttackChart />
        <RecentActivity />
      </div>
    </div>
  )
}
