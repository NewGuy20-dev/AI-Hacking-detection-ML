'use client'

import { Card, CardContent } from '@/components/ui/Card'
import { useStatsStore } from '@/stores/statsStore'
import { Shield, ShieldAlert, ShieldCheck, Activity } from 'lucide-react'

export function StatsGrid() {
  const { total, malicious, safe } = useStatsStore()
  const threatRate = total > 0 ? ((malicious / total) * 100).toFixed(1) : '0'

  const stats = [
    { label: 'Total Scans', value: total, icon: Activity, color: 'text-primary' },
    { label: 'Threats Found', value: malicious, icon: ShieldAlert, color: 'text-danger' },
    { label: 'Safe Inputs', value: safe, icon: ShieldCheck, color: 'text-success' },
    { label: 'Threat Rate', value: `${threatRate}%`, icon: Shield, color: 'text-warning' },
  ]

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat) => (
        <Card key={stat.label} className="text-center hover:shadow-lg transition-all duration-200 hover:-translate-y-1">
          <CardContent className="p-6">
            <div className={`w-12 h-12 mx-auto mb-3 rounded-full flex items-center justify-center bg-clay-border/20 dark:bg-clay-dark-border/20`}>
              <stat.icon className={`w-6 h-6 ${stat.color}`} />
            </div>
            <div className="text-3xl font-bold mb-1">{stat.value}</div>
            <div className="text-sm text-clay-muted dark:text-clay-dark-muted font-medium">{stat.label}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
