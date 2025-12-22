'use client'

import { Card } from '@/components/ui'
import { useStatsStore } from '@/stores/statsStore'
import { Shield, ShieldAlert, ShieldCheck, Activity } from 'lucide-react'

export function StatsGrid() {
  const { total, malicious, safe } = useStatsStore()
  const threatRate = total > 0 ? ((malicious / total) * 100).toFixed(1) : '0'

  const stats = [
    { label: 'Total Scans', value: total, icon: Activity, color: 'text-blue-500' },
    { label: 'Threats Found', value: malicious, icon: ShieldAlert, color: 'text-red-500' },
    { label: 'Safe Inputs', value: safe, icon: ShieldCheck, color: 'text-green-500' },
    { label: 'Threat Rate', value: `${threatRate}%`, icon: Shield, color: 'text-orange-500' },
  ]

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat) => (
        <Card key={stat.label} className="text-center">
          <stat.icon className={`w-8 h-8 mx-auto mb-2 ${stat.color}`} />
          <div className="text-2xl font-bold text-gray-800 dark:text-gray-100">{stat.value}</div>
          <div className="text-sm text-gray-500 dark:text-gray-400">{stat.label}</div>
        </Card>
      ))}
    </div>
  )
}
