'use client'

import { Card, CardContent } from '@/components/ui/Card'
import { useStatsStore } from '@/stores/statsStore'
import { Shield, ShieldAlert, ShieldCheck, Activity } from 'lucide-react'
import { cn } from '@/lib/utils'

export function StatsGrid() {
  const { total, malicious, safe } = useStatsStore()
  const threatRate = total > 0 ? ((malicious / total) * 100).toFixed(1) : '0'

  const stats = [
    { label: 'TOTAL_SCANS', value: total, icon: Activity, color: 'text-primary', glow: 'shadow-neon-primary', border: 'border-primary/50' },
    { label: 'THREATS_FOUND', value: malicious, icon: ShieldAlert, color: 'text-danger', glow: 'shadow-neon-danger', border: 'border-danger/50' },
    { label: 'SAFE_INPUTS', value: safe, icon: ShieldCheck, color: 'text-success', glow: 'shadow-[0_0_10px_rgba(0,255,102,0.5)]', border: 'border-success/50' },
    { label: 'THREAT_RATE', value: `${threatRate}%`, icon: Shield, color: 'text-warning', glow: 'shadow-[0_0_10px_rgba(255,176,0,0.5)]', border: 'border-warning/50' },
  ]

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
      {stats.map((stat) => (
        <Card key={stat.label} variant="inset" className={cn("group overflow-visible bg-black/40 backdrop-blur-sm border", stat.border)}>
          <CardContent className="p-6 relative">
            <div className="absolute top-0 right-0 w-8 h-8 opacity-20 group-hover:opacity-100 transition-opacity flex justify-end p-2 cursor-pointer">
              {/* Optional top right corner decoration */}
              <div className="w-2 h-2 rounded-full bg-current animate-pulse object-right-top" style={{ color: 'var(--tw-colors-primary)' }} />
            </div>

            <div className={`w-12 h-12 mb-4 rounded-sm flex items-center justify-center bg-black border ${stat.border} ${stat.glow} transition-all duration-300 group-hover:scale-110 group-hover:rotate-3`}>
              <stat.icon className={`w-6 h-6 ${stat.color} group-hover:animate-pulse`} />
            </div>

            <div className="flex flex-col">
              <span className={`text-4xl font-bold font-mono tracking-tighter ${stat.color} text-shadow-sm`}>
                {stat.value}
              </span>
              <span className="text-xs text-cyber-muted font-mono tracking-widest mt-1 uppercase group-hover:text-white transition-colors">
                [ {stat.label} ]
              </span>
            </div>

            {/* Bottom animated border line */}
            <div className="absolute bottom-0 left-0 h-[2px] w-0 group-hover:w-full transition-all duration-500 bg-current opacity-50" style={{ color: 'var(--tw-colors-primary)' }} />
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
