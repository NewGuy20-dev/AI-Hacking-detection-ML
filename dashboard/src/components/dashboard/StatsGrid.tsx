'use client'

import { useEffect, useState, useMemo, useRef } from 'react'
import { useStatsStore } from '@/stores/statsStore'
import { useHistoryStore } from '@/stores/historyStore'
import { Shield, AlertTriangle, CheckCircle, Activity, TrendingUp, TrendingDown, Sparkles } from 'lucide-react'
import { CardSkeleton } from '@/components/ui/Skeleton'

function useCountUp(end: number, duration = 600) {
  const [count, setCount] = useState(0)
  const prevEnd = useRef(end)

  useEffect(() => {
    if (end === prevEnd.current) return
    prevEnd.current = end
    
    const start = count
    const startTime = performance.now()
    
    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setCount(Math.floor(start + (end - start) * eased))
      
      if (progress < 1) requestAnimationFrame(animate)
    }
    
    requestAnimationFrame(animate)
  }, [end, duration])

  useEffect(() => { setCount(end) }, [])

  return count
}

export function StatsGrid() {
  const [mounted, setMounted] = useState(false)
  const total = useStatsStore((s) => s.total)
  const malicious = useStatsStore((s) => s.malicious)
  const safe = useStatsStore((s) => s.safe)
  const dailyStats = useStatsStore((s) => s.dailyStats)
  
  useEffect(() => {
    useStatsStore.persist.rehydrate()
    useHistoryStore.persist.rehydrate()
    setMounted(true)
  }, [])

  const animatedTotal = useCountUp(mounted ? total : 0)
  const animatedMalicious = useCountUp(mounted ? malicious : 0)
  const animatedSafe = useCountUp(mounted ? safe : 0)

  const trends = useMemo(() => {
    if (dailyStats.length < 2) return { total: 0, threats: 0, clean: 0 }
    
    const recent = dailyStats.slice(-3)
    const older = dailyStats.slice(-6, -3)
    
    const recentTotal = recent.reduce((a, b) => a + b.threats + b.clean, 0)
    const olderTotal = older.reduce((a, b) => a + b.threats + b.clean, 0) || 1
    const recentThreats = recent.reduce((a, b) => a + b.threats, 0)
    const olderThreats = older.reduce((a, b) => a + b.threats, 0) || 1
    const recentClean = recent.reduce((a, b) => a + b.clean, 0)
    const olderClean = older.reduce((a, b) => a + b.clean, 0) || 1

    return {
      total: ((recentTotal - olderTotal) / olderTotal) * 100,
      threats: ((recentThreats - olderThreats) / olderThreats) * 100,
      clean: ((recentClean - olderClean) / olderClean) * 100,
    }
  }, [dailyStats])

  if (!mounted) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => <CardSkeleton key={i} />)}
      </div>
    )
  }

  const detectionRate = total > 0 ? ((malicious / total) * 100).toFixed(1) : '0.0'

  const cards = [
    { 
      title: 'Total Scans', 
      value: animatedTotal, 
      icon: Activity, 
      gradient: 'from-blue-500 to-cyan-400',
      bgGlow: 'bg-blue-500/20',
      iconBg: 'bg-gradient-to-br from-blue-500 to-cyan-400',
      trend: trends.total,
      format: 'number'
    },
    { 
      title: 'Threats Detected', 
      value: animatedMalicious, 
      icon: AlertTriangle, 
      gradient: 'from-red-500 to-orange-400',
      bgGlow: 'bg-red-500/20',
      iconBg: 'bg-gradient-to-br from-red-500 to-orange-400',
      trend: trends.threats,
      invertTrend: true,
      format: 'number'
    },
    { 
      title: 'Clean Results', 
      value: animatedSafe, 
      icon: CheckCircle, 
      gradient: 'from-emerald-500 to-green-400',
      bgGlow: 'bg-emerald-500/20',
      iconBg: 'bg-gradient-to-br from-emerald-500 to-green-400',
      trend: trends.clean,
      format: 'number'
    },
    { 
      title: 'Detection Rate', 
      value: detectionRate, 
      icon: Shield, 
      gradient: 'from-amber-500 to-yellow-400',
      bgGlow: 'bg-amber-500/20',
      iconBg: 'bg-gradient-to-br from-amber-500 to-yellow-400',
      trend: 0,
      format: 'percent'
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, i) => {
        const trendUp = card.invertTrend ? card.trend < 0 : card.trend > 0
        const trendColor = card.trend === 0 ? 'text-muted-foreground' : trendUp ? 'text-emerald-400' : 'text-red-400'
        const TrendIcon = card.trend >= 0 ? TrendingUp : TrendingDown
        const hasData = card.format === 'number' ? card.value > 0 : parseFloat(String(card.value)) > 0
        
        return (
          <div 
            key={i} 
            className="group relative overflow-hidden rounded-2xl bg-card/50 backdrop-blur-sm border border-border/50 p-5 transition-all duration-300 hover:border-border hover:shadow-lg hover:shadow-black/5 dark:hover:shadow-black/20 hover:-translate-y-1 animate-in"
            style={{ animationDelay: `${i * 80}ms` }}
          >
            {/* Background glow effect */}
            <div className={`absolute -top-12 -right-12 w-32 h-32 ${card.bgGlow} rounded-full blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
            
            {/* Sparkle indicator for active stats */}
            {hasData && (
              <div className="absolute top-3 right-3">
                <Sparkles className="w-3 h-3 text-muted-foreground/50" />
              </div>
            )}
            
            <div className="relative flex items-start justify-between gap-4">
              <div className="space-y-3 flex-1">
                <p className="text-sm text-muted-foreground font-medium tracking-wide uppercase text-[11px]">
                  {card.title}
                </p>
                
                <div className="flex items-baseline gap-1">
                  <span className={`text-4xl font-bold tabular-nums bg-gradient-to-r ${card.gradient} bg-clip-text text-transparent`}>
                    {card.format === 'number' ? card.value.toLocaleString() : card.value}
                  </span>
                  {card.format === 'percent' && (
                    <span className={`text-2xl font-bold bg-gradient-to-r ${card.gradient} bg-clip-text text-transparent`}>%</span>
                  )}
                </div>
                
                <div className="flex items-center gap-1.5 pt-1">
                  <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full ${
                    card.trend === 0 ? 'bg-muted/50' : trendUp ? 'bg-emerald-500/10' : 'bg-red-500/10'
                  }`}>
                    <TrendIcon className={`w-3 h-3 ${trendColor} ${card.trend < 0 ? 'rotate-180' : ''}`} />
                    <span className={`text-xs font-semibold ${trendColor}`}>
                      {card.trend === 0 ? 'No data' : `${card.trend > 0 ? '+' : ''}${card.trend.toFixed(0)}%`}
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Icon with gradient background */}
              <div className={`${card.iconBg} p-3 rounded-xl shadow-lg transition-all duration-300 group-hover:scale-110 group-hover:rotate-3`}>
                <card.icon className="w-6 h-6 text-white" strokeWidth={2.5} />
              </div>
            </div>
            
            {/* Bottom accent line */}
            <div className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r ${card.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />
          </div>
        )
      })}
    </div>
  )
}
