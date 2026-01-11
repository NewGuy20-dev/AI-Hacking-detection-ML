'use client'

import { useEffect, useState, useMemo } from 'react'
import { Card, CardHeader, CardContent } from '@/components/ui/Card'
import { useStatsStore } from '@/stores/statsStore'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { TrendingUp, TrendingDown, Calendar } from 'lucide-react'
import { ChartSkeleton } from '@/components/ui/Skeleton'

type TimeRange = '7d' | '30d' | 'all'

export function AttackChart() {
  const [mounted, setMounted] = useState(false)
  const [timeRange, setTimeRange] = useState<TimeRange>('7d')
  const dailyStats = useStatsStore((s) => s.dailyStats)
  
  useEffect(() => {
    useStatsStore.persist.rehydrate()
    setMounted(true)
  }, [])

  const { chartData, trend } = useMemo(() => {
    if (!dailyStats.length) {
      // Generate sample data if no real data
      const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
      return {
        chartData: days.map(name => ({ name, threats: 0, clean: 0 })),
        trend: 0
      }
    }

    const limit = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : dailyStats.length
    const data = dailyStats.slice(-limit).map(d => ({
      name: new Date(d.date).toLocaleDateString('en-US', { weekday: 'short' }),
      threats: d.threats,
      clean: d.clean
    }))

    // Calculate trend
    const recent = dailyStats.slice(-3)
    const older = dailyStats.slice(-6, -3)
    const recentThreats = recent.reduce((a, b) => a + b.threats, 0)
    const olderThreats = older.reduce((a, b) => a + b.threats, 0) || 1
    const trendValue = ((recentThreats - olderThreats) / olderThreats) * 100

    return { chartData: data, trend: trendValue }
  }, [dailyStats, timeRange])

  if (!mounted) return <ChartSkeleton />

  const trendUp = trend > 0
  const trendText = trend === 0 ? 'No change' : `${trendUp ? '+' : ''}${trend.toFixed(0)}% threats`

  return (
    <Card className="hover-lift glass border-0 animate-in">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold">Threat Analysis</h3>
            <p className="text-sm text-muted-foreground">Detection trends over time</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 p-1 bg-muted rounded-lg">
              {(['7d', '30d', 'all'] as TimeRange[]).map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    timeRange === range ? 'bg-primary text-primary-foreground' : 'hover:bg-muted-foreground/10'
                  }`}
                >
                  {range === 'all' ? 'All' : range}
                </button>
              ))}
            </div>
            <div className={`flex items-center gap-1 px-3 py-1 rounded-lg ${trendUp ? 'bg-destructive/10' : 'bg-success/10'}`}>
              {trendUp ? <TrendingUp className="w-4 h-4 text-destructive" /> : <TrendingDown className="w-4 h-4 text-success" />}
              <span className={`text-sm font-medium ${trendUp ? 'text-destructive' : 'text-success'}`}>{trendText}</span>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="threats" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--destructive))" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="hsl(var(--destructive))" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="clean" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--success))" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="hsl(var(--success))" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'hsl(var(--card))', 
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
              }}
            />
            <Legend />
            <Area type="monotone" dataKey="threats" stroke="hsl(var(--destructive))" fillOpacity={1} fill="url(#threats)" strokeWidth={2} />
            <Area type="monotone" dataKey="clean" stroke="hsl(var(--success))" fillOpacity={1} fill="url(#clean)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
