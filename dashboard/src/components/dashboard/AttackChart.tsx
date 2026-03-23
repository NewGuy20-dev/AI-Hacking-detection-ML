'use client'

import { Card } from '@/components/ui'
import { useStatsStore } from '@/stores/statsStore'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'
import { Terminal } from 'lucide-react'

// Cyber-Tactical Neon Color Palette
const COLORS = ['#ff003c', '#ffb000', '#00f3ff', '#bf00ff', '#00ff66']

export function AttackChart() {
  const { byType } = useStatsStore()

  const data = Object.entries(byType).map(([name, value]) => ({
    name: name.replace('_', ' ').toUpperCase(),
    value,
  }))

  if (data.length === 0) {
    return (
      <Card className="h-64 flex flex-col items-center justify-center bg-black/50 border-cyber-border/50 border-dashed relative overflow-hidden group">
        <div className="absolute inset-0 bg-cyber-grid opacity-20 pointer-events-none" />
        <Terminal className="w-10 h-10 text-cyber-muted mb-4 group-hover:text-primary transition-colors" />
        <p className="text-cyber-muted font-mono tracking-widest text-sm">[ AWAITING_THREAT_DATA ]</p>
      </Card>
    )
  }

  return (
    <Card className="bg-black/60 border border-cyber-border/80 backdrop-blur-md p-6 relative overflow-hidden group">
      {/* Decorative corners */}
      <div className="absolute top-0 left-0 w-4 h-4 border-t border-l border-primary/50" />
      <div className="absolute bottom-0 right-0 w-4 h-4 border-b border-r border-primary/50" />

      <h3 className="font-bold text-lg text-white mb-6 flex items-center justify-between font-mono tracking-widest uppercase text-shadow-sm">
        <span className="flex items-center gap-3">
          <span className="w-2 h-2 bg-primary rounded-full animate-pulse shadow-neon-primary" />
          ATTACK_VECTORS
        </span>
        <span className="text-xs text-cyber-muted">/// REAL_TIME</span>
      </h3>
      
      <div className="h-64 relative z-10">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={65}
              outerRadius={85}
              paddingAngle={4}
              dataKey="value"
              stroke="none"
              animationBegin={200}
              animationDuration={1500}
              animationEasing="ease-out"
            >
              {data.map((_, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={COLORS[index % COLORS.length]} 
                  style={{ filter: `drop-shadow(0 0 8px ${COLORS[index % COLORS.length]}80)` }}
                />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(5, 5, 5, 0.95)',
                borderColor: '#1a1f2e',
                border: '1px solid #1a1f2e',
                fontFamily: 'var(--font-fira-code)',
                fontSize: '12px',
                borderRadius: '2px',
                color: '#00f3ff',
                boxShadow: '0 0 15px rgba(0, 243, 255, 0.1)'
              }}
              itemStyle={{ color: '#00f3ff', fontWeight: 'bold' }}
              cursor={{ fill: 'rgba(0, 243, 255, 0.1)' }}
            />
            <Legend 
              wrapperStyle={{ paddingTop: '20px', fontFamily: 'var(--font-fira-code)', fontSize: '11px' }}
              iconType="square"
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}
