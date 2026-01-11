'use client'

import { Card } from '@/components/ui'
import { useStatsStore } from '@/stores/statsStore'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'

const COLORS = ['#fb7185', '#fbbf24', '#38bdf8', '#34d399', '#c084fc']

export function AttackChart() {
  const { byType } = useStatsStore()

  const data = Object.entries(byType).map(([name, value]) => ({
    name: name.replace('_', ' '),
    value,
  }))

  if (data.length === 0) {
    return (
      <Card className="h-64 flex items-center justify-center bg-glass-dark border border-white/5 backdrop-blur-md">
        <p className="text-slate-400">No attack data yet</p>
      </Card>
    )
  }

  return (
    <Card className="bg-glass-dark border border-white/5 backdrop-blur-md shadow-glass p-6">
      <h3 className="font-semibold text-lg text-white mb-6 flex items-center gap-2">
        <span className="w-1 h-6 bg-secondary rounded-full shadow-[0_0_10px_#818cf8]" />
        Attack Distribution
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
              stroke="none"
            >
              {data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} className="drop-shadow-[0_0_8px_rgba(0,0,0,0.3)]" />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                borderColor: 'rgba(255,255,255,0.1)',
                backdropFilter: 'blur(8px)',
                borderRadius: '8px',
                color: '#fff'
              }}
              itemStyle={{ color: '#fff' }}
            />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}
