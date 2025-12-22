'use client'

import { Card } from '@/components/ui'
import { useStatsStore } from '@/stores/statsStore'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'

const COLORS = ['#fc8181', '#fbd38d', '#90cdf4', '#9ae6b4', '#d6bcfa']

export function AttackChart() {
  const { byType } = useStatsStore()
  
  const data = Object.entries(byType).map(([name, value]) => ({
    name: name.replace('_', ' '),
    value,
  }))

  if (data.length === 0) {
    return (
      <Card className="h-64 flex items-center justify-center">
        <p className="text-gray-500 dark:text-gray-400">No attack data yet</p>
      </Card>
    )
  }

  return (
    <Card>
      <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-4">Attack Distribution</h3>
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
            >
              {data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}
