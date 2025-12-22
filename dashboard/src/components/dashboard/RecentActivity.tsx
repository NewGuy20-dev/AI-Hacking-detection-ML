'use client'

import { Card, Badge } from '@/components/ui'
import { useHistoryStore } from '@/stores/historyStore'
import { getSeverity } from '@/lib/utils'
import { Clock } from 'lucide-react'

export function RecentActivity() {
  const { history } = useHistoryStore()
  const recent = history.slice(0, 5)

  if (recent.length === 0) {
    return (
      <Card>
        <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-4">Recent Activity</h3>
        <p className="text-gray-500 dark:text-gray-400 text-center py-8">No scans yet</p>
      </Card>
    )
  }

  return (
    <Card>
      <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-4">Recent Activity</h3>
      <div className="space-y-3">
        {recent.map((item) => {
          const severity = getSeverity(item.result.confidence)
          const variant = item.result.is_attack 
            ? severity.label.toLowerCase() as 'critical' | 'high' | 'medium' | 'low'
            : 'low'
          return (
            <div
              key={item.id}
              className="flex items-center justify-between p-3 rounded-clay-sm bg-clay-bg dark:bg-clay-bg-dark"
            >
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-800 dark:text-gray-100 truncate">
                  {item.input}
                </p>
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                  <Clock className="w-3 h-3" />
                  <span>{new Date(item.timestamp).toLocaleTimeString()}</span>
                  <span>•</span>
                  <span className="capitalize">{item.type}</span>
                </div>
              </div>
              <Badge variant={variant}>
                {item.result.is_attack ? '⚠️ Threat' : '✅ Safe'}
              </Badge>
            </div>
          )
        })}
      </div>
    </Card>
  )
}
