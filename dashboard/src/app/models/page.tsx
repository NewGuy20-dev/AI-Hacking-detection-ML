'use client'

import { useQuery } from '@tanstack/react-query'
import { Card, Badge } from '@/components/ui'
import { api } from '@/lib/api'
import { Cpu, Activity, Zap } from 'lucide-react'

const MODEL_PERFORMANCE = [
  { name: 'Payload CNN', accuracy: '99.89%', type: 'pytorch' },
  { name: 'URL CNN', accuracy: '97.47%', type: 'pytorch' },
  { name: 'TimeSeries LSTM', accuracy: '75.38%', type: 'pytorch' },
]

export default function ModelsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['models'],
    queryFn: api.ready,
    refetchInterval: 30000,
  })

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">🧠 Models</h1>

      <Card>
        <div className="flex items-center gap-3 mb-4">
          <Activity className="w-5 h-5 text-blue-500" />
          <h3 className="font-semibold text-gray-800 dark:text-gray-100">API Status</h3>
        </div>
        {isLoading ? (
          <p className="text-gray-500">Checking...</p>
        ) : error ? (
          <Badge variant="critical">🔴 Offline</Badge>
        ) : (
          <div className="space-y-2">
            <Badge variant="low">🟢 {data?.status}</Badge>
            <p className="text-sm text-gray-500">Uptime: {Math.floor(data?.uptime_seconds || 0)}s</p>
          </div>
        )}
      </Card>

      <Card>
        <div className="flex items-center gap-3 mb-4">
          <Cpu className="w-5 h-5 text-purple-500" />
          <h3 className="font-semibold text-gray-800 dark:text-gray-100">Loaded Models</h3>
        </div>
        {data?.models_loaded ? (
          <div className="flex flex-wrap gap-2">
            {data.models_loaded.pytorch.map((m) => (
              <Badge key={m} variant="info">{m}</Badge>
            ))}
            {data.models_loaded.sklearn.map((m) => (
              <Badge key={m} variant="medium">{m}</Badge>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No models loaded</p>
        )}
      </Card>

      <Card>
        <div className="flex items-center gap-3 mb-4">
          <Zap className="w-5 h-5 text-yellow-500" />
          <h3 className="font-semibold text-gray-800 dark:text-gray-100">Model Performance</h3>
        </div>
        <div className="space-y-3">
          {MODEL_PERFORMANCE.map((m) => (
            <div
              key={m.name}
              className="flex items-center justify-between p-3 rounded-clay-sm bg-clay-bg dark:bg-clay-bg-dark"
            >
              <div>
                <p className="font-medium text-gray-800 dark:text-gray-100">{m.name}</p>
                <p className="text-xs text-gray-500">{m.type}</p>
              </div>
              <span className="font-semibold text-green-600">{m.accuracy}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-4">⚠️ Known Limitations</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between p-2 bg-clay-bg dark:bg-clay-bg-dark rounded">
            <span>&lt;3 emoji</span>
            <span className="text-orange-500">~95% false positive</span>
          </div>
          <div className="flex justify-between p-2 bg-clay-bg dark:bg-clay-bg-dark rounded">
            <span>SELECT FROM menu</span>
            <span className="text-orange-500">~72% flagged</span>
          </div>
        </div>
      </Card>
    </div>
  )
}
