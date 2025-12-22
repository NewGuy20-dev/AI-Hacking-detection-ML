'use client'

import { Card, Button, Badge } from '@/components/ui'
import { useHistoryStore } from '@/stores/historyStore'
import { getSeverity } from '@/lib/utils'
import { Trash2, Download, Clock, FileCode, Globe } from 'lucide-react'

export default function HistoryPage() {
  const { history, clearHistory } = useHistoryStore()

  const exportJSON = () => {
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'scan-history.json'
    a.click()
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">📜 History</h1>
        <div className="flex gap-3">
          <Button onClick={exportJSON} disabled={history.length === 0}>
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button onClick={clearHistory} disabled={history.length === 0}>
            <Trash2 className="w-4 h-4 mr-2" />
            Clear
          </Button>
        </div>
      </div>

      {history.length === 0 ? (
        <Card className="text-center py-12">
          <Clock className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <p className="text-gray-500 dark:text-gray-400">No scan history yet</p>
        </Card>
      ) : (
        <div className="space-y-3">
          {history.map((item) => {
            const severity = getSeverity(item.result.confidence)
            const variant = item.result.is_attack
              ? (severity.label.toLowerCase() as 'critical' | 'high' | 'medium' | 'low')
              : 'low'
            return (
              <Card key={item.id} variant={item.result.is_attack ? 'danger' : 'success'}>
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      {item.type === 'payload' ? (
                        <FileCode className="w-4 h-4 text-gray-500" />
                      ) : (
                        <Globe className="w-4 h-4 text-gray-500" />
                      )}
                      <span className="text-xs text-gray-500 uppercase">{item.type}</span>
                    </div>
                    <p className="font-mono text-sm break-all text-gray-800 dark:text-gray-100">
                      {item.input}
                    </p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                      <span>{new Date(item.timestamp).toLocaleString()}</span>
                      <span>{item.result.processing_time_ms.toFixed(1)}ms</span>
                      {item.result.attack_type && <span>{item.result.attack_type}</span>}
                    </div>
                  </div>
                  <Badge variant={variant}>
                    {severity.emoji} {item.result.is_attack ? 'Threat' : 'Safe'}
                  </Badge>
                </div>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  )
}
