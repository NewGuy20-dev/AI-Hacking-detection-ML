'use client'

import { Card, Badge, Progress } from '@/components/ui'
import { PredictResponse } from '@/types/api'
import { getSeverity, formatConfidence } from '@/lib/utils'
import { Shield, ShieldAlert, Clock } from 'lucide-react'

interface ResultCardProps {
  result: PredictResponse | null
  input?: string
}

export function ResultCard({ result, input }: ResultCardProps) {
  if (!result) {
    return (
      <Card className="text-center py-12">
        <Shield className="w-12 h-12 mx-auto text-gray-400 mb-4" />
        <p className="text-gray-500 dark:text-gray-400">Enter a payload or URL to analyze</p>
      </Card>
    )
  }

  const severity = getSeverity(result.confidence)
  const severityVariant = severity.label.toLowerCase() as 'critical' | 'high' | 'medium' | 'low'

  return (
    <Card variant={result.is_attack ? 'danger' : 'success'}>
      <div className="flex justify-between items-start flex-wrap gap-4">
        <div className="flex items-center gap-3">
          {result.is_attack ? (
            <ShieldAlert className="w-8 h-8 text-red-500" />
          ) : (
            <Shield className="w-8 h-8 text-green-500" />
          )}
          <div>
            <h3 className="text-xl font-bold text-gray-800 dark:text-gray-100">
              {result.is_attack ? '⚠️ MALICIOUS' : '✅ SAFE'}
            </h3>
            {result.attack_type && (
              <p className="text-gray-600 dark:text-gray-400">{result.attack_type.replace('_', ' ')}</p>
            )}
          </div>
        </div>
        <Badge variant={severityVariant}>
          {severity.emoji} {severity.label}
        </Badge>
      </div>

      <div className="mt-6">
        <div className="flex justify-between mb-2">
          <span className="font-medium text-gray-700 dark:text-gray-300">Confidence</span>
          <span className="font-semibold">{formatConfidence(result.confidence)}</span>
        </div>
        <Progress value={result.confidence * 100} showLabel={false} />
      </div>

      {input && (
        <div className="mt-4 p-3 rounded-clay-sm bg-clay-bg dark:bg-clay-bg-dark">
          <code className="text-sm break-all text-gray-700 dark:text-gray-300">{input}</code>
        </div>
      )}

      <div className="mt-4 flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
        <Clock className="w-4 h-4" />
        <span>{result.processing_time_ms.toFixed(1)}ms</span>
      </div>
    </Card>
  )
}
