'use client'

import { useState } from 'react'
import { Button, Textarea } from '@/components/ui'
import { ResultCard } from './ResultCard'
import { usePayloadScan } from '@/hooks/usePredict'
import { Search } from 'lucide-react'

const EXAMPLES = [
  "' OR '1'='1",
  "<script>alert('XSS')</script>",
  "; cat /etc/passwd",
  "Hello, this is a normal message",
]

export function PayloadScanner() {
  const [payload, setPayload] = useState('')
  const { mutate, data, isPending, reset } = usePayloadScan()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (payload.trim()) mutate(payload)
  }

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Textarea
          value={payload}
          onChange={(e) => setPayload(e.target.value)}
          placeholder="Enter payload to analyze (e.g., ' OR 1=1--)"
          rows={4}
        />
        <div className="flex gap-3">
          <Button type="submit" variant="primary" loading={isPending}>
            <Search className="w-4 h-4 mr-2" />
            Analyze Payload
          </Button>
          {data && (
            <Button type="button" onClick={() => { reset(); setPayload('') }}>
              Clear
            </Button>
          )}
        </div>
      </form>

      <div className="flex flex-wrap gap-2">
        <span className="text-sm text-gray-500 dark:text-gray-400">Try:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            onClick={() => setPayload(ex)}
            className="text-sm px-3 py-1 rounded-full bg-clay-card dark:bg-clay-card-dark shadow-clay dark:shadow-clay-dark hover:shadow-clay-hover transition-all"
          >
            {ex.slice(0, 20)}...
          </button>
        ))}
      </div>

      <ResultCard result={data || null} input={data ? payload : undefined} />
    </div>
  )
}
