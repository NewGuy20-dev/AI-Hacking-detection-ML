'use client'

import { useState } from 'react'
import { Button, Input } from '@/components/ui'
import { ResultCard } from './ResultCard'
import { useURLScan } from '@/hooks/usePredict'
import { Globe } from 'lucide-react'

const EXAMPLES = [
  'https://google.com',
  'http://paypa1-secure.tk/login',
  'https://github.com',
]

export function URLScanner() {
  const [url, setUrl] = useState('')
  const { mutate, data, isPending, reset } = useURLScan()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (url.trim()) mutate(url)
  }

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter URL to analyze (e.g., https://example.com)"
          type="url"
        />
        <div className="flex gap-3">
          <Button type="submit" variant="primary" loading={isPending}>
            <Globe className="w-4 h-4 mr-2" />
            Analyze URL
          </Button>
          {data && (
            <Button type="button" onClick={() => { reset(); setUrl('') }}>
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
            onClick={() => setUrl(ex)}
            className="text-sm px-3 py-1 rounded-full bg-clay-card dark:bg-clay-card-dark shadow-clay dark:shadow-clay-dark hover:shadow-clay-hover transition-all"
          >
            {ex}
          </button>
        ))}
      </div>

      <ResultCard result={data || null} input={data ? url : undefined} />
    </div>
  )
}
