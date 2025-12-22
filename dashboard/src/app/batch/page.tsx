'use client'

import { useState, useCallback } from 'react'
import { Card, Button, Progress } from '@/components/ui'
import { useBatchScan } from '@/hooks/usePredict'
import { Upload, Download, FileText } from 'lucide-react'

export default function BatchPage() {
  const [file, setFile] = useState<File | null>(null)
  const [lines, setLines] = useState<string[]>([])
  const { mutate, data, isPending } = useBatchScan()

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f?.type === 'text/plain') processFile(f)
  }, [])

  const processFile = async (f: File) => {
    setFile(f)
    const text = await f.text()
    const parsed = text.split('\n').filter((l) => l.trim()).slice(0, 100)
    setLines(parsed)
  }

  const handleSubmit = () => {
    if (lines.length > 0) mutate(lines)
  }

  const exportCSV = () => {
    if (!data) return
    const csv = ['Input,Verdict,Confidence']
    data.results.forEach((r, i) => {
      csv.push(`"${lines[i]}",${r.is_attack ? 'Malicious' : 'Safe'},${(r.confidence * 100).toFixed(1)}%`)
    })
    const blob = new Blob([csv.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'scan-results.csv'
    a.click()
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">📁 Batch Analysis</h1>

      <Card
        className="border-2 border-dashed border-gray-300 dark:border-gray-600 text-center py-12 cursor-pointer"
        onDrop={handleDrop}
        onDragOver={(e: React.DragEvent) => e.preventDefault()}
      >
        <input
          type="file"
          accept=".txt"
          className="hidden"
          id="file-upload"
          onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])}
        />
        <label htmlFor="file-upload" className="cursor-pointer">
          <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <p className="text-gray-600 dark:text-gray-400">
            {file ? file.name : 'Drop a .txt file or click to upload'}
          </p>
          {lines.length > 0 && (
            <p className="text-sm text-gray-500 mt-2">{lines.length} lines loaded</p>
          )}
        </label>
      </Card>

      <div className="flex gap-3">
        <Button variant="primary" onClick={handleSubmit} loading={isPending} disabled={lines.length === 0}>
          <FileText className="w-4 h-4 mr-2" />
          Analyze {lines.length} Items
        </Button>
        {data && (
          <Button onClick={exportCSV}>
            <Download className="w-4 h-4 mr-2" />
            Export CSV
          </Button>
        )}
      </div>

      {data && (
        <Card>
          <div className="flex justify-between mb-4">
            <span className="font-semibold">Results</span>
            <span className="text-sm text-gray-500">{data.total_processing_time_ms.toFixed(0)}ms total</span>
          </div>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {data.results.map((r, i) => (
              <div
                key={i}
                className="flex items-center justify-between p-3 rounded-clay-sm bg-clay-bg dark:bg-clay-bg-dark"
              >
                <span className="text-sm truncate flex-1 mr-4">{lines[i]?.slice(0, 50)}</span>
                <span className={r.is_attack ? 'text-red-500 font-semibold' : 'text-green-500'}>
                  {r.is_attack ? '⚠️ Malicious' : '✅ Safe'}
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
