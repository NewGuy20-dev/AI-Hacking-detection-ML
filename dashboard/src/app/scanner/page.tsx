'use client'

import { Tabs } from '@/components/ui'
import { PayloadScanner, URLScanner } from '@/components/scanner'
import { FileCode, Globe } from 'lucide-react'

const tabs = [
  { id: 'payload', label: 'Payload', icon: <FileCode className="w-4 h-4" /> },
  { id: 'url', label: 'URL', icon: <Globe className="w-4 h-4" /> },
]

export default function ScannerPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-6">
        🔍 Scanner
      </h1>
      <Tabs tabs={tabs} defaultTab="payload">
        {(activeTab) => (
          <>
            {activeTab === 'payload' && <PayloadScanner />}
            {activeTab === 'url' && <URLScanner />}
          </>
        )}
      </Tabs>
    </div>
  )
}
