'use client'

import { Tabs } from '@/components/ui'
import { PayloadScanner, URLScanner } from '@/components/scanner'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { FileCode, Globe } from 'lucide-react'

const tabs = [
  { id: 'payload', label: 'Payload', icon: <FileCode className="w-4 h-4" /> },
  { id: 'url', label: 'URL', icon: <Globe className="w-4 h-4" /> },
]

export default function ScannerPage() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-primary">
            🔍 Scanner
          </CardTitle>
        </CardHeader>
      </Card>
      
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
