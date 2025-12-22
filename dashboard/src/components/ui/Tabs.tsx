'use client'

import { cn } from '@/lib/utils'
import { useState } from 'react'

interface Tab {
  id: string
  label: string
  icon?: React.ReactNode
}

interface TabsProps {
  tabs: Tab[]
  defaultTab?: string
  onChange?: (id: string) => void
  children: (activeTab: string) => React.ReactNode
}

export function Tabs({ tabs, defaultTab, onChange, children }: TabsProps) {
  const [active, setActive] = useState(defaultTab || tabs[0]?.id)

  const handleChange = (id: string) => {
    setActive(id)
    onChange?.(id)
  }

  return (
    <div>
      <div
        className={cn(
          'inline-flex gap-2 p-2 rounded-clay-lg',
          'bg-clay-card dark:bg-clay-card-dark',
          'shadow-clay dark:shadow-clay-dark'
        )}
      >
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => handleChange(tab.id)}
            className={cn(
              'px-4 py-2 rounded-clay-sm font-medium transition-all duration-200',
              'flex items-center gap-2',
              active === tab.id
                ? 'bg-clay-bg dark:bg-clay-bg-dark shadow-clay-inset dark:shadow-clay-dark-inset text-gray-800 dark:text-gray-100'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
            )}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>
      <div className="mt-6">{children(active)}</div>
    </div>
  )
}
