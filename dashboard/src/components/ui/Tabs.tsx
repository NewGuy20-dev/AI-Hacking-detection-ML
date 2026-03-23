'use client'

import { cn } from '@/lib/utils'
import { ReactNode, useState } from 'react'

interface Tab {
  id: string
  label: string
  icon?: ReactNode
}

interface TabsProps {
  tabs: Tab[]
  defaultTab?: string
  children: (activeTab: string) => ReactNode
  className?: string
}

export function Tabs({ tabs, defaultTab, children, className }: TabsProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id)

  return (
    <div className={cn('w-full', className)}>
      <div className="bg-black border border-cyber-border p-1 mb-6 inline-flex">
        <div className="flex space-x-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'flex items-center gap-2 px-6 py-2 border font-mono text-xs uppercase tracking-widest font-bold transition-all duration-300 relative',
                activeTab === tab.id
                  ? 'bg-primary/20 border-primary/50 text-primary shadow-[inset_0_0_15px_rgba(0,243,255,0.15)]'
                  : 'border-transparent text-cyber-muted hover:text-white hover:bg-white/5'
              )}
            >
              {activeTab === tab.id && (
                <div className="absolute bottom-0 left-0 w-full h-[2px] bg-primary shadow-neon-primary" />
              )}
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>
      <div className="animate-in fade-in slide-in-from-bottom-2 duration-300">
        {children(activeTab)}
      </div>
    </div>
  )
}
