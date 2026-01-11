'use client'

import Link from 'next/link'
import { ThemeToggle } from './ThemeToggle'
import { Shield, Bell } from 'lucide-react'
import { cn } from '@/lib/utils'

interface HeaderProps {
  sidebarCollapsed: boolean
}

export function Header({ sidebarCollapsed }: HeaderProps) {
  return (
    <header 
      className={cn(
        'sticky top-0 z-30 glass border-b border-border px-6 py-4 transition-all duration-300',
        sidebarCollapsed ? 'md:ml-16' : 'md:ml-64'
      )}
      role="banner"
    >
      {/* Skip to main content link */}
      <a 
        href="#main-content" 
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-primary-foreground focus:rounded-lg"
      >
        Skip to main content
      </a>
      
      <div className="flex items-center justify-between">
        <Link 
          href="/" 
          className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          aria-label="AI Hacking Detection - Go to dashboard"
        >
          <Shield className="w-8 h-8 text-primary" aria-hidden="true" />
          <span className="text-xl font-bold hidden sm:block">AI Hacking Detection</span>
        </Link>
        
        <div className="flex items-center gap-3" role="group" aria-label="Header actions">
          <button 
            className="p-2 rounded-xl hover:bg-muted transition-colors relative" 
            aria-label="Notifications - 1 unread"
          >
            <Bell className="w-5 h-5" aria-hidden="true" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-destructive rounded-full" aria-hidden="true" />
          </button>
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
