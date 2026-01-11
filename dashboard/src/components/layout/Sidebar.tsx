'use client'

import { useState, useEffect, useCallback } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import { LayoutDashboard, Search, FolderUp, History, Cpu, Menu, X, ChevronLeft, Shield } from 'lucide-react'

const navItems = [
  { href: '/', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/scanner', label: 'Scanner', icon: Search },
  { href: '/batch', label: 'Batch', icon: FolderUp },
  { href: '/history', label: 'History', icon: History },
  { href: '/models', label: 'Models', icon: Cpu },
]

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const pathname = usePathname()
  const [isMobile, setIsMobile] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768)
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  // Close on Escape
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape' && mobileOpen) setMobileOpen(false)
  }, [mobileOpen])

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  // Mobile overlay
  if (isMobile) {
    return (
      <>
        <button
          onClick={() => setMobileOpen(true)}
          className="fixed top-4 left-4 z-50 p-2 glass rounded-xl md:hidden"
          aria-label="Open navigation menu"
          aria-expanded={mobileOpen}
          aria-controls="mobile-nav"
        >
          <Menu className="w-6 h-6" aria-hidden="true" />
        </button>

        {mobileOpen && (
          <div className="fixed inset-0 z-50 md:hidden" role="dialog" aria-modal="true" aria-label="Navigation menu">
            <div 
              className="fixed inset-0 bg-black/50" 
              onClick={() => setMobileOpen(false)}
              aria-hidden="true"
            />
            <aside 
              id="mobile-nav"
              className="fixed left-0 top-0 h-full w-64 glass border-r border-border p-4 animate-in"
              role="navigation"
              aria-label="Main navigation"
            >
              <div className="flex items-center justify-between mb-6">
                <span className="text-lg font-bold">Menu</span>
                <button 
                  onClick={() => setMobileOpen(false)} 
                  aria-label="Close navigation menu"
                  className="p-2 rounded-lg hover:bg-muted transition-colors"
                >
                  <X className="w-5 h-5" aria-hidden="true" />
                </button>
              </div>
              <nav className="space-y-2" role="menubar" aria-label="Main">
                {navItems.map((item, index) => {
                  const isActive = pathname === item.href
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setMobileOpen(false)}
                      role="menuitem"
                      aria-current={isActive ? 'page' : undefined}
                      tabIndex={0}
                      className={cn(
                        'flex items-center gap-3 px-4 py-3 rounded-xl transition-all',
                        isActive ? 'bg-primary/10 text-primary' : 'hover:bg-muted'
                      )}
                    >
                      <item.icon className="w-5 h-5" aria-hidden="true" />
                      <span>{item.label}</span>
                    </Link>
                  )
                })}
              </nav>
            </aside>
          </div>
        )}
      </>
    )
  }

  // Desktop sidebar
  return (
    <aside 
      className={cn(
        'fixed left-0 top-0 h-full bg-card/80 backdrop-blur-xl border-r border-border/50 transition-all duration-300 z-40',
        collapsed ? 'w-16' : 'w-64'
      )}
      role="navigation"
      aria-label="Main navigation"
    >
      {/* Logo/Brand */}
      <div className={cn(
        'h-16 flex items-center border-b border-border/50 px-4',
        collapsed ? 'justify-center' : 'gap-3'
      )}>
        <div className="p-2 bg-gradient-to-br from-primary to-purple-600 rounded-xl">
          <Shield className="w-5 h-5 text-white" />
        </div>
        {!collapsed && (
          <span className="font-bold text-sm truncate">AI Security</span>
        )}
      </div>
      
      <nav className="p-3 space-y-1" role="menubar" aria-label="Main">
        {navItems.map((item, index) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              role="menuitem"
              aria-current={isActive ? 'page' : undefined}
              className={cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all relative group',
                isActive 
                  ? 'bg-primary/10 text-primary' 
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/50',
                collapsed && 'justify-center px-2'
              )}
              title={collapsed ? item.label : undefined}
              aria-label={collapsed ? item.label : undefined}
            >
              <item.icon className={cn('w-5 h-5 shrink-0', isActive && 'text-primary')} aria-hidden="true" />
              {!collapsed && <span className="font-medium text-sm">{item.label}</span>}
              {isActive && !collapsed && (
                <div className="absolute right-2 w-1.5 h-1.5 bg-primary rounded-full" aria-hidden="true" />
              )}
              
              {collapsed && (
                <span 
                  className="absolute left-full ml-2 px-2 py-1 bg-card border border-border rounded-lg text-xs font-medium opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap shadow-lg z-50"
                  role="tooltip"
                >
                  {item.label}
                </span>
              )}
            </Link>
          )
        })}
      </nav>

      {/* Collapse Toggle - Bottom */}
      <div className="absolute bottom-4 left-0 right-0 px-3">
        <button
          onClick={onToggle}
          className={cn(
            'w-full flex items-center gap-2 px-3 py-2 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-all text-sm',
            collapsed && 'justify-center px-2'
          )}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-expanded={!collapsed}
        >
          <ChevronLeft className={cn('w-4 h-4 transition-transform', collapsed && 'rotate-180')} aria-hidden="true" />
          {!collapsed && <span className="font-medium">Collapse</span>}
        </button>
      </div>
    </aside>
  )
}
