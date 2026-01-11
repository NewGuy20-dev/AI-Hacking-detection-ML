'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import { LayoutDashboard, Search, FolderUp, History, Cpu } from 'lucide-react'

const navItems = [
  { href: '/', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/scanner', label: 'Scanner', icon: Search },
  { href: '/batch', label: 'Batch', icon: FolderUp },
  { href: '/history', label: 'History', icon: History },
  { href: '/models', label: 'Models', icon: Cpu },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-64 shrink-0">
      <nav className="sticky top-24 space-y-2">
        {navItems.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 px-4 py-3 rounded-clay transition-all duration-200 relative',
                isActive
                  ? 'clay-card bg-primary/10 text-primary border-primary/20'
                  : 'clay-card hover:bg-clay-border/50 dark:hover:bg-clay-dark-border/50'
              )}
            >
              <item.icon className={cn(
                "w-5 h-5 transition-transform duration-200",
                isActive ? "text-primary" : "text-clay-muted dark:text-clay-dark-muted"
              )} />
              <span className={cn(
                "font-medium transition-colors",
                isActive ? "text-primary" : ""
              )}>{item.label}</span>
              {isActive && (
                <div className="absolute right-2 top-1/2 -translate-y-1/2 w-2 h-2 bg-primary rounded-full" />
              )}
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}
