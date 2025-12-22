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
                'flex items-center gap-3 px-4 py-3 rounded-clay-sm transition-all duration-200',
                'hover:shadow-clay-hover hover:-translate-y-0.5',
                isActive
                  ? 'bg-clay-card dark:bg-clay-card-dark shadow-clay dark:shadow-clay-dark text-blue-600 dark:text-blue-400'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
              )}
            >
              <item.icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}
