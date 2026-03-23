'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import { LayoutDashboard, Search, FolderUp, History, Cpu } from 'lucide-react'

const navItems = [
  { href: '/', label: 'DASHBOARD', icon: LayoutDashboard },
  { href: '/scanner', label: 'SCANNER', icon: Search },
  { href: '/batch', label: 'BATCH_JOB', icon: FolderUp },
  { href: '/history', label: 'HISTORY', icon: History },
  { href: '/models', label: 'MODELS', icon: Cpu },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-64 shrink-0 font-mono">
      <nav className="sticky top-24 space-y-1">
        <div className="text-[10px] uppercase text-cyber-muted tracking-widest mb-4 px-4">
          // Modules Selection
        </div>
        {navItems.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 px-4 py-3 transition-all duration-300 relative group overflow-hidden border border-transparent',
                isActive
                  ? 'bg-primary/10 text-primary border-primary/30'
                  : 'hover:bg-white/5 text-cyber-muted hover:text-white hover:border-cyber-border'
              )}
            >
              {/* Scanline hover effect */ }
              <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent w-0 group-hover:w-full transition-all duration-500 z-0" />
              
              <div className="relative z-10 flex items-center justify-center w-6 h-6">
                {isActive && <div className="absolute inset-0 bg-primary/20 blur-md rounded-full" />}
                <item.icon className={cn(
                  "w-4 h-4 transition-transform duration-300 group-hover:scale-110",
                  isActive ? "text-primary shadow-neon-primary" : ""
                )} />
              </div>

              <span className={cn(
                "text-sm font-medium tracking-wider relative z-10 transition-colors",
                isActive ? "text-primary" : ""
              )}>
                {isActive ? `[ ${item.label} ]` : item.label}
              </span>
              
              {isActive && (
                <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary shadow-neon-primary" />
              )}
            </Link>
          )
        })}
      </nav>
      
      <div className="mt-12 px-4 border-t border-cyber-border pt-4 text-xs text-cyber-muted font-mono space-y-2">
        <div className="flex justify-between items-center">
          <span>SYS.LATENCY</span>
          <span className="text-success">12ms</span>
        </div>
        <div className="flex justify-between items-center">
          <span>MEM.USAGE</span>
          <span className="text-warning">84%</span>
        </div>
        <div className="flex justify-between items-center">
          <span>UPLINK</span>
          <span className="text-primary">SECURE</span>
        </div>
      </div>
    </aside>
  )
}
