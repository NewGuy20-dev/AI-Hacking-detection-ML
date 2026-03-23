'use client'

import Link from 'next/link'
import { ThemeToggle } from './ThemeToggle'
import { ShieldAlert, Terminal } from 'lucide-react'

export function Header() {
  return (
    <header className="py-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
      <div className="flex items-center gap-4 group">
        <Link href="/" className="flex items-center gap-3 transition-opacity">
          <div className="relative">
            <ShieldAlert className="w-8 h-8 text-primary group-hover:text-white transition-colors animate-pulse-slow" />
            <div className="absolute inset-0 bg-primary/20 blur-md group-hover:bg-primary/40 transition-colors rounded-full" />
          </div>
          <div className="flex flex-col">
            <span className="text-xl font-bold font-mono tracking-widest text-white group-hover:text-primary transition-colors uppercase">
              Omni-Sec <span className="opacity-50">v4.0</span>
            </span>
            <span className="text-[10px] text-primary/70 font-mono tracking-widest uppercase flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-danger rounded-full animate-pulse-fast inline-block" />
              Live Threat Detection Network
            </span>
          </div>
        </Link>
      </div>
      <div className="flex items-center gap-4">
        <div className="text-xs font-mono text-cyan-500/50 hidden sm:flex items-center gap-2 border border-cyber-border px-3 py-1 bg-black/40">
          <Terminal className="w-3 h-3" />
          SYS.OP: ONLINE
        </div>
      </div>
    </header>
  )
}
