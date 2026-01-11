'use client'

import Link from 'next/link'
import { ThemeToggle } from './ThemeToggle'
import { Shield } from 'lucide-react'

export function Header() {
  return (
    <header className="sticky top-0 z-50 clay-card mb-6 p-4">
      <div className="flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <Shield className="w-8 h-8 text-primary" />
          <span className="text-xl font-bold">
            AI Hacking Detection
          </span>
        </Link>
        <ThemeToggle />
      </div>
    </header>
  )
}
