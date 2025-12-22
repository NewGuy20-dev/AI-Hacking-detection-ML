'use client'

import Link from 'next/link'
import { ThemeToggle } from './ThemeToggle'
import { Shield } from 'lucide-react'

export function Header() {
  return (
    <header className="sticky top-0 z-50 bg-clay-card dark:bg-clay-card-dark shadow-clay dark:shadow-clay-dark rounded-clay mb-6 p-4">
      <div className="flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3">
          <Shield className="w-8 h-8 text-blue-500" />
          <span className="text-xl font-bold text-gray-800 dark:text-gray-100">
            AI Hacking Detection
          </span>
        </Link>
        <ThemeToggle />
      </div>
    </header>
  )
}
