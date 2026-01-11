'use client'

import { useTheme } from 'next-themes'
import { Sun, Moon } from 'lucide-react'
import { useEffect, useState } from 'react'

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => setMounted(true), [])

  if (!mounted) return null

  return (
    <div className="flex gap-1 p-1 clay-card" role="group" aria-label="Theme selection">
      <button
        onClick={() => setTheme('light')}
        className={`p-2 rounded-lg transition-all duration-200 ${
          theme === 'light'
            ? 'bg-primary text-white shadow-clay-inset'
            : 'hover:bg-clay-border dark:hover:bg-clay-dark-border'
        }`}
        aria-label="Switch to light mode"
        aria-pressed={theme === 'light'}
      >
        <Sun className="w-4 h-4" aria-hidden="true" />
      </button>
      <button
        onClick={() => setTheme('dark')}
        className={`p-2 rounded-lg transition-all duration-200 ${
          theme === 'dark'
            ? 'bg-primary text-white shadow-clay-inset'
            : 'hover:bg-clay-border dark:hover:bg-clay-dark-border'
        }`}
        aria-label="Switch to dark mode"
        aria-pressed={theme === 'dark'}
      >
        <Moon className="w-4 h-4" aria-hidden="true" />
      </button>
    </div>
  )
}
