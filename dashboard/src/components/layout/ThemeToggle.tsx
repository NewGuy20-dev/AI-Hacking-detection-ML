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
    <div className="flex gap-1 p-1 rounded-full bg-clay-card dark:bg-clay-card-dark shadow-clay dark:shadow-clay-dark">
      <button
        onClick={() => setTheme('light')}
        className={`p-2 rounded-full transition-all ${
          theme === 'light'
            ? 'bg-clay-bg dark:bg-clay-bg-dark shadow-clay-inset'
            : 'hover:bg-clay-bg/50'
        }`}
        aria-label="Light mode"
      >
        <Sun className="w-4 h-4" />
      </button>
      <button
        onClick={() => setTheme('dark')}
        className={`p-2 rounded-full transition-all ${
          theme === 'dark'
            ? 'bg-clay-bg dark:bg-clay-bg-dark shadow-clay-inset dark:shadow-clay-dark-inset'
            : 'hover:bg-clay-bg/50'
        }`}
        aria-label="Dark mode"
      >
        <Moon className="w-4 h-4" />
      </button>
    </div>
  )
}
