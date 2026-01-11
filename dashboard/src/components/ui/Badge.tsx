import { cn } from '@/lib/utils'
import { ReactNode } from 'react'

interface BadgeProps {
  children: ReactNode
  className?: string
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'
}

export function Badge({ children, className, variant = 'default' }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
        {
          'bg-clay-border/20 dark:bg-clay-dark-border/20 text-clay-text dark:text-clay-dark-text': variant === 'default',
          'bg-success/20 text-success': variant === 'success',
          'bg-warning/20 text-warning': variant === 'warning',
          'bg-danger/20 text-danger': variant === 'danger',
          'bg-info/20 text-info': variant === 'info',
        },
        className
      )}
    >
      {children}
    </span>
  )
}
