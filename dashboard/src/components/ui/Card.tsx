import { cn } from '@/lib/utils'
import { ReactNode } from 'react'

interface CardProps {
  children: ReactNode
  className?: string
  variant?: 'default' | 'danger' | 'success'
}

export function Card({ children, className, variant = 'default' }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-clay p-6 transition-all duration-200',
        'bg-clay-card dark:bg-clay-card-dark',
        'shadow-clay dark:shadow-clay-dark',
        'hover:shadow-clay-hover hover:-translate-y-0.5',
        variant === 'danger' && 'border-l-4 border-clay-danger-dark',
        variant === 'success' && 'border-l-4 border-clay-success',
        className
      )}
    >
      {children}
    </div>
  )
}
