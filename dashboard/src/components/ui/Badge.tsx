import { cn } from '@/lib/utils'

interface BadgeProps {
  variant: 'critical' | 'high' | 'medium' | 'low' | 'info'
  children: React.ReactNode
  className?: string
}

const variants = {
  critical: 'bg-clay-danger text-red-900',
  high: 'bg-clay-warning text-orange-900',
  medium: 'bg-yellow-100 text-yellow-900',
  low: 'bg-clay-success text-green-900',
  info: 'bg-clay-info text-blue-900',
}

export function Badge({ variant, children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold',
        'shadow-clay dark:shadow-clay-dark',
        variants[variant],
        className
      )}
    >
      {children}
    </span>
  )
}
