import { cn } from '@/lib/utils'

interface BadgeProps {
  children: React.ReactNode
  variant?: 'default' | 'success' | 'destructive' | 'warning'
  className?: string
}

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
        {
          'bg-primary/10 text-primary': variant === 'default',
          'bg-success/10 text-success': variant === 'success',
          'bg-destructive/10 text-destructive': variant === 'destructive',
          'bg-warning/10 text-warning': variant === 'warning',
        },
        className
      )}
    >
      {children}
    </span>
  )
}
