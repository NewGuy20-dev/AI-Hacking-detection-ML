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
        'inline-flex items-center px-2 py-0.5 border text-[10px] font-bold font-mono tracking-widest uppercase transition-all whitespace-nowrap',
        {
          'bg-cyber-border/20 border-cyber-border text-cyber-text shadow-sm': variant === 'default',
          'bg-success/10 border-success/30 text-success shadow-[0_0_10px_rgba(0,255,102,0.2)]': variant === 'success',
          'bg-warning/10 border-warning/30 text-warning shadow-[0_0_10px_rgba(255,176,0,0.2)]': variant === 'warning',
          'bg-danger/10 border-danger/30 text-danger shadow-neon-danger': variant === 'danger',
          'bg-info/10 border-info/30 text-info shadow-[0_0_10px_rgba(0,174,255,0.2)]': variant === 'info',
        },
        className
      )}
    >
      {children}
    </span>
  )
}
