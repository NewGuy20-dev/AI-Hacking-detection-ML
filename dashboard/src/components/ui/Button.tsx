import { cn } from '@/lib/utils'
import { ButtonHTMLAttributes, forwardRef } from 'react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'primary' | 'secondary' | 'danger' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  loading?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'md', loading = false, ...props }, ref) => {
    return (
      <button
        className={cn(
          'cyber-button focus:outline-none focus:ring-0 disabled:opacity-50 disabled:cursor-not-allowed group',
          {
            'text-cyber-text border-cyber-border hover:text-primary': variant === 'default',
            'border-primary text-primary shadow-neon-primary bg-primary/10 hover:bg-primary/20': variant === 'primary',
            'border-secondary text-secondary hover:shadow-[0_0_15px_rgba(191,0,255,0.5)] hover:border-secondary': variant === 'secondary',
            'border-danger text-danger hover:shadow-neon-danger': variant === 'danger',
            'border-transparent bg-transparent shadow-none hover:border-cyber-border hover:bg-white/5': variant === 'ghost',
            'px-3 py-1.5 text-xs': size === 'sm',
            'px-4 py-2 text-sm': size === 'md',
            'px-6 py-3 text-base': size === 'lg',
          },
          className
        )}
        disabled={loading || props.disabled}
        ref={ref}
        {...props}
      >
        <div className="absolute inset-0 bg-primary/10 w-0 group-hover:w-full transition-all duration-300 ease-out z-0 mix-blend-screen" />
        <span className="relative z-10 flex items-center gap-2">
          {loading ? <span className="animate-spin mr-2">⟳</span> : null}
          {props.children}
        </span>
      </button>
    )
  }
)

Button.displayName = 'Button'
