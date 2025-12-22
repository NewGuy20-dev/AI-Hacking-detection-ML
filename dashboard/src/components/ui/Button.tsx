import { cn } from '@/lib/utils'
import { ButtonHTMLAttributes, forwardRef } from 'react'
import { Loader2 } from 'lucide-react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'primary'
  loading?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', loading, children, disabled, ...props }, ref) => {
    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={cn(
          'px-6 py-3 rounded-clay-sm font-semibold transition-all duration-200',
          'shadow-clay dark:shadow-clay-dark',
          'hover:shadow-clay-hover hover:-translate-y-0.5',
          'active:shadow-clay-pressed active:translate-y-0',
          'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0',
          variant === 'default' && 'bg-clay-card dark:bg-clay-card-dark text-gray-800 dark:text-gray-100',
          variant === 'primary' && 'bg-gradient-to-r from-clay-info to-blue-400 text-white',
          className
        )}
        {...props}
      >
        {loading ? (
          <span className="flex items-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            Processing...
          </span>
        ) : (
          children
        )}
      </button>
    )
  }
)

Button.displayName = 'Button'
