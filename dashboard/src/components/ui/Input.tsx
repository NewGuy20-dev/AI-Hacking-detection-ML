import { cn } from '@/lib/utils'
import { InputHTMLAttributes, forwardRef } from 'react'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={cn(
          'w-full px-4 py-3 rounded-clay-sm transition-all duration-200',
          'bg-clay-card dark:bg-clay-card-dark',
          'shadow-clay-inset dark:shadow-clay-dark-inset',
          'border border-white/20 dark:border-white/10',
          'text-gray-800 dark:text-gray-100',
          'placeholder:text-gray-400 dark:placeholder:text-gray-500',
          'focus:outline-none focus:ring-2 focus:ring-clay-info/50',
          className
        )}
        {...props}
      />
    )
  }
)

Input.displayName = 'Input'
