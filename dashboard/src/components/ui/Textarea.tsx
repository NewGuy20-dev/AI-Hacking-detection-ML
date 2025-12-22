import { cn } from '@/lib/utils'
import { TextareaHTMLAttributes, forwardRef } from 'react'

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        className={cn(
          'w-full px-4 py-3 rounded-clay-sm transition-all duration-200 resize-none',
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

Textarea.displayName = 'Textarea'
