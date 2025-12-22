import { cn } from '@/lib/utils'

interface ProgressProps {
  value: number
  max?: number
  className?: string
  showLabel?: boolean
}

export function Progress({ value, max = 100, className, showLabel = true }: ProgressProps) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100))
  const color =
    percentage > 70 ? 'from-clay-danger-dark to-red-500' :
    percentage > 40 ? 'from-clay-warning to-orange-400' :
    'from-clay-success to-green-400'

  return (
    <div className={cn('w-full', className)}>
      <div
        className={cn(
          'h-3 rounded-full overflow-hidden',
          'bg-clay-bg dark:bg-clay-bg-dark',
          'shadow-clay-inset dark:shadow-clay-dark-inset'
        )}
      >
        <div
          className={cn('h-full rounded-full transition-all duration-500 bg-gradient-to-r', color)}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && (
        <div className="flex justify-end mt-1">
          <span className="text-sm text-gray-600 dark:text-gray-400">{percentage.toFixed(1)}%</span>
        </div>
      )}
    </div>
  )
}
