import { clsx, type ClassValue } from 'clsx'

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs)
}

export function getSeverity(confidence: number): { label: string; color: string; emoji: string } {
  if (confidence > 0.95) return { label: 'CRITICAL', color: 'bg-red-500', emoji: '🔴' }
  if (confidence > 0.85) return { label: 'HIGH', color: 'bg-orange-500', emoji: '🟠' }
  if (confidence > 0.7) return { label: 'MEDIUM', color: 'bg-yellow-500', emoji: '🟡' }
  return { label: 'LOW', color: 'bg-green-500', emoji: '🟢' }
}

export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`
}
