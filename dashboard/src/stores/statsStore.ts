import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { Stats } from '@/types/api'

interface StatsStore extends Stats {
  increment: (isAttack: boolean, attackType?: string) => void
  reset: () => void
}

export const useStatsStore = create<StatsStore>()(
  persist(
    (set) => ({
      total: 0,
      malicious: 0,
      safe: 0,
      byType: {},
      increment: (isAttack, attackType) =>
        set((state) => ({
          total: state.total + 1,
          malicious: isAttack ? state.malicious + 1 : state.malicious,
          safe: isAttack ? state.safe : state.safe + 1,
          byType: attackType
            ? { ...state.byType, [attackType]: (state.byType[attackType] || 0) + 1 }
            : state.byType,
        })),
      reset: () => set({ total: 0, malicious: 0, safe: 0, byType: {} }),
    }),
    { name: 'scan-stats' }
  )
)
