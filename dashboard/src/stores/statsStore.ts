import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { Stats } from '@/types/api'

interface DailyStats {
  date: string
  threats: number
  clean: number
}

interface StatsStore extends Stats {
  dailyStats: DailyStats[]
  increment: (isAttack: boolean, attackType?: string) => void
  reset: () => void
}

const getTodayKey = () => new Date().toISOString().split('T')[0]

export const useStatsStore = create<StatsStore>()(
  persist(
    (set, get) => ({
      total: 0,
      malicious: 0,
      safe: 0,
      byType: {},
      dailyStats: [],
      increment: (isAttack, attackType) =>
        set((state) => {
          const today = getTodayKey()
          const existingDay = state.dailyStats.find(d => d.date === today)
          
          let newDailyStats: DailyStats[]
          if (existingDay) {
            newDailyStats = state.dailyStats.map(d => 
              d.date === today 
                ? { ...d, threats: d.threats + (isAttack ? 1 : 0), clean: d.clean + (isAttack ? 0 : 1) }
                : d
            )
          } else {
            newDailyStats = [...state.dailyStats.slice(-6), { date: today, threats: isAttack ? 1 : 0, clean: isAttack ? 0 : 1 }]
          }

          return {
            total: state.total + 1,
            malicious: isAttack ? state.malicious + 1 : state.malicious,
            safe: isAttack ? state.safe : state.safe + 1,
            byType: attackType
              ? { ...state.byType, [attackType]: (state.byType[attackType] || 0) + 1 }
              : state.byType,
            dailyStats: newDailyStats,
          }
        }),
      reset: () => set({ total: 0, malicious: 0, safe: 0, byType: {}, dailyStats: [] }),
    }),
    { 
      name: 'scan-stats',
      skipHydration: true,
    }
  )
)
