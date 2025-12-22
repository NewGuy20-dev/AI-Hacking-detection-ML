import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { ScanHistoryItem, PredictResponse } from '@/types/api'

interface HistoryStore {
  history: ScanHistoryItem[]
  addScan: (input: string, type: 'payload' | 'url', result: PredictResponse) => void
  clearHistory: () => void
}

export const useHistoryStore = create<HistoryStore>()(
  persist(
    (set) => ({
      history: [],
      addScan: (input, type, result) =>
        set((state) => ({
          history: [
            {
              id: crypto.randomUUID(),
              input: input.slice(0, 100),
              type,
              result,
              timestamp: new Date(),
            },
            ...state.history.slice(0, 99),
          ],
        })),
      clearHistory: () => set({ history: [] }),
    }),
    { name: 'scan-history' }
  )
)
