'use client'

import { useMutation } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { useHistoryStore } from '@/stores/historyStore'
import { useStatsStore } from '@/stores/statsStore'

export function usePayloadScan() {
  const addScan = useHistoryStore((s) => s.addScan)
  const increment = useStatsStore((s) => s.increment)

  return useMutation({
    mutationFn: api.predictPayload,
    onSuccess: (data, payload) => {
      addScan(payload, 'payload', data)
      increment(data.is_attack, data.attack_type || undefined)
    },
  })
}

export function useURLScan() {
  const addScan = useHistoryStore((s) => s.addScan)
  const increment = useStatsStore((s) => s.increment)

  return useMutation({
    mutationFn: api.predictURL,
    onSuccess: (data, url) => {
      addScan(url, 'url', data)
      increment(data.is_attack, data.attack_type || undefined)
    },
  })
}

export function useBatchScan() {
  return useMutation({
    mutationFn: api.predictBatch,
  })
}
