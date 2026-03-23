import { PredictResponse, BatchResponse, HealthResponse, ReadinessResponse, TimeSeriesRequest } from '@/types/api'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${endpoint}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    throw new Error(`API Error: ${res.status}`)
  }
  return res.json()
}

export const api = {
  predictPayload: (payload: string) =>
    fetchAPI<PredictResponse>('/api/v1/predict/payload', {
      method: 'POST',
      body: JSON.stringify({ payload }),
    }),

  predictURL: (url: string) =>
    fetchAPI<PredictResponse>('/api/v1/predict/url', {
      method: 'POST',
      body: JSON.stringify({ url }),
    }),

  predictBatch: (payloads: string[], urls?: string[]) =>
    fetchAPI<BatchResponse>('/api/v1/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ payloads, urls }),
    }),

  predictTimeSeries: (data: TimeSeriesRequest) =>
    fetchAPI<PredictResponse>('/api/v1/predict/timeseries', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  health: () => fetchAPI<HealthResponse>('/health'),

  ready: () => fetchAPI<ReadinessResponse>('/health/ready'),
}
