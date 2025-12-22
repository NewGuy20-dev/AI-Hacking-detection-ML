export interface PredictResponse {
  is_attack: boolean
  confidence: number
  attack_type: string | null
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
  processing_time_ms: number
}

export interface BatchResponse {
  results: PredictResponse[]
  total_processing_time_ms: number
}

export interface HealthResponse {
  status: string
  uptime_seconds: number
}

export interface ReadinessResponse {
  status: string
  models_loaded: {
    pytorch: string[]
    sklearn: string[]
  }
  uptime_seconds: number
}

export interface ScanHistoryItem {
  id: string
  input: string
  type: 'payload' | 'url'
  result: PredictResponse
  timestamp: Date
}

export interface Stats {
  total: number
  malicious: number
  safe: number
  byType: Record<string, number>
}
