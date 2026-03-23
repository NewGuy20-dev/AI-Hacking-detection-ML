# API Integration Guide

## Overview

The AI Hacking Detection system consists of a FastAPI backend and Next.js frontend, fully integrated and ready for deployment.

## Backend API (FastAPI)

### Base URL
- Development: `http://localhost:8000`
- Production: Configure via environment variable

### Endpoints

#### 1. Health Check
```
GET /health
```
Returns server health status and uptime.

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600.5
}
```

#### 2. Readiness Check
```
GET /health/ready
```
Returns model loading status.

**Response:**
```json
{
  "status": "ready",
  "models_loaded": {
    "pytorch": ["payload_cnn", "url_cnn", "timeseries_lstm"],
    "sklearn": ["payload_rf", "url_rf"]
  },
  "uptime_seconds": 3600.5
}
```

#### 3. Payload Prediction
```
POST /api/v1/predict/payload
```
Analyzes a text payload for attack patterns.

**Request:**
```json
{
  "payload": "' OR '1'='1",
  "include_explanation": false
}
```

**Response:**
```json
{
  "is_attack": true,
  "confidence": 0.98,
  "attack_type": "SQL_INJECTION",
  "severity": "CRITICAL",
  "processing_time_ms": 12.5
}
```

#### 4. URL Prediction
```
POST /api/v1/predict/url
```
Analyzes a URL for maliciousness.

**Request:**
```json
{
  "url": "http://malicious-site.com/phishing",
  "include_explanation": false
}
```

**Response:**
```json
{
  "is_attack": true,
  "confidence": 0.87,
  "attack_type": "MALICIOUS_URL",
  "severity": "HIGH",
  "processing_time_ms": 8.3
}
```

#### 5. Batch Prediction
```
POST /api/v1/predict/batch
```
Analyzes multiple payloads and/or URLs in a single request.

**Request:**
```json
{
  "payloads": ["' OR '1'='1", "Hello world"],
  "urls": ["http://example.com", "http://malicious.com"]
}
```

**Response:**
```json
{
  "results": [
    {
      "is_attack": true,
      "confidence": 0.98,
      "attack_type": "SQL_INJECTION",
      "severity": "CRITICAL",
      "processing_time_ms": 0
    },
    {
      "is_attack": false,
      "confidence": 0.05,
      "attack_type": null,
      "severity": "LOW",
      "processing_time_ms": 0
    }
  ],
  "total_processing_time_ms": 25.7
}
```

#### 6. TimeSeries Prediction
```
POST /api/v1/predict/timeseries
```
Analyzes time-series events for attack patterns.

**Request:**
```json
{
  "events": [
    {"timestamp": 1, "action": "login", "ip": "192.168.1.1"},
    {"timestamp": 2, "action": "access", "ip": "192.168.1.1"}
  ],
  "window_size": 10
}
```

**Response:**
```json
{
  "is_attack": false,
  "confidence": 0.34,
  "attack_type": null,
  "severity": "LOW",
  "processing_time_ms": 45.2
}
```

## Frontend (Next.js)

### Configuration

Set the API URL in `.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### API Client

Located at `dashboard/src/lib/api.ts`, provides typed methods for all endpoints:

```typescript
import { api } from '@/lib/api'

// Payload scan
const result = await api.predictPayload("' OR '1'='1")

// URL scan
const result = await api.predictURL("http://example.com")

// Batch scan
const result = await api.predictBatch(
  ["payload1", "payload2"],
  ["http://url1.com"]
)

// TimeSeries scan
const result = await api.predictTimeSeries({
  events: [...],
  window_size: 10
})

// Health checks
const health = await api.health()
const ready = await api.ready()
```

### React Query Hooks

Located at `dashboard/src/hooks/usePredict.ts`:

```typescript
import { usePayloadScan, useURLScan, useBatchScan, useTimeSeriesScan } from '@/hooks/usePredict'

// In your component
const { mutate, data, isPending } = usePayloadScan()

// Trigger scan
mutate("' OR '1'='1")

// Access result
if (data) {
  console.log(data.is_attack, data.confidence)
}
```

### Pages

1. **Scanner** (`/scanner`) - Single payload/URL analysis
2. **Batch** (`/batch`) - Bulk file upload and analysis
3. **Models** (`/models`) - Model status and performance metrics
4. **History** (`/history`) - Scan history and statistics

## Running the System

### Backend
```bash
# Install dependencies
py -3 -m pip install -r requirements.txt

# Start API server
py -3 -m uvicorn src.api.server:app --reload

# Or use the startup script
./start-api.sh  # Linux/WSL
./start-api.ps1 # Windows PowerShell
```

### Frontend
```bash
cd dashboard

# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
npm start
```

### Testing

```bash
# Backend tests
py -3 -m pytest tests/test_api_integration.py -v

# Frontend tests
cd dashboard
npm test
```

## CORS Configuration

The backend allows requests from:
- `http://localhost:3000`
- `http://127.0.0.1:3000`
- `https://*.vercel.app`

Modify `src/api/server.py` to add additional origins.

## Error Handling

All endpoints return standard HTTP status codes:
- `200` - Success
- `422` - Validation error (invalid input)
- `500` - Server error
- `503` - Service unavailable (models not loaded)

Frontend automatically handles errors via React Query and displays user-friendly messages.

## Performance

- **Payload/URL**: ~10-50ms per request
- **Batch**: ~20-100ms for 100 items
- **TimeSeries**: ~40-80ms per request

Actual performance depends on model complexity and hardware.

## Known Issues

1. **TimeSeries Model**: Currently has low recall (52.5%) for C2 and Bruteforce attacks due to threshold calibration. See TODO list for remediation plan.

2. **Benign Pre-filter**: Some benign inputs may be incorrectly flagged. The pre-filter helps reduce false positives.

3. **Batch Limits**: Maximum 100 items per batch request to prevent memory issues.

## Next Steps

1. Adjust thresholds for C2/Bruteforce categories (see TODO task #4)
2. Add real-time WebSocket support for streaming predictions
3. Implement authentication/authorization
4. Add rate limiting
5. Deploy to production environment
