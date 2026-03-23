# System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js)                       │
│                      http://localhost:3000                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Scanner    │  │    Batch     │  │   Models     │          │
│  │    /scanner  │  │    /batch    │  │   /models    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                   │
│         └──────────────────┴──────────────────┘                  │
│                            │                                      │
│                  ┌─────────▼─────────┐                          │
│                  │   React Query     │                          │
│                  │   Hooks Layer     │                          │
│                  └─────────┬─────────┘                          │
│                            │                                      │
│         ┌──────────────────┼──────────────────┐                 │
│         │                  │                  │                  │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐         │
│  │usePayloadScan│  │ useBatchScan │  │useTimesSeries│         │
│  │  useURLScan  │  │              │  │     Scan     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                  │
│                            │                                      │
│                  ┌─────────▼─────────┐                          │
│                  │   API Client      │                          │
│                  │  (src/lib/api.ts) │                          │
│                  └─────────┬─────────┘                          │
│                            │                                      │
└────────────────────────────┼──────────────────────────────────────┘
                             │
                             │ HTTP/JSON
                             │ CORS Enabled
                             │
┌────────────────────────────▼──────────────────────────────────────┐
│                      BACKEND (FastAPI)                            │
│                    http://localhost:8000                          │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    API Routes                              │  │
│  │                                                            │  │
│  │  GET  /health              → Health Check                 │  │
│  │  GET  /health/ready        → Readiness Check              │  │
│  │  POST /api/v1/predict/payload    → Payload Analysis       │  │
│  │  POST /api/v1/predict/url        → URL Analysis           │  │
│  │  POST /api/v1/predict/batch      → Batch Analysis         │  │
│  │  POST /api/v1/predict/timeseries → TimeSeries Analysis    │  │
│  │                                                            │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │                                       │
│                  ┌────────▼────────┐                             │
│                  │  BatchHybrid    │                             │
│                  │   Predictor     │                             │
│                  └────────┬────────┘                             │
│                           │                                       │
│         ┌─────────────────┼─────────────────┐                   │
│         │                 │                 │                    │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐            │
│  │  PyTorch    │  │   Sklearn   │  │  Benign     │            │
│  │   Models    │  │   Models    │  │ Pre-filter  │            │
│  │             │  │             │  │             │            │
│  │ • Payload   │  │ • Payload   │  │ • Regex     │            │
│  │   CNN       │  │   RF        │  │ • Patterns  │            │
│  │ • URL CNN   │  │ • URL RF    │  │             │            │
│  │ • TimeSeries│  │             │  │             │            │
│  │   LSTM      │  │             │  │             │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                             │
                             │
                    ┌────────▼────────┐
                    │  Model Files    │
                    │  (models/*.pkl) │
                    │  (models/*.pt)  │
                    └─────────────────┘
```

## Data Flow

### 1. Single Scan (Payload/URL)
```
User Input → Scanner Page → usePayloadScan/useURLScan Hook
    → api.predictPayload/predictURL → POST /api/v1/predict/{type}
    → BatchHybridPredictor → Model Inference → Response
    → React Query Cache → UI Update → History Store
```

### 2. Batch Scan
```
File Upload → Batch Page → Parse Lines → useBatchScan Hook
    → api.predictBatch → POST /api/v1/predict/batch
    → Benign Pre-filter (fast path) → BatchHybridPredictor
    → Bulk Model Inference → Batch Response
    → Results Table → Export CSV
```

### 3. Model Status Check
```
Models Page → useQuery → api.ready → GET /health/ready
    → Check Loaded Models → Response
    → Display Model Cards → Auto-refresh (30s)
```

## Key Features

### Frontend
- ✅ Type-safe API client with TypeScript
- ✅ React Query for caching and state management
- ✅ Automatic retry on failure
- ✅ Loading states and error handling
- ✅ Real-time history tracking
- ✅ Statistics aggregation
- ✅ CSV export for batch results

### Backend
- ✅ FastAPI with automatic OpenAPI docs
- ✅ Pydantic validation for all inputs
- ✅ CORS enabled for frontend
- ✅ Model lifecycle management
- ✅ Benign pre-filtering for performance
- ✅ Shadow logging for monitoring
- ✅ Threshold-based classification
- ✅ Multi-model ensemble predictions

## Performance Optimizations

1. **Benign Pre-filter**: Fast regex-based filtering before ML inference
2. **Batch Processing**: Vectorized predictions for multiple inputs
3. **Model Caching**: Models loaded once at startup
4. **React Query**: Automatic caching and deduplication
5. **Lazy Loading**: Frontend components load on demand

## Security Features

1. **Input Validation**: Max length limits on all inputs
2. **CORS**: Restricted to known origins
3. **Error Sanitization**: No sensitive data in error messages
4. **Rate Limiting**: Ready for implementation (TODO)
5. **Authentication**: Ready for implementation (TODO)
