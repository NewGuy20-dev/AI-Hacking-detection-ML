# Integration Testing Checklist

## Pre-flight Checks

- [ ] Backend dependencies installed: `py -3 -m pip install -r requirements.txt`
- [ ] Frontend dependencies installed: `cd dashboard && npm install`
- [ ] Models exist in `models/` directory
- [ ] `.env.local` configured in dashboard: `NEXT_PUBLIC_API_URL=http://localhost:8000`

## Backend Tests

### 1. Start Backend Server
```bash
py -3 -m uvicorn src.api.server:app --reload
```

**Expected Output:**
```
Loading models...
Loaded X PyTorch, Y sklearn models
INFO:     Uvicorn running on http://127.0.0.1:8000
```

- [ ] Server starts without errors
- [ ] Models load successfully
- [ ] No import errors

### 2. Test Health Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/health/ready
```

- [ ] `/health` returns `{"status": "healthy", "uptime_seconds": ...}`
- [ ] `/health/ready` returns model list
- [ ] Both return 200 status code

### 3. Test Prediction Endpoints

**Payload Prediction:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/payload \
  -H "Content-Type: application/json" \
  -d '{"payload": "'"'"' OR '"'"'1'"'"'='"'"'1"}'
```

- [ ] Returns prediction with `is_attack`, `confidence`, `severity`
- [ ] Confidence is between 0 and 1
- [ ] Processing time is reasonable (<100ms)

**URL Prediction:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/url \
  -H "Content-Type: application/json" \
  -d '{"url": "http://malicious-site.com/phishing"}'
```

- [ ] Returns prediction result
- [ ] Attack type is "MALICIOUS_URL" if detected

**Batch Prediction:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"payloads": ["'"'"' OR 1=1", "Hello world"], "urls": ["http://example.com"]}'
```

- [ ] Returns array of results
- [ ] Result count matches input count (3 in this case)
- [ ] Total processing time included

**TimeSeries Prediction:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/timeseries \
  -H "Content-Type: application/json" \
  -d '{"events": [{"timestamp": 1, "action": "login"}], "window_size": 10}'
```

- [ ] Returns prediction (may fail if model not loaded)
- [ ] Error message is clear if model unavailable

### 4. Run Integration Tests
```bash
py -3 -m pytest tests/test_api_integration.py -v
```

- [ ] All tests pass
- [ ] No import errors
- [ ] Test coverage is adequate

### 5. Check API Documentation
Open http://localhost:8000/docs

- [ ] Swagger UI loads
- [ ] All 6 endpoints visible
- [ ] Can test endpoints interactively
- [ ] Request/response schemas shown

## Frontend Tests

### 1. Start Frontend Server
```bash
cd dashboard
npm run dev
```

**Expected Output:**
```
ready - started server on 0.0.0.0:3000
```

- [ ] Server starts without errors
- [ ] No TypeScript errors
- [ ] No build warnings

### 2. Test Scanner Page
Navigate to http://localhost:3000/scanner

**Payload Tab:**
- [ ] Page loads without errors
- [ ] Can type in textarea
- [ ] Example payloads are clickable
- [ ] Submit button works
- [ ] Loading spinner shows during scan
- [ ] Results display correctly
- [ ] Confidence percentage shows
- [ ] Severity badge displays
- [ ] Attack type shows if detected

**URL Tab:**
- [ ] Can switch to URL tab
- [ ] URL input field works
- [ ] Submit button works
- [ ] Results display correctly

### 3. Test Batch Page
Navigate to http://localhost:3000/batch

- [ ] Page loads without errors
- [ ] Can drag and drop .txt file
- [ ] Can click to browse files
- [ ] File upload shows line count
- [ ] Scan button works
- [ ] Progress/loading state shows
- [ ] Results table displays
- [ ] Summary stats show (Total, Threats, Safe)
- [ ] Can export CSV
- [ ] CSV contains correct data
- [ ] Can clear results

### 4. Test Models Page
Navigate to http://localhost:3000/models

- [ ] Page loads without errors
- [ ] API status shows "Online" (green)
- [ ] Uptime displays
- [ ] Loaded models list shows
- [ ] Model performance cards display
- [ ] Accuracy percentages show
- [ ] Known limitations section shows
- [ ] Refresh button works

### 5. Test History Page
Navigate to http://localhost:3000/history

- [ ] Page loads without errors
- [ ] Recent scans show (if any performed)
- [ ] Statistics display correctly
- [ ] Can filter by type (payload/url)

### 6. Browser Console Check
Open DevTools → Console

- [ ] No JavaScript errors
- [ ] No failed network requests
- [ ] API calls return 200 status
- [ ] No CORS errors

### 7. Network Tab Check
Open DevTools → Network

**Perform a scan and verify:**
- [ ] Request goes to correct endpoint
- [ ] Request payload is correct JSON
- [ ] Response is valid JSON
- [ ] Response time is reasonable
- [ ] Status code is 200

## Integration Tests

### 1. End-to-End Payload Scan
1. Open http://localhost:3000/scanner
2. Enter: `' OR '1'='1`
3. Click "Execute Scan"

**Expected:**
- [ ] Loading state shows
- [ ] Result appears within 1-2 seconds
- [ ] Shows as attack (red/danger)
- [ ] Confidence > 0.8
- [ ] Attack type: "SQL_INJECTION"
- [ ] Severity: "CRITICAL" or "HIGH"

### 2. End-to-End URL Scan
1. Switch to URL tab
2. Enter: `http://example.com`
3. Click "Execute Scan"

**Expected:**
- [ ] Result appears
- [ ] Shows as safe (green) or low confidence
- [ ] Processing time < 100ms

### 3. End-to-End Batch Scan
1. Navigate to http://localhost:3000/batch
2. Create test file `test.txt`:
   ```
   ' OR '1'='1
   Hello world
   <script>alert('xss')</script>
   Normal text here
   ```
3. Upload file
4. Click "Initiate Scan"

**Expected:**
- [ ] All 4 lines processed
- [ ] Results show mix of attacks and safe
- [ ] Summary stats are correct
- [ ] Can export CSV
- [ ] CSV has 5 rows (header + 4 results)

### 4. Error Handling Test
1. Stop backend server
2. Try to scan something

**Expected:**
- [ ] Error message displays
- [ ] No crash or blank screen
- [ ] User-friendly error text
- [ ] Can retry after restarting backend

### 5. Models Page Live Update
1. Navigate to http://localhost:3000/models
2. Note the uptime
3. Wait 30 seconds

**Expected:**
- [ ] Page auto-refreshes
- [ ] Uptime increases
- [ ] No manual refresh needed

## Performance Tests

### 1. Single Scan Latency
- [ ] Payload scan: < 50ms
- [ ] URL scan: < 50ms
- [ ] Results appear instantly

### 2. Batch Scan Performance
Upload 100 lines:
- [ ] Completes in < 5 seconds
- [ ] No browser freeze
- [ ] Results render smoothly

### 3. Memory Usage
- [ ] Backend memory stable (no leaks)
- [ ] Frontend memory stable
- [ ] Can perform 10+ scans without issues

## Cross-Browser Tests (Optional)

Test in multiple browsers:
- [ ] Chrome/Edge
- [ ] Firefox
- [ ] Safari (if available)

## Mobile Responsive (Optional)

Test on mobile viewport:
- [ ] Layout adapts
- [ ] Buttons are clickable
- [ ] Text is readable
- [ ] No horizontal scroll

## Final Checklist

- [ ] All backend endpoints working
- [ ] All frontend pages working
- [ ] No console errors
- [ ] No network errors
- [ ] Integration tests pass
- [ ] Documentation is accurate
- [ ] Ready for deployment

## Issues Found

Document any issues here:

1. 
2. 
3. 

## Notes

Add any observations or recommendations:

1. 
2. 
3. 
