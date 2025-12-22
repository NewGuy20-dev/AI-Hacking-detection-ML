# Start API server and optionally Cloudflare Tunnel
# Run with: .\start-api.ps1

Write-Host "🚀 Starting AI Hacking Detection API..." -ForegroundColor Cyan

# Check if uvicorn is available
$uvicorn = Get-Command uvicorn -ErrorAction SilentlyContinue
if (-not $uvicorn) {
    Write-Host "❌ uvicorn not found. Install with: pip install uvicorn" -ForegroundColor Red
    exit 1
}

# Start FastAPI server in background
Write-Host "📡 Starting FastAPI on http://localhost:8000" -ForegroundColor Green
$apiJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
}

# Wait for API to start
Write-Host "⏳ Waiting for API to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check API health
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5
    Write-Host "✅ API is running! Status: $($response.status)" -ForegroundColor Green
} catch {
    Write-Host "⚠️ API may still be loading models..." -ForegroundColor Yellow
}

# Check for cloudflared
$cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
if ($cloudflared) {
    Write-Host ""
    Write-Host "🌐 Cloudflare Tunnel available. Options:" -ForegroundColor Cyan
    Write-Host "  1. Quick tunnel: cloudflared tunnel --url http://localhost:8000"
    Write-Host "  2. Named tunnel: cloudflared tunnel run <tunnel-name>"
    Write-Host ""
    $response = Read-Host "Start quick tunnel? (y/n)"
    if ($response -eq 'y') {
        cloudflared tunnel --url http://localhost:8000
    }
} else {
    Write-Host ""
    Write-Host "💡 Tip: Install cloudflared to expose API via Cloudflare Tunnel" -ForegroundColor Yellow
    Write-Host "   winget install cloudflare.cloudflared" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Press Ctrl+C to stop the API server" -ForegroundColor Gray

# Wait for job
try {
    Receive-Job -Job $apiJob -Wait
} finally {
    Stop-Job -Job $apiJob
    Remove-Job -Job $apiJob
}
