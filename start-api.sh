#!/bin/bash
# Start API server and optionally Cloudflare Tunnel

set -e

echo "🚀 Starting AI Hacking Detection API..."

# Check if uvicorn is available
if ! command -v uvicorn &> /dev/null; then
    echo "❌ uvicorn not found. Install with: pip install uvicorn"
    exit 1
fi

# Start FastAPI server
echo "📡 Starting FastAPI on http://localhost:8000"
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API to be ready
echo "⏳ Waiting for API to start..."
sleep 5

# Check if cloudflared is available
if command -v cloudflared &> /dev/null; then
    echo ""
    echo "🌐 Cloudflare Tunnel available. Options:"
    echo "  1. Quick tunnel (temporary URL): cloudflared tunnel --url http://localhost:8000"
    echo "  2. Named tunnel: cloudflared tunnel run <tunnel-name>"
    echo ""
    read -p "Start quick tunnel? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cloudflared tunnel --url http://localhost:8000
    fi
else
    echo ""
    echo "💡 Tip: Install cloudflared to expose API via Cloudflare Tunnel"
    echo "   https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/"
fi

# Keep running
wait $API_PID
