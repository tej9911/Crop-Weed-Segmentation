#!/usr/bin/env bash
# Start the development server and open the app in the default browser.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-5000}"
URL="http://127.0.0.1:${PORT}"

python -m backend.app &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM

# Wait up to 60 seconds for the server to answer before opening the browser.
for _ in $(seq 1 60); do
    if python -c "import urllib.request; urllib.request.urlopen('${URL}/api/health', timeout=1)" 2>/dev/null; then
        break
    fi
    sleep 1
done

if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
    open "$URL" || true
else
    echo "Open $URL in your browser."
fi

wait "$SERVER_PID"
