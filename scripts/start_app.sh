#!/bin/bash
cd "$(dirname "$0")/.."
python -m backend.app &
sleep 2
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:5000
elif command -v open &>/dev/null; then
    open http://localhost:5000
fi
wait
