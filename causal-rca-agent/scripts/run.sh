#!/usr/bin/env bash
set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   Causal RCA Agent — Starting        ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Start Ollama if not already running
if ! pgrep -x "ollama" > /dev/null; then
  echo "▶ Starting Ollama server..."
  ollama serve &>/dev/null &
  sleep 3
else
  echo "▶ Ollama already running"
fi

echo "▶ Starting FastAPI server on port 8000..."
echo "▶ Open port 8000 in Codespaces when prompted"
echo ""

uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
