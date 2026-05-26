#!/usr/bin/env bash
set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   Causal RCA Agent — Setup           ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 1. Python deps ────────────────────────────────────────────────────────────
echo "▶ Installing Python dependencies..."
pip install -r requirements.txt --quiet --break-system-packages

# ── 2. Env file ───────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  cp .env.example .env
  echo "▶ Created .env from .env.example"
else
  echo "▶ .env already exists — skipping"
fi

# ── 3. Ollama install check ───────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo "▶ Installing Ollama..."
  curl -fsSL https://ollama.ai/install.sh | sh
else
  echo "▶ Ollama already installed"
fi

# ── 4. Start Ollama in background ─────────────────────────────────────────────
echo "▶ Starting Ollama server..."
ollama serve &>/dev/null &
sleep 3

# ── 5. Pull models ────────────────────────────────────────────────────────────
echo "▶ Pulling tinyllama (small model — 637MB)..."
ollama pull tinyllama

echo "▶ Pulling phi3 (large model — 2.3GB)..."
ollama pull phi3

echo ""
echo "✅ Setup complete."
echo "   Run: ./scripts/run.sh"
echo ""
