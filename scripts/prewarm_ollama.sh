#!/bin/bash
# ==============================================================================
# Pre-warm Ollama models into a shared cache (run ONCE on a login/shared node)
# - Downloads all required models to $SHARED_OLLAMA_MODELS
# - Uses a local ephemeral server for consistent pulls
# - Verifies results and records versions for provenance
# ==============================================================================

set -Eeuo pipefail

# --- Settings you control -----------------------------------------------------
SHARED_OLLAMA_MODELS="${HOME}/.ollama_shared_models"   # shared/home/project path
ALL_MODELS_TO_DOWNLOAD=("nomic-embed-text" "mxbai-embed-large" "llama3.1:8b")  # adjust
# If you truly have "all-minilm", confirm the exact tag on your mirror and add it

mkdir -p "${SHARED_OLLAMA_MODELS}"
echo "✅ Shared model cache dir: ${SHARED_OLLAMA_MODELS}"

# --- Record versions for provenance (client-only, sanitized) ------------------
# We do not require a running server; extract just the client version number.
CLIENT_VER="$(ollama --version 2>&1 | sed -n 's/.*client version is \([0-9.]\+\).*/\1/p')"

{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  # Store only a clean key + value so downstream parsing is robust.
  echo "ollama_client_version: ${CLIENT_VER:-unknown}"
} > "${SHARED_OLLAMA_MODELS}/PREWARM_METADATA.txt"

# --- Start ephemeral local server bound to the shared model dir ---------------
export OLLAMA_MODELS="${SHARED_OLLAMA_MODELS}"
PORT=12345
export OLLAMA_HOST="127.0.0.1:${PORT}"

echo "🔥 Starting temporary Ollama server to pre-warm cache..."
( ollama serve > /tmp/ollama_prewarm.log 2>&1 ) & O_PID=$!
trap 'echo "Cleaning up server..."; kill $O_PID 2>/dev/null || true' EXIT

# Wait for readiness
for i in {1..90}; do
  if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null; then
    echo "✅ Server is up."
    break
  fi
  sleep 1
done

# --- Pull models sequentially (kinder to mirrors & bandwidth) -----------------
echo "🔥 Pulling required models..."
for model in "${ALL_MODELS_TO_DOWNLOAD[@]}"; do
  echo "--- pulling ${model} ---"
  if ! OLLAMA_HOST="${OLLAMA_HOST}" ollama pull "${model}"; then
    echo "❌ Failed to pull ${model}. See /tmp/ollama_prewarm.log"
    exit 1
  fi
done

echo "🧾 Listing models present in shared cache:"
OLLAMA_HOST="${OLLAMA_HOST}" ollama list || true

echo "🎉 Pre-warm complete at: ${SHARED_OLLAMA_MODELS}"


