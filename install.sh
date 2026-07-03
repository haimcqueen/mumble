#!/bin/zsh
# mumble installer — sets up a local Python environment and pre-downloads
# the speech model so the first real launch is instant.
set -e
cd "$(dirname "$0")"

echo "── mumble installer ─────────────────────────────────"

# 1. Apple Silicon check (Parakeet runs on the Neural Engine via MLX)
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: mumble needs an Apple Silicon Mac (M1 or newer)." >&2
  exit 1
fi

# 2. Python 3.10+
PY=$(command -v python3 || true)
if [[ -z "$PY" ]]; then
  echo "ERROR: python3 not found. Install it with:  brew install python" >&2
  exit 1
fi
PYVER=$($PY -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  echo "ERROR: Python $PYVER found, but 3.10+ is required." >&2
  exit 1
fi
echo "✓ Python $PYVER"

# 3. Virtual environment + dependencies
if [[ ! -d .venv ]]; then
  echo "creating virtual environment ..."
  $PY -m venv .venv
fi
source .venv/bin/activate
echo "installing dependencies (a few minutes on first run) ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✓ dependencies installed"

# 4. Pre-download the speech model (~600 MB, one time)
echo "downloading the speech model (~600 MB, one time) ..."
python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("mlx-community/parakeet-tdt-0.6b-v2")
print(f"✓ model ready: {p}")
EOF

# 5. Optional cleanup model via Ollama
if command -v ollama >/dev/null 2>&1; then
  if curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✓ Ollama is running — transcript cleanup will be enabled"
  else
    echo "· Ollama installed but not running — start it (or open the Ollama"
    echo "  app) to enable transcript cleanup. mumble works without it."
  fi
else
  echo "· Ollama not found — mumble will insert raw transcripts."
  echo "  For automatic cleanup:  brew install ollama && ollama pull qwen2.5:7b"
fi

echo ""
echo "── done ─────────────────────────────────────────────"
echo "Start dictating:   ./dictate.sh"
echo "Hands-free mode:   ./dictate.sh --wake \"start dictation\""
