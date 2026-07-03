#!/bin/zsh
# mumble installer: sets up a local Python environment and pre-downloads
# the speech model so the first real launch is instant.
set -e
cd "$(dirname "$0")"

echo "── mumble installer ─────────────────────────────────"

# 1. Apple Silicon check (Parakeet runs on the Neural Engine via MLX)
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: mumble needs an Apple Silicon Mac (M1 or newer)." >&2
  exit 1
fi

# 2. Virtual environment + dependencies.
# Prefer uv: its standalone Python is complete (includes the lzma module that
# librosa/Parakeet needs, which many pyenv/Homebrew builds omit). Fall back to
# python3 -m venv, but verify lzma so we fail loudly instead of at first use.
if command -v uv >/dev/null 2>&1; then
  echo "using uv to create the environment ..."
  uv venv --python 3.12 .venv
  source .venv/bin/activate
  echo "installing dependencies (a few minutes on first run) ..."
  uv pip install -r requirements.txt
else
  PY=$(command -v python3 || true)
  if [[ -z "$PY" ]]; then
    echo "ERROR: neither uv nor python3 found." >&2
    echo "Install uv (recommended):  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
  if ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
    echo "ERROR: Python 3.10+ required." >&2
    exit 1
  fi
  [[ -d .venv ]] || $PY -m venv .venv
  source .venv/bin/activate
  if ! python -c "import lzma" 2>/dev/null; then
    echo "ERROR: your python3 was built without the lzma module, which the" >&2
    echo "speech model needs. Easiest fix: install uv and re-run:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "(or rebuild Python with xz:  brew install xz  then reinstall Python)" >&2
    exit 1
  fi
  echo "installing dependencies (a few minutes on first run) ..."
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt
fi
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
    echo "✓ Ollama is running, transcript cleanup will be enabled"
  else
    echo "· Ollama installed but not running. Start it (or open the Ollama"
    echo "  app) to enable transcript cleanup. mumble works without it."
  fi
else
  echo "· Ollama not found. mumble will insert raw transcripts."
  echo "  For automatic cleanup:  brew install ollama && ollama pull qwen2.5:7b"
fi

echo ""
echo "── done ─────────────────────────────────────────────"
echo "Start dictating:   ./dictate.sh"
echo "Hands-free mode:   ./dictate.sh --wake \"start dictation\""
