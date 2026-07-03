#!/bin/zsh
# Launch local push-to-talk dictation. Pass-through args, e.g.:
#   ./dictate.sh                # hold Fn to dictate (raw transcript, fastest)
#   ./dictate.sh --clean        # + LLM cleanup (removes filler words)
#   ./dictate.sh --key rcmd     # use Right-Command as the push-to-talk key
cd "$(dirname "$0")"
source .venv/bin/activate
exec python dictate.py "$@"
