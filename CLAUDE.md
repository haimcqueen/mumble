# mumble — setup guide for coding agents

You're helping a user install and run **mumble**, a fully local push-to-talk
dictation app for macOS. Hold `Fn`, speak, release — transcribed text appears
in the focused app. Everything runs on-device (Parakeet via MLX for speech-to-
text, optional Ollama LLM for cleanup). This file tells you how to get it
running. Read it fully before acting.

## Hard requirements (check first, stop if unmet)

- **Apple Silicon Mac (M1 or newer).** Run `uname -m` → must be `arm64`.
  Intel Macs are NOT supported (Parakeet needs the Neural Engine).
- **macOS 14+**.
- Internet for the one-time ~600 MB model download.

## Setup

1. **Install `uv` if missing** — it provides a complete standalone Python.
   Prefer this over the system `python3`, which is often built without the
   `lzma` module that a dependency (librosa) needs.
   ```bash
   command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Run the installer** (idempotent — safe to re-run):
   ```bash
   ./install.sh
   ```
   It creates `.venv`, installs deps, and pre-downloads the model. This takes
   a few minutes on first run.
3. **(Optional) cleanup LLM.** For automatic filler-word removal and numbered
   lists, the user needs [Ollama](https://ollama.com) running with a model:
   ```bash
   brew install ollama && ollama pull qwen2.5:7b
   ```
   Without it, mumble inserts raw transcripts — that's fine, not an error.

## Permissions — you CANNOT do this for the user

mumble needs three macOS permissions, granted to **the terminal app that runs
it** (Terminal, iTerm, etc.). You cannot click these toggles — guide the user.
On first launch mumble fires the prompts and prints a live `✅/❌` status line.

| Permission | Why | Settings pane |
|---|---|---|
| Microphone | hear the user | Privacy & Security → Microphone |
| Input Monitoring | detect the `Fn` key | Privacy & Security → Input Monitoring |
| Accessibility | insert text into the focused app | Privacy & Security → Accessibility |

After granting Input Monitoring **or** Accessibility, mumble restarts itself
automatically to pick them up (a running process can't see those grants).

Also tell the user to set **System Settings → Keyboard → "Press 🌐 key to" →
Do Nothing** and turn **Dictation off**, so `Fn` doesn't trigger Apple's own
features (double-tapping Fn otherwise launches macOS Dictation).

## Run

```bash
./dictate.sh                          # hold Fn to dictate (with cleanup)
./dictate.sh --raw                    # fastest, skip LLM cleanup
./dictate.sh --wake "start dictation" # hands-free, no keys — say the phrase
```

Success looks like this line printed to the terminal:
```
ready [clean (qwen2.5:7b)] — hold Fn and speak, release to insert.
```
plus a floating "hold Fn to dictate" pill at the top-center of the screen.

## Verify without a live mic

You can confirm the model/pipeline works without the user speaking:
```bash
say -o /tmp/t.aiff "this is a test of local dictation"
afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/t.aiff /tmp/t.wav
source .venv/bin/activate && python dictate.py --test /tmp/t.wav
```
Expect a `[stt ...]` line with the transcript. Add `--wake`/permission testing
only with the user present, since it needs the mic and their toggles.

## Known gotchas (all already handled in code — don't "fix" them)

- **`ModuleNotFoundError: _lzma`** → their `python3` lacks lzma. Use `uv`.
- **numba tries to build 0.53.1 and fails on 3.12** → `requirements.txt` pins
  `numba>=0.60`; make sure you installed from it.
- **Onboarding seems stuck on ❌ accessibility after granting** → fixed; it
  reads the grant in a subprocess and restarts. If on an old checkout, `git
  pull`.
- **AirPods stem can't start/stop dictation.** macOS reserves it for its own
  mic control — this is a documented OS limitation, not a bug. Use the keyboard
  or `--wake` mode. AirPods work fine as the *microphone*.
- **`Ctrl-C` to quit.** Only one instance runs at a time (lock file).

## Repo map

- `dictate.py` — the whole app (~1,300 lines): hotkey tap, audio capture,
  Parakeet STT, LLM cleanup, HUD, text insertion, wake-word mode.
- `dictate.sh` / `install.sh` — launcher / installer.
- `build_dmg.sh` — builds the downloadable `.dmg`.
- `ARCHITECTURE.md` — design doc + the planned native-Swift rewrite.
- `README.md` — user-facing docs.
