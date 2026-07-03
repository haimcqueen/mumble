<h1 align="center">mumble</h1>

<p align="center">
  <b>Private, local push-to-talk dictation for macOS.</b><br>
  Hold a key, speak, release. Your words appear in whatever app you're using.<br>
  No cloud, no subscription, no account, fully secure. Everything runs on your Mac. You own everything.
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#privacy">Privacy</a> ·
  <a href="#limitations">Limitations</a>
</p>

---

mumble is a free, open-source alternative to Wispr Flow, Willow Voice, and superwhisper. It uses NVIDIA's **Parakeet** speech model running on Apple Silicon's Neural Engine for near-instant transcription, and an optional local LLM to clean up your rambling into polished text, all without a single byte leaving your machine.

Built for Apple Silicon (M1 or newer). 
**$0 forever.**

## Features

- 🎙️ **Hold-to-talk:** hold `Fn`, speak, release. Text lands at your cursor in ~half a second.
- 🧹 **Automatic cleanup:** a local LLM strips filler words ("um", "uh"), fixes punctuation, and turns spoken lists ("first… second… third…") into clean numbered lists.
- 🙌 **Hands-free mode:** double-tap `Fn` and just talk; it stops automatically when you go quiet.
- 🗣️ **Wake-word mode:** say *"start dictation"*, talk, say *"stop dictation"*. No keys at all, dictate from across the room.
- 〰️ **Live waveform:** an elegant floating pill shows it's listening, with audible start/stop cues so you know it's working without looking.
- 🎧 **AirPods aware:** auto-switches to your AirPods mic when you put them on, even mid-session.
- 🔒 **100% local & private:** Parakeet + your LLM run on-device. No network calls, no telemetry, no accounts.
- ⚡ **Fast:** ~0.4s to transcribe 10 seconds of speech on an M1 Pro; the model stays warm in memory.

## Install

**Requirements:** Apple Silicon Mac (M1+), macOS 14+, [Python 3.10+](https://www.python.org/), and [Ollama](https://ollama.com) (optional, for cleanup).

```bash
git clone https://github.com/haimcqueen/mumble.git
cd mumble
./install.sh
```

The installer creates a virtual environment, installs dependencies, and pre-downloads the speech model (~600 MB). For the optional text cleanup, install Ollama and pull a model:

```bash
brew install ollama
ollama pull qwen2.5:7b
```

Then run it:

```bash
./dictate.sh
```

On first launch mumble will prompt you for three macOS permissions and walk you through granting them (see [Permissions](#permissions)).

> **Using a coding agent?** Just point it at this repo and ask it to set mumble up. Between `install.sh` and [ARCHITECTURE.md](ARCHITECTURE.md), it has everything it needs to install, configure, and run it for you.

## Usage

```bash
./dictate.sh                       # hold Fn to dictate (with cleanup)
./dictate.sh --raw                 # skip LLM cleanup (fastest, raw transcript)
./dictate.sh --wake "start dictation"   # hands-free wake-word mode
./dictate.sh --key rcmd            # use Right-Command instead of Fn
```

**Everyday dictation:** Click into any text field, **hold `Fn`**, speak, **release**. Done.

**Hands-free:** **Double-tap `Fn`** to start, then talk. It stops after 3 seconds of silence (or tap `Fn` again). Great for longer thoughts.

**Wake-word (no keyboard):** Run with `--wake "start dictation"`. Say your wake phrase, dictate, then say *"stop dictation"*. Perfect for pacing around while you think.

<details>
<summary><b>All options</b></summary>

| Flag | Description |
|---|---|
| `--raw` | Skip LLM cleanup; insert the raw transcript (fastest) |
| `--key {fn,rcmd}` | Push-to-talk key (default: `fn`) |
| `--clean-model MODEL` | Ollama model for cleanup (default: `qwen2.5:7b`) |
| `--silence SEC` | Hands-free: stop after this many seconds of silence (default: 3) |
| `--wake PHRASE` | Always-listen mode with this wake phrase |
| `--stop-phrase PHRASE` | Phrase that ends wake-mode dictation (default: "stop dictation") |
| `--wake-timeout SEC` | Auto-stop wake dictation after silence (default: 45) |
| `--no-hud` | Disable the floating waveform overlay |
| `--no-sound` | Disable the audible start/stop cues |
| `--test AUDIO.wav` | Transcribe an audio file and exit (no mic needed) |

</details>

## Start automatically at login

By default you re-launch mumble after a restart. To have it start on its own,
install a login item — it preserves whatever mode you pass:

```bash
./dictate.sh --enable-autostart                          # push-to-talk
./dictate.sh --enable-autostart --wake "start dictation" # hands-free
./dictate.sh --disable-autostart                         # turn it off
```

> **One-time catch:** the login item runs as a separate process from your
> terminal, so on the *first* auto-launch macOS asks you to grant Microphone
> and Accessibility once more (to a "Python" entry) — grant them and it sticks
> for every login after. A future signed app will make this seamless.

## How it works

```
  Fn key ──▶ CGEvent tap ──▶ controller ──▶ AVAudioEngine (warm mic + pre-roll)
                                                    │
                                              16 kHz audio
                                                    ▼
                                         Parakeet TDT (Core ML / ANE)
                                                    │
                                                raw text
                                                    ▼
                                    Ollama LLM cleanup (optional, local)
                                                    │
                                              polished text
                                                    ▼
                                    Pasteboard + synthetic ⌘V into focused app
```

- **Speech-to-text:** [Parakeet TDT 0.6B](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2) via [parakeet-mlx](https://github.com/senstella/parakeet-mlx): runs on the Neural Engine, excellent English accuracy, punctuation included.
- **Cleanup:** any local [Ollama](https://ollama.com) model (default `qwen2.5:7b`) rewrites the transcript with a fixed prompt. Fully optional and offline.
- **Insertion:** copies text to the pasteboard, sends a synthetic ⌘V, then restores your previous clipboard. Works in virtually every app.

## Permissions

macOS requires three permissions, granted to the app that runs mumble (your Terminal, or the bundled app). mumble fires all the prompts on first launch and shows live status while you grant them:

| Permission | Why |
|---|---|
| **Microphone** | To hear you |
| **Input Monitoring** | To detect the `Fn` key globally |
| **Accessibility** | To insert text into the focused app |

Wake-word mode (`--wake`) needs only Microphone + Accessibility.

> **Tip:** Set System Settings → Keyboard → "Press 🌐 key to" → **Do Nothing**, and turn off built-in **Dictation**, so the `Fn` key doesn't trigger macOS's own features.

## Privacy

mumble makes **zero network connections** during use. The speech model is downloaded once from Hugging Face at install time; after that, your audio and text never leave the machine. There's no telemetry, no analytics, and no account. Read the ~1,000 lines of `dictate.py` yourself; that's the whole thing.

## Limitations

- **Apple Silicon only.** Parakeet-mlx needs the Neural Engine. Intel Macs aren't supported.
- **The AirPods stem can't trigger dictation.** macOS reserves the AirPods Pro stem squeeze for its own mic/call control and never delivers it to third-party apps. Use AirPods for the *mic*; trigger with the keyboard or a wake word. (The AirPods *stem-as-button* feature is a macOS limitation, not a bug here; see [details](#).)
- **Not on the Mac App Store.** Global key detection is incompatible with the App Store sandbox, so mumble is distributed directly. This is normal for this class of app.
- **Cleanup needs Ollama running.** Without it, mumble inserts the raw transcript and tells you so.

## Roadmap

- [ ] Native Swift menu-bar app (smaller, faster, signable)
- [ ] Custom vocabulary & text replacements
- [ ] Dedicated wake-word engine (lower CPU than transcribe-everything)
- [ ] Windows support
- [ ] Notarized, signed builds

## Credits

Built on [parakeet-mlx](https://github.com/senstella/parakeet-mlx), [MLX](https://github.com/ml-explore/mlx), [Ollama](https://ollama.com), and [PyObjC](https://pyobjc.readthedocs.io/). Inspired by [Wispr Flow](https://wisprflow.ai) and [FreeFlow](https://github.com/zachlatta/freeflow).

## License

[MIT](LICENSE) © 2026 Hai Bui
