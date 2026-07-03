# Local Push-to-Talk Dictation for macOS

A fully local Whisper Flow / Willow Voice–style dictation app: **hold Fn → speak → release → text appears in the focused app**. No network calls, no API costs, everything on-device.

Target hardware: Apple Silicon (developed on M1 Pro, macOS 15.6). Windows port considered but deferred.

---

## 1. Overall architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Menu-bar app (LSUIElement)                    │
│                                                                  │
│  ┌──────────────┐   key down/up    ┌───────────────────────┐    │
│  │ HotkeyMonitor │ ───────────────▶ │  DictationController  │    │
│  │ (CGEventTap)  │                  │   (state machine)     │    │
│  └──────────────┘                  └───┬───────────┬───────┘    │
│                                        │           │            │
│                          start/stop    │           │ show/hide  │
│                                        ▼           ▼            │
│  ┌──────────────┐  16kHz mono   ┌───────────┐ ┌──────────┐     │
│  │ AudioCapture  │ ────────────▶ │ STTEngine │ │ HUD panel │     │
│  │ (AVAudioEngine│  ring buffer  │ (whisper. │ │(NSPanel,  │     │
│  │  pre-warmed)  │               │  cpp /    │ │ level +   │     │
│  └──────────────┘               │ Parakeet) │ │ partials) │     │
│                                  └─────┬─────┘ └──────────┘     │
│                                        │ raw text               │
│                                        ▼                        │
│                                  ┌───────────┐                  │
│                                  │PostProcess│ vocab, replace-  │
│                                  │           │ ments, commands  │
│                                  └─────┬─────┘                  │
│                                        │ final text             │
│                                        ▼                        │
│                                  ┌───────────┐                  │
│                                  │TextInjector│ paste / CGEvent │
│                                  │           │  / AX API        │
│                                  └───────────┘                  │
│                                                                  │
│  Settings (UserDefaults/JSON) · History (SQLite) · Onboarding    │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Responsibility | Key API |
|---|---|---|
| `HotkeyMonitor` | Detect Fn press/hold/release globally | `CGEventTap` (flagsChanged) |
| `AudioCapture` | Mic → 16 kHz mono Float32 ring buffer; engine pre-warmed | `AVAudioEngine` |
| `STTEngine` | Local speech-to-text, streaming partials + final pass | whisper.cpp (Metal) or Parakeet (Core ML) behind a protocol |
| `PostProcessor` | Punctuation fixes, custom vocabulary, snippets, command mode | pure Swift |
| `TextInjector` | Insert text into the frontmost app | `NSPasteboard` + synthetic ⌘V, `CGEventKeyboardSetUnicodeString`, `AXUIElement` |
| `DictationController` | State machine tying it all together | n/a |
| `HUD` | Floating non-activating overlay: mic level, partial text, state | `NSPanel` (`.nonactivatingPanel`) |
| `OnboardingFlow` | Walk user through Mic / Input Monitoring / Accessibility grants | TCC checks |

### State machine (`DictationController`)

```
idle ──keyDown──▶ arming(80ms debounce) ──held──▶ recording ──keyUp──▶ finalizing ──▶ inserting ──▶ idle
                        │ released early                                    │ error
                        ▼                                                   ▼
                      idle (treat as tap: ignore or toggle-mode)          idle + error HUD
```

The 60–100 ms debounce distinguishes an accidental Fn tap from a hold, and lets you support **tap-to-toggle** (hands-free mode) alongside **hold-to-talk** later.

Everything runs **in-process** in one Swift app. No IPC, no sidecar server. That's the single biggest reliability and latency win over Python/Electron architectures. (See §10 for the alternative stacks and why they lose.)

---

## 2. Global Fn key detection: capabilities and platform limitations

### How it works

The Fn/🌐 key **is observable** on macOS. It arrives as a `flagsChanged` event with keycode `63` (`kVK_Function`) and the `.maskSecondaryFn` flag. Use a `CGEventTap` rather than `NSEvent.addGlobalMonitorForEvents`: an event tap can be created listen-only or active, tells you when it's disabled, and also sees key events you may want later (e.g. Esc to cancel).

```swift
let tap = CGEvent.tapCreate(
    tap: .cgSessionEventTap,
    place: .headInsertEventTap,
    options: .listenOnly,                       // .defaultTap if you need to swallow events
    eventsOfInterest: 1 << CGEventType.flagsChanged.rawValue,
    callback: { _, type, event, refcon in
        let keyCode = event.getIntegerValueField(.keyboardEventKeycode)
        if keyCode == 63 {                       // kVK_Function
            let fnDown = event.flags.contains(.maskSecondaryFn)
            // dispatch fnDown/up to DictationController (hop off this thread fast!)
        }
        return Unmanaged.passUnretained(event)
    },
    userInfo: nil)
```

Rules for the tap callback:
- **Return within milliseconds.** Do nothing but read the keycode/flags and post to a queue. If the callback stalls, macOS disables the tap (you'll get `tapDisabledByTimeout`; re-enable with `CGEvent.tapEnable`).
- **Check `keyCode == 63`, not just the flag.** Arrow keys, Page Up/Down, and Home/End also set `.maskSecondaryFn` on many keyboards.
- Track down/up by the flag's presence, since `flagsChanged` fires for both transitions.

### Platform limitations (the honest list)

1. **System Fn binding conflicts.** macOS binds the 🌐/Fn key to emoji picker / input-source switching / **Apple's own Dictation** (double-press by default). During onboarding, detect and ask the user to set *System Settings → Keyboard → "Press 🌐 key to" → **Do Nothing*** and disable the built-in Dictation shortcut. This can be pre-checked by reading `defaults read com.apple.HIToolbox AppleFnUsageType` (0 = Do Nothing) and deep-linking to the settings pane. You cannot change it programmatically without user involvement.
2. **You can't fully "own" Fn.** Even with an active (`.defaultTap`) tap swallowing the `flagsChanged` event, Fn's effect on other keys (F-row media functions, embedded numpad on some keyboards) is applied at driver/firmware level. In practice this doesn't matter for push-to-talk (you only observe it), but you cannot repurpose Fn *away* from those hardware behaviors.
3. **External non-Apple keyboards often never send Fn to the OS.** Many third-party keyboards handle Fn entirely in firmware. **Ship a configurable fallback hotkey** (good candidates: Right ⌘ used alone, Right ⌥, F5, the dictation key on newer Macs, or a double-tap-and-hold of a modifier). The `HotkeyMonitor` should abstract "PTT key" so Fn is just the default.
4. **Secure Input blocks event taps.** When a password field (or apps like Terminal with "Secure Keyboard Entry") enables secure input, your tap receives no key events. The PTT key silently stops working. Poll `IsSecureEventInputEnabled()` and show a menu-bar/HUD indicator ("dictation unavailable: secure input active") instead of appearing broken.
5. **Permissions.** A `flagsChanged` event tap requires **Input Monitoring** (TCC). An active tap that swallows events additionally behaves best with **Accessibility**. See §9.
6. **Sandboxing.** `CGEventTap` + posting synthetic events are incompatible with the App Sandbox → **no Mac App Store distribution**. Direct-distribute with Developer ID + notarization (§8). This is why Whisper Flow etc. are direct downloads.

---

## 3. Audio capture

**Engine: `AVAudioEngine`, started once at launch and kept warm.**

- Install a tap on `inputNode` and convert to **16 kHz mono Float32** (what every STT model wants) with `AVAudioConverter`. Do the conversion on the audio thread's buffer callback; it's cheap.
- Write into a **ring buffer** continuously… but only *retain* from ~300 ms before key-down. Two benefits:
  - **Zero start latency.** `AVAudioEngine.start()` can take 100–300 ms (worse with Bluetooth mics, which also need a route change). If the engine only starts on key-down, the first syllable gets clipped. Keep the engine running; gate on the key.
  - **Pre-roll.** People start speaking slightly before/exactly as they press. Prepending 300 ms of pre-roll audio measurably reduces clipped first words.
- If always-on capture feels wrong privacy-wise (the orange mic indicator stays lit), offer a "start engine on demand" setting and accept the latency; default to warm-engine + pre-roll and explain it in onboarding. Nothing leaves the ring buffer unless the key is held.
- Enable **voice processing** (`inputNode.setVoiceProcessingEnabled(true)`) optionally: echo cancellation + noise suppression helps in calls/noisy rooms, but it can color audio; make it a setting, default off, and A/B it for WER.
- Handle device changes (`AVAudioEngineConfigurationChange` notification): rebuild the tap when the user plugs in AirPods mid-session.
- Cap recording length (e.g. 5 min) to bound memory and transcription time.

---

## 4. Local speech-to-text: engines and tradeoffs

All options below are free, open-weight, on-device. The design hides them behind one protocol so you can swap:

```swift
protocol STTEngine {
    func load() async throws
    func feed(_ samples: [Float])                    // streaming input while key held
    var partials: AsyncStream<String> { get }        // optional live partials for HUD
    func finalize() async throws -> Transcript       // called on key release
    func reset()
}
```

### The candidates

| Engine | Runtime on Apple Silicon | Speed (M1 Pro, 10 s audio) | Accuracy | Languages | Punct./caps | Integration |
|---|---|---|---|---|---|---|
| **Parakeet TDT 0.6B v2/v3** (NVIDIA, open) | Core ML via **FluidAudio** (Swift) or MLX via `parakeet-mlx` (Python) | ~0.2–0.5 s (RTF ≈ 0.03) | Excellent English (top of open leaderboards, beats whisper-large on many English sets) | v2: EN; v3: 25 European langs | Yes, built in | FluidAudio = native Swift pkg, ANE-accelerated |
| **whisper.cpp** + `large-v3-turbo` (q5) | C/C++ w/ Metal; optional Core ML encoder | ~1–2 s | Excellent, robust to accents/noise | ~99 | Yes | SwiftPM package, embeds directly |
| **whisper.cpp** + `small`/`base` (q5) | same | ~0.3–0.6 s | Good, noticeably weaker on names/jargon | ~99 | Yes | same |
| **faster-whisper** (CTranslate2) | **CPU-only on macOS** (no Metal backend) | slow on Mac | same models as whisper.cpp | ~99 | Yes | Python; needs sidecar process |
| **Moonshine** (tiny/base) | ONNX/Core ML | very fast, built for streaming | Decent EN, below the above | EN | Yes | more DIY |
| **Vosk** | CPU | fast | Clearly worse | many | Weak | easy but not worth it |
| **Apple `SFSpeechRecognizer`** (on-device) | system | streaming, fast | OK, weaker than the above on jargon | many | Yes | zero model shipping; `SpeechAnalyzer` (better) needs macOS 26, not available on 15.6 |

### Recommendation

- **Primary: Parakeet TDT 0.6B via FluidAudio (Core ML).** It is the same family Whisper Flow-class apps have moved to: near-instant on Apple Silicon (runs on the ANE, barely touches CPU/GPU), excellent English WER, punctuation and capitalization included, native Swift integration, MIT-ish licensing (CC-BY-4.0 model). For a dictation product where p95 latency is the product, this wins.
- **Bundled fallback: whisper.cpp with `large-v3-turbo` q5_0** (~574 MB) for multilingual users and as an accuracy cross-check; `ggml-base.en` as a low-RAM option. whisper.cpp is also the engine you'll reuse verbatim on Windows (Vulkan/CUDA backends), which pays for the abstraction.
- **Skip faster-whisper for the macOS product**: CTranslate2 has no Metal support, so it's the one popular option that's actually *slow* on your hardware, and it drags in a Python runtime. It becomes interesting again on Windows/NVIDIA.

### Streaming vs. transcribe-on-release

Whisper-family models are not natively streaming; Parakeet (RNN-T/TDT) is friendlier to chunked decoding. Strategy:

- **v1 (Milestone 2): transcribe on release only.** With Parakeet, a 15 s utterance transcribes in well under a second. Release-to-text ≈ 400–700 ms feels instant. Ship this first; it may be all you need.
- **v2 (Milestone 4): chunked pre-transcription while held.** Every ~2 s of accumulated audio, transcribe the whole buffer so far (Parakeet is fast enough to redo from scratch) and show it as HUD partials. On release, transcribe only since-last-chunk + merge, or re-run the full buffer if < 30 s. This gives live feedback *and* cuts release latency for long dictations. Avoid token-level streaming hacks (local agreement etc.) until this proves insufficient. They add real complexity for marginal gain at dictation lengths.

---

## 5. Text insertion into the active application

Three mechanisms, used as a cascade (this mirrors what the commercial apps do):

1. **Pasteboard + synthetic ⌘V (default).**
   - Save current `NSPasteboard` contents (all types you can round-trip), write the transcript, post ⌘V key-down/up via `CGEvent` to the session tap, restore the old contents after ~150–300 ms.
   - Pros: instant regardless of length, handles Unicode/emoji/CJK, works in ~99% of apps including Electron.
   - Cons: clobbers clipboard briefly (mitigate: restore + mark the transient item with `org.nspasteboard.TransientType` so clipboard managers ignore it); a few apps (some terminals, remote desktops) treat paste specially.
2. **Direct typing via `CGEventKeyboardSetUnicodeString` (fallback / user-selectable per-app).**
   - Post keyboard events carrying the unicode string in ≤20-char chunks with tiny inter-event delays.
   - Pros: no clipboard involvement, looks like typing (good for terminals, vim, remote sessions).
   - Cons: slower for long text; some apps drop rapid synthetic events; IME interactions.
3. **Accessibility API (`AXUIElement`) insertion (opportunistic).**
   - Read the focused element (`kAXFocusedUIElementAttribute`), and where it exposes `AXValue`/`AXSelectedText` as settable, insert at the caret directly.
   - Pros: cleanest semantics, no clipboard, can *read* surrounding text (enables smart spacing/capitalization relative to existing text, and context-aware formatting later).
   - Cons: support is inconsistent (web views and Electron apps are patchy). Treat as enhancement, not the workhorse.

Practical details:
- Keep a small **per-app strategy table** (bundle ID → paste | type | AX), user-overridable. Terminals default to *type*; everything else *paste*.
- Use AX (when readable) to decide whether to prepend a space / capitalize (caret mid-sentence vs. new line).
- Both paste-posting and typing require the **Accessibility** permission.
- In secure-input contexts you generally *can* still post events, but you already can't hear the hotkey (§2.4), so the point is moot. Surface the indicator instead.

---

## 6. Local post-processing (all optional, all on-device)

Pipeline of pure, testable stages between STT output and injection:

1. **Normalization**: trim, collapse whitespace, strip leading punctuation artifacts, decide leading space/capitalization from AX context.
2. **Custom vocabulary**: two complementary mechanisms:
   - *Bias at decode time:* whisper.cpp `initial_prompt` seeded with the user's terms ("Kubernetes, Supabase, Hai Bui, PostHog…"); helps the model pick the right spelling. (Parakeet has no prompt; rely on the next mechanism.)
   - *Post-hoc replacement:* user-defined replacement rules with fuzzy matching (case-insensitive, small edit distance): "post hog → PostHog", "super base → Supabase". Store in a simple JSON/SQLite table with an editor UI in Settings.
3. **Spoken punctuation & symbols** (dictation-mode): "new line", "new paragraph", "comma" as literal commands, toggleable; off by default since models already punctuate.
4. **Snippets**: "insert my address" → expansion table.
5. **Command mode** (later milestone): a prefix wake-word ("command: …") switches from *insert text* to *execute*: "select all", "delete last sentence", "make that a bullet list". Implement as intent-matching over a fixed grammar first (regex/keyword: deterministic, testable); a tiny local LLM is *not* required for v1.
6. **LLM polish (optional, still local)**: a "clean this up" toggle that pipes the transcript through a small local model (Qwen 2.5 1.5B/3B instruct via **MLX** or llama.cpp, or Gemma 3 4B) with a rewrite prompt: remove filler words, fix grammar, apply tone presets per-app (Slack casual vs. email formal). Adds ~300–900 ms on M1 Pro. Make it an explicit mode (e.g. hold Fn+Shift), never the default path, so core dictation stays instant.

---

## 7. Performance: the latency budget

Target: **key-release → text inserted ≤ 500 ms** for utterances under 30 s (Parakeet path). Where the time goes and how to keep it down:

| Stage | Naive | Optimized | How |
|---|---|---|---|
| Audio start | 100–300 ms lost at *start* | ~0 | Warm engine + ring buffer + 300 ms pre-roll (§3) |
| End-of-speech capture | up to one buffer (~100 ms) | ~20 ms | Small tap buffer (1024–2048 frames) |
| Model load | 1–10 s | 0 on hot path | Load at launch; keep resident; `mlock` optional. Reload on wake-from-sleep in background |
| Transcription 10 s audio | 1–2 s (whisper large) | 0.2–0.5 s | Parakeet/ANE; or whisper-turbo q5 + Metal, greedy (beam 1), `no_context` |
| While-held pre-transcription | n/a | hides most of the above | Chunked re-transcription every ~2 s (§4); on release only the tail is new |
| VAD trim | n/a | saves proportional time | Trim leading/trailing silence (Silero VAD via FluidAudio, or whisper.cpp's built-in) before final pass |
| Post-processing | ~0 | ~0 | Pure string ops |
| Injection | 5–30 ms | 5–30 ms | Paste path |

Other rules:
- **Never touch the main thread** from the tap callback or audio thread; the pipeline is `actor`-isolated with `AsyncStream`s between stages.
- Memory: Parakeet ~600 MB, whisper-turbo-q5 ~600 MB. Load one engine at a time, offer "unload after N min idle" for low-RAM users.
- Instrument every stage with `os_signpost` from day one; latency regressions are product bugs here.
- Energy: ANE inference keeps the fans off; avoid polling loops (the tap and audio callbacks are all event-driven).

---

## 8. Packaging as a background app

- **App shape:** menu-bar-only agent, `LSUIElement = YES` (no Dock icon, no app switcher). `NSStatusItem` with state icon (idle / recording / transcribing / error / secure-input-blocked), menu for Settings, History, Pause, Quit.
- **Settings window:** SwiftUI, opened on demand (hotkey choice, engine/model picker + downloader, vocabulary editor, per-app insertion strategy, launch-at-login toggle).
- **Model storage:** don't bundle multi-hundred-MB models in the .app; download on first run to `~/Library/Application Support/<app>/models/` with checksum verification (bundle only the smallest model for out-of-box function).
- **Launch at login:** `SMAppService.mainApp.register()` (modern replacement for launch agents; gives the user-visible Login Items entry).
- **Distribution:** no sandbox possible (§2.6) ⇒ Developer ID signing + hardened runtime + **notarization**, ship as DMG/zip. Sparkle 2 for auto-updates. For purely personal use you can skip all of this and run the ad-hoc-signed debug build. TCC permissions work the same, though they must be re-granted when the binary changes unless you sign with a stable identity.
- **Crash/hang hygiene:** watchdog that re-enables a disabled event tap; auto-restart audio engine on route change; single-instance lock.
- **History:** last N transcripts in local SQLite (with an off switch and "never store" mode), invaluable for recovering text when an app rejected insertion.

---

## 9. Permissions & macOS APIs (complete list)

| Permission (TCC) | Why | Triggered by | Notes |
|---|---|---|---|
| **Microphone** | capture | first `AVAudioEngine` input tap | `NSMicrophoneUsageDescription` in Info.plist; standard prompt |
| **Input Monitoring** | see Fn key globally | `CGEventTap` creation | No automatic prompt in all cases. Use `IOHIDRequestAccess(kIOHIDRequestTypeListenEvent)` to trigger it; user may need to toggle in System Settings; **app restart often required** after grant |
| **Accessibility** | post ⌘V / typing events; read focused element | `CGEvent.post`, `AXUIElement` | Check `AXIsProcessTrustedWithOptions` (can show prompt); deep-link: `x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility` |
| *(no network, no screen recording, no automation needed)* | | | |

Build a dedicated **onboarding window** that shows live status of all three grants (poll every second while visible), with buttons deep-linking to the right Settings panes, plus the Fn-key system-setting check from §2.1. Getting this flow right is half the perceived quality of the app.

Key APIs used: `CGEventTap`/`CGEvent` (Quartz Event Services), `AVAudioEngine`/`AVAudioConverter`, `NSPasteboard`, `AXUIElement*`, `NSPanel` (`.nonactivatingPanel`, `.floating` level) for the HUD, `SMAppService`, `IsSecureEventInputEnabled`, `os_signpost`.

---

## 10. Technology stack

**Recommendation: native Swift.**

| Stack | Verdict |
|---|---|
| **Swift + AppKit/SwiftUI + FluidAudio (Parakeet) + whisper.cpp SwiftPM** | ✅ **Chosen.** Every hard part of this app (event taps, TCC, AX, pasteboard, audio routes, non-activating panels) is a macOS-native API; wrappers only add failure modes. Single process, single binary, lowest latency, smallest memory. |
| Python (pynput/sounddevice/faster-whisper + rumps) | ✅ as a *throwaway 1-day spike* (via PyObjC for the Fn tap), ❌ as the product: PyInstaller + TCC is fragile, faster-whisper is CPU-only on Mac, GIL vs. audio callbacks, ugly permission attribution. |
| Rust (tauri + cpal + whisper-rs + rdev/enigo) | Defensible *if Windows is certain and near-term*. But rdev/enigo don't handle Fn or AX insertion well on macOS. You end up writing the same native code behind FFI. Choose only if you want one core for both OSes from day one. |
| Electron/Tauri UI-centric | ❌: this app has almost no UI; paying 200 MB of Chromium for a menu-bar item is the wrong trade. |

**Windows-port posture:** keep `STTEngine`, `PostProcessor`, vocabulary/history logic UI-free and platform-free (protocol-driven, no AppKit imports). The engines port cleanly (whisper.cpp → Vulkan/CUDA; Parakeet → onnxruntime/faster-whisper on NVIDIA). The platform layer (hotkey via Low-Level Keyboard Hook `WH_KEYBOARD_LL`, audio via WASAPI, insertion via `SendInput`, packaging via MSIX) is a rewrite regardless of stack. Accept it rather than compromising the macOS product. Note: many Windows keyboards *never* send Fn to the OS at all, so the default PTT key there should be something else (e.g. Right Ctrl).

### Repo layout

```
whisper/
├─ ARCHITECTURE.md
├─ App/                          # Xcode project (or XcodeGen/Tuist)
│  ├─ WhisperFlowLocal/
│  │  ├─ AppMain.swift           # @main, LSUIElement, status item
│  │  ├─ DictationController.swift
│  │  ├─ Hotkey/
│  │  │  ├─ HotkeyMonitor.swift  # CGEventTap wrapper, PTT-key abstraction
│  │  │  └─ SecureInputWatcher.swift
│  │  ├─ Audio/
│  │  │  ├─ AudioCapture.swift   # AVAudioEngine + ring buffer + pre-roll
│  │  │  └─ VAD.swift
│  │  ├─ STT/                    # platform-free
│  │  │  ├─ STTEngine.swift      # protocol
│  │  │  ├─ ParakeetEngine.swift # FluidAudio
│  │  │  └─ WhisperCppEngine.swift
│  │  ├─ PostProcess/            # platform-free
│  │  │  ├─ Pipeline.swift
│  │  │  ├─ Vocabulary.swift
│  │  │  └─ Commands.swift
│  │  ├─ Insertion/
│  │  │  ├─ TextInjector.swift   # strategy cascade
│  │  │  ├─ PasteInjector.swift
│  │  │  ├─ TypeInjector.swift
│  │  │  └─ AXInjector.swift
│  │  ├─ UI/
│  │  │  ├─ HUDPanel.swift
│  │  │  ├─ SettingsView.swift
│  │  │  ├─ OnboardingView.swift
│  │  │  └─ StatusItem.swift
│  │  └─ Storage/
│  │     ├─ Settings.swift
│  │     ├─ History.swift        # SQLite
│  │     └─ ModelStore.swift     # download/verify models
│  └─ Tests/                     # PostProcess + state machine + injector strategy tests
└─ spike/                        # (optional) day-1 Python proof of pipeline
```

---

## 11. Implementation plan: milestones

**M0: Spike (0.5–1 day, optional but recommended).**
Python script: PyObjC `CGEventTap` for Fn → `sounddevice` record → `parakeet-mlx` (or `mlx-whisper`) transcribe → `pbcopy`+AppleScript ⌘V paste. Proves the full loop on your machine, calibrates model quality/latency expectations. Throw it away.

**M1: App skeleton + hotkey (2–3 days).**
Menu-bar app (LSUIElement), status icon states, onboarding window with live TCC status for Mic/Input Monitoring/Accessibility + Fn-usage system-setting check. `HotkeyMonitor` with debounced press/hold/release, fallback-hotkey abstraction, tap-disabled watchdog, `SecureInputWatcher`.
*Exit criteria: icon reliably flips state on Fn hold/release everywhere, including after sleep/wake; secure-input indicator works.*

**M2: Audio + transcribe-on-release (3–4 days).**
Warm `AVAudioEngine`, ring buffer, pre-roll, device-change handling. Integrate FluidAudio/Parakeet (and model downloader). On release: VAD-trim → transcribe → log + copy to clipboard (no injection yet). `os_signpost` instrumentation.
*Exit criteria: release-to-text < 700 ms for 10 s utterances; no clipped first words; survives AirPods connect/disconnect.*

**M3: Text insertion (2–3 days).**
Paste injector with clipboard save/restore + transient-type marking; type injector; per-app strategy table; AX-based leading-space/capitalization. History store (recover lost text).
*Exit criteria: dictation lands correctly in TextEdit, Safari/Chrome text areas, Slack, Terminal, VS Code; clipboard restored; long transcripts (500+ words) insert instantly.*

**M4: HUD + while-held partials (3–4 days).**
Non-activating floating HUD (mic level, state, partial text). Chunked pre-transcription every ~2 s while held; merged final pass on release. Esc-to-cancel.
*Exit criteria: partials visible while speaking; release latency for 60 s dictation ≈ same as for 5 s.*

**M5: Post-processing & vocabulary (3–5 days).**
Pipeline stages, replacement-rule engine + Settings editor UI, snippets, spoken-punctuation toggle, whisper `initial_prompt` biasing, whisper.cpp as selectable second engine. Optional: command-mode grammar; optional local-LLM polish mode (MLX).
*Exit criteria: your personal jargon list transcribes correctly; rules editable without restart.*

**M6: Productionize (3–5 days).**
Settings polish, launch-at-login (`SMAppService`), model manager UI, single-instance lock, Sparkle updates, Developer ID signing + notarization, DMG. Soak testing: 24 h running, sleep/wake cycles, fast repeated PTT, Bluetooth churn, low-RAM behavior.
*Exit criteria: notarized build a stranger could install and self-serve through onboarding.*

**M7: Later.** Windows port (shared core + WH_KEYBOARD_LL/WASAPI/SendInput platform layer), tap-to-toggle hands-free mode, per-app tone presets via local LLM, context-aware formatting from AX text reading, multilingual auto-detect.

Total: **~3–4 weeks of focused work to M6.**

---

## 12. Cost

Zero marginal cost by construction: open-weight models (Parakeet CC-BY-4.0, Whisper MIT-licensed weights, whisper.cpp MIT, FluidAudio Apache-2.0), all inference on your M1 Pro, no telemetry, no accounts. The only "cost" is a one-time $99/yr Apple Developer ID **if** you want to distribute notarized builds to others, unnecessary for personal use.
