#!/usr/bin/env python3
"""
dictate.py — fully local push-to-talk dictation for macOS.

Hold Fn, speak, release: the transcript is inserted at the cursor of the
frontmost app. Press Esc while holding to cancel. Everything runs on-device
(Parakeet via MLX for speech-to-text, optional Ollama model for cleanup).

Usage:
    python dictate.py                     # hold Fn to dictate
    python dictate.py --key rcmd          # use Right-Command instead of Fn
    python dictate.py --clean             # clean transcript with local LLM
    python dictate.py --test audio.wav    # transcribe a file and exit

Permissions (grant to your terminal app in System Settings > Privacy & Security):
    Microphone, Input Monitoring, Accessibility
"""

import argparse
import subprocess
import sys
import queue
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
PREROLL_SEC = 0.3          # audio kept from before the key went down
MIN_HOLD_SEC = 0.15        # ignore accidental taps shorter than this
MAX_RECORD_SEC = 300
DOUBLE_TAP_SEC = 0.45      # two Fn taps within this window -> hands-free mode
VOICE_RMS = 0.012          # mic RMS above this counts as "someone is talking"
NO_SPEECH_CANCEL_SEC = 15  # hands-free with no speech at all -> give up
KEYCODE_FN = 63
KEYCODE_RCMD = 54
KEYCODE_ESC = 53
KEYCODE_V = 9
EVENT_SYSTEM_DEFINED = 14      # kCGEventSystemDefined (media keys)
NX_KEYTYPE_PLAY = 16           # AirPods single stem-press → play/pause

CLEAN_PROMPT = (
    "Clean up this dictated, possibly rambling text:\n"
    "- fix punctuation and capitalization\n"
    "- remove filler words (um, uh, like, you know), false starts, and "
    "repeated phrases\n"
    "- if the speaker enumerates points (first... second... also... the "
    "third thing...), format them as a numbered list (1. 2. 3.), one item "
    "per line, with a short intro sentence if they gave one\n"
    "- otherwise keep it as flowing prose\n"
    "- keep the speaker's wording and meaning; do not add new content or "
    "commentary\n"
    "Output ONLY the cleaned text, nothing else.\n\n"
    "Dictated text:\n"
)


# ---------------------------------------------------------------- audio

class Recorder:
    """Always-warm microphone stream with a pre-roll ring buffer."""

    def __init__(self):
        import sounddevice as sd
        self._lock = threading.Lock()
        self._recording = False
        self._preroll = deque()
        self._preroll_samples = 0
        self._chunks = []
        self.level = 0.0            # live mic RMS, read by the HUD
        self.last_voice = 0.0       # when we last heard speech
        self.speech_started = False
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=1024, callback=self._callback)
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        block = indata[:, 0].copy()
        self.level = float(np.sqrt(np.mean(block ** 2)))
        with self._lock:
            if self._recording:
                if self.level > VOICE_RMS:
                    self.last_voice = time.time()
                    self.speech_started = True
                if sum(len(c) for c in self._chunks) < MAX_RECORD_SEC * SAMPLE_RATE:
                    self._chunks.append(block)
            else:
                self._preroll.append(block)
                self._preroll_samples += len(block)
                while self._preroll_samples > PREROLL_SEC * SAMPLE_RATE:
                    self._preroll_samples -= len(self._preroll.popleft())

    def start(self):
        with self._lock:
            self._chunks = list(self._preroll)
            self.speech_started = False
            self.last_voice = time.time()
            self._recording = True

    def stop(self) -> np.ndarray:
        with self._lock:
            self._recording = False
            audio = (np.concatenate(self._chunks)
                     if self._chunks else np.zeros(0, dtype=np.float32))
            self._chunks = []
            return audio

    def cancel(self):
        with self._lock:
            self._recording = False
            self._chunks = []


def write_wav(path: str, audio: np.ndarray):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- STT

class Transcriber:
    def __init__(self, model_name: str):
        import os
        # If the model is already cached, skip HF's online update check —
        # it can hang for minutes when rate-limited.
        cache = (Path.home() / ".cache/huggingface/hub"
                 / ("models--" + model_name.replace("/", "--")))
        if cache.exists():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from parakeet_mlx import from_pretrained
        t0 = time.time()
        print(f"loading model {model_name} ...", flush=True)
        self.model = from_pretrained(model_name)
        # warm-up pass so the first real dictation isn't slow
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write_wav(f.name, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
            self.model.transcribe(f.name)
            Path(f.name).unlink(missing_ok=True)
        print(f"model ready in {time.time() - t0:.1f}s", flush=True)

    def transcribe(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write_wav(f.name, audio)
            result = self.model.transcribe(f.name)
            Path(f.name).unlink(missing_ok=True)
        return result.text.strip()

    def transcribe_file(self, path: str) -> str:
        return self.model.transcribe(path).text.strip()


# ---------------------------------------------------------------- cleanup

def airpods_connected() -> bool:
    """True if AirPods are among the current audio input devices."""
    import os
    from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
    # AVCapture logs a one-time Continuity-Camera warning to stderr; mute it.
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        devs = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeAudio) or []
        names = [(d.localizedName() or "") for d in devs]
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
    return any("airpods" in n.lower() for n in names)


def cleanup_available(model: str) -> bool:
    """True if Ollama is running and the cleanup model is installed."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(
                "http://localhost:11434/api/tags", timeout=3) as resp:
            names = [m.get("name", "")
                     for m in json.loads(resp.read()).get("models", [])]
        base = model.split(":")[0]
        return any(n == model or n.split(":")[0] == base for n in names)
    except Exception:
        return False


def clean_text(text: str, model: str) -> str:
    import json
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({
                "model": model,
                "prompt": CLEAN_PROMPT + text,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.1},
            }).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            cleaned = json.loads(resp.read())["response"].strip()
        return cleaned if cleaned else text
    except Exception as e:
        print(f"  cleanup failed ({e}); using raw transcript", flush=True)
        return text


# ---------------------------------------------------------------- insertion

class Inserter:
    """Paste-based insertion: set clipboard, synthesize Cmd+V, restore clipboard."""

    def __init__(self):
        from AppKit import NSPasteboard, NSPasteboardTypeString
        import Quartz
        self.NSPasteboard = NSPasteboard
        self.NSPasteboardTypeString = NSPasteboardTypeString
        self.Quartz = Quartz

    def insert(self, text: str):
        Q = self.Quartz
        pb = self.NSPasteboard.generalPasteboard()
        old = pb.stringForType_(self.NSPasteboardTypeString)
        pb.clearContents()
        pb.setString_forType_(text, self.NSPasteboardTypeString)
        my_count = pb.changeCount()
        time.sleep(0.05)  # let the pasteboard settle before the app reads it

        for down in (True, False):
            ev = Q.CGEventCreateKeyboardEvent(None, KEYCODE_V, down)
            Q.CGEventSetFlags(ev, Q.kCGEventFlagMaskCommand)
            Q.CGEventPost(Q.kCGHIDEventTap, ev)
            time.sleep(0.01)

        def restore():
            time.sleep(0.35)
            if pb.changeCount() == my_count and old is not None:
                pb.clearContents()
                pb.setString_forType_(old, self.NSPasteboardTypeString)
        threading.Thread(target=restore, daemon=True).start()


# ---------------------------------------------------------------- HUD

class HUD:
    """A small black pill at the top-center of the active screen with live
    waveform bars — inspired by FreeFlow's overlay. Three looks:
      listening() live mic waveform · processing() gentle ripple ·
      flash(text) a brief text message. All are thread-safe.
    """

    N_BARS = 5
    BAR_W, BAR_GAP = 3.0, 4.0
    MIN_BAR, MAX_BAR = 3.0, 15.0
    PILL_H = 28.0
    PILL_W = 74.0                     # compact width for the waveform looks

    def __init__(self, recorder):
        from AppKit import (NSApplication, NSPanel, NSTextField, NSColor,
                            NSFont, NSBackingStoreBuffered,
                            NSWindowStyleMaskBorderless)
        from Foundation import (NSMakeRect, NSTimer, NSRunLoop,
                                NSRunLoopCommonModes)
        import Quartz
        try:                          # silence a harmless CGColor pointer warning
            import objc
            import warnings
            warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)
        except Exception:
            pass

        self.recorder = recorder
        self._mode = "hidden"
        self._token = 0
        self._h = [self.MIN_BAR] * self.N_BARS
        self._shape = [0.55, 0.8, 1.0, 0.8, 0.55]   # center bars taller

        NSApplication.sharedApplication().setActivationPolicy_(1)  # no Dock icon

        style = NSWindowStyleMaskBorderless | (1 << 7)  # non-activating panel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, -1000, self.PILL_W, self.PILL_H), style,
            NSBackingStoreBuffered, False)
        panel.setLevel_(1000)                # screen-saver level: above all
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)            # soft drop shadow
        panel.setIgnoresMouseEvents_(True)
        panel.setCollectionBehavior_(
            (1 << 0) | (1 << 4) | (1 << 8))  # all Spaces · stationary · over full-screen

        content = panel.contentView()
        content.setWantsLayer_(True)
        content.layer().setBackgroundColor_(
            NSColor.blackColor().colorWithAlphaComponent_(0.9).CGColor())
        content.layer().setCornerRadius_(self.PILL_H / 2)
        content.layer().setMasksToBounds_(True)

        white = NSColor.whiteColor().CGColor()
        total = self.N_BARS * self.BAR_W + (self.N_BARS - 1) * self.BAR_GAP
        x0 = (self.PILL_W - total) / 2.0
        self._bar_x = [x0 + i * (self.BAR_W + self.BAR_GAP)
                       for i in range(self.N_BARS)]
        self.bars = []
        for i in range(self.N_BARS):
            bar = Quartz.CALayer.layer()
            bar.setBackgroundColor_(white)
            bar.setCornerRadius_(self.BAR_W / 2)
            bar.setFrame_(NSMakeRect(self._bar_x[i],
                                     (self.PILL_H - self.MIN_BAR) / 2,
                                     self.BAR_W, self.MIN_BAR))
            content.layer().addSublayer_(bar)
            self.bars.append(bar)

        field = NSTextField.labelWithString_("")
        field.setAlignment_(1)               # centered
        field.setTextColor_(NSColor.whiteColor())
        field.setFont_(NSFont.systemFontOfSize_weight_(12.5, 0.23))
        field.setHidden_(True)
        content.addSubview_(field)
        self.panel, self.field = panel, field
        self._Quartz = Quartz

        timer = NSTimer.timerWithTimeInterval_repeats_block_(
            1 / 30.0, True, self._tick)
        NSRunLoop.mainRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
        self._timer = timer

    # ---- public, thread-safe -------------------------------------------
    def listening(self):
        self._on_main(self._enter, "wave")

    def processing(self):
        self._on_main(self._enter, "proc")

    def flash(self, text, seconds=2.2):
        self._on_main(self._flash, text, seconds)

    def hide(self):
        self._on_main(self._hide)

    # ---- internals ------------------------------------------------------
    def _on_main(self, fn, *args):
        from Foundation import NSThread
        if NSThread.isMainThread():
            fn(*args)
        else:
            from PyObjCTools import AppHelper
            AppHelper.callAfter(fn, *args)

    def _place(self, width):
        """Top-center of the screen the mouse is on, tucked under the menu
        bar. Screen frames are NOT (0,0)-based on multi-monitor setups."""
        from AppKit import NSScreen, NSEvent
        from Foundation import NSMakeRect
        mouse = NSEvent.mouseLocation()
        target = NSScreen.mainScreen() or NSScreen.screens()[0]
        for s in NSScreen.screens():
            f = s.frame()
            if (f.origin.x <= mouse.x <= f.origin.x + f.size.width
                    and f.origin.y <= mouse.y <= f.origin.y + f.size.height):
                target = s
                break
        f, vf = target.frame(), target.visibleFrame()
        x = f.origin.x + (f.size.width - width) / 2.0
        y = vf.origin.y + vf.size.height - self.PILL_H - 6.0
        self.panel.setFrame_display_(
            NSMakeRect(x, y, width, self.PILL_H), True)

    def _enter(self, mode):
        if self._mode == mode and self.panel.isVisible():
            return                            # already showing this look
        self._token += 1
        self._mode = mode
        self.field.setHidden_(True)
        for b in self.bars:
            b.setHidden_(False)
        self._place(self.PILL_W)
        self.panel.orderFrontRegardless()

    def _flash(self, text, seconds):
        from AppKit import NSAttributedString, NSFontAttributeName
        from Foundation import NSMakeRect
        self._token += 1
        tok = self._token
        self._mode = "text"
        for b in self.bars:
            b.setHidden_(True)
        size = NSAttributedString.alloc().initWithString_attributes_(
            text, {NSFontAttributeName: self.field.font()}).size()
        width = min(max(size.width + 40, self.PILL_W), 380.0)
        self.field.setStringValue_(text)
        self.field.setFrame_(
            NSMakeRect(16, (self.PILL_H - 18) / 2, width - 32, 18))
        self.field.setHidden_(False)
        self._place(width)
        self.panel.orderFrontRegardless()
        threading.Timer(seconds, lambda: self._flash_done(tok)).start()

    def _flash_done(self, tok):
        if self._token == tok:
            self._on_main(self._hide)

    def _hide(self):
        self._mode = "hidden"
        self.panel.orderOut_(None)

    def _tick(self, _timer):
        if self._mode not in ("wave", "proc"):
            return
        import math
        from Foundation import NSMakeRect
        t = time.time()
        if self._mode == "wave":
            a = max(0.06, min(1.0, self.recorder.level / 0.08))
            for i in range(self.N_BARS):
                tgt = (self.MIN_BAR + (self.MAX_BAR - self.MIN_BAR) * a
                       * self._shape[i] * (0.7 + 0.3 * math.sin(t * 9 + i * 1.3)))
                self._h[i] += (tgt - self._h[i]) * 0.4
        else:                                # gentle traveling ripple
            for i in range(self.N_BARS):
                tgt = (self.MIN_BAR + (self.MAX_BAR - self.MIN_BAR) * 0.5
                       * (0.5 + 0.5 * math.sin(t * 6 - i * 0.9)))
                self._h[i] += (tgt - self._h[i]) * 0.4
        Q = self._Quartz
        Q.CATransaction.begin()
        Q.CATransaction.setDisableActions_(True)
        for i, b in enumerate(self.bars):
            h = max(self.MIN_BAR, self._h[i])
            b.setFrame_(NSMakeRect(self._bar_x[i], (self.PILL_H - h) / 2.0,
                                   self.BAR_W, h))
        Q.CATransaction.commit()


class NoHUD:
    def listening(self): pass
    def processing(self): pass
    def flash(self, text, seconds=2.2): pass
    def hide(self): pass


class Cues:
    """Short audible cues played to the default output (e.g. your AirPods),
    so you know when dictation starts/stops without looking at the screen."""

    SOUNDS = {"start": "Pop", "stop": "Tink", "cancel": "Funk"}

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._snd = {}
        if enabled:
            from AppKit import NSSound
            for key, name in self.SOUNDS.items():
                s = NSSound.soundNamed_(name)
                if s is not None:
                    self._snd[key] = s

    def play(self, key):
        if not self.enabled:
            return
        s = self._snd.get(key)
        if s is not None:
            if s.isPlaying():
                s.stop()
            s.play()


# ---------------------------------------------------------------- controller

class Controller:
    """States: idle -> ptt (hold key) or handsfree (double-tap) -> idle.

    Hands-free ends on: another Fn tap, Esc (cancel), sustained silence,
    or the max-length cap.
    """

    def __init__(self, recorder, transcriber, inserter, hud,
                 clean_model=None, silence_sec=3.0, cues=None):
        self.recorder = recorder
        self.transcriber = transcriber
        self.inserter = inserter
        self.hud = hud
        self.clean_model = clean_model
        self.silence_sec = silence_sec
        self.cues = cues or Cues(enabled=False)
        self.state = "idle"
        self._down_at = 0.0
        self._last_tap = 0.0
        self._lock = threading.Lock()

    def on_key_down(self):
        with self._lock:
            now = time.time()
            if self.state == "handsfree":       # a tap ends hands-free
                self._stop_locked(cancel=False)
                return
            if self.state != "idle":
                return
            self.recorder.start()
            if now - self._last_tap < DOUBLE_TAP_SEC:
                self.state = "handsfree"
                self._last_tap = 0.0
                self.hud.listening()
                self.cues.play("start")
                print("\n● hands-free — tap Fn to stop, or just stop "
                      "talking", flush=True)
                threading.Thread(target=self._silence_watch,
                                 daemon=True).start()
            else:
                self.state = "ptt"
                self._down_at = now
                self.hud.listening()
                self.cues.play("start")
                print("\n● recording ...", flush=True)

    def on_key_up(self):
        with self._lock:
            if self.state != "ptt":
                return
            if time.time() - self._down_at < MIN_HOLD_SEC:
                # quick tap: not dictation, maybe half of a double-tap
                self._last_tap = time.time()
                self.state = "idle"
                self.recorder.cancel()
                self.hud.hide()
                return
            self._stop_locked(cancel=False)

    def toggle_handsfree(self):
        """Start/stop hands-free dictation — bound to the AirPods stem press
        (a discrete tap, so it toggles rather than holds)."""
        with self._lock:
            if self.state == "idle":
                self.recorder.start()
                self.state = "handsfree"
                self.hud.listening()
                self.cues.play("start")
                print("\n● hands-free (AirPods) — press again to stop, or "
                      "just stop talking", flush=True)
                threading.Thread(target=self._silence_watch,
                                 daemon=True).start()
            elif self.state == "handsfree":
                self._stop_locked(cancel=False)
            # during a held-key PTT session: ignore

    def on_cancel(self):
        with self._lock:
            if self.state in ("ptt", "handsfree"):
                self._stop_locked(cancel=True)
                print("  (cancelled)", flush=True)

    def _stop_locked(self, cancel):
        self.state = "idle"
        if cancel:
            self.recorder.cancel()
            self.cues.play("cancel")
            self.hud.hide()
            return
        audio = self.recorder.stop()
        if len(audio) < SAMPLE_RATE * 0.3:
            self.hud.hide()
            return
        self.cues.play("stop")
        self.hud.processing()
        threading.Thread(target=self._finish, args=(audio,),
                         daemon=True).start()

    def _silence_watch(self):
        started = time.time()
        while True:
            time.sleep(0.15)
            with self._lock:
                if self.state != "handsfree":
                    return
                now = time.time()
                if (self.recorder.speech_started
                        and now - self.recorder.last_voice > self.silence_sec):
                    print(f"  ({self.silence_sec:.0f}s of silence — done)",
                          flush=True)
                    self._stop_locked(cancel=False)
                    return
                if (not self.recorder.speech_started
                        and now - started > NO_SPEECH_CANCEL_SEC):
                    print("  (heard nothing — cancelled)", flush=True)
                    self._stop_locked(cancel=True)
                    return
                if now - started > MAX_RECORD_SEC:
                    self._stop_locked(cancel=False)
                    return

    def _finish(self, audio):
        dur = len(audio) / SAMPLE_RATE
        t0 = time.time()
        text = self.transcriber.transcribe(audio)
        t_stt = time.time() - t0
        if not text:
            print("  (no speech detected)", flush=True)
            self.hud.hide()
            return
        t_clean = 0.0
        if self.clean_model:
            t1 = time.time()
            text = clean_text(text, self.clean_model)
            t_clean = time.time() - t1
        self.inserter.insert(text)
        self.hud.hide()
        total = time.time() - t0
        stats = f"stt {t_stt:.2f}s"
        if self.clean_model:
            stats += f" + clean {t_clean:.2f}s"
        print(f"▸ [{dur:.1f}s audio | {stats} | total {total:.2f}s] {text}",
              flush=True)


# ---------------------------------------------------------------- wake word

WAKE_SPEECH_RMS = 0.015        # a bit above VOICE_RMS to ignore room noise
WAKE_HANG_SEC = 0.7            # silence after speech that ends an utterance
WAKE_MIN_SEG = 0.3            # ignore utterances shorter than this
WAKE_MAX_SEG = 20.0


def _norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def find_phrase(text: str, phrase: str):
    """Locate a spoken phrase inside a transcript, tolerantly.

    Returns (found, before, after) where before/after are the normalized
    words surrounding the phrase. Uses exact word-sequence match first, then
    a fuzzy fallback so "start dictation" still fires on "start diction"."""
    import difflib
    words, pw = _norm(text), _norm(phrase)
    if not pw:
        return (False, "", text)
    n = len(pw)
    for i in range(len(words) - n + 1):
        window = words[i:i + n]
        if window == pw or difflib.SequenceMatcher(
                None, " ".join(window), " ".join(pw)).ratio() >= 0.82:
            return (True, " ".join(words[:i]), " ".join(words[i + n:]))
    return (False, "", text)


class WakeListener:
    """Always-on, hands-free dictation gated by spoken phrases.

    Continuously segments speech with a simple energy VAD and transcribes
    each utterance locally. While idle it only watches for the wake phrase;
    once heard it inserts everything you say until the stop phrase (or an
    inactivity timeout), so you never touch the keyboard.
    """

    def __init__(self, transcriber, inserter, cues, hud, clean_model,
                 wake, stop_phrase, inactivity=45.0, level_sink=None):
        import sounddevice as sd
        self.transcriber = transcriber
        self.inserter = inserter
        self.cues = cues
        self.hud = hud
        self.clean_model = clean_model
        self.wake = wake
        self.stop_phrase = stop_phrase
        self.inactivity = inactivity
        self.level_sink = level_sink       # HUD waveform reads .level from here

        self.dictating = False
        self._last_activity = time.time()
        self._q = queue.Queue()
        self._active = False
        self._buf = []
        self._preroll = deque()
        self._preroll_n = 0
        self._below = 0

        self._sd = sd
        self._stream = None
        self._open_stream()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._device_watch, daemon=True).start()
        if inactivity:
            threading.Thread(target=self._idle_watch, daemon=True).start()

    def _open_stream(self):
        self._active = False
        self._buf = []
        self._below = 0
        self._preroll.clear()
        self._preroll_n = 0
        self._stream = self._sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=1024, callback=self._cb)
        self._stream.start()

    def _default_input_name(self):
        """Current system default audio input (reflects live hot-plugging)."""
        import os
        saved = os.dup(2)                        # mute one-time AVCapture warning
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
            from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
            d = AVCaptureDevice.defaultDeviceWithMediaType_(AVMediaTypeAudio)
            return d.localizedName() if d is not None else None
        except Exception:
            return None
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            os.close(devnull)

    def _device_watch(self):
        """Follow the system default mic — reopen on AirPods connect/remove."""
        current = self._default_input_name()
        while True:
            time.sleep(2.0)
            name = self._default_input_name()
            if name and name != current:
                current = name
                print(f"🎧 mic changed → {name}; switching …", flush=True)
                try:
                    self._stream.stop()
                    self._stream.close()
                    # refresh PortAudio's device list so the new default is seen
                    self._sd._terminate()
                    self._sd._initialize()
                    self._open_stream()
                except Exception as e:
                    print(f"  (mic switch failed: {e})", flush=True)

    # audio thread: segment speech, hand finished utterances to the queue
    def _cb(self, indata, frames, time_info, status):
        import numpy as np
        block = indata[:, 0].copy()
        level = float(np.sqrt(np.mean(block ** 2)))
        if self.level_sink is not None:
            self.level_sink.level = level
        if level > WAKE_SPEECH_RMS:
            if not self._active:
                self._active = True
                self._buf = list(self._preroll)
            self._buf.append(block)
            self._below = 0
        elif self._active:
            self._buf.append(block)
            self._below += frames
            if self._below > WAKE_HANG_SEC * SAMPLE_RATE:
                self._emit()
        else:
            self._preroll.append(block)
            self._preroll_n += frames
            while self._preroll_n > 0.3 * SAMPLE_RATE:
                self._preroll_n -= len(self._preroll.popleft())
        if self._active and sum(len(c) for c in self._buf) > WAKE_MAX_SEG * SAMPLE_RATE:
            self._emit()

    def _emit(self):
        import numpy as np
        seg = np.concatenate(self._buf) if self._buf else np.zeros(0, "float32")
        self._active = False
        self._buf = []
        self._below = 0
        if len(seg) > WAKE_MIN_SEG * SAMPLE_RATE:
            self._q.put(seg)

    def _worker(self):
        while True:
            seg = self._q.get()
            try:
                text = self.transcriber.transcribe(seg)
            except Exception as e:
                print(f"  (transcribe error: {e})", flush=True)
                continue
            if text:
                self._handle(text)

    def _handle(self, text):
        if not self.dictating:
            found, _before, after = find_phrase(text, self.wake)
            if found:
                self.dictating = True
                self._last_activity = time.time()
                self.cues.play("start")
                self.hud.listening()
                print(f"\n● wake heard — dictating (say “{self.stop_phrase}” "
                      f"to stop)", flush=True)
                if after.strip():
                    self._insert(after)
            else:
                print(f"  · (idle, ignored: {text!r})", flush=True)
        else:
            self._last_activity = time.time()
            found, before, _after = find_phrase(text, self.stop_phrase)
            if found:
                if before.strip():
                    self._insert(before)
                self.dictating = False
                self.cues.play("stop")
                self.hud.hide()
                print("  (stop heard — back to idle)", flush=True)
            else:
                self._insert(text)

    def _insert(self, text):
        out = clean_text(text, self.clean_model) if self.clean_model else text
        out = out.strip()
        if not out:
            return
        self.hud.processing()
        self.inserter.insert(out + " ")
        self.hud.listening() if self.dictating else self.hud.hide()
        print(f"▸ {out}", flush=True)

    def _idle_watch(self):
        while True:
            time.sleep(1.0)
            if (self.dictating
                    and time.time() - self._last_activity > self.inactivity):
                self.dictating = False
                self.cues.play("stop")
                self.hud.hide()
                print(f"  ({self.inactivity:.0f}s inactivity — back to idle)",
                      flush=True)


# ---------------------------------------------------------------- hotkey tap

def run_event_loop(controller, keycode, airpods_mode="auto"):
    import Quartz
    from AppKit import NSEvent

    flag_for_key = {
        KEYCODE_FN: Quartz.kCGEventFlagMaskSecondaryFn,
        KEYCODE_RCMD: Quartz.kCGEventFlagMaskCommand,
    }
    watch_flag = flag_for_key[keycode]
    taps = {}

    def add_tap(name, mask, option, cb):
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
            option, mask, cb, None)
        if tap is None:
            return None
        taps[name] = tap
        src = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetCurrent(), src, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        return tap

    # ---- keyboard tap (listen-only): Fn / right-⌘ + Esc ----------------
    def kbd_cb(proxy, type_, event, refcon):
        if type_ in (Quartz.kCGEventTapDisabledByTimeout,
                     Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(taps["kbd"], True)
            return event
        code = Quartz.CGEventGetIntegerValueField(
            event, Quartz.kCGKeyboardEventKeycode)
        if type_ == Quartz.kCGEventFlagsChanged and code == keycode:
            down = bool(Quartz.CGEventGetFlags(event) & watch_flag)
            controller.on_key_down() if down else controller.on_key_up()
        elif type_ == Quartz.kCGEventKeyDown and code == KEYCODE_ESC:
            controller.on_cancel()
        return event

    kbd_mask = (Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
                | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown))
    if add_tap("kbd", kbd_mask,
               Quartz.kCGEventTapOptionListenOnly, kbd_cb) is None:
        sys.exit(
            "\nERROR: could not create the keyboard event tap.\n"
            "Grant *Input Monitoring* to your terminal app:\n"
            "  System Settings > Privacy & Security > Input Monitoring\n"
            "then fully quit and reopen the terminal and run this again.")

    # ---- media-key tap (active) for the AirPods stem press --------------
    # Separate active tap so the keyboard/typing path stays listen-only.
    state = {"enabled": airpods_mode == "on"
             or (airpods_mode == "auto" and airpods_connected())}
    if airpods_mode != "off":
        def media_cb(proxy, type_, event, refcon):
            if type_ in (Quartz.kCGEventTapDisabledByTimeout,
                         Quartz.kCGEventTapDisabledByUserInput):
                Quartz.CGEventTapEnable(taps["media"], True)
                return event
            if type_ == EVENT_SYSTEM_DEFINED and state["enabled"]:
                ns = NSEvent.eventWithCGEvent_(event)
                if ns is not None and ns.subtype() == 8:   # media-key subtype
                    data1 = ns.data1()
                    key = (data1 & 0xFFFF0000) >> 16
                    down = ((data1 & 0xFF00) >> 8) == 0x0A
                    if key == NX_KEYTYPE_PLAY:
                        if down:
                            controller.toggle_handsfree()
                        return None                        # swallow the press
            return event

        if add_tap("media", Quartz.CGEventMaskBit(EVENT_SYSTEM_DEFINED),
                   Quartz.kCGEventTapOptionDefault, media_cb) is None:
            print("NOTE: couldn't create the media-key tap; AirPods stem "
                  "control is unavailable this session.", flush=True)
        elif airpods_mode == "auto":
            def poll_airpods():
                while True:
                    time.sleep(4)
                    now = airpods_connected()
                    if now != state["enabled"]:
                        state["enabled"] = now
                        print("🎧 AirPods connected — squeeze a stem to "
                              "dictate." if now else
                              "🎧 AirPods disconnected — stem control off.",
                              flush=True)
            threading.Thread(target=poll_airpods, daemon=True).start()
    # Flash the HUD at startup: visual confirmation that the overlay works
    # and shows where it will appear.
    key_name = "Fn" if keycode == KEYCODE_FN else "right-⌘"
    controller.hud.flash(f"hold {key_name} to dictate", seconds=2.2)
    # NSApplication's loop (not bare CFRunLoopRun) so the HUD panel renders
    from AppKit import NSApplication
    NSApplication.sharedApplication().run()


# ---------------------------------------------------------------- permissions

_SETTINGS_PANE = {
    "microphone": "Privacy_Microphone",
    "input monitoring": "Privacy_ListenEvent",
    "accessibility": "Privacy_Accessibility",
}


def _perm_status():
    """Return {permission: granted_bool} for the three TCC permissions."""
    import ctypes
    from ApplicationServices import AXIsProcessTrusted
    from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
    iokit = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/IOKit.framework/IOKit")
    iokit.IOHIDCheckAccess.restype = ctypes.c_int
    return {
        "microphone":
            AVCaptureDevice.authorizationStatusForMediaType_(
                AVMediaTypeAudio) == 3,          # AVAuthorizationStatusAuthorized
        "input monitoring":
            iokit.IOHIDCheckAccess(1) == 0,      # ListenEvent -> Granted
        "accessibility": bool(AXIsProcessTrusted()),
    }


def _fire_prompts(missing):
    """Trigger the system permission dialogs for every missing permission."""
    import ctypes
    if "microphone" in missing:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeAudio, lambda ok: None)
    if "accessibility" in missing:
        from ApplicationServices import (AXIsProcessTrustedWithOptions,
                                         kAXTrustedCheckOptionPrompt)
        AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
    if "input monitoring" in missing:
        iokit = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/IOKit.framework/IOKit")
        iokit.IOHIDRequestAccess.restype = ctypes.c_bool
        iokit.IOHIDRequestAccess(1)


def ensure_permissions(needed=("microphone", "input monitoring",
                               "accessibility")):
    """Surface the needed permissions at startup and wait until granted.

    Fires all system prompts immediately; for toggles the user must flip
    (Accessibility / Input Monitoring the dialogs deep-link into System
    Settings). Polls live and auto-restarts the script once everything is
    granted, since a fresh Input Monitoring grant only applies to a new
    process.
    """
    import os
    status = {k: v for k, v in _perm_status().items() if k in needed}
    missing = [k for k, ok in status.items() if not ok]
    if not missing:
        return

    needed_restart = "input monitoring" in missing
    print("── permission setup ─────────────────────────────────")
    print("This app needs: " + ", ".join(needed) + ".")
    print("Approve the dialogs that just appeared; for Accessibility / "
          "Input Monitoring flip the toggle next to your terminal app "
          "in the System Settings window they open.\n")
    _fire_prompts(missing)
    time.sleep(1.0)

    opened_pane = None
    spinner = 0
    while True:
        status = {k: v for k, v in _perm_status().items() if k in needed}
        missing = [k for k, ok in status.items() if not ok]
        line = "  ".join(
            f"{'✅' if ok else '❌'} {name}" for name, ok in status.items())
        print(f"\r{line}   {'⏳' if spinner % 2 else '…'} ",
              end="", flush=True)
        spinner += 1
        if not missing:
            print("\nall permissions granted ✔")
            break
        # If a dialog was dismissed / previously denied, open the exact
        # System Settings pane for the first still-missing permission.
        if spinner >= 6 and opened_pane != missing[0]:
            opened_pane = missing[0]
            subprocess.run(
                ["open", "x-apple.systempreferences:com.apple.preference"
                 f".security?{_SETTINGS_PANE[missing[0]]}"], check=False)
        time.sleep(1.0)

    if needed_restart:
        print("restarting to pick up the Input Monitoring grant ...\n")
        os.execv(sys.executable, [sys.executable] + sys.argv)


def preflight(keycode):
    if keycode == KEYCODE_FN:
        try:
            out = subprocess.run(
                ["defaults", "read", "com.apple.HIToolbox", "AppleFnUsageType"],
                capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip() not in ("0", ""):
                print('NOTE: the 🌐/Fn key is bound to a system action. Set '
                      '"Press 🌐 key to" = "Do Nothing" in System Settings > '
                      "Keyboard to avoid the emoji picker / Apple dictation "
                      "popping up.", flush=True)
        except Exception:
            pass


def run_wake_mode(args):
    """Always-listen, no-keys dictation gated by spoken wake/stop phrases."""
    import types
    ensure_permissions(needed=("microphone", "accessibility"))
    transcriber = Transcriber(args.model)
    inserter = Inserter()
    cues = Cues(enabled=not args.no_sound)
    level = types.SimpleNamespace(level=0.0)
    hud = NoHUD() if args.no_hud else HUD(level)
    clean_model = None if args.raw else args.clean_model
    if clean_model and not cleanup_available(clean_model):
        print(f"NOTE: cleanup model '{clean_model}' isn't reachable via "
              f"Ollama — inserting raw transcripts this session.", flush=True)
        clean_model = None

    WakeListener(transcriber, inserter, cues, hud, clean_model,
                 args.wake, args.stop_phrase, args.wake_timeout,
                 level_sink=level)

    mode = f"clean ({clean_model})" if clean_model else "raw"
    print(f"\nlistening [{mode}] — say “{args.wake}” to start dictating, "
          f"“{args.stop_phrase}” to stop.\n        Text lands in whatever "
          f"app is focused. Ctrl-C quits.", flush=True)
    hud.flash(f"say “{args.wake}” to dictate", seconds=3.0)
    from AppKit import NSApplication
    NSApplication.sharedApplication().run()


def hud_test():
    """Show the HUD with a synthetic waveform and print WindowServer facts."""
    import math
    import os

    class FakeRecorder:
        level = 0.0

    rec = FakeRecorder()
    hud = HUD(rec)
    hud.flash("hold Fn to dictate", seconds=2.0)

    import Quartz
    from Foundation import NSRunLoop, NSDate
    t0 = time.time()
    reported = False
    while time.time() - t0 < 8.0:
        rec.level = abs(math.sin((time.time() - t0) * 6)) * 0.1
        if 2.0 < time.time() - t0 < 5.0:
            hud.listening()          # live waveform look
        elif time.time() - t0 >= 5.0:
            hud.processing()         # ripple look
        NSRunLoop.mainRunLoop().runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(0.05))
        if not reported and time.time() - t0 > 3.0:
            reported = True
            wins = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID) or []
            mine = [w for w in wins
                    if w.get("kCGWindowOwnerPID") == os.getpid()]
            print(f"panel.isVisible(): {hud.panel.isVisible()}")
            print(f"windows this process has ON SCREEN "
                  f"(per WindowServer): {len(mine)}")
            for w in mine:
                b = w.get("kCGWindowBounds", {})
                print(f"  bounds x={b.get('X')} y={b.get('Y')} "
                      f"w={b.get('Width')} h={b.get('Height')} "
                      f"alpha={w.get('kCGWindowAlpha')} "
                      f"layer={w.get('kCGWindowLayer')}")
            from AppKit import NSScreen
            pf = hud.panel.frame()
            on_a_display = False
            for i, s in enumerate(NSScreen.screens()):
                f = s.frame()
                inside = (f.origin.x <= pf.origin.x
                          and pf.origin.x + pf.size.width
                          <= f.origin.x + f.size.width
                          and f.origin.y <= pf.origin.y
                          and pf.origin.y + pf.size.height
                          <= f.origin.y + f.size.height)
                on_a_display = on_a_display or inside
                print(f"display {i}: origin=({f.origin.x:.0f},"
                      f"{f.origin.y:.0f}) size={f.size.width:.0f}x"
                      f"{f.size.height:.0f}"
                      + ("  <- HUD is on this display" if inside else ""))
            print("panel fully inside a display: "
                  + ("YES ✅" if on_a_display else
                     "NO ❌ — this is the bug, panel is in dead space"))
    hud.hide()
    print("hud test done — a small black pill should have shown at the "
          "top-center of the display your mouse is on: a text message, "
          "then a live waveform, then a ripple.")


def main():
    # Default OS handling for Ctrl-C: the CFRunLoop (and stalled network
    # calls) never hand control back to Python, so a Python-level SIGINT
    # handler would never fire and Ctrl-C would appear dead.
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2",
                    help="MLX STT model (use ...-v3 for multilingual)")
    ap.add_argument("--key", choices=["fn", "rcmd"], default="fn",
                    help="push-to-talk key (default: fn)")
    ap.add_argument("--raw", action="store_true",
                    help="skip LLM cleanup and insert the raw transcript "
                         "(fastest; cleanup is on by default)")
    ap.add_argument("--clean-model", default="qwen2.5:7b")
    ap.add_argument("--silence", type=float, default=3.0, metavar="SEC",
                    help="hands-free: stop after this many seconds of "
                         "silence (default 3)")
    ap.add_argument("--no-hud", action="store_true",
                    help="disable the floating waveform overlay")
    ap.add_argument("--airpods", dest="airpods", action="store_true",
                    default=None,
                    help="force AirPods stem control on (default: auto — on "
                         "when AirPods are detected)")
    ap.add_argument("--no-airpods", dest="airpods", action="store_false",
                    help="disable AirPods stem control")
    ap.add_argument("--no-sound", action="store_true",
                    help="disable the audible start/stop cues")
    ap.add_argument("--wake", metavar="PHRASE",
                    help="always-listen mode: say this phrase to begin "
                         "dictating hands-free (no keys). e.g. --wake "
                         "\"start dictation\"")
    ap.add_argument("--stop-phrase", default="stop dictation",
                    help="phrase that ends dictation in --wake mode "
                         "(default: \"stop dictation\")")
    ap.add_argument("--wake-timeout", type=float, default=45.0, metavar="SEC",
                    help="in --wake mode, auto-stop dictating after this "
                         "many seconds of silence (default 45)")
    ap.add_argument("--test", metavar="AUDIO",
                    help="transcribe an audio file and exit")
    ap.add_argument("--hud-test", action="store_true",
                    help="show the HUD for 6s with a fake waveform, print "
                         "window diagnostics, and exit")
    args = ap.parse_args()

    if args.hud_test:
        hud_test()
        return

    if args.test:
        transcriber = Transcriber(args.model)
        t0 = time.time()
        text = transcriber.transcribe_file(args.test)
        t_stt = time.time() - t0
        print(f"[stt {t_stt:.2f}s] {text}")
        if not args.raw:
            t1 = time.time()
            cleaned = clean_text(text, args.clean_model)
            print(f"[clean {time.time() - t1:.2f}s] {cleaned}")
        return

    keycode = KEYCODE_FN if args.key == "fn" else KEYCODE_RCMD

    # Single-instance lock: a second copy would double-record and double-paste.
    import fcntl
    lock = open(Path(tempfile.gettempdir()) / "dictate.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another dictate.py is already running — quit it first "
                 "(Ctrl-C in its terminal), or use that one.")

    if args.wake:             # always-listen, hands-free, no keys
        run_wake_mode(args)
        return

    ensure_permissions()      # surface all TCC prompts immediately at spawn
    preflight(keycode)
    transcriber = Transcriber(args.model)
    recorder = Recorder()
    inserter = Inserter()
    hud = NoHUD() if args.no_hud else HUD(recorder)
    cues = Cues(enabled=not args.no_sound)
    clean_model = None if args.raw else args.clean_model
    if clean_model and not cleanup_available(clean_model):
        print(f"NOTE: cleanup model '{clean_model}' isn't reachable via "
              f"Ollama — inserting raw transcripts this session. Start "
              f"Ollama (`ollama serve`) and `ollama pull {clean_model}` to "
              f"enable cleanup, or pass --raw to silence this.", flush=True)
        clean_model = None
    controller = Controller(recorder, transcriber, inserter, hud,
                            clean_model, args.silence, cues)

    # AirPods stem control is OFF by default: macOS captures the AirPods Pro
    # stem squeeze for its own mic/call control and never delivers it to us
    # as a media key (confirmed by testing — you just get "Cannot Control
    # Mic" popups). --airpods still forces the attempt for experimentation.
    if args.airpods is True:
        airpods_mode = "on"
    else:
        airpods_mode = "off"

    mode = f"clean ({clean_model})" if clean_model else "raw"
    key_name = "Fn" if args.key == "fn" else "Right-Command"
    print(f"\nready [{mode}] — hold {key_name} and speak, release to insert."
          f"\n        double-tap {key_name} for hands-free (stops after "
          f"{args.silence:.0f}s of silence or another tap).", flush=True)
    if airpods_mode != "off":
        detected = airpods_mode == "on" or airpods_connected()
        print(f"        AirPods stem control: {airpods_mode}"
              + (" — AirPods detected, squeeze a stem to dictate."
                 if detected else " — will activate when AirPods connect."),
              flush=True)
    print("        Esc cancels. Ctrl-C quits.", flush=True)
    run_event_loop(controller, keycode, airpods_mode=airpods_mode)


if __name__ == "__main__":
    main()
