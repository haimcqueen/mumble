#!/usr/bin/env python3
"""
probe_stem.py: does an AirPods stem squeeze reach us as a media key?

This is the go/no-go test for "Option B". It opens NO microphone, so your
AirPods stay in normal music (A2DP) mode rather than call mode. It installs
an active media-key event tap and reports anything it sees.

Run it, make sure music is paused, then SQUEEZE an AirPod stem a few times:
  • "PLAY/PAUSE" lines  -> Option B is viable (we can use the stem to start).
  • nothing at all      -> macOS is capturing the stem; Option B won't work.

Requires Input Monitoring + Accessibility for your terminal (already granted
if dictation worked). Ctrl-C to quit.
"""

import signal
import sys
import time

import Quartz
from AppKit import NSEvent

LOG_PATH = "/tmp/stem_probe.log"
EVENT_SYSTEM_DEFINED = 14
KEY_NAMES = {
    16: "PLAY/PAUSE", 17: "NEXT", 18: "PREVIOUS",
    19: "FAST-FWD", 20: "REWIND", 7: "MUTE", 0: "SOUND-UP", 1: "SOUND-DOWN",
}


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    taps = {}
    seen = {"count": 0}

    def cb(proxy, type_, event, refcon):
        if type_ in (Quartz.kCGEventTapDisabledByTimeout,
                     Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(taps["t"], True)
            return event
        if type_ == EVENT_SYSTEM_DEFINED:
            ns = NSEvent.eventWithCGEvent_(event)
            if ns is not None and ns.subtype() == 8:
                data1 = ns.data1()
                key = (data1 & 0xFFFF0000) >> 16
                state = (data1 & 0xFF00) >> 8
                edge = "down" if state == 0x0A else "up"
                name = KEY_NAMES.get(key, f"key#{key}")
                seen["count"] += 1
                print(f"  ✅ media key: {name:12s} {edge}   "
                      f"(#{seen['count']})", flush=True)
                if key == 16 and edge == "down":
                    print("     ^ THIS is the AirPods stem squeeze. Option B "
                          "is VIABLE. Swallowing it so music won't toggle.",
                          flush=True)
                if key == 16:
                    return None          # swallow play/pause so music is quiet
        return event

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionDefault,
        Quartz.CGEventMaskBit(EVENT_SYSTEM_DEFINED), cb, None)
    if tap is None:
        sys.exit(
            "ERROR: couldn't create the media-key tap.\n"
            "Grant Input Monitoring (and Accessibility) to your terminal app, "
            "then reopen it and retry.")
    taps["t"] = tap
    src = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
    Quartz.CFRunLoopAddSource(
        Quartz.CFRunLoopGetCurrent(), src, Quartz.kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap, True)

    print("probe ready: NO microphone is open.")
    print("Pause any music, then squeeze an AirPod stem a few times.")
    print("Watching for media keys (Ctrl-C to quit)...\n", flush=True)
    Quartz.CFRunLoopRun()


if __name__ == "__main__":
    main()
