"""
Tempo clock + lookahead scheduler shared by the metronome, the step sequencer and the looper.

Everything musical is expressed in *beats*; beats are converted to absolute audio frames only
when an event is handed to the AudioEngine, a few tens of milliseconds ahead of time, so
timing is sample-accurate regardless of Python thread jitter. Tempo changes keep the current
beat position, and recorded loops follow the new tempo.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import synth

BEATS_PER_BAR = 4
LOOKAHEAD = 0.08          # seconds scheduled ahead of the audio clock
START_LATENCY = 0.05      # transport starts this far in the future so beat 0 is never late
NUM_TRACKS = 8
LOOP_BARS = (1, 2, 4, 8)


class Transport:
    def __init__(self, audio, bpm: float = 120.0):
        self.audio = audio
        self.bpm = float(bpm)
        self.playing = False
        self.metronome = False
        self.origin = 0.0     # audio frame of beat 0
        self.generation = 0   # bumped on every start, so the scheduler notices restarts
        self._taps: List[float] = []
        self.lock = threading.RLock()

    @property
    def fpb(self) -> float:
        return self.audio.samplerate * 60.0 / self.bpm

    def beat_at(self, frame: float) -> float:
        return (frame - self.origin) / self.fpb

    def frame_at(self, beat: float) -> int:
        return int(round(self.origin + beat * self.fpb))

    def beat(self) -> float:
        return self.beat_at(self.audio.now()) if self.playing else 0.0

    def start(self) -> None:
        with self.lock:
            if not self.playing:
                self.origin = self.audio.now() + START_LATENCY * self.audio.samplerate
                self.generation += 1
                self.playing = True

    def stop(self) -> None:
        with self.lock:
            self.playing = False

    def set_bpm(self, bpm: float) -> None:
        with self.lock:
            bpm = max(40.0, min(300.0, float(bpm)))
            if self.playing:  # keep the current musical position
                now = self.audio.now()
                b = self.beat_at(now)
                self.bpm = bpm
                self.origin = now - b * self.fpb
            else:
                self.bpm = bpm

    def tap(self) -> Optional[float]:
        """Tap tempo: average of the last few intervals; returns the new BPM if set."""
        t = time.perf_counter()
        self._taps = [x for x in self._taps if t - x < 2.5][-5:] + [t]
        if len(self._taps) >= 2:
            iv = np.diff(self._taps)
            self.set_bpm(round(60.0 / float(np.mean(iv)), 1))
            return self.bpm
        return None


# ======================================================================= looper
@dataclass
class LoopEvent:
    beat: float                 # offset inside the loop
    kind: str                   # "play" | "stop"
    key: str
    data: Optional[np.ndarray] = None
    gain: float = 1.0
    loop: bool = False
    release: float = 0.0


@dataclass
class Track:
    state: str = "empty"        # empty | armed | recording | playing | muted
    length: float = 0.0         # beats
    origin: float = 0.0         # beat at which the loop's beat 0 falls
    bars: int = 2               # requested length while recording
    events: List[LoopEvent] = field(default_factory=list)
    overdub: bool = False
    history: List[List[LoopEvent]] = field(default_factory=list)  # undo stack for overdubs
    _shift: Dict[str, float] = field(default_factory=dict)        # quantize shift per key

    @property
    def has_content(self) -> bool:
        return self.state in ("playing", "muted") and self.length > 0


class Looper:
    """Records what you play (on any page) as timed audio events and replays it in sync."""

    def __init__(self, transport: Transport, audio):
        self.t = transport
        self.audio = audio
        self.tracks = [Track() for _ in range(NUM_TRACKS)]
        self.bars = 2
        self.quantize = False
        self.lock = threading.RLock()
        self.is_held: Callable[[str], bool] = lambda key: False

    @staticmethod
    def prefix(i: int) -> str:
        return f"L{i}|"

    # ------------------------------------------------------------- controls
    def main(self, i: int) -> None:
        """The big pad of a track: record / finish / mute / unmute."""
        with self.lock:
            tr = self.tracks[i]
            if tr.state == "empty":
                self._arm(i)
            elif tr.state == "armed":
                tr.state = "empty"
            elif tr.state == "recording":
                self._finish(i, early=True)
            elif tr.state == "playing":
                tr.state = "muted"
                tr.overdub = False
                self.audio.stop_prefix(self.prefix(i), 0.05)
            elif tr.state == "muted":
                tr.state = "playing"
                self.t.start()

    def toggle_overdub(self, i: int) -> None:
        with self.lock:
            tr = self.tracks[i]
            if not tr.has_content:
                return
            if not tr.overdub:
                tr.history = (tr.history + [list(tr.events)])[-10:]
                tr.state = "playing"
                tr._shift.clear()
                self.t.start()
            tr.overdub = not tr.overdub

    def undo(self, i: int) -> None:
        with self.lock:
            tr = self.tracks[i]
            if tr.history:
                tr.events = tr.history.pop()
                tr.overdub = False
                self.audio.stop_prefix(self.prefix(i), 0.05)

    def clear(self, i: int) -> None:
        with self.lock:
            self.audio.stop_prefix(self.prefix(i), 0.05)
            self.tracks[i] = Track(bars=self.bars)

    def clear_all(self) -> None:
        for i in range(NUM_TRACKS):
            self.clear(i)

    def _arm(self, i: int) -> None:
        tr = self.tracks[i]
        tr.bars, tr.events, tr.history, tr.overdub = self.bars, [], [], False
        tr._shift.clear()
        tr.length = self.bars * BEATS_PER_BAR
        if not self.t.playing:
            self.t.start()
            tr.origin = 0.0
            tr.state = "recording"
        else:  # start on the next bar line
            b = self.t.beat()
            nxt = math.ceil(b / BEATS_PER_BAR - 1e-9) * BEATS_PER_BAR
            if nxt - b > BEATS_PER_BAR - 0.25:  # pressed right on the downbeat: start now
                nxt = math.floor(b / BEATS_PER_BAR) * BEATS_PER_BAR
            tr.origin = nxt
            tr.state = "armed" if nxt > b else "recording"

    def _finish(self, i: int, early: bool = False) -> None:
        tr = self.tracks[i]
        if early:
            # Round the length up to whole bars; recording continues until that bar line
            # (the scheduler then closes it), so nothing played before it is lost.
            played = max(0.0, self.t.beat() - tr.origin)
            tr.length = max(1, math.ceil(played / BEATS_PER_BAR - 1e-6)) * BEATS_PER_BAR
            return
        # close notes still held / looping samples at the loop end
        last: Dict[str, LoopEvent] = {}
        for ev in tr.events:
            last[ev.key] = ev
        for key, ev in last.items():
            if ev.kind == "play" and (ev.loop or self.is_held(key)):
                tr.events.append(LoopEvent(tr.length - 1e-3, "stop", key, release=0.05))
        tr.events = [e for e in tr.events if e.beat < tr.length]
        tr.events.sort(key=lambda e: e.beat)
        tr.state = "playing" if tr.events else "empty"

    # ------------------------------------------------------------- capture
    def capture(self, kind: str, key: str, data=None, gain=1.0, loop=False, release=0.0) -> None:
        """Called for every live play/stop. Adds it to recording / overdubbing tracks."""
        if not self.t.playing:
            return
        b = self.t.beat()
        with self.lock:
            for tr in self.tracks:
                if tr.state in ("recording", "armed"):
                    off = b - tr.origin
                    if off < -0.125:      # too early for an armed track
                        continue
                    off = max(0.0, off)
                    if off >= tr.length:
                        continue
                elif tr.overdub and tr.has_content:
                    off = (b - tr.origin) % tr.length
                else:
                    continue
                if self.quantize:
                    if kind == "play":
                        q = round(off * 4) / 4
                        tr._shift[key] = q - off
                        off = q % (tr.length if tr.has_content else 1e9)
                    else:
                        off = max(0.0, off + tr._shift.get(key, 0.0))
                tr.events.append(LoopEvent(off, kind, key, data, gain, loop, release))
                if tr.has_content:
                    tr.events.sort(key=lambda e: e.beat)

    # ------------------------------------------------------------- scheduling
    def schedule(self, b0: float, b1: float) -> None:
        with self.lock:
            for i, tr in enumerate(self.tracks):
                if tr.state == "armed" and b1 > tr.origin:
                    tr.state = "recording"
                if tr.state == "recording" and self.t.beat() >= tr.origin + tr.length:
                    self._finish(i)
                if tr.state not in ("playing", "recording") or tr.length <= 0:
                    continue
                # A track still recording already schedules its second cycle (which starts
                # inside the lookahead window) from the events recorded so far.
                first = tr.origin + tr.length if tr.state == "recording" else -math.inf
                pre = self.prefix(i)
                for ev in list(tr.events):
                    if ev.beat >= tr.length:
                        continue
                    k = math.ceil((b0 - tr.origin - ev.beat) / tr.length - 1e-12)
                    tb = tr.origin + ev.beat + k * tr.length
                    while tb < b1:
                        if tb >= b0 and tb >= first:
                            at = self.t.frame_at(tb)
                            if ev.kind == "play":
                                self.audio.play_data(pre + ev.key, ev.data, ev.gain, ev.loop, at=at)
                            else:
                                self.audio.stop(pre + ev.key, ev.release, at=at)
                        tb += tr.length

    def on_stop(self) -> None:
        """Transport stopped: silence loops and cancel recordings in progress."""
        self.audio.stop_prefix("L", 0.05)
        with self.lock:
            for tr in self.tracks:
                tr.overdub = False
                if tr.state in ("armed", "recording"):
                    tr.state, tr.events = "empty", []

    def reset_alignment(self) -> None:
        """Transport (re)started: every loop starts again from its beginning at beat 0."""
        with self.lock:
            for tr in self.tracks:
                if tr.has_content:
                    tr.origin = 0.0

    def progress(self, i: int) -> float:
        tr = self.tracks[i]
        if not self.t.playing:
            return 0.0
        b = self.t.beat()
        if tr.state == "recording":
            return max(0.0, min(1.0, (b - tr.origin) / tr.length))
        if tr.has_content:
            return ((b - tr.origin) % tr.length) / tr.length
        return 0.0

    # ------------------------------------------------------------- bounce
    def bounce(self, i: int) -> Tuple[np.ndarray, int]:
        """Mix a loop down to a seamless stereo buffer (exactly one loop long)."""
        sr = self.audio.samplerate
        with self.lock:
            tr = self.tracks[i]
            if not tr.has_content:
                raise ValueError("track is empty")
            evs, L = list(tr.events), tr.length
        n = int(round(L * self.t.fpb))
        out = np.zeros((n, 2), dtype=np.float64)
        for j, ev in enumerate(evs):
            if ev.kind != "play" or ev.data is None:
                continue
            start = int(ev.beat * self.t.fpb)
            # the matching stop: next stop of this key after the play (wrapping around)
            dur, rel = None, 0.0
            later = [e for e in evs[j + 1:] if e.key == ev.key] + [e for e in evs[:j] if e.key == ev.key]
            for e in later:
                if e.kind == "stop":
                    dur = int(((e.beat - ev.beat) % L) * self.t.fpb)
                    rel = e.release
                    break
                if e.kind == "play":
                    dur = int(((e.beat - ev.beat) % L or L) * self.t.fpb)
                    rel = 0.006
                    break
            d = ev.data if ev.data.ndim == 2 else np.repeat(ev.data[:, None], 2, axis=1)
            if ev.loop:
                total = dur if dur is not None else n
                d = np.tile(d, (total // max(1, len(d)) + 1, 1))[:total]
            if dur is not None:
                fade = max(64, int(rel * sr))
                d = d[: dur + fade].astype(np.float64)
                k = min(fade, max(0, len(d) - dur))
                if k:
                    d[dur:dur + k] *= np.linspace(1, 0, k)[:, None]
            seg = d * ev.gain
            pos = 0
            while pos < len(seg):  # wrap the tail around to the loop start
                at = (start + pos) % n
                m = min(len(seg) - pos, n - at)
                out[at:at + m] += seg[pos:pos + m]
                pos += m
        peak = np.max(np.abs(out)) or 1.0
        if peak > 0.99:
            out *= 0.99 / peak
        return out.astype(np.float32), sr

    def snapshot(self) -> list:
        return [{"state": t.state, "bars": round(t.length / BEATS_PER_BAR, 2) if t.length else t.bars,
                 "events": sum(1 for e in t.events if e.kind == "play"), "overdub": t.overdub,
                 "undo": bool(t.history), "progress": round(self.progress(i), 3)}
                for i, t in enumerate(self.tracks)]


# ======================================================================= scheduler
class Scheduler:
    """Background thread that feeds the AudioEngine ahead of time."""

    def __init__(self, transport: Transport, looper: Looper,
                 seq_cb: Callable[[int, int], None], step_cb: Callable[[], None]):
        self.t, self.looper = transport, looper
        self.seq_cb = seq_cb          # seq_cb(step16, frame): schedule sequencer step
        self.step_cb = step_cb        # visual tick (LED refresh)
        self._done = 0.0              # beats already scheduled
        self._was_playing = False
        self._run = True
        self._click = {}
        self._gen = -1
        self._step = None

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="scheduler").start()

    def _loop(self) -> None:
        while self._run:
            try:
                self._tick()
            except Exception as e:  # never let the clock die
                print(f"[scheduler] {e!r}")
            time.sleep(0.004)

    def _tick(self) -> None:
        t = self.t
        # Only read the transport under its lock; never call out while holding it.
        with t.lock:
            playing, gen = t.playing, t.generation
            horizon = t.beat_at(t.audio.now() + LOOKAHEAD * t.audio.samplerate) if playing else 0.0
        if not playing:
            if self._was_playing:
                self._was_playing = False
                t.audio.cancel_future()
                self.looper.on_stop()
                self._step = None
                self.step_cb()
            return
        if not self._was_playing or gen != self._gen:
            self._was_playing, self._gen, self._done = True, gen, -1e-9
            self.looper.reset_alignment()
        b0, b1 = self._done, horizon
        if b1 > b0:
            if t.metronome:
                sr = t.audio.samplerate
                for beat in range(math.ceil(b0), math.ceil(b1)):
                    if beat < b0:
                        continue
                    acc = beat % BEATS_PER_BAR == 0
                    if (sr, acc) not in self._click:
                        self._click[(sr, acc)] = synth.click(sr, acc)
                    t.audio.play_data("__click__", self._click[(sr, acc)], 1.0, at=t.frame_at(beat))
            for s in range(math.ceil(b0 * 4 - 1e-9), math.ceil(b1 * 4 - 1e-9)):
                if s / 4 >= b0:
                    self.seq_cb(s, s / 4)  # sequencer: 16th-note steps
            self.looper.schedule(b0, b1)
            self._done = b1
        step = math.floor(t.beat() * 4)
        if step != self._step:  # visual tick for LEDs / GUI
            self._step = step
            self.step_cb()
