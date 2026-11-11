"""Soundboard logic: maps Launchpad buttons -> sounds + LED feedback, and serves the GUI."""
from __future__ import annotations

import copy
import json
import math
import os
import queue
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import soundfile as sf

from lpdriver import BRIGHTNESS_LEVELS, Button, Color, Launchpad, OFF, all_buttons
from . import fx, synth
from .audio import AUDIO_EXTS, AudioEngine
from .transport import BEATS_PER_BAR, LOOP_BARS, NUM_TRACKS, Looper, Scheduler, Transport

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SOUNDS = os.path.join(ROOT, "sounds")
CONFIG = os.path.join(ROOT, "config.json")
NUM_PAGES = 8
PAGE_TYPES = ("soundboard", "piano", "drums", "sequencer", "looper")
INSTRUMENT_TYPES = ("piano", "drums", "sequencer")
# Where sound comes out: the machine running this program, the browser showing the GUI, or both.
# macOS is usually run on the laptop the Launchpad is plugged into; Linux often on a headless box.
DEFAULT_OUTPUT = "host" if sys.platform == "darwin" else "browser"

# Instrument page settings ("inst" key of a page)
DEFAULT_INST = {
    "patch": "keys",        # see synth.PATCHES
    "slots": list(synth.PATCHES)[:6],  # patches on side buttons 3-8
    "layout": "piano",      # piano (white/black key rows) | scale (in-key grid, rows a 4th apart)
    "root": 0,              # 0 = C .. 11 = B
    "scale": "minor",       # see synth.SCALES
    "octave": 2,            # lowest octave shown (C2 = MIDI 36)
    "volume": 0.8,
    "kit": "808",           # see synth.KITS
    "swing": 0.0,           # sequencer swing 0 .. 0.5 (fraction of a 16th the off-beats are late)
    "fx": {},               # see fx.DEFAULT_FX
    "colors": {"key": [0, 1], "black": [1, 0], "root": [0, 3], "pressed": [3, 3]},
}
# Step sequencer ("seq" key of a sequencer page). Rows top -> bottom.
DEFAULT_SEQ = {
    "length": 16,
    "rows": [12, 10, 8, 5, 4, 3, 2, 0],  # Rim, Hi tom, Lo tom, Open hat, Closed hat, Clap, Snare, Kick
    "pattern": ["0" * 32] * 8,
}
SEQ_LENGTHS = (8, 16, 32)
PIANO_WHITE = [0, 2, 4, 5, 7, 9, 11, 12]
PIANO_BLACK = [None, 1, 3, None, 6, 8, 10, None]  # black key sits above the white key to its right
BLACK_KEYS = {1, 3, 6, 8, 10}

DEFAULT_BUTTON = {
    "action": "sound",      # sound | stop_all | stop_page
    "sound": "",
    "label": "",
    "mode": "oneshot",      # oneshot (retrigger) | hold (plays while held) | toggle
    "loop": False,
    "volume": 1.0,
    "idle": [0, 1],         # [red 0-3, green 0-3]
    "pressed": [3, 3],
    "playing": [0, 3],
    "flash": True,          # flash the LED while the sound is playing
    "fx": {},               # pitch / distortion / ... see fx.DEFAULT_FX
}


def default_config() -> dict:
    pages = [{"name": f"Page {i + 1}", "buttons": {}} for i in range(NUM_PAGES)]
    palette = [[0, 1], [1, 0], [1, 1], [0, 2], [2, 0], [1, 2], [2, 1], [2, 2]]
    files = sorted(f for f in os.listdir(SOUNDS) if os.path.splitext(f)[1].lower() in AUDIO_EXTS) \
        if os.path.isdir(SOUNDS) else []
    for i, f in enumerate(files[:64]):
        b = dict(DEFAULT_BUTTON, sound=f, label=os.path.splitext(f)[0], idle=palette[i % 8])
        pages[0]["buttons"][f"pad{i % 8}{i // 8}"] = b
    pages[0]["buttons"]["side7"] = dict(DEFAULT_BUTTON, action="stop_all", label="STOP ALL",
                                        idle=[1, 0], pressed=[3, 0])
    cfg = {"version": 2, "page": 0, "brightness": 5, "master_volume": 0.9, "pages": pages,
           "bpm": 120.0, "metronome": False, "loop_bars": 2, "quantize": False}
    migrate(cfg)
    return cfg


def migrate(cfg: dict) -> bool:
    """Add page types; seed empty pages with the instruments the first time each kind exists."""
    changed = False
    for p in cfg["pages"]:
        if p.get("type") not in PAGE_TYPES:
            p["type"] = "soundboard"
            changed = True

    def seed(flag, plan):
        nonlocal changed
        if cfg.get(flag):
            return
        for i, t, name in plan:
            p = cfg["pages"][i]
            if p["type"] == "soundboard" and not p["buttons"]:
                p["type"] = t
                if p["name"] == f"Page {i + 1}":
                    p["name"] = name
        cfg[flag] = True
        changed = True

    seed("instruments_seeded", ((6, "piano", "Piano"), (7, "drums", "Drums")))
    seed("v2_seeded", ((4, "sequencer", "Beats"), (5, "looper", "Looper")))
    for k, v in (("bpm", 120.0), ("metronome", False), ("loop_bars", 2), ("quantize", False),
                 ("audio_output", DEFAULT_OUTPUT)):
        if k not in cfg:
            cfg[k] = v
            changed = True
    return changed


class Soundboard:
    def __init__(self):
        os.makedirs(SOUNDS, exist_ok=True)
        self.cfg = self._load_config()
        self.lock = threading.RLock()
        self.cond = threading.Condition()
        self.version = 0
        self.last_press: Optional[str] = None
        self.gui_held: set = set()
        out = self.cfg.get("audio_output", DEFAULT_OUTPUT)
        self.audio = AudioEngine(output=out if out in AudioEngine.OUTPUTS else DEFAULT_OUTPUT)
        self.audio.master = self.cfg["master_volume"]
        self.fxcache = fx.FxCache()
        n, d = BRIGHTNESS_LEVELS[self.cfg["brightness"]]
        self.lp = Launchpad(brightness=(n, d))
        self.lp.on_button(self._on_hw_button)
        self.lp.on_connection(self._on_connection)
        self._last_playing: set = set()
        self._desired: Dict[Button, Tuple[Color, bool]] = {}
        self.active: Dict[str, Tuple[str, float]] = {}  # button id -> (voice key, release s)
        self.held_notes: Dict[str, int] = {}            # button id -> MIDI note being played
        self._warm_q: "queue.Queue" = queue.Queue()
        self.seq_view: Dict[int, int] = {}              # sequencer page -> visible 8-step block
        # tempo, looper, sequencer
        self.transport = Transport(self.audio, self.cfg["bpm"])
        self.transport.metronome = bool(self.cfg["metronome"])
        self.looper = Looper(self.transport, self.audio)
        self.looper.bars = self.cfg["loop_bars"] if self.cfg["loop_bars"] in LOOP_BARS else 2
        self.looper.quantize = bool(self.cfg["quantize"])
        self.looper.is_held = lambda key: any(k == key for k, _ in list(self.active.values()))
        self.scheduler = Scheduler(self.transport, self.looper, self._seq_step, self._tick_refresh)
        self._seq_pages: tuple = ()
        self._update_seq_cache()

    # ------------------------------------------------------------- config
    def _load_config(self) -> dict:
        if os.path.exists(CONFIG):
            try:
                with open(CONFIG) as f:
                    cfg = json.load(f)
                base = default_config()
                base.update(cfg)
                base["instruments_seeded"] = cfg.get("instruments_seeded", False)
                base["v2_seeded"] = cfg.get("v2_seeded", False)
                while len(base["pages"]) < NUM_PAGES:
                    base["pages"].append({"name": f"Page {len(base['pages']) + 1}", "buttons": {}})
                if migrate(base):
                    self._write(base)
                return base
            except (OSError, ValueError) as e:
                print("Config unreadable, using defaults:", e)
        cfg = default_config()
        self._write(cfg)
        return cfg

    def save(self) -> None:
        with self.lock:
            self._write(self.cfg)
            self._update_seq_cache()

    @staticmethod
    def _write(cfg: dict) -> None:
        tmp = CONFIG + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=1)
        os.replace(tmp, CONFIG)

    def button_cfg(self, page: int, bid: str) -> Optional[dict]:
        b = self.cfg["pages"][page]["buttons"].get(bid)
        return dict(DEFAULT_BUTTON, **b) if b else None

    def sounds(self):
        return sorted(f for f in os.listdir(SOUNDS)
                      if os.path.splitext(f)[1].lower() in AUDIO_EXTS and not f.startswith("."))

    # ------------------------------------------------------------- audio helpers
    def _play(self, key: str, data, gain: float = 1.0, loop: bool = False) -> None:
        """Play something live; the looper records it if a track is recording."""
        self.audio.play_data(key, data, gain, loop)
        self.looper.capture("play", key, data, gain, loop)

    def _stop(self, key: str, release: float = 0.0) -> None:
        self.audio.stop(key, release)
        self.looper.capture("stop", key, release=release)

    def sample(self, path: str, fxs: Optional[dict] = None):
        """A sound file with its effects applied (cached)."""
        data = self.audio.load(path)
        if not fxs or fx.is_neutral(fxs):
            return data
        fxs = fx.clean(fxs)
        return self.fxcache.get((path, fx.key_of(fxs)),
                                lambda: fx.apply(data, self.audio.samplerate, fxs))

    # ------------------------------------------------------------- instruments
    def page_type(self, page: int) -> str:
        return self.cfg["pages"][page].get("type", "soundboard")

    def inst_cfg(self, page: int) -> dict:
        raw = self.cfg["pages"][page].get("inst", {})
        inst = dict(DEFAULT_INST, **raw)
        inst["colors"] = dict(DEFAULT_INST["colors"], **raw.get("colors", {}))
        inst["fx"] = fx.clean(raw.get("fx", {}))
        slots = [s for s in inst["slots"] if s in synth.PATCHES][:6]
        inst["slots"] = slots + [s for s in DEFAULT_INST["slots"] if s not in slots][: 6 - len(slots)]
        return inst

    def seq_cfg(self, page: int) -> dict:
        raw = self.cfg["pages"][page].get("seq", {})
        seq = dict(DEFAULT_SEQ, **raw)
        rows = [int(r) for r in seq["rows"]][:8]
        seq["rows"] = rows + DEFAULT_SEQ["rows"][len(rows):]
        pat = [str(r).ljust(32, "0")[:32] for r in seq["pattern"]][:8]
        seq["pattern"] = pat + ["0" * 32] * (8 - len(pat))
        if seq["length"] not in SEQ_LENGTHS:
            seq["length"] = 16
        return seq

    def _update_seq_cache(self) -> None:
        """Immutable snapshot of every sequencer page for the scheduler thread (lock-free)."""
        out = []
        for p in range(NUM_PAGES):
            if self.page_type(p) == "sequencer":
                inst, seq = self.inst_cfg(p), self.seq_cfg(p)
                out.append((p, seq["length"], tuple(seq["rows"]), tuple(seq["pattern"]),
                            inst["kit"], inst["fx"], inst["volume"], inst["swing"]))
        self._seq_pages = tuple(out)

    def instrument_map(self, page: int) -> Dict[str, tuple]:
        """What every button does on an instrument / looper page.

        Piano:     ("note", midi, is_black_key); side 1/2 = octave up/down; side 3-8 = sound slots.
        Drums:     ("drum", index) as 16 big 2x2 pads; side 1-5 = kit; side 8 = stop.
        Sequencer: ("step", row, step); side 1-4 = step block; 5 play; 6 kit; 7 length; 8 clear.
        Looper:    rows = tracks: cols 1-6 record/mute, col 7 overdub, col 8 clear;
                   side: play, metronome, tempo -, tempo +, tap, loop length, quantize, clear all.
        """
        t = self.page_type(page)
        m: Dict[str, tuple] = {}
        if t == "piano":
            inst = self.inst_cfg(page)
            base = 12 * (inst["octave"] + 1)
            scale = synth.SCALES.get(inst["scale"], synth.SCALES["major"])
            for y in range(8):
                row = 7 - y  # count rows from the bottom, like a keyboard
                for x in range(8):
                    if inst["layout"] == "scale":
                        if inst["scale"] == "chromatic":
                            midi = base + inst["root"] + x + 5 * row
                        else:
                            deg = x + 3 * row  # each row starts a 4th higher
                            midi = base + inst["root"] + 12 * (deg // len(scale)) + scale[deg % len(scale)]
                        black = midi % 12 in BLACK_KEYS and inst["scale"] == "chromatic"
                    else:
                        off = (PIANO_WHITE if row % 2 == 0 else PIANO_BLACK)[x]
                        if off is None:
                            continue
                        midi, black = base + 12 * (row // 2) + off, row % 2 == 1
                    if 12 <= midi <= 120:
                        m[f"pad{x}{y}"] = ("note", midi, black)
            m["side0"], m["side1"] = ("oct", 1), ("oct", -1)
            for i, pid in enumerate(inst["slots"]):
                m[f"side{2 + i}"] = ("patch", pid)
        elif t == "drums":
            for y in range(8):
                for x in range(8):
                    m[f"pad{x}{y}"] = ("drum", (3 - y // 2) * 4 + x // 2)
            for i, k in enumerate(synth.KITS):
                m[f"side{i}"] = ("kit", k)
            m["side7"] = ("stop",)
        elif t == "sequencer":
            view = self.seq_view.get(page, 0)
            for y in range(8):
                for x in range(8):
                    m[f"pad{x}{y}"] = ("step", y, view * 8 + x)
            for i in range(4):
                m[f"side{i}"] = ("view", i)
            m["side4"], m["side5"], m["side6"], m["side7"] = ("play",), ("kitnext",), ("len",), ("clear",)
        elif t == "looper":
            for y in range(8):
                for x in range(8):
                    m[f"pad{x}{y}"] = ("track", y) if x < 6 else ("dub", y) if x == 6 else ("tclear", y)
            for i, a in enumerate((("play",), ("metro",), ("tempo", -1), ("tempo", 1), ("tap",),
                                   ("bars",), ("quant",), ("clearall",))):
                m[f"side{i}"] = a
        return m

    def set_inst(self, page: int, **changes) -> None:
        with self.lock:
            pg = self.cfg["pages"][page]
            inst = pg.setdefault("inst", {})
            valid = {"patch": lambda v: v in synth.PATCHES, "kit": lambda v: v in synth.KITS,
                     "layout": lambda v: v in ("piano", "scale"), "scale": lambda v: v in synth.SCALES,
                     "root": lambda v: isinstance(v, int) and 0 <= v <= 11,
                     "octave": lambda v: isinstance(v, int),
                     "volume": lambda v: isinstance(v, (int, float)) and 0 <= v <= 1.5,
                     "swing": lambda v: isinstance(v, (int, float)) and 0 <= v <= 0.5,
                     "slots": lambda v: isinstance(v, list) and len(v) == 6
                     and all(s in synth.PATCHES for s in v)}
            for k, v in changes.items():
                if k == "colors" and isinstance(v, dict):
                    inst.setdefault("colors", {}).update(
                        {c: [max(0, min(3, int(x))) for x in rgb][:2] for c, rgb in v.items()
                         if c in DEFAULT_INST["colors"]})
                elif k == "fx" and isinstance(v, dict):
                    merged = dict(inst.get("fx", {}), **v)
                    inst["fx"] = {a: b for a, b in fx.clean(merged).items() if b != fx.DEFAULT_FX[a]}
                elif k in valid and valid[k](v):
                    inst[k] = v
            inst["octave"] = max(0, min(5, int(self.inst_cfg(page)["octave"])))
            self.save()
        self.prewarm(page)

    def set_seq(self, page: int, **changes) -> None:
        with self.lock:
            seq = self.cfg["pages"][page].setdefault("seq", {})
            cur = self.seq_cfg(page)
            if changes.get("length") in SEQ_LENGTHS:
                seq["length"] = changes["length"]
                if self.seq_view.get(page, 0) * 8 >= seq["length"]:
                    self.seq_view[page] = 0
            if isinstance(changes.get("rows"), dict):
                rows = list(cur["rows"])
                for y, d in changes["rows"].items():
                    if 0 <= int(y) < 8 and isinstance(d, int) and 0 <= d < len(synth.DRUMS):
                        rows[int(y)] = d
                seq["rows"] = rows
            if changes.get("clear"):
                seq["pattern"] = ["0" * 32] * 8
            if isinstance(changes.get("pattern"), list) and len(changes["pattern"]) == 8:
                seq["pattern"] = ["".join("1" if c == "1" else "0" for c in str(r))[:32].ljust(32, "0")
                                  for r in changes["pattern"]]
            if isinstance(changes.get("toggle"), list):
                y, s = (int(v) for v in changes["toggle"])
                pat = list(cur["pattern"])
                row = list(pat[y])
                row[s] = "0" if row[s] == "1" else "1"
                pat[y] = "".join(row)
                seq["pattern"] = pat
            self.save()
        self.prewarm(page)

    def prewarm(self, page: int) -> None:
        """Render the notes/drums of a page in the background so the first hit is instant."""
        self._warm_q.put(page)

    def _warm_loop(self) -> None:
        sr = self.audio.samplerate
        while True:
            page = self._warm_q.get()
            with self.lock:
                t, inst, m = self.page_type(page), self.inst_cfg(page), self.instrument_map(page)
            try:
                if t == "piano":
                    for midi in sorted({a[1] for a in m.values() if a[0] == "note"}):
                        synth.render_note(inst["patch"], midi, sr, inst["fx"])
                elif t in ("drums", "sequencer"):
                    for i in range(len(synth.DRUMS)):
                        synth.render_drum_fx(inst["kit"], i, sr, inst["fx"])
            except Exception as e:
                print("prewarm failed:", e)

    def _handle_instrument(self, page: int, bid: str) -> None:
        """A button went down on an instrument / looper page."""
        act = self.instrument_map(page).get(bid)
        if not act:
            return
        inst, sr, kind = self.inst_cfg(page), self.audio.samplerate, act[0]
        if kind == "note":
            key = f"{page}:{bid}"
            self._play(key, synth.render_note(inst["patch"], act[1], sr, inst["fx"]), inst["volume"])
            self.active[bid] = (key, synth.RELEASE[inst["patch"]])
            self.held_notes[bid] = act[1]
        elif kind == "drum":
            if act[1] == synth.CLOSED_HAT:  # closed hat chokes the open hat, like real hi-hats
                self._stop(f"{page}:drum{synth.OPEN_HAT}", 0.02)
            self._play(f"{page}:drum{act[1]}",
                       synth.render_drum_fx(inst["kit"], act[1], sr, inst["fx"]), inst["volume"])
        elif kind == "oct":
            self.set_inst(page, octave=inst["octave"] + act[1])
        elif kind == "patch":
            self.set_inst(page, patch=act[1])
        elif kind == "kit":
            self.set_inst(page, kit=act[1])
        elif kind == "stop":
            self.audio.stop_all()
        # ---- sequencer
        elif kind == "step":
            if act[2] < self.seq_cfg(page)["length"]:
                self.set_seq(page, toggle=[act[1], act[2]])
                if self.seq_cfg(page)["pattern"][act[1]][act[2]] == "1" and not self.transport.playing:
                    d = self.seq_cfg(page)["rows"][act[1]]  # audition the sound
                    self.audio.play_data(f"{page}:seq{d}",
                                         synth.render_drum_fx(inst["kit"], d, sr, inst["fx"]), inst["volume"])
        elif kind == "view":
            if act[1] * 8 < self.seq_cfg(page)["length"]:
                self.seq_view[page] = act[1]
        elif kind == "kitnext":
            kits = list(synth.KITS)
            self.set_inst(page, kit=kits[(kits.index(inst["kit"]) + 1) % len(kits)])
        elif kind == "len":
            ln = self.seq_cfg(page)["length"]
            self.set_seq(page, length=SEQ_LENGTHS[(SEQ_LENGTHS.index(ln) + 1) % len(SEQ_LENGTHS)])
        elif kind == "clear":
            self.set_seq(page, clear=True)
        # ---- transport / looper
        elif kind == "play":
            self.set_transport(play=not self.transport.playing)
        elif kind == "metro":
            self.set_transport(metronome=not self.transport.metronome)
        elif kind == "tempo":
            self.set_transport(bpm=round(self.transport.bpm) + act[1])
        elif kind == "tap":
            self.set_transport(tap=True)
        elif kind == "bars":
            self.set_transport(bars=LOOP_BARS[(LOOP_BARS.index(self.looper.bars) + 1) % len(LOOP_BARS)])
        elif kind == "quant":
            self.set_transport(quantize=not self.looper.quantize)
        elif kind == "clearall":
            self.looper.clear_all()
        elif kind == "track":
            self.looper.main(act[1])
        elif kind == "dub":
            self.looper.toggle_overdub(act[1])
        elif kind == "tclear":
            self.looper.clear(act[1])

    # ------------------------------------------------------------- transport
    def set_transport(self, **c) -> None:
        t = self.transport
        if "play" in c:
            if c["play"]:
                t.start()
            else:
                t.stop()
        if "bpm" in c:
            t.set_bpm(float(c["bpm"]))
        if c.get("tap"):
            t.tap()
        if "metronome" in c:
            t.metronome = bool(c["metronome"])
        if c.get("bars") in LOOP_BARS:
            self.looper.bars = c["bars"]
        if "quantize" in c:
            self.looper.quantize = bool(c["quantize"])
        with self.lock:
            self.cfg.update(bpm=t.bpm, metronome=t.metronome, loop_bars=self.looper.bars,
                            quantize=self.looper.quantize)
            self.save()
        self.refresh()

    def _seq_step(self, step: int, beat: float) -> None:
        """Scheduler callback for every 16th note: queue the sequencer hits (runs lock-free)."""
        sr, t = self.audio.samplerate, self.transport
        for page, length, rows, pattern, kit, fxs, vol, swing in self._seq_pages:
            s = step % length
            b = beat + (swing * 0.25 if s % 2 else 0.0)
            at = t.frame_at(b)
            for y in range(8):
                if pattern[y][s] == "1":
                    d = rows[y]
                    if d == synth.CLOSED_HAT:
                        self.audio.stop(f"{page}:seq{synth.OPEN_HAT}", 0.02, at=at)
                    self.audio.play_data(f"{page}:seq{d}", synth.render_drum_fx(kit, d, sr, fxs), vol, at=at)

    def seq_step_now(self, page: int) -> Optional[int]:
        if not self.transport.playing or self.page_type(page) != "sequencer":
            return None
        b = self.transport.beat()
        return None if b < 0 else int(math.floor(b * 4)) % self.seq_cfg(page)["length"]

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.lp.start()
        threading.Thread(target=self._poll_loop, daemon=True).start()
        threading.Thread(target=self._warm_loop, daemon=True).start()
        for i in range(NUM_PAGES):
            if self.page_type(i) in INSTRUMENT_TYPES:
                self.prewarm(i)
        self.scheduler.start()
        self.refresh()

    def stop(self) -> None:
        self.transport.stop()
        self.audio.stop_all()
        self.lp.stop()
        self.audio.close()

    def _on_connection(self, connected: bool) -> None:
        print("Launchpad", "connected" if connected else "disconnected")
        if connected:
            threading.Thread(target=self._intro, daemon=True).start()
        self.bump()

    def _intro(self) -> None:
        """Startup animation, rendered tear-free with hardware double buffering."""
        def frames():
            for i in range(18):
                fr = {}
                for b in all_buttons():
                    v = math.sin((b.x + b.y + 1) * 0.7 - i * 0.6)
                    fr[b] = Color(round(1.5 + 1.5 * v), round(1.5 - 1.5 * v))
                yield fr
        self.lp.animate(frames(), fps=14)
        self.refresh(force=True)

    # ------------------------------------------------------------- input
    def _on_hw_button(self, ev) -> None:
        self.handle(ev.button.id, ev.pressed)

    def handle(self, bid: str, pressed: bool, from_gui: bool = False) -> None:
        with self.lock:
            if from_gui:
                (self.gui_held.add if pressed else self.gui_held.discard)(bid)
            if pressed:
                self.last_press = bid
            elif bid in self.active:
                # Release whatever this button started, even if the page changed meanwhile.
                key, release = self.active.pop(bid)
                self.held_notes.pop(bid, None)
                self._stop(key, release)
            if bid.startswith("top"):
                if pressed:
                    self.cfg["page"] = int(bid[3:])
                    self.save()
                self.refresh()
                return
            page = self.cfg["page"]
            if self.page_type(page) != "soundboard":
                if pressed:
                    try:
                        self._handle_instrument(page, bid)
                    except Exception as e:
                        print(f"Instrument error: {e!r}")
                self.refresh()
                return
            b = self.button_cfg(page, bid)
            key = f"{page}:{bid}"
            if b:
                if b["action"] == "stop_all" and pressed:
                    self.audio.stop_all()
                elif b["action"] == "stop_page" and pressed:
                    for k in self.audio.playing():
                        if k.startswith(f"{page}:"):
                            self._stop(k)
                elif b["action"] == "sound" and b["sound"]:
                    path = os.path.join(SOUNDS, b["sound"])
                    try:
                        if pressed:
                            if b["mode"] == "toggle" and key in self.audio.playing():
                                self._stop(key)
                            else:
                                self._play(key, self.sample(path, b["fx"]), b["volume"], b["loop"])
                                if b["mode"] == "hold":
                                    self.active[bid] = (key, 0.0)
                    except Exception as e:
                        print(f"Cannot play {path}: {e}")
            self.refresh()

    # ------------------------------------------------------------- LEDs
    def compute_leds(self) -> Dict[Button, Tuple[Color, bool]]:
        page = self.cfg["page"]
        playing = self.audio.playing()
        held = {b.id for b in self.lp.held} | self.gui_held
        ptype = self.page_type(page)
        imap = self.instrument_map(page) if ptype != "soundboard" else None
        recording = any(t.state in ("armed", "recording") or t.overdub for t in self.looper.tracks)
        looping = any(t.state == "playing" for t in self.looper.tracks) and self.transport.playing
        out: Dict[Button, Tuple[Color, bool]] = {}
        for btn in all_buttons():
            bid = btn.id
            if btn.kind == "top":
                p = btn.x
                pt = self.page_type(p)
                active = any(k.startswith(f"{p}:") for k in playing)
                if bid in held:
                    out[btn] = (Color(3, 0), False)
                elif pt == "looper" and recording:
                    out[btn] = (Color(3, 0), True)    # flashing red: looper recording
                elif p == page:
                    out[btn] = (Color(3, 3), False)
                elif active or (pt == "looper" and looping):
                    out[btn] = (Color(0, 3), True)
                elif pt == "piano":
                    out[btn] = (Color(2, 1), False)   # orange-ish: piano page
                elif pt == "drums":
                    out[btn] = (Color(1, 0), False)   # red: drum page
                elif pt == "sequencer":
                    out[btn] = (Color(1, 2), False)   # lime: sequencer
                elif pt == "looper":
                    out[btn] = (Color(1, 1), False)   # amber: looper
                elif self.cfg["pages"][p]["buttons"]:
                    out[btn] = (Color(0, 1), False)   # green: soundboard page with sounds
                else:
                    out[btn] = (OFF, False)
                continue
            if imap is not None:
                if ptype == "sequencer":
                    out[btn] = self._seq_led(page, imap, bid, held)
                elif ptype == "looper":
                    out[btn] = self._looper_led(imap, bid, held)
                else:
                    out[btn] = self._instrument_led(page, imap, bid, held)
                continue
            b = self.button_cfg(page, bid)
            if bid in held:
                c = b["pressed"] if b else [1, 1]
                out[btn] = (Color(*c), False)
            elif b and f"{page}:{bid}" in playing:
                out[btn] = (Color(*b["playing"]), bool(b["flash"]))
            elif b and (b["sound"] or b["action"] != "sound"):
                out[btn] = (Color(*b["idle"]), False)
            else:
                out[btn] = (OFF, False)
        return out

    def _instrument_led(self, page: int, m: Dict[str, tuple], bid: str, held: set) -> Tuple[Color, bool]:
        act = m.get(bid)
        if not act:
            return OFF, False
        inst = self.inst_cfg(page)
        col = inst["colors"]
        down = bid in held
        kind = act[0]
        if kind == "note":
            if down or act[1] in self.held_notes.values():  # also light duplicates of the note
                return Color(*col["pressed"]), False
            if act[1] % 12 == inst["root"]:
                return Color(*col["root"]), False
            return Color(*(col["black"] if act[2] else col["key"])), False
        if kind == "drum":
            if any(m.get(h, (None,))[0] == "drum" and m[h][1] == act[1] for h in held):
                return Color(*col["pressed"]), False  # light the whole 2x2 pad
            return Color(*synth.DRUMS[act[1]][1]), False
        if kind == "oct":
            limit = 5 if act[1] > 0 else 0
            if inst["octave"] == limit:
                return OFF, False
            return (Color(3, 3) if down else Color(1, 1)), False
        if kind in ("patch", "kit"):
            sel = inst["patch" if kind == "patch" else "kit"] == act[1]
            return (Color(0, 3) if sel else Color(0, 1) if kind == "patch" else Color(1, 0)), False
        return (Color(3, 0) if down else Color(1, 0)), False  # stop

    def _seq_led(self, page: int, m: Dict[str, tuple], bid: str, held: set) -> Tuple[Color, bool]:
        act = m.get(bid)
        seq = self.seq_cfg(page)
        cur = self.seq_step_now(page)
        view = self.seq_view.get(page, 0)
        kind = act[0]
        if kind == "step":
            y, s = act[1], act[2]
            if s >= seq["length"]:
                return OFF, False
            on = seq["pattern"][y][s] == "1"
            if s == cur:
                return (Color(3, 3) if on else Color(1, 1)), False
            if on:
                return Color(*synth.DRUMS[seq["rows"][y]][1]), False
            return (Color(0, 1) if s % 4 == 0 and y == 7 else OFF), False  # beat markers, bottom row
        if kind == "view":
            if act[1] * 8 >= seq["length"]:
                return OFF, False
            if act[1] == view:
                return Color(0, 3), False
            return (Color(2, 2) if cur is not None and cur // 8 == act[1] else Color(0, 1)), False
        if kind == "play":
            return (Color(0, 3) if self.transport.playing else Color(1, 0)), False
        if kind == "kitnext":
            return (Color(3, 3) if bid in held else Color(1, 1)), False
        if kind == "len":
            return Color(0, 1 + SEQ_LENGTHS.index(seq["length"])), False
        return (Color(3, 0) if bid in held else Color(1, 0)), False  # clear

    def _looper_led(self, m: Dict[str, tuple], bid: str, held: set) -> Tuple[Color, bool]:
        act = m.get(bid)
        kind, down = act[0], bid in held
        if kind in ("track", "dub", "tclear"):
            y = act[1]
            tr = self.looper.tracks[y]
            x = int(bid[3])
            if kind == "dub":
                if not tr.has_content:
                    return OFF, False
                return (Color(3, 0), True) if tr.overdub else (Color(2, 1), False)
            if kind == "tclear":
                return ((Color(3, 0) if down else Color(1, 0)) if tr.state != "empty" else OFF), False
            cell = int(self.looper.progress(y) * 6)
            if tr.state == "empty":
                return Color(1, 0), False
            if tr.state == "armed":
                return Color(3, 0), True
            if tr.state == "recording":
                return (Color(3, 0) if x <= cell else Color(1, 0)), False
            if tr.state == "muted":
                return Color(1, 1), False
            on = self.transport.playing and x == cell
            if tr.overdub:
                return (Color(3, 0) if on else Color(2, 1)), False
            return (Color(0, 3) if on else Color(0, 1)), False
        t = self.transport
        if kind == "play":
            return (Color(0, 3) if t.playing else Color(1, 0)), False
        if kind == "metro":
            return (Color(2, 3) if t.metronome else Color(0, 1)), False
        if kind in ("tempo", "tap"):
            if kind == "tap" and t.playing and (t.beat() % 1) < 0.25:
                return Color(3, 3), False  # blinks on the beat
            return (Color(3, 3) if down else Color(1, 1)), False
        if kind == "bars":
            return [Color(0, 1), Color(0, 2), Color(0, 3), Color(2, 3)][LOOP_BARS.index(self.looper.bars)], False
        if kind == "quant":
            return (Color(0, 3) if self.looper.quantize else Color(0, 1)), False
        return (Color(3, 0) if down else Color(1, 0)), False  # clear all

    def labels(self, page: int) -> Dict[str, str]:
        """Pad captions for the GUI on instrument / looper pages."""
        out = {}
        t = self.page_type(page)
        seq = self.seq_cfg(page) if t == "sequencer" else None
        for bid, act in self.instrument_map(page).items():
            kind = act[0]
            if kind == "note":
                out[bid] = synth.note_name(act[1])
            elif kind == "drum":
                x, y = int(bid[3]), int(bid[4])
                out[bid] = synth.DRUMS[act[1]][0] if x % 2 == 0 and y % 2 == 1 else ""
            elif kind == "oct":
                out[bid] = "Oct +" if act[1] > 0 else "Oct −"
            elif kind == "patch":
                out[bid] = synth.PATCH_SHORT[act[1]]
            elif kind == "kit":
                out[bid] = act[1]
            elif kind == "stop":
                out[bid] = "Stop"
            elif kind == "step":
                out[bid] = synth.DRUMS[seq["rows"][act[1]]][0] if bid[3] == "0" else \
                    (str(act[2] + 1) if act[1] == 7 and act[2] < seq["length"] else "")
            elif kind == "view":
                out[bid] = f"{act[1] * 8 + 1}–{act[1] * 8 + 8}" if act[1] * 8 < seq["length"] else ""
            elif kind == "track":
                if bid[3] == "0":
                    tr = self.looper.tracks[act[1]]
                    out[bid] = f"Loop {act[1] + 1} · " + {"empty": "rec", "armed": "armed", "recording": "REC",
                                                         "playing": "play", "muted": "muted"}[tr.state]
            else:
                out[bid] = {"dub": "Dub", "tclear": "Clear", "play": "Play", "metro": "Click",
                            "tempo": "BPM " + ("+" if act[-1] == 1 else "−"), "tap": "Tap",
                            "bars": f"{self.looper.bars} bar", "quant": "Quant", "clearall": "Clr all",
                            "kitnext": "Kit", "len": f"{seq['length'] if seq else ''} st",
                            "clear": "Clear"}.get(kind, "")
        return out

    def refresh(self, force: bool = False) -> None:
        with self.lock:
            desired = self.compute_leds()
            self._desired = desired
        self.lp.set_leds(desired, force=force)
        self.bump()

    def _tick_refresh(self) -> None:
        """Called by the scheduler on every 16th note: animate playheads / loop progress."""
        with self.lock:
            desired = self.compute_leds()
            changed = desired != self._desired
            self._desired = desired
        if changed:
            self.lp.set_leds(desired)
        self.bump()

    def _poll_loop(self) -> None:
        """Notice sounds that finished on their own so their LEDs go back to idle."""
        while True:
            time.sleep(0.03)
            now = self.audio.playing()
            if now != self._last_playing:
                self._last_playing = now
                self.refresh()

    # ------------------------------------------------------------- GUI state
    def bump(self) -> None:
        with self.cond:
            self.version += 1
            self.cond.notify_all()

    def snapshot(self) -> dict:
        with self.lock:
            page = self.cfg["page"]
            leds = {b.id: [c.red, c.green, f] for b, (c, f) in self._desired.items()}
            playing = sorted(k.split(":", 1)[1] for k in self.audio.playing()
                             if k.startswith(f"{page}:"))
            ptype = self.page_type(page)
            t = self.transport
            return {"version": self.version, "connected": self.lp.connected, "page": page,
                    "leds": leds, "playing": playing, "last_press": self.last_press,
                    "held": sorted({b.id for b in self.lp.held}),
                    "page_type": ptype,
                    "page_types": [self.page_type(i) for i in range(NUM_PAGES)],
                    "inst": self.inst_cfg(page) if ptype in INSTRUMENT_TYPES else None,
                    "seq": dict(self.seq_cfg(page), view=self.seq_view.get(page, 0),
                                step=self.seq_step_now(page)) if ptype == "sequencer" else None,
                    "labels": self.labels(page) if ptype != "soundboard" else None,
                    "transport": {"playing": t.playing, "bpm": round(t.bpm, 1), "metronome": t.metronome,
                                  "beat": round(t.beat(), 2), "bars": self.looper.bars,
                                  "quantize": self.looper.quantize},
                    "tracks": self.looper.snapshot(),
                    "audio": {"output": self.audio.output, "device": self.audio.device_name,
                              "listeners": self.audio.listeners}}

    def full_state(self) -> dict:
        with self.lock:
            return {"config": copy.deepcopy(self.cfg), "sounds": self.sounds(),
                    "defaults": DEFAULT_BUTTON, "fx_defaults": fx.DEFAULT_FX, "fx_limits": fx.LIMITS,
                    "synth": {"patches": synth.PATCHES, "kits": synth.KITS,
                              "scales": list(synth.SCALES), "notes": synth.NOTE_NAMES,
                              "drums": [d[0] for d in synth.DRUMS]},
                    "loop_bars": LOOP_BARS, "seq_lengths": SEQ_LENGTHS, "num_tracks": NUM_TRACKS,
                    "audio_device": self.audio.device_name, "host_audio": self.audio.host_available,
                    "samplerate": self.audio.samplerate,
                    "brightness_levels": [f"{n}/{d}" for n, d in BRIGHTNESS_LEVELS],
                    "state": self.snapshot()}

    def bounce(self, track: int) -> str:
        data, sr = self.looper.bounce(track)
        bars = round(len(data) / (self.transport.fpb * BEATS_PER_BAR))
        name = f"loop{track + 1}-{bars}bar-{round(self.transport.bpm)}bpm-{time.strftime('%H%M%S')}.wav"
        sf.write(os.path.join(SOUNDS, name), data, sr, subtype="PCM_16")
        return name


# ======================================================================= HTTP
SAFE_NAME = re.compile(r"[^A-Za-z0-9 ._()\-]+")


def make_handler(sb: Soundboard):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/state":
                self._json(sb.full_state())
            elif path == "/api/events":
                self._events()
            elif path == "/api/audio":
                self._audio_stream()
            else:
                self._json({"error": "not found"}, 404)

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with sb.cond:
                        if sb.version == seen:
                            sb.cond.wait(timeout=15)
                    if sb.version != seen:
                        snap = sb.snapshot()
                        seen = snap["version"]
                        self.wfile.write(f"data: {json.dumps(snap)}\n\n".encode())
                    else:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _audio_stream(self):
            """Raw little-endian int16 stereo PCM at sb.audio.samplerate, until the client leaves."""
            q = sb.audio.subscribe()
            sb.bump()
            self.close_connection = True
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("X-Samplerate", str(sb.audio.samplerate))
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    try:
                        buf = [q.get(timeout=1.0)]
                    except queue.Empty:  # output switched to host only: keep the socket alive
                        buf = [bytes(4 * sb.audio.BLOCK)]
                    while len(buf) < 16:     # send what has piled up in one write
                        try:
                            buf.append(q.get_nowait())
                        except queue.Empty:
                            break
                    self.wfile.write(b"".join(buf))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                sb.audio.unsubscribe(q)
                sb.bump()

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                if path == "/api/upload":
                    name = SAFE_NAME.sub("_", os.path.basename(self.headers.get("X-Filename", "")))
                    if os.path.splitext(name)[1].lower() not in AUDIO_EXTS:
                        return self._json({"error": "unsupported file type"}, 400)
                    if int(self.headers.get("Content-Length") or 0) > 200 * 1024 * 1024:
                        return self._json({"error": "file too large"}, 400)
                    dest = os.path.join(SOUNDS, name)
                    with open(dest, "wb") as f:
                        f.write(self._body())
                    sb.audio.invalidate(dest)
                    sb.fxcache.drop(lambda k: k[0] == dest)
                    return self._json({"ok": True, "name": name, "sounds": sb.sounds()})
                data = json.loads(self._body() or b"{}")
                if path == "/api/press":
                    sb.handle(data["id"], bool(data["down"]), from_gui=True)
                elif path == "/api/button":
                    page, bid, conf = int(data["page"]), data["id"], data.get("config")
                    with sb.lock:
                        btns = sb.cfg["pages"][page]["buttons"]
                        if conf is None:
                            btns.pop(bid, None)
                        else:
                            b = {k: conf.get(k, v) for k, v in DEFAULT_BUTTON.items()}
                            b["fx"] = {a: v for a, v in fx.clean(b["fx"]).items() if v != fx.DEFAULT_FX[a]}
                            btns[bid] = b
                        sb.save()
                    sb.refresh()
                elif path == "/api/instrument":
                    page = int(data["page"])
                    if data.get("type") in PAGE_TYPES:
                        with sb.lock:
                            sb.cfg["pages"][page]["type"] = data["type"]
                            sb.save()
                    sb.set_inst(page, **(data.get("inst") or {}))
                    if data.get("seq"):
                        sb.set_seq(page, **data["seq"])
                    sb.refresh()
                elif path == "/api/transport":
                    sb.set_transport(**{k: v for k, v in data.items()
                                        if k in ("play", "bpm", "tap", "metronome", "bars", "quantize")})
                elif path == "/api/looper":
                    i, action = int(data["track"]), data["action"]
                    if not 0 <= i < NUM_TRACKS:
                        raise ValueError("bad track")
                    if action == "bounce":
                        name = sb.bounce(i)
                        return self._json({"ok": True, "name": name, "sounds": sb.sounds()})
                    {"main": sb.looper.main, "overdub": sb.looper.toggle_overdub, "undo": sb.looper.undo,
                     "clear": sb.looper.clear}[action](i)
                    sb.refresh()
                elif path == "/api/page":
                    with sb.lock:
                        p = int(data["page"])
                        if "name" in data:
                            sb.cfg["pages"][p]["name"] = str(data["name"])[:40]
                        else:
                            sb.cfg["page"] = p
                        sb.save()
                    sb.refresh()
                elif path == "/api/settings":
                    with sb.lock:
                        if "brightness" in data:
                            i = max(0, min(len(BRIGHTNESS_LEVELS) - 1, int(data["brightness"])))
                            sb.cfg["brightness"] = i
                            sb.lp.set_brightness(*BRIGHTNESS_LEVELS[i])
                        if data.get("audio_output") in AudioEngine.OUTPUTS:
                            sb.cfg["audio_output"] = sb.audio.set_output(data["audio_output"])
                            sb.bump()
                        if "master_volume" in data:
                            sb.cfg["master_volume"] = max(0.0, min(1.0, float(data["master_volume"])))
                            sb.audio.master = sb.cfg["master_volume"]
                        sb.save()
                elif path == "/api/preview":
                    p = os.path.join(SOUNDS, os.path.basename(data["sound"]))
                    sb.audio.play_data("__preview__", sb.sample(p, data.get("fx")), float(data.get("volume", 1.0)))
                elif path == "/api/stop_all":
                    sb.audio.stop_all()
                elif path == "/api/test_leds":
                    sb.lp.test_leds(int(data.get("level", 3)), 1.2)
                elif path == "/api/animate":
                    threading.Thread(target=sb._intro, daemon=True).start()
                else:
                    return self._json({"error": "not found"}, 404)
                self._json({"ok": True})
            except Exception as e:
                self._json({"error": str(e)}, 400)

    return Handler


def serve(port: int = 8765, open_browser: bool = True, host: str = "127.0.0.1") -> None:
    sb = Soundboard()
    sb.start()
    httpd = ThreadingHTTPServer((host, port), make_handler(sb))
    httpd.daemon_threads = True
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    print(f"Launchpad soundboard running at {url}  (audio output: {sb.audio.output}, "
          f"{sb.audio.device_name}; Ctrl+C to quit)")
    if host in ("0.0.0.0", ""):
        print("Listening on every network interface: open http://<this machine's IP>:%d from other devices" % port)
    if open_browser and (os.name != "posix" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
                         or os.uname().sysname == "Darwin"):
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sb.stop()
        httpd.server_close()
