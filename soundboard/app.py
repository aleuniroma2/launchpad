"""Soundboard logic: maps Launchpad buttons -> sounds + LED feedback, and serves the GUI."""
from __future__ import annotations

import copy
import json
import queue
import math
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

from lpdriver import BRIGHTNESS_LEVELS, Button, Color, Launchpad, OFF, all_buttons
from . import synth
from .audio import AUDIO_EXTS, AudioEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SOUNDS = os.path.join(ROOT, "sounds")
CONFIG = os.path.join(ROOT, "config.json")
NUM_PAGES = 8
PAGE_TYPES = ("soundboard", "piano", "drums")

# Instrument page settings ("inst" key of a page)
DEFAULT_INST = {
    "patch": "keys",        # see synth.PATCHES
    "layout": "piano",      # piano (white/black key rows) | scale (in-key grid, rows a 4th apart)
    "root": 0,              # 0 = C .. 11 = B
    "scale": "minor",       # see synth.SCALES
    "octave": 2,            # lowest octave shown (C2 = MIDI 36)
    "volume": 0.8,
    "kit": "808",           # see synth.KITS
    "colors": {"key": [0, 1], "black": [1, 0], "root": [0, 3], "pressed": [3, 3]},
}
PIANO_WHITE = [0, 2, 4, 5, 7, 9, 11, 12]
PIANO_BLACK = [None, 1, 3, None, 6, 8, 10, None]  # black key sits above the white key to its right
BLACK_KEYS = {1, 3, 6, 8, 10}
PATCH_SHORT = {"keys": "EP", "pluck": "Pluck", "lead": "Lead", "bass": "Bass", "pad": "Pad", "organ": "Organ"}

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
    cfg = {"version": 1, "page": 0, "brightness": 5, "master_volume": 0.9, "pages": pages}
    migrate(cfg)
    return cfg


def migrate(cfg: dict) -> bool:
    """Add page types; the first time, turn empty pages 7 and 8 into Piano and Drums."""
    changed = False
    for p in cfg["pages"]:
        if p.get("type") not in PAGE_TYPES:
            p["type"] = "soundboard"
            changed = True
    if not cfg.get("instruments_seeded"):
        for i, (t, name) in ((6, ("piano", "Piano")), (7, ("drums", "Drums"))):
            p = cfg["pages"][i]
            if p["type"] == "soundboard" and not p["buttons"]:
                p["type"] = t
                if p["name"] == f"Page {i + 1}":
                    p["name"] = name
        cfg["instruments_seeded"] = True
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
        self.audio = AudioEngine()
        self.audio.master = self.cfg["master_volume"]
        n, d = BRIGHTNESS_LEVELS[self.cfg["brightness"]]
        self.lp = Launchpad(brightness=(n, d))
        self.lp.on_button(self._on_hw_button)
        self.lp.on_connection(self._on_connection)
        self._last_playing: set = set()
        self._desired: Dict[Button, Tuple[Color, bool]] = {}
        self.active: Dict[str, Tuple[str, float]] = {}  # button id -> (voice key, release s)
        self.held_notes: Dict[str, int] = {}            # button id -> MIDI note being played
        self._warm_q: "queue.Queue" = queue.Queue()

    # ------------------------------------------------------------- config
    def _load_config(self) -> dict:
        if os.path.exists(CONFIG):
            try:
                with open(CONFIG) as f:
                    cfg = json.load(f)
                base = default_config()
                base.update(cfg)
                base["instruments_seeded"] = cfg.get("instruments_seeded", False)
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

    # ------------------------------------------------------------- instruments
    def page_type(self, page: int) -> str:
        return self.cfg["pages"][page].get("type", "soundboard")

    def inst_cfg(self, page: int) -> dict:
        raw = self.cfg["pages"][page].get("inst", {})
        inst = dict(DEFAULT_INST, **raw)
        inst["colors"] = dict(DEFAULT_INST["colors"], **raw.get("colors", {}))
        return inst

    def instrument_map(self, page: int) -> Dict[str, tuple]:
        """What every button does on an instrument page.

        Piano:  ("note", midi, is_black_key); side 1/2 = octave up/down; side 3-8 = sound.
        Drums:  ("drum", index) as 16 big 2x2 pads; side 1-3 = kit; side 8 = stop.
        """
        t, inst = self.page_type(page), self.inst_cfg(page)
        m: Dict[str, tuple] = {}
        if t == "piano":
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
            for i, pid in enumerate(synth.PATCHES):
                m[f"side{2 + i}"] = ("patch", pid)
        elif t == "drums":
            for y in range(8):
                for x in range(8):
                    m[f"pad{x}{y}"] = ("drum", (3 - y // 2) * 4 + x // 2)
            for i, k in enumerate(synth.KITS):
                m[f"side{i}"] = ("kit", k)
            m["side7"] = ("stop",)
        return m

    def set_inst(self, page: int, **changes) -> None:
        with self.lock:
            pg = self.cfg["pages"][page]
            inst = pg.setdefault("inst", {})
            valid = {"patch": lambda v: v in synth.PATCHES, "kit": lambda v: v in synth.KITS,
                     "layout": lambda v: v in ("piano", "scale"), "scale": lambda v: v in synth.SCALES,
                     "root": lambda v: isinstance(v, int) and 0 <= v <= 11,
                     "octave": lambda v: isinstance(v, int),
                     "volume": lambda v: isinstance(v, (int, float)) and 0 <= v <= 1.5}
            for k, v in changes.items():
                if k == "colors" and isinstance(v, dict):
                    inst.setdefault("colors", {}).update(
                        {c: [max(0, min(3, int(x))) for x in rgb][:2] for c, rgb in v.items()
                         if c in DEFAULT_INST["colors"]})
                elif k in valid and valid[k](v):
                    inst[k] = v
            inst["octave"] = max(0, min(5, int(self.inst_cfg(page)["octave"])))
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
                        synth.render_note(inst["patch"], midi, sr)
                elif t == "drums":
                    for i in range(len(synth.DRUMS)):
                        synth.render_drum(inst["kit"], i, sr)
            except Exception as e:
                print("prewarm failed:", e)

    def _handle_instrument(self, page: int, bid: str) -> None:
        """A button went down on an instrument page."""
        act = self.instrument_map(page).get(bid)
        if not act:
            return
        inst, sr, kind = self.inst_cfg(page), self.audio.samplerate, act[0]
        if kind == "note":
            key = f"{page}:{bid}"
            self.audio.play_data(key, synth.render_note(inst["patch"], act[1], sr), inst["volume"])
            self.active[bid] = (key, synth.RELEASE[inst["patch"]])
            self.held_notes[bid] = act[1]
        elif kind == "drum":
            if act[1] == synth.CLOSED_HAT:  # closed hat chokes the open hat, like real hi-hats
                self.audio.stop(f"{page}:drum{synth.OPEN_HAT}", 0.02)
            self.audio.play_data(f"{page}:drum{act[1]}",
                                 synth.render_drum(inst["kit"], act[1], sr), inst["volume"])
        elif kind == "oct":
            self.set_inst(page, octave=inst["octave"] + act[1])
        elif kind == "patch":
            self.set_inst(page, patch=act[1])
        elif kind == "kit":
            self.set_inst(page, kit=act[1])
        elif kind == "stop":
            self.audio.stop_all()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.lp.start()
        threading.Thread(target=self._poll_loop, daemon=True).start()
        threading.Thread(target=self._warm_loop, daemon=True).start()
        for i in range(NUM_PAGES):
            if self.page_type(i) != "soundboard":
                self.prewarm(i)
        self.refresh()

    def stop(self) -> None:
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
                self.audio.stop(key, release)
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
                            self.audio.stop(k)
                elif b["action"] == "sound" and b["sound"]:
                    path = os.path.join(SOUNDS, b["sound"])
                    try:
                        if pressed:
                            if b["mode"] == "toggle" and key in self.audio.playing():
                                self.audio.stop(key)
                            else:
                                self.audio.play(key, path, b["volume"], b["loop"])
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
        imap = self.instrument_map(page) if self.page_type(page) != "soundboard" else None
        out: Dict[Button, Tuple[Color, bool]] = {}
        for btn in all_buttons():
            bid = btn.id
            if btn.kind == "top":
                p = btn.x
                active = any(k.startswith(f"{p}:") for k in playing)
                if bid in held:
                    out[btn] = (Color(3, 0), False)
                elif p == page:
                    out[btn] = (Color(3, 3), False)
                elif active:
                    out[btn] = (Color(0, 3), True)
                elif self.page_type(p) == "piano":
                    out[btn] = (Color(2, 1), False)   # orange-ish: piano page
                elif self.page_type(p) == "drums":
                    out[btn] = (Color(1, 0), False)   # red: drum page
                elif self.cfg["pages"][p]["buttons"]:
                    out[btn] = (Color(0, 1), False)   # green: soundboard page with sounds
                else:
                    out[btn] = (OFF, False)
                continue
            if imap is not None:
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

    def labels(self, page: int) -> Dict[str, str]:
        """Pad captions for the GUI on instrument pages."""
        out = {}
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
                out[bid] = PATCH_SHORT[act[1]]
            elif kind == "kit":
                out[bid] = act[1]
            elif kind == "stop":
                out[bid] = "Stop"
        return out

    def refresh(self, force: bool = False) -> None:
        with self.lock:
            desired = self.compute_leds()
            self._desired = desired
        self.lp.set_leds(desired, force=force)
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
            return {"version": self.version, "connected": self.lp.connected, "page": page,
                    "leds": leds, "playing": playing, "last_press": self.last_press,
                    "held": sorted({b.id for b in self.lp.held}),
                    "page_type": ptype,
                    "page_types": [self.page_type(i) for i in range(NUM_PAGES)],
                    "inst": self.inst_cfg(page) if ptype != "soundboard" else None,
                    "labels": self.labels(page) if ptype != "soundboard" else None}

    def full_state(self) -> dict:
        with self.lock:
            return {"config": copy.deepcopy(self.cfg), "sounds": self.sounds(),
                    "defaults": DEFAULT_BUTTON,
                    "synth": {"patches": synth.PATCHES, "kits": synth.KITS,
                              "scales": list(synth.SCALES), "notes": synth.NOTE_NAMES,
                              "drums": [d[0] for d in synth.DRUMS]},
                    "brightness_levels": [f"{n}/{d}" for n, d in BRIGHTNESS_LEVELS],
                    "state": self.snapshot()}


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
                            btns[bid] = {k: conf.get(k, v) for k, v in DEFAULT_BUTTON.items()}
                        sb.save()
                    sb.refresh()
                elif path == "/api/instrument":
                    page = int(data["page"])
                    if data.get("type") in PAGE_TYPES:
                        with sb.lock:
                            sb.cfg["pages"][page]["type"] = data["type"]
                            sb.save()
                    sb.set_inst(page, **(data.get("inst") or {}))
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
                        if "master_volume" in data:
                            sb.cfg["master_volume"] = max(0.0, min(1.0, float(data["master_volume"])))
                            sb.audio.master = sb.cfg["master_volume"]
                        sb.save()
                elif path == "/api/preview":
                    sb.audio.play("__preview__", os.path.join(SOUNDS, os.path.basename(data["sound"])),
                                  float(data.get("volume", 1.0)))
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


def serve(port: int = 8765, open_browser: bool = True) -> None:
    sb = Soundboard()
    sb.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(sb))
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{port}"
    print(f"Launchpad soundboard running at {url}  (Ctrl+C to quit)")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sb.stop()
        httpd.server_close()
