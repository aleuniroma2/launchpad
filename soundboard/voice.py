"""Voice page: plays the CHIMERA physical-modelling voice synth (served at /voice/) from the Launchpad.

The grid and side buttons are mapped onto CHIMERA's own controls (modes, keys, vowels, body, breath,
song / speech, beatbox, bestiary and the voice encoders). This module only decides *what each button
means* and queues events; the browser page (voice/launchpad-bridge.ts, built into CHIMERA) executes
them against CHIMERA's real store, so CHIMERA's own on-screen controls stay in sync.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

# ---- CHIMERA's catalogue (src/wasm catalog, src/song/tunes.ts, voice controls)
MODES = ("play", "speak", "sing", "bestiary", "beatbox")
MODE_LABEL = {"play": "Play", "speak": "Speak", "sing": "Sing", "bestiary": "Beast", "beatbox": "Beat"}
BODIES = ("male", "female", "child")
BASE_NOTE = {"male": 48, "female": 57, "child": 60}   # lowest key per body (Keys.tsx)
VOWELS = ["i", "ɪ", "e", "ɛ", "æ", "a", "ɑ", "ɒ", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɐ", "ɝ",
          "y", "ø", "œ", "ɨ", "ɯ", "ɤ", "ɿ", "ʅ", "u̠", "ʊ̠"]
BEAT_SOUNDS = [("B", "Kick"), ("t", "Hat"), ("tss", "Open"), ("Pf", "Snare"), ("K", "Rim"),
               ("^K", "In-snr"), ("!", "Click"), ("P!", "Pop"), ("kch", "Crash"), ("m", "Hum")]
BEAT_PRESETS = ["Boots and cats", "Four on the floor", "Breakbeat", "Rimshot shuffle",
                "Inward snare", "Clicks and pops", "Hum bass"]
CREATURES = {"songbird": ("Bird", ["flute", "trill", "chirps", "buzz"]),
             "seal": ("Seal", ["trill", "zap", "whistle", "chug"]),
             "alligator": ("Gator", ["bellow", "growl", "hiss", "dance"]),
             "elephant": ("Eleph", ["rumble", "trumpet", "squeak"])}
TUNES = ["dreamy", "joyful", "somber", "playful", "gentle", "twinkle",
         "song:Ночной ветер", "song:Daisy Bell", "song:さくら さくら"]
TUNE_LABEL = ["Dream", "Joy", "Somber", "Play", "Gentle", "Twinkle", "Ночной", "Daisy", "さくら"]
# the voice encoders, in CHIMERA's order (VoiceControls.tsx); the bridge knows their ranges
TWEAK_PAGES = {"voice": ["pitch", "breath", "pressed", "vibrato"],
               "tone": ["bright", "breathy", "rough", "fry"],
               "glide": ["glide"],
               "mouth": ["rounding", "lipsopen", "protrude", "jaw"],
               "tongue": ["root", "tipcurl"],
               "throat": ["velum", "larynx", "twang", "falsefolds"]}
TWEAK_SCREENS = [("voice", "tone"), ("glide", "mouth"), ("tongue", "throat")]
PARAM_LABEL = {"pitch": "Pitch", "breath": "Breath", "pressed": "Pressed", "vibrato": "Vibrato",
               "bright": "Bright", "breathy": "Breathy", "rough": "Rough", "fry": "Fry", "glide": "Glide",
               "rounding": "Round", "lipsopen": "Lips", "protrude": "Protr", "jaw": "Jaw",
               "root": "Root", "tipcurl": "Curl", "velum": "Palate", "larynx": "Larynx",
               "twang": "Twang", "falsefolds": "Folds"}

# LED colours (red, green) 0-3
OFFC = (0, 0)
WHITE_KEY, BLACK_KEY, ROOT_KEY, DOWN = (0, 2), (0, 1), (2, 3), (3, 0)
CTRL_OFF, CTRL_ON, ACCENT, STOPC = (1, 1), (0, 3), (3, 3), (3, 0)

PIANO_WHITE = [0, 2, 4, 5, 7, 9, 11, 12]       # one octave of white keys across 8 pads
PIANO_BLACK = [1, 3, None, 6, 8, 10, None, None]


class EventBus:
    """Ordered events for the /voice/ page(s): readers wait for seq > what they have seen."""

    def __init__(self):
        self.cond = threading.Condition()
        self.seq = 0
        self.events: List[dict] = []

    def emit(self, cmd: str, **kw) -> None:
        with self.cond:
            self.seq += 1
            self.events.append(dict(kw, cmd=cmd, seq=self.seq))
            del self.events[:-200]
            self.cond.notify_all()


class VoiceState:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.lock = threading.RLock()
        self.mode = "sing"           # CHIMERA opens on Sing
        self.tweak = False           # the voice-encoder overlay (any mode)
        self.tweak_screen = 0
        self.octave = 0              # -2..2, like CHIMERA's Z / X
        self.body = "male"
        self.breath = False
        self.whisper = False
        self.vbank = 0               # vowel bank 0..3 (8 vowels each)
        self.vowel: Optional[str] = None
        self.tune = 0
        self.ipa = False
        self.lang = "EN"
        self.creature = "songbird"
        self.key_call: Optional[str] = None   # None = keys hold the pitch
        self.beat_playing = False
        self.mic = True
        self.beat_preset: Optional[int] = 0
        self.speaking = False        # approximate: the primary button toggles it
        self.levels: Dict[str, int] = {}      # encoder id -> last level 0..7
        self.held: Dict[str, int] = {}        # pad id -> MIDI note held

    # ------------------------------------------------------------ layout
    def _keys(self, m: Dict[str, tuple], rows: range) -> None:
        """Piano keys (white row + black row above) on grid rows `rows`, 3 octaves from the body's base."""
        base = BASE_NOTE[self.body] + 12 * self.octave
        ys = list(rows)
        for i, y in enumerate(reversed(ys)):          # bottom-most listed row first
            off = (PIANO_WHITE if i % 2 == 0 else PIANO_BLACK)
            for x in range(8):
                if off[x] is not None:
                    m[f"pad{x}{y}"] = ("note", base + 12 * (i // 2) + off[x], i % 2 == 1)

    def layout(self) -> Dict[str, tuple]:
        m: Dict[str, tuple] = {}
        for i, md in enumerate(MODES):
            m[f"side{i}"] = ("mode", md)
        m["side5"] = ("tweak",)
        m["side6"] = ("screen",) if self.tweak else ("body", None)
        m["side7"] = ("stop",)
        if self.tweak:
            a, b = TWEAK_SCREENS[self.tweak_screen]
            for half, page in enumerate((a, b)):
                for r, pid in enumerate(TWEAK_PAGES[page]):
                    for x in range(8):
                        m[f"pad{x}{half * 4 + r}"] = ("param", pid, x)
            return m
        md = self.mode
        if md in ("play", "speak", "sing"):
            self._keys(m, range(0, 6))
        if md == "play":
            for i, a in enumerate((("oct", -1), ("oct", 1), ("breath",), ("whisper",),
                                   ("body", "male"), ("body", "female"), ("body", "child"), ("vbank",))):
                m[f"pad{i}6"] = a
            for i in range(8):
                v = self.vbank * 8 + i
                if v < len(VOWELS):
                    m[f"pad{i}7"] = ("vowel", VOWELS[v])
        elif md == "sing":
            for i, a in enumerate((("oct", -1), ("oct", 1), ("primary",), ("tune_step", -1),
                                   ("tune_step", 1), ("fromstart",), ("body", None), ("tune", 8))):
                m[f"pad{i}6"] = a
            for i in range(8):
                m[f"pad{i}7"] = ("tune", i)
        elif md == "speak":
            for i, a in enumerate((("primary",), ("ipa",), ("rate", -1), ("rate", 1),
                                   ("lang", "EN"), ("lang", "TR"), ("fromstart",), ("body", None))):
                m[f"pad{i}6"] = a
            for i in range(8):
                m[f"pad{i}7"] = ("example", i)
        elif md == "bestiary":
            self._keys(m, range(0, 4))
            for i, cid in enumerate(CREATURES):
                m[f"pad{i}4"] = ("creature", cid)
            for i, a in enumerate((("breath",), ("oct", -1), ("oct", 1), ("keycall", None))):
                m[f"pad{4 + i}4"] = a
            for i, call in enumerate(CREATURES[self.creature][1]):
                m[f"pad{i}5"] = ("call", call)
                m[f"pad{4 + i}5"] = ("keycall", call)
            for i in range(8):
                m[f"pad{i}6"] = ("note", BASE_NOTE[self.body] + 12 * self.octave + 12 + [0, 2, 4, 5, 7, 9, 11, 12][i], False)
        elif md == "beatbox":
            for i, (sid, _) in enumerate(BEAT_SOUNDS):
                m[f"pad{i % 8}{i // 8}"] = ("beat", sid)
            for i in range(6):
                m[f"pad{2 + i}1"] = ("preset", i)
            self._keys(m, range(2, 6))
            for i, a in enumerate((("beat_play",), ("tempo", -5), ("tempo", 5), ("mic",),
                                   ("preset", 6), ("oct", -1), ("oct", 1))):
                m[f"pad{i}6"] = a
        return m

    def labels(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        lay = self.layout()
        names = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
        for bid, a in lay.items():
            k = a[0]
            if k == "note":
                out[bid] = f"{names[a[1] % 12]}{a[1] // 12 - 1}"
            elif k == "mode":
                out[bid] = MODE_LABEL[a[1]]
            elif k == "tweak":
                out[bid] = "Tweak"
            elif k == "screen":
                out[bid] = "Next ▸"
            elif k == "body":
                out[bid] = {None: self.body.capitalize(), "male": "Male", "female": "Female", "child": "Child"}[a[1]]
            elif k == "stop":
                out[bid] = "Stop"
            elif k == "oct":
                out[bid] = "Oct +" if a[1] > 0 else "Oct −"
            elif k == "breath":
                out[bid] = "Breathe"
            elif k == "whisper":
                out[bid] = "Whisper"
            elif k == "vbank":
                out[bid] = f"Vowels {self.vbank + 1}/4"
            elif k == "vowel":
                out[bid] = a[1]
            elif k == "primary":
                out[bid] = "Say" if self.mode == "speak" else "Sing"
            elif k == "tune_step":
                out[bid] = "Tune ◂" if a[1] < 0 else "Tune ▸"
            elif k == "tune":
                out[bid] = TUNE_LABEL[a[1]]
            elif k == "fromstart":
                out[bid] = "Restart"
            elif k == "ipa":
                out[bid] = "IPA"
            elif k == "rate":
                out[bid] = "Rate −" if a[1] < 0 else "Rate +"
            elif k == "lang":
                out[bid] = a[1]
            elif k == "example":
                out[bid] = f"Ex {a[1] + 1}"
            elif k == "creature":
                out[bid] = CREATURES[a[1]][0]
            elif k == "call":
                out[bid] = a[1]
            elif k == "keycall":
                out[bid] = "Keys=pitch" if a[1] is None else "Key:" + a[1]
            elif k == "beat":
                out[bid] = dict(BEAT_SOUNDS)[a[1]]
            elif k == "preset":
                out[bid] = f"Beat {a[1] + 1}"
            elif k == "beat_play":
                out[bid] = "Stop" if self.beat_playing else "Play"
            elif k == "tempo":
                out[bid] = "BPM −" if a[1] < 0 else "BPM +"
            elif k == "mic":
                out[bid] = "Mic"
            elif k == "param":
                out[bid] = PARAM_LABEL[a[1]] if a[2] == 0 else ""
        return out

    # ------------------------------------------------------------ LEDs
    def led(self, bid: str, down: bool) -> Tuple[Tuple[int, int], bool]:
        a = self.layout().get(bid)
        if not a:
            return OFFC, False
        k = a[0]
        if k == "note":
            if down or a[1] in self.held.values():
                return DOWN, False
            return (ROOT_KEY if a[1] % 12 == 0 else BLACK_KEY if a[2] else WHITE_KEY), False
        if down and k != "stop":
            return DOWN, False
        on = False
        if k == "mode":
            on = a[1] == self.mode and not self.tweak
        elif k == "tweak":
            on = self.tweak
        elif k == "stop":
            return STOPC, False
        elif k == "body":
            if a[1] is None:
                return (2, 1), False
            on = a[1] == self.body
        elif k == "screen":
            return (2, 1), False
        elif k == "breath":
            on = self.breath
        elif k == "whisper":
            on = self.whisper
        elif k == "vowel":
            on = a[1] == self.vowel
        elif k == "tune":
            on = a[1] == self.tune
        elif k == "ipa":
            on = self.ipa
        elif k == "lang":
            on = a[1] == self.lang
        elif k == "creature":
            on = a[1] == self.creature
        elif k == "keycall":
            on = a[1] == self.key_call
        elif k == "beat_play":
            return (CTRL_ON, self.beat_playing) if self.beat_playing else (CTRL_OFF, False)
        elif k == "mic":
            on = self.mic
        elif k == "preset":
            on = a[1] == self.beat_preset
        elif k == "param":
            lv = self.levels.get(a[1])
            if lv is not None and a[2] <= lv:
                return (CTRL_ON if a[2] == lv else (0, 1)), False
            return (1, 0) if a[2] == 0 else OFFC, False
        elif k == "beat":
            return (2, 2), False
        elif k in ("primary", "call"):
            return (ACCENT if k == "primary" else (2, 2)), False
        else:
            return (2, 1), False
        return (CTRL_ON if on else CTRL_OFF), False

    # ------------------------------------------------------------ input
    def release_all(self) -> None:
        with self.lock:
            for bid, midi in list(self.held.items()):
                self.bus.emit("note_off", midi=midi)
            self.held.clear()

    def up(self, bid: str) -> None:
        """Any button released (on any page): lift a held voice note."""
        with self.lock:
            midi = self.held.pop(bid, None)
            if midi is not None and midi not in self.held.values():
                self.bus.emit("note_off", midi=midi)

    def press(self, bid: str, down: bool) -> None:
        if not down:
            self.up(bid)
            return
        with self.lock:
            a = self.layout().get(bid)
            if not a:
                return
            k, e = a[0], self.bus.emit
            if k == "note":
                self.held[bid] = a[1]
                e("note_on", midi=a[1], vel=96)
            elif k == "mode":
                self.release_all()
                self.mode, self.tweak = a[1], False
                e("mode", mode=a[1])
            elif k == "tweak":
                self.tweak = not self.tweak
            elif k == "screen":
                self.tweak_screen = (self.tweak_screen + 1) % len(TWEAK_SCREENS)
            elif k == "body":
                name = a[1] or BODIES[(BODIES.index(self.body) + 1) % 3]
                self.release_all()
                self.body = name
                e("body", name=name)
            elif k == "oct":
                self.release_all()
                self.octave = max(-2, min(2, self.octave + a[1]))
            elif k == "stop":
                self.release_all()
                self.breath = self.beat_playing = False
                e("stop")
            elif k == "breath":
                self.breath = not self.breath
                e("breath", on=self.breath)
            elif k == "whisper":
                self.whisper = not self.whisper
                e("whisper", on=self.whisper)
            elif k == "vbank":
                self.vbank = (self.vbank + 1) % 4
            elif k == "vowel":
                self.vowel = a[1]
                e("vowel", ipa=a[1])
            elif k == "primary":
                e("primary")
            elif k == "tune":
                self.tune = a[1]
                e("tune", id=TUNES[a[1]])
            elif k == "tune_step":
                self.tune = (self.tune + a[1]) % len(TUNES)
                e("tune", id=TUNES[self.tune])
            elif k == "fromstart":
                e("fromstart")
            elif k == "ipa":
                self.ipa = not self.ipa
                e("ipa")
            elif k == "rate":
                e("rate", step=a[1])
            elif k == "lang":
                self.lang = a[1]
                e("lang", lang=a[1])
            elif k == "example":
                e("example", index=a[1])
            elif k == "creature":
                self.creature, self.key_call = a[1], None
                self.release_all()
                e("creature", id=a[1])
            elif k == "call":
                e("call", id=a[1])
            elif k == "keycall":
                self.key_call = a[1]
                self.release_all()
                e("keycall", id=a[1])
            elif k == "beat":
                e("beat", id=a[1])
            elif k == "preset":
                self.beat_preset = a[1]
                e("preset", name=BEAT_PRESETS[a[1]])
            elif k == "beat_play":
                self.beat_playing = not self.beat_playing
                e("beat_play", on=self.beat_playing)
            elif k == "tempo":
                e("tempo", delta=a[1])
            elif k == "mic":
                self.mic = not self.mic
                e("mic", on=self.mic)
            elif k == "param":
                self.levels[a[1]] = a[2]
                e("param", id=a[1], norm=a[2] / 7)

    def snapshot(self) -> dict:
        return {"mode": self.mode, "tweak": self.tweak, "octave": self.octave, "body": self.body,
                "creature": self.creature, "tune": self.tune}
