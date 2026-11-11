"""
Tiny offline synthesizer: renders instrument notes and drum hits into numpy buffers.
Everything is generated from oscillators and noise, so no sample files are needed.
Buffers are mono float32 and cached, so each note is rendered only once.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from . import fx as _fx

# ------------------------------------------------------------------ catalogues
PATCHES: Dict[str, str] = {        # id -> display name (first 6 = default side-button slots)
    "keys": "E-Piano",
    "pluck": "Pluck",
    "lead": "Supersaw lead",
    "bass": "Bass",
    "pad": "Pad",
    "organ": "Organ",
    "bell": "FM bell",
    "strings": "Strings",
    "brass": "Brass",
    "acid": "Acid bass",
    "marimba": "Marimba",
    "guitar": "Guitar (plucked)",
    "vox": "Vox choir",
    "chip": "Chiptune",
    "sub808": "808 sub bass",
}
PATCH_SHORT: Dict[str, str] = {
    "keys": "EP", "pluck": "Pluck", "lead": "Lead", "bass": "Bass", "pad": "Pad", "organ": "Organ",
    "bell": "Bell", "strings": "Str", "brass": "Brass", "acid": "Acid", "marimba": "Mrmb",
    "guitar": "Gtr", "vox": "Vox", "chip": "Chip", "sub808": "808",
}
RELEASE = {"keys": 0.15, "pluck": 0.1, "lead": 0.12, "bass": 0.08, "pad": 0.9, "organ": 0.06,
           "bell": 0.6, "strings": 0.7, "brass": 0.15, "acid": 0.05, "marimba": 0.2, "guitar": 0.15,
           "vox": 0.5, "chip": 0.03, "sub808": 0.12}

KITS: Dict[str, str] = {"808": "TR-808 style", "909": "TR-909 style", "electro": "Electro",
                        "lofi": "Lo-fi (crushed)", "industrial": "Industrial (distorted)"}

# (name, LED colour [red, green]) in pad order: bottom-left first, left->right, bottom->top
DRUMS: List[Tuple[str, List[int]]] = [
    ("Kick", [3, 0]), ("Sub kick", [2, 0]), ("Snare", [3, 2]), ("Clap", [3, 1]),
    ("Closed hat", [2, 3]), ("Open hat", [1, 3]), ("Shaker", [1, 2]), ("Ride", [0, 2]),
    ("Low tom", [2, 1]), ("Mid tom", [2, 1]), ("High tom", [2, 1]), ("Crash", [0, 3]),
    ("Rim", [1, 1]), ("Cowbell", [2, 2]), ("Conga", [1, 0]), ("Zap", [0, 1]),
]
CLOSED_HAT, OPEN_HAT = 4, 5

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES: Dict[str, List[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "harmonic minor": [0, 2, 3, 5, 7, 8, 11],
    "major pentatonic": [0, 2, 4, 7, 9],
    "minor pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "chromatic": list(range(12)),
}


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def mtof(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


# ------------------------------------------------------------------ DSP helpers
def _t(seconds: float, sr: int) -> np.ndarray:
    return np.arange(int(seconds * sr)) / sr


def _filter(x: np.ndarray, sr: int, lo: float = 0.0, hi: float = 0.0, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth-shaped low/high/band-pass done in the frequency domain."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    H = np.ones_like(f)
    if hi:
        H /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        with np.errstate(divide="ignore"):
            H /= np.sqrt(1 + (lo / np.maximum(f, 1e-9)) ** (2 * order))
    return np.fft.irfft(X * H, len(x))


def _osc(freq: float, t: np.ndarray) -> np.ndarray:
    """Sine oscillator; wraps the phase first because np.sin is slow for huge arguments."""
    return np.sin(2 * np.pi * ((freq * t) % 1.0))


def _saw(freq: float, t: np.ndarray, sr: int) -> np.ndarray:
    """Band-limited (PolyBLEP) sawtooth."""
    dt = freq / sr
    ph = (freq * t) % 1.0
    y = 2 * ph - 1
    m = ph < dt
    p = ph[m] / dt
    y[m] -= p + p - p * p - 1
    m = ph > 1 - dt
    p = (ph[m] - 1) / dt
    y[m] -= p * p + p + p + 1
    return y


def _reso(x: np.ndarray, sr: int, fc: float, q: float = 0.7) -> np.ndarray:
    """Resonant 2-pole low-pass (frequency domain); q > 0.7 gives a peak at the cutoff."""
    X = np.fft.rfft(x)
    r = np.fft.rfftfreq(len(x), 1 / sr) / fc
    return np.fft.irfft(X / np.sqrt((1 - r ** 2) ** 2 + (r / q) ** 2), len(x))


def _formant(x: np.ndarray, sr: int, peaks) -> np.ndarray:
    """Sum of Gaussian band-pass bumps (centre Hz, width Hz, gain)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    H = sum(g * np.exp(-0.5 * ((f - c) / w) ** 2) for c, w, g in peaks)
    return np.fft.irfft(X * H, len(x))


def _karplus(f: float, seconds: float, sr: int, damp: float, seed: int) -> np.ndarray:
    """Karplus-Strong plucked string, vectorised one period at a time.
    Rendered with an integer delay, then resampled to the exact pitch."""
    period = max(2, int(sr / f))
    n = int(seconds * sr * (period + 1) * f / sr) + period
    rng = np.random.default_rng(seed)
    y = np.empty(n + period + 1)
    burst = rng.uniform(-1, 1, period + 1)
    y[: period + 1] = _filter(burst, sr, hi=min(sr / 2.2, f * 14))
    i = period + 1
    while i < len(y):
        j = min(len(y), i + period)
        a = y[i - period - 1: j - period - 1]
        b = y[i - period: j - period]
        y[i:j] = damp * (0.5 * (a + b))
        i = j
    exact = sr / f
    src = np.arange(0, len(y) - 1, (period + 0.5) / exact)[: int(seconds * sr)]  # loop delay = N + 0.5
    return np.interp(src, np.arange(len(y)), y)


def _adsr(t: np.ndarray, a: float, d: float, s: float) -> np.ndarray:
    a = max(a, 1e-4)
    return np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / d))


def _finish(y: np.ndarray, sr: int, peak: float = 0.45) -> np.ndarray:
    n = min(len(y), int(0.02 * sr))
    y[-n:] *= np.linspace(1, 0, n)          # no click at the end of the buffer
    y[: int(0.002 * sr)] *= np.linspace(0, 1, int(0.002 * sr))  # nor at the start
    m = np.max(np.abs(y)) or 1.0
    return (y * (peak / m)).astype(np.float32)


# ------------------------------------------------------------------ instruments
def render_note(patch: str, midi: float, sr: int, fx=None) -> np.ndarray:
    """A note with optional effects; fx["pitch"]/["fine"] transpose instead of resampling."""
    if not fx or _fx.is_neutral(fx):
        return _render_note(patch, midi, sr)
    fx = _fx.clean(fx)
    return _render_fx(patch, midi + _fx.semitones(fx), sr, _fx.key_of(fx))


@lru_cache(maxsize=256)
def _render_fx(patch: str, midi: float, sr: int, fxkey) -> np.ndarray:
    return _fx.apply(_render_note(patch, midi, sr), sr, dict(fxkey), pitch=False)


@lru_cache(maxsize=256)
def _render_note(patch: str, midi: float, sr: int) -> np.ndarray:
    midi = max(0.0, min(127.0, float(midi)))
    f = mtof(midi)
    nyq = sr / 2
    if patch == "keys":  # FM electric piano
        t = _t(3.0, sr)
        idx = 1.6 * np.exp(-t / 0.25) + 0.25
        tau = 1.6 * (261.6 / f) ** 0.45
        y = np.sin(2 * np.pi * ((f * t) % 1.0) + idx * _osc(f, t)) * np.exp(-t / tau)
        if 14 * f < nyq:  # metallic tine attack
            y += 0.25 * _osc(14 * f, t) * np.exp(-t / 0.015)
    elif patch == "pluck":  # additive plucked string, high harmonics die first
        t = _t(2.5, sr)
        y = np.zeros_like(t)
        tau = 0.9 * (261.6 / f) ** 0.4
        for k in range(1, 31):
            if k * f >= nyq:
                break
            y += _osc(k * f, t) / k * np.exp(-t * (1 + 0.5 * k) / tau)
    elif patch == "lead":  # 3 detuned saws, filter "opens" on attack
        t = _t(4.0, sr)
        raw = sum(_saw(f * 2 ** (c / 1200), t, sr) for c in (-11, 0, 9))
        bright = _filter(raw, sr, hi=min(9000, f * 12))
        dark = _filter(raw, sr, hi=min(4000, f * 4))
        e = np.exp(-t / 0.15)
        y = (bright * e + dark * (1 - e)) * _adsr(t, 0.005, 0.3, 0.75)
    elif patch == "bass":  # played an octave down: saw + sine sub, plucky filter
        f /= 2
        t = _t(3.0, sr)
        raw = _saw(f, t, sr) + 0.9 * _osc(f, t)
        bright = _filter(raw, sr, hi=min(5000, f * 16))
        dark = _filter(raw, sr, hi=f * 3)
        e = np.exp(-t / 0.07)
        y = np.tanh(1.5 * (bright * e + dark * (1 - e))) * _adsr(t, 0.003, 0.5, 0.65)
    elif patch == "pad":  # 4 detuned saws, slow attack, dark
        t = _t(4.0, sr)
        raw = sum(_saw(f * 2 ** (c / 1200), t, sr) for c in (-15, -6, 7, 16))
        y = _filter(raw, sr, hi=1600) * _adsr(t, 0.45, 1.0, 0.85)
    elif patch == "organ":  # drawbar organ with a little tremolo
        t = _t(4.0, sr)
        y = np.zeros_like(t)
        for mult, amp in ((0.5, 0.6), (1, 1.0), (2, 0.6), (3, 0.35), (4, 0.3), (6, 0.15)):
            if mult * f < nyq:
                y += amp * _osc(mult * f, t)
        y *= (1 + 0.12 * _osc(6.0, t)) * _adsr(t, 0.006, 0.1, 1.0)
        y[: int(0.004 * sr)] += np.random.default_rng(int(midi)).normal(0, 0.3, int(0.004 * sr))
    elif patch == "bell":  # FM bell: inharmonic 3.5:1 modulator, long ring
        t = _t(4.0, sr)
        idx = 3.0 * np.exp(-t / 0.6) + 0.3
        tau = 2.2 * (261.6 / f) ** 0.5
        y = np.sin(2 * np.pi * ((f * t) % 1.0) + idx * _osc(3.5 * f, t)) * np.exp(-t / tau)
        if 2.76 * f < nyq:
            y += 0.3 * _osc(2.76 * f, t) * np.exp(-t / (tau * 0.4))
    elif patch == "strings":  # ensemble of detuned saws, slow bow, delayed vibrato
        t = _t(4.0, sr)
        vib = 1 + 0.004 * np.minimum(t / 0.6, 1) * np.sin(2 * np.pi * 5.2 * t)
        raw = sum(_saw(f * 2 ** (c / 1200), t, sr) for c in (-12, -5, 0, 6, 13))
        raw = np.interp(np.cumsum(vib) - 1, np.arange(len(raw)), raw)  # vibrato by time-warp
        y = _filter(raw, sr, lo=f * 0.7, hi=min(5000, f * 10)) * _adsr(t, 0.28, 1.5, 0.9)
    elif patch == "brass":  # saws with a filter that swells open
        t = _t(4.0, sr)
        raw = _saw(f, t, sr) + 0.7 * _saw(f * 1.003, t, sr)
        bright = _filter(raw, sr, hi=min(7000, f * 9))
        dark = _filter(raw, sr, hi=min(1800, f * 2.5))
        e = np.minimum(t / 0.08, 1) * (0.55 + 0.45 * np.exp(-t / 0.35))
        y = np.tanh(1.3 * (bright * e + dark * (1 - e))) * _adsr(t, 0.03, 0.6, 0.8)
    elif patch == "acid":  # 303-style: saw, resonant filter sweeping down, a bit of drive
        f /= 2
        t = _t(2.0, sr)
        raw = _saw(f, t, sr)
        e = np.exp(-t / 0.18)
        hi_, mid_, lo_ = (_reso(raw, sr, min(nyq * 0.8, c), q=6.0) for c in (f * 22, f * 7, f * 2.5))
        y = hi_ * e ** 2 + mid_ * 2 * e * (1 - e) + lo_ * (1 - e) ** 2
        y = np.tanh(2.0 * y / (np.max(np.abs(y)) or 1)) * _adsr(t, 0.002, 0.4, 0.7)
    elif patch == "marimba":  # wooden bar: fundamental + tuned overtones, short decay
        t = _t(2.0, sr)
        tau = 0.5 * (261.6 / f) ** 0.5
        y = _osc(f, t) * np.exp(-t / tau)
        if 3.93 * f < nyq:
            y += 0.35 * _osc(3.93 * f, t) * np.exp(-t / (tau * 0.25))
        if 9.2 * f < nyq:
            y += 0.12 * _osc(9.2 * f, t) * np.exp(-t / 0.02)
    elif patch == "guitar":  # Karplus-Strong string with a touch of body resonance
        y = _karplus(f, 3.0, sr, damp=0.996 if f < 400 else 0.992, seed=int(midi))
        y = y + 0.3 * _filter(y, sr, lo=90, hi=300)
    elif patch == "vox":  # "aah" formants on a vibrating saw
        t = _t(4.0, sr)
        vib = 1 + 0.006 * np.minimum(t / 0.4, 1) * np.sin(2 * np.pi * 5.0 * t)
        ph = np.cumsum(f * vib) / sr % 1.0
        raw = 2 * ph - 1 + 0.3 * np.random.default_rng(int(midi)).normal(0, 0.05, len(t))
        y = _formant(raw, sr, ((800, 80, 1.0), (1150, 90, 0.6), (2900, 120, 0.25), (3300, 130, 0.15)))
        y = y + 0.4 * _formant(raw, sr, ((700, 100, 1.0), (1200, 110, 0.5)))  # second "singer"
        y *= _adsr(t, 0.18, 1.0, 0.9)
    elif patch == "chip":  # 25% pulse wave, 8-bit flavour
        t = _t(3.0, sr)
        y = np.where((f * t) % 1.0 < 0.25, 1.0, -1.0) * _adsr(t, 0.001, 0.15, 0.6)
        y = _filter(y, sr, hi=min(nyq * 0.9, 12000))
    elif patch == "sub808":  # long sine boom with a pitch dip and saturation
        f /= 2
        t = _t(3.0, sr)
        freq = f * (1 + 0.5 * np.exp(-t / 0.03))
        y = np.tanh(2.2 * np.sin(2 * np.pi * np.cumsum(freq) / sr)) * np.exp(-t / 1.4)
    else:
        raise ValueError(f"unknown patch {patch}")
    return _finish(y, sr)


# ------------------------------------------------------------------ drums
_KIT_PARAMS = {
    "808": dict(k0=140, k1=47, kd=0.75, kclick=0.0, kdrive=1.0, st=185, sn=0.17, hat=1.0, tone=1.0),
    "909": dict(k0=240, k1=54, kd=0.38, kclick=0.6, kdrive=1.6, st=200, sn=0.21, hat=1.35, tone=1.15),
    "electro": dict(k0=320, k1=44, kd=0.28, kclick=0.3, kdrive=3.0, st=230, sn=0.12, hat=1.7, tone=1.3),
    "lofi": dict(k0=150, k1=50, kd=0.55, kclick=0.2, kdrive=1.4, st=190, sn=0.15, hat=0.9, tone=0.95),
    "industrial": dict(k0=280, k1=42, kd=0.45, kclick=0.8, kdrive=4.0, st=170, sn=0.25, hat=1.5, tone=0.85),
}
_KIT_FX = {"lofi": {"crush": 0.5, "tone": 0.62},
           "industrial": {"drive": 0.55, "reverb": 0.22}}


def render_drum_fx(kit: str, idx: int, sr: int, fx=None) -> np.ndarray:
    """A drum hit with optional user effects (pitch here resamples, like tuning a sample)."""
    if not fx or _fx.is_neutral(fx):
        return render_drum(kit, idx, sr)
    return _drum_fx(kit, idx, sr, _fx.key_of(_fx.clean(fx)))


@lru_cache(maxsize=256)
def _drum_fx(kit: str, idx: int, sr: int, fxkey) -> np.ndarray:
    return _fx.apply(render_drum(kit, idx, sr), sr, dict(fxkey))


def _metal(t: np.ndarray, scale: float) -> np.ndarray:
    """808-style metallic source: six square waves at inharmonic ratios."""
    return sum(np.sign(_osc(fr * scale, t))
               for fr in (205.3, 304.4, 369.6, 522.7, 540.0, 800.0))


def _tom(t, f0, f1, decay):
    ph = 2 * np.pi * np.cumsum(f1 + (f0 - f1) * np.exp(-t / 0.05)) / (len(t) / (t[-1] + t[1]))
    return np.sin(ph) * np.exp(-t / decay)


@lru_cache(maxsize=96)
def render_drum(kit: str, idx: int, sr: int) -> np.ndarray:
    p = _KIT_PARAMS[kit]
    rng = np.random.default_rng(idx * 31 + len(kit))
    name = DRUMS[idx][0]

    def noise(sec):
        return rng.uniform(-1, 1, int(sec * sr))

    if name in ("Kick", "Sub kick"):
        sub = name == "Sub kick"
        t = _t(1.6 if sub else 1.0, sr)
        f0, f1 = (p["k0"] * 0.6, p["k1"] * 0.8) if sub else (p["k0"], p["k1"])
        freq = f1 + (f0 - f1) * np.exp(-t / 0.035)
        y = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t / (p["kd"] * (1.8 if sub else 1)))
        y = np.tanh(p["kdrive"] * y)
        if p["kclick"]:
            c = _filter(noise(t[-1] + 1 / sr)[: len(t)], sr, lo=2000) * np.exp(-t / 0.004)
            y += p["kclick"] * c
    elif name == "Snare":
        t = _t(0.6, sr)
        tone = (_osc(p["st"], t) + 0.5 * _osc(p["st"] * 1.6, t)) * np.exp(-t / 0.06)
        y = 0.6 * tone + _filter(noise(0.6), sr, lo=1200, hi=9000) * np.exp(-t / p["sn"])
    elif name == "Clap":
        t = _t(0.7, sr)
        env = np.zeros_like(t)
        for d in (0.0, 0.011, 0.022):
            env += (t >= d) * np.exp(-np.maximum(t - d, 0) / 0.006)
        env += (t >= 0.03) * np.exp(-np.maximum(t - 0.03, 0) / 0.15) * 0.7
        y = _filter(noise(0.7), sr, lo=900, hi=2600) * env
    elif name in ("Closed hat", "Open hat"):
        t = _t(0.15 if name == "Closed hat" else 0.7, sr)
        src = _metal(t, p["hat"]) + 0.6 * noise(t[-1] + 1 / sr)[: len(t)]
        y = _filter(src, sr, lo=7000, order=3) * np.exp(-t / (0.035 if name == "Closed hat" else 0.25))
    elif name == "Shaker":
        t = _t(0.25, sr)
        env = np.minimum(t / 0.03, 1) * np.exp(-t / 0.05)
        y = _filter(noise(0.25), sr, lo=5000) * env
    elif name in ("Ride", "Crash"):
        t = _t(2.2, sr)
        src = _metal(t, p["hat"] * (0.8 if name == "Ride" else 1.1)) + 1.2 * noise(2.2)[: len(t)]
        y = _filter(src, sr, lo=4000 if name == "Crash" else 6000, order=2) * \
            np.exp(-t / (0.9 if name == "Crash" else 0.6))
        if name == "Ride":
            y += 0.4 * _osc(3100, t) * np.exp(-t / 0.4)
    elif name in ("Low tom", "Mid tom", "High tom"):
        base = {"Low tom": 90, "Mid tom": 130, "High tom": 185}[name] * p["tone"]
        t = _t(0.9, sr)
        y = _tom(t, base * 1.6, base, 0.3) + 0.1 * _filter(noise(0.9), sr, hi=3000) * np.exp(-t / 0.02)
    elif name == "Rim":
        t = _t(0.12, sr)
        y = (_osc(1700, t) + 0.6 * _osc(480, t)) * np.exp(-t / 0.012)
    elif name == "Cowbell":
        t = _t(0.6, sr)
        sq = np.sign(_osc(540, t)) + np.sign(_osc(800, t))
        y = _filter(sq, sr, lo=500, hi=4000) * (0.6 * np.exp(-t / 0.02) + 0.4 * np.exp(-t / 0.18))
    elif name == "Conga":
        t = _t(0.5, sr)
        y = _tom(t, 400 * p["tone"], 330 * p["tone"], 0.12)
    elif name == "Zap":
        t = _t(0.35, sr)
        freq = 80 + 2400 * np.exp(-t / 0.03)
        y = np.sign(np.sin(2 * np.pi * np.cumsum(freq) / sr)) * np.exp(-t / 0.08)
        y = _filter(y, sr, hi=6000)
    else:
        raise ValueError(name)
    y = _finish(np.asarray(y, dtype=np.float64), sr, peak=0.6)
    if kit in _KIT_FX:
        y = _fx.apply(y, sr, _KIT_FX[kit])
    return y


def click(sr: int, accent: bool) -> np.ndarray:
    """Metronome tick."""
    t = _t(0.05, sr)
    y = _osc(1800 if accent else 1200, t) * np.exp(-t / 0.008)
    return _finish(y, sr, peak=0.5 if accent else 0.35)
