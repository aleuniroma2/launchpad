"""
Offline effects applied to a sound before it is played: pitch, distortion, bit-crusher,
low-pass "tone", echo, reverb and reverse. Processed buffers are cached, so a pad with
effects costs nothing extra after its first hit.

Settings are a dict like DEFAULT_FX; any missing key falls back to its neutral value.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Dict, Hashable, Tuple

import numpy as np

DEFAULT_FX: Dict[str, float] = {
    "pitch": 0.0,      # semitones, -24 .. +24 (samples/drums: like a sampler, also changes speed)
    "fine": 0.0,       # cents, -100 .. +100
    "drive": 0.0,      # distortion 0 .. 1
    "crush": 0.0,      # bit-crusher / sample-rate reducer 0 .. 1
    "tone": 1.0,       # low-pass: 0 = very dark .. 1 = open
    "echo": 0.0,       # echo mix 0 .. 1
    "echo_time": 0.3,  # seconds between repeats, 0.05 .. 1.0
    "reverb": 0.0,     # reverb mix 0 .. 1
    "reverse": False,  # play backwards
}
LIMITS = {"pitch": (-24, 24), "fine": (-100, 100), "drive": (0, 1), "crush": (0, 1), "tone": (0, 1),
          "echo": (0, 1), "echo_time": (0.05, 1.0), "reverb": (0, 1)}


def clean(raw) -> Dict[str, float]:
    """Validate/clamp user-supplied settings into a full FX dict."""
    out = dict(DEFAULT_FX)
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if k == "reverse":
            out[k] = bool(v)
        elif k in LIMITS and isinstance(v, (int, float)) and not isinstance(v, bool):
            lo, hi = LIMITS[k]
            out[k] = float(max(lo, min(hi, v)))
    return out


def is_neutral(fx: Dict[str, float], ignore_pitch: bool = False) -> bool:
    for k, v in DEFAULT_FX.items():
        if k == "echo_time" or (ignore_pitch and k in ("pitch", "fine")):
            continue
        if fx.get(k, v) != v:
            return False
    return True


def key_of(fx: Dict[str, float]) -> Tuple:
    return tuple(sorted((k, fx.get(k, v)) for k, v in DEFAULT_FX.items()))


def semitones(fx: Dict[str, float]) -> float:
    return fx.get("pitch", 0.0) + fx.get("fine", 0.0) / 100.0


# ------------------------------------------------------------------ DSP
def _lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1 / sr)[:, None]
    return np.fft.irfft(X / np.sqrt(1 + (f / cutoff) ** 4), x.shape[0], axis=0)


def _convolve(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    n = x.shape[0] + ir.shape[0] - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, size, axis=0) * np.fft.rfft(ir, size, axis=0), size, axis=0)[:n]


_IR_CACHE: Dict[int, np.ndarray] = {}


def _reverb_ir(sr: int) -> np.ndarray:
    """Stereo decaying-noise impulse response (~2.2 s), darker as it decays."""
    if sr not in _IR_CACHE:
        rng = np.random.default_rng(7)
        t = np.arange(int(2.2 * sr)) / sr
        noise = rng.standard_normal((len(t), 2))
        bright = noise * np.exp(-t / 0.25)[:, None]
        dark = _lowpass(noise, sr, 2500) * np.exp(-t / 0.6)[:, None]
        ir = bright * 0.35 + dark
        ir[: int(0.012 * sr)] = 0  # pre-delay
        _IR_CACHE[sr] = (ir / np.sqrt(np.sum(ir ** 2) / 2)).astype(np.float64)
    return _IR_CACHE[sr]


def apply(data: np.ndarray, sr: int, fx: Dict[str, float], pitch: bool = True) -> np.ndarray:
    """Return a processed float32 copy of `data` ((n,) mono or (n, 2) stereo)."""
    fx = clean(fx)
    if is_neutral(fx, ignore_pitch=not pitch) or len(data) < 2:
        return data
    mono = data.ndim == 1
    x = np.asarray(data, dtype=np.float64)
    x = x[:, None] if mono else x.copy()
    peak_in = float(np.max(np.abs(x))) or 1.0

    if fx["reverse"]:
        x = x[::-1].copy()
    st = semitones(fx) if pitch else 0.0
    if st:
        ratio = 2 ** (st / 12)
        n = max(2, int(len(x) / ratio))
        src = np.linspace(0, len(x) - 1, n)
        idx = np.arange(len(x))
        x = np.stack([np.interp(src, idx, x[:, c]) for c in range(x.shape[1])], axis=1)
    if fx["drive"]:
        g = 1 + 40 * fx["drive"] ** 1.5
        x = np.tanh(g * x / peak_in + 0.15 * fx["drive"]) - np.tanh(0.15 * fx["drive"])
    if fx["crush"]:
        bits = 16 - 13 * fx["crush"]
        hold = 1 + int(fx["crush"] ** 2 * 20)
        if hold > 1:
            x = np.repeat(x[::hold], hold, axis=0)[: len(x)]
        q = 2 ** (bits - 1)
        x = np.round(x / (np.max(np.abs(x)) or 1) * q) / q
    if fx["tone"] < 0.999:
        x = _lowpass(x, sr, 150 * (20000 / 150) ** fx["tone"])
    if fx["echo"]:
        d, fb = max(1, int(fx["echo_time"] * sr)), 0.5
        taps = int(np.ceil(np.log(0.02) / np.log(fb)))
        y = np.zeros((len(x) + taps * d, x.shape[1]))
        y[: len(x)] += x
        for k in range(1, taps + 1):
            y[k * d: k * d + len(x)] += fx["echo"] * fb ** (k - 1) * x
        x = y
    if fx["reverb"]:
        ir = _reverb_ir(sr)
        if x.shape[1] == 1:
            ir = ir[:, :1]
        wet = _convolve(x, ir) * 0.5
        dry = np.zeros_like(wet)
        dry[: len(x)] = x
        x = dry * (1 - 0.5 * fx["reverb"]) + wet * fx["reverb"]

    peak = float(np.max(np.abs(x))) or 1.0
    x *= peak_in / peak  # keep the original loudness
    n = min(len(x), int(0.01 * sr))
    x[-n:] *= np.linspace(1, 0, n)[:, None]  # click-free ending
    if fx["reverse"]:
        m = min(len(x), int(0.005 * sr))
        x[:m] *= np.linspace(0, 1, m)[:, None]
    x = x.astype(np.float32)
    return x[:, 0].copy() if mono else np.ascontiguousarray(x)


class FxCache:
    """Small LRU of processed buffers, bounded by total memory."""

    def __init__(self, max_bytes: int = 600 * 1024 * 1024):
        self.max_bytes = max_bytes
        self._d: "OrderedDict[Hashable, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: Hashable, make) -> np.ndarray:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        val = make()
        with self._lock:
            if key not in self._d:
                self._d[key] = val
                self._bytes += val.nbytes
                while self._bytes > self.max_bytes and len(self._d) > 1:
                    _, old = self._d.popitem(last=False)
                    self._bytes -= old.nbytes
        return val

    def drop(self, pred) -> None:
        with self._lock:
            for k in [k for k in self._d if pred(k)]:
                self._bytes -= self._d.pop(k).nbytes
