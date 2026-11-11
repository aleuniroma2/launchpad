"""Low-latency polyphonic sample player (sounddevice + soundfile + numpy)."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3", ".caf"}


@dataclass
class Voice:
    key: str
    data: np.ndarray  # (frames, 2) stereo or (frames,) mono float32
    pos: int = 0
    gain: float = 1.0
    loop: bool = False
    fade_out: int = 0  # frames remaining in a fade-out (0 = not fading)
    fade_len: int = 256


class AudioEngine:
    FADE = 256  # frames of fade-out on stop, avoids clicks

    def __init__(self, samplerate: Optional[int] = None, max_voices: int = 48):
        dev = sd.query_devices(kind="output")
        self.samplerate = int(samplerate or dev["default_samplerate"])
        self.max_voices = max_voices
        self.master = 0.9
        self._cache: Dict[str, np.ndarray] = {}
        self._voices: List[Voice] = []
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(samplerate=self.samplerate, channels=2, dtype="float32",
                                       blocksize=256, latency="low", callback=self._callback)
        self._stream.start()

    # ---------------------------------------------------------------- loading
    def load(self, path: str) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
        if sr != self.samplerate and len(data) > 1:
            n = int(round(len(data) * self.samplerate / sr))
            src = np.linspace(0, len(data) - 1, n)
            idx = np.arange(len(data))
            data = np.stack([np.interp(src, idx, data[:, c]) for c in range(2)], axis=1)
        data = np.ascontiguousarray(data, dtype=np.float32)
        self._cache[path] = data
        return data

    def invalidate(self, path: str) -> None:
        self._cache.pop(path, None)

    # ---------------------------------------------------------------- control
    def play(self, key: str, path: str, gain: float = 1.0, loop: bool = False) -> None:
        self.play_data(key, self.load(path), gain, loop)

    def play_data(self, key: str, data: np.ndarray, gain: float = 1.0, loop: bool = False) -> None:
        """Play an in-memory buffer (e.g. a synthesized note)."""
        with self._lock:
            for v in self._voices:  # retrigger: fade out the previous instance
                if v.key == key and not v.fade_out:
                    v.fade_out = self.FADE
            if len(self._voices) >= self.max_voices:
                self._voices.pop(0)
            self._voices.append(Voice(key, data, 0, gain, loop))

    def stop(self, key: str, release: float = 0.0) -> None:
        """Stop a voice, fading out over `release` seconds (at least a few ms)."""
        n = max(self.FADE, int(release * self.samplerate))
        with self._lock:
            for v in self._voices:
                if v.key == key and not v.fade_out:
                    v.fade_out = v.fade_len = n

    def stop_all(self) -> None:
        with self._lock:
            for v in self._voices:
                if not v.fade_out:
                    v.fade_out = self.FADE

    def playing(self) -> set:
        with self._lock:
            return {v.key for v in self._voices if not v.fade_out}

    def progress(self) -> Dict[str, float]:
        with self._lock:
            return {v.key: v.pos / max(1, len(v.data)) for v in self._voices if not v.fade_out}

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    # ---------------------------------------------------------------- mixer
    def _callback(self, out, frames, _time, _status):
        out.fill(0)
        with self._lock:
            alive = []
            for v in self._voices:
                written = 0
                while written < frames:
                    n = min(frames - written, len(v.data) - v.pos)
                    if n <= 0:
                        if v.loop:
                            v.pos = 0
                            continue
                        break
                    chunk = v.data[v.pos:v.pos + n] * v.gain
                    if chunk.ndim == 1:
                        chunk = np.repeat(chunk[:, None], 2, axis=1)
                    if v.fade_out:
                        k = min(n, v.fade_out)
                        ramp = np.linspace(v.fade_out / v.fade_len, (v.fade_out - k) / v.fade_len,
                                           k, dtype=np.float32)[:, None]
                        chunk[:k] *= ramp
                        chunk[k:] = 0
                        v.fade_out -= k
                        if v.fade_out <= 0:
                            v.fade_out = -1  # done
                    out[written:written + n] += chunk
                    v.pos += n
                    written += n
                    if v.fade_out == -1:
                        break
                finished = v.fade_out == -1 or (not v.loop and v.pos >= len(v.data))
                if not finished:
                    alive.append(v)
            self._voices = alive
        out *= self.master
        np.clip(out, -1.0, 1.0, out=out)
