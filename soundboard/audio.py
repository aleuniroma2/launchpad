"""Low-latency polyphonic sample player (sounddevice + soundfile + numpy).

Besides immediate playback, voices can be scheduled on an absolute sample clock
(`play_data(..., at=frame)` / `stop(..., at=frame)`), which is what the step sequencer
and the looper use for sample-accurate timing.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
except OSError as _e:  # PortAudio missing (e.g. Linux without libportaudio2)
    sd = None
    _SD_ERROR = _e

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
    start: int = 0     # absolute frame at which the voice starts
    stop_at: Optional[int] = None   # absolute frame at which a scheduled release begins
    stop_len: int = 256


class _NullStream:
    """Stand-in output stream when no audio device exists: keeps the clock running."""

    def __init__(self, samplerate: int, blocksize: int, callback):
        self.sr, self.bs, self.cb = samplerate, blocksize, callback
        self._run = False

    def start(self):
        self._run = True
        threading.Thread(target=self._loop, daemon=True, name="null-audio").start()

    def _loop(self):
        buf = np.zeros((self.bs, 2), dtype=np.float32)
        period, nxt = self.bs / self.sr, time.perf_counter()
        while self._run:
            self.cb(buf, self.bs, None, None)
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def stop(self):
        self._run = False

    def close(self):
        pass


class AudioEngine:
    FADE = 256  # frames of fade-out on stop, avoids clicks
    BLOCK = 256

    OUTPUTS = ("host", "browser", "both")

    def __init__(self, samplerate: Optional[int] = None, max_voices: int = 64,
                 device: Optional[str] = None, output: str = "host"):
        device = device or os.environ.get("LAUNCHPAD_AUDIO_DEVICE") or None
        if device and device.isdigit():
            device = int(device)
        self._device = device
        self.master = 0.9
        self.max_voices = max_voices
        self._cache: Dict[str, np.ndarray] = {}
        self._voices: List[Voice] = []
        self._lock = threading.Lock()
        self._switch = threading.Lock()
        self.frame = 0                       # absolute frames rendered so far
        self._cb_time = time.perf_counter()  # wall clock at the last callback
        self._subs: List["queue.Queue[bytes]"] = []  # browser listeners (int16 stereo PCM blocks)
        self._dev_info = None
        if sd is not None:
            try:
                self._dev_info = sd.query_devices(device, kind="output")
            except Exception as e:
                print(f"[audio] no usable output device ({e})")
        else:
            print(f"[audio] PortAudio not available ({_SD_ERROR}). On Linux: sudo apt install libportaudio2")
        # One fixed sample rate for the whole session, so caches stay valid when switching output.
        self.samplerate = int(samplerate or (self._dev_info["default_samplerate"] if self._dev_info else 48000))
        self.output = "browser"
        self.to_browser = True
        self.device_name = "none"
        self._stream = None
        self.set_output(output)

    @property
    def host_available(self) -> bool:
        return self._dev_info is not None

    def set_output(self, output: str) -> str:
        """Switch between the host sound card, streaming to browsers, or both (live)."""
        if output not in self.OUTPUTS:
            raise ValueError(output)
        with self._switch:
            old = self._stream
            if old is not None:
                old.stop()
                old.close()
            stream = None
            if output in ("host", "both") and self._dev_info is not None:
                try:
                    stream = sd.OutputStream(samplerate=self.samplerate, channels=2, dtype="float32",
                                             blocksize=self.BLOCK, latency="low", device=self._device,
                                             callback=self._callback)
                    self.device_name = self._dev_info["name"]
                except Exception as e:
                    print(f"[audio] cannot open the host output ({e}); using the browser instead")
            if stream is None:
                if output in ("host", "both") and self._dev_info is None:
                    print("[audio] no host audio device: sound only plays in the browser")
                self.device_name = "browser"
                stream = _NullStream(self.samplerate, self.BLOCK, self._callback)  # paced clock
            self.output = output
            # stream to browsers when asked to, or when there is no sound card to play on
            self.to_browser = output != "host" or isinstance(stream, _NullStream)
            self._stream = stream
            stream.start()
        return self.output

    # ---------------------------------------------------------------- browser streaming
    def subscribe(self) -> "queue.Queue[bytes]":
        q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    @property
    def listeners(self) -> int:
        return len(self._subs)

    # ---------------------------------------------------------------- clock
    def now(self) -> int:
        """Best estimate of the absolute frame being heard right now."""
        return self.frame + int((time.perf_counter() - self._cb_time) * self.samplerate)

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

    def play_data(self, key: str, data: np.ndarray, gain: float = 1.0, loop: bool = False,
                  at: Optional[int] = None) -> None:
        """Play an in-memory buffer now, or at absolute frame `at`."""
        with self._lock:
            start = self.frame if at is None else int(at)
            for v in self._voices:  # retrigger: fade out the previous instance
                if v.key == key and not v.fade_out:
                    if at is None:
                        v.fade_out = self.FADE
                    elif v.stop_at is None or v.stop_at > start:
                        v.stop_at, v.stop_len = start, self.FADE
            if len(self._voices) >= self.max_voices:
                self._voices.pop(0)
            self._voices.append(Voice(key, data, 0, gain, loop, start=start))

    def stop(self, key: str, release: float = 0.0, at: Optional[int] = None) -> None:
        """Stop a voice, fading out over `release` seconds (at least a few ms)."""
        n = max(self.FADE, int(release * self.samplerate))
        with self._lock:
            for v in self._voices:
                if v.key == key and not v.fade_out:
                    if at is None:
                        v.fade_out = v.fade_len = n
                    elif v.start < at and (v.stop_at is None or v.stop_at > at):
                        v.stop_at, v.stop_len = int(at), n

    def stop_prefix(self, prefix: str, release: float = 0.0) -> None:
        """Stop every voice whose key starts with `prefix` (also cancels scheduled ones)."""
        n = max(self.FADE, int(release * self.samplerate))
        with self._lock:
            keep = []
            for v in self._voices:
                if v.key.startswith(prefix):
                    if v.start > self.frame:
                        continue  # not started yet: just drop it
                    if not v.fade_out:
                        v.fade_out = v.fade_len = n
                keep.append(v)
            self._voices = keep

    def cancel_future(self) -> None:
        """Drop every voice scheduled to start later (e.g. when the transport stops)."""
        with self._lock:
            self._voices = [v for v in self._voices if v.start <= self.frame]

    def stop_all(self) -> None:
        with self._lock:
            self._voices = [v for v in self._voices if v.start <= self.frame]
            for v in self._voices:
                if not v.fade_out:
                    v.fade_out = self.FADE

    def playing(self) -> set:
        with self._lock:
            return {v.key for v in self._voices if not v.fade_out and v.start <= self.frame}

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
            t0 = self.frame
            alive = []
            for v in self._voices:
                if v.start >= t0 + frames:      # scheduled for a later block
                    alive.append(v)
                    continue
                if v.stop_at is not None and v.stop_at < t0 + frames and not v.fade_out:
                    v.fade_out = v.fade_len = v.stop_len
                    v.stop_at = None
                written = max(0, v.start - t0)  # sample-accurate start inside the block
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
            self.frame = t0 + frames
            self._cb_time = time.perf_counter()
        out *= self.master
        np.clip(out, -1.0, 1.0, out=out)
        if self._subs and self.to_browser:
            pcm = (out * 32767).astype("<i2").tobytes()
            for q in list(self._subs):
                try:
                    q.put_nowait(pcm)
                except queue.Full:  # a stalled browser: drop audio rather than grow memory
                    pass
