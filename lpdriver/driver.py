"""
Open-source userspace driver for the original Novation Launchpad (2009, USB 1235:000e).

The original Launchpad is NOT USB-MIDI class compliant: it exposes a single
vendor-specific interface (class 0xFF) with two 8-byte interrupt endpoints:

    EP 0x81  IN   button events
    EP 0x02  OUT  LED / control commands

The payload on both endpoints is a plain MIDI byte stream (with running status),
so we talk to it directly via libusb instead of needing Novation's kext.

Hardware capabilities exposed here:
  * 80 buttons: 8x8 grid, 8 round "scene" buttons on the right, 8 round buttons on top
  * bi-colour LEDs: red 0-3 x green 0-3 (16 colours incl. off)
  * hardware LED flashing
  * double buffering (tear-free animation)
  * LED brightness via multiplexing duty cycle
  * rapid LED update (2 LEDs per message, 80 LEDs per frame)
  * X-Y and drum-rack note layouts
  * built-in all-LED test mode (low / medium / full)
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import usb.core
import usb.util

VID, PID = 0x1235, 0x000E
EP_IN, EP_OUT = 0x81, 0x02

NOTE_ON, NOTE_OFF, CC, RAPID = 0x90, 0x80, 0xB0, 0x92

# Velocity flag bits for LED messages
FLAG_COPY = 0x04   # write to both buffers
FLAG_CLEAR = 0x08  # clear the other buffer's copy of this LED
FLAGS_NORMAL = FLAG_COPY | FLAG_CLEAR  # steady LED
FLAGS_FLASH = FLAG_CLEAR               # LED flashes when flash mode is running
FLAGS_BUFFER = 0                       # write only to the "update" buffer

# Duty-cycle presets (numerator, denominator); hardware default is 1/5.
BRIGHTNESS_LEVELS: List[Tuple[int, int]] = [
    (1, 18), (1, 12), (1, 9), (1, 7), (1, 6), (1, 5), (1, 4),
    (1, 3), (2, 5), (8, 16), (2, 3), (3, 4), (16, 16),
]


@dataclass(frozen=True)
class Color:
    """Launchpad LED colour: red and green each 0 (off) .. 3 (full)."""
    red: int = 0
    green: int = 0

    def velocity(self, flags: int = FLAGS_NORMAL) -> int:
        r = max(0, min(3, int(self.red)))
        g = max(0, min(3, int(self.green)))
        return (g << 4) | r | flags

    @property
    def off(self) -> bool:
        return self.red == 0 and self.green == 0


OFF = Color(0, 0)
NAMED_COLORS: Dict[str, Color] = {
    "off": OFF,
    "red_low": Color(1, 0), "red": Color(3, 0),
    "green_low": Color(0, 1), "green": Color(0, 3),
    "amber_low": Color(1, 1), "amber": Color(3, 3),
    "yellow": Color(2, 3), "orange": Color(3, 2), "lime": Color(1, 3),
}


@dataclass(frozen=True)
class Button:
    """kind: 'pad' (x,y 0-7), 'side' (y 0-7, right column), 'top' (x 0-7, top row)."""
    kind: str
    x: int
    y: int

    @property
    def id(self) -> str:
        if self.kind == "pad":
            return f"pad{self.x}{self.y}"
        if self.kind == "side":
            return f"side{self.y}"
        return f"top{self.x}"

    @staticmethod
    def from_id(bid: str) -> "Button":
        if bid.startswith("pad"):
            return Button("pad", int(bid[3]), int(bid[4]))
        if bid.startswith("side"):
            return Button("side", 8, int(bid[4:]))
        if bid.startswith("top"):
            return Button("top", int(bid[3:]), -1)
        raise ValueError(bid)


def all_buttons() -> List[Button]:
    """All 80 buttons, in rapid-update order (grid rows, side column, top row)."""
    out = [Button("pad", x, y) for y in range(8) for x in range(8)]
    out += [Button("side", 8, y) for y in range(8)]
    out += [Button("top", x, -1) for x in range(8)]
    return out


@dataclass
class ButtonEvent:
    button: Button
    pressed: bool
    timestamp: float


class Launchpad:
    """
    Thread-safe driver. Usage:

        lp = Launchpad()
        lp.on_button(lambda ev: print(ev))
        lp.start()                 # opens device (and keeps reconnecting if unplugged)
        lp.set_led(Button("pad", 0, 0), Color(3, 0))
    """

    def __init__(self, auto_reconnect: bool = True, layout: str = "xy",
                 brightness: Tuple[int, int] = (1, 5), libusb_path: Optional[str] = None):
        self.auto_reconnect = auto_reconnect
        self._layout = layout
        self._brightness = brightness
        self._libusb_path = libusb_path
        self._dev = None
        self._connected = threading.Event()
        self._running = False
        self._out: "queue.Queue[bytes]" = queue.Queue()
        self._lock = threading.RLock()
        self._anim_lock = threading.Lock()
        # Shadow LED state so we can redraw on reconnect and answer queries
        self._leds: Dict[Button, Tuple[Color, bool]] = {b: (OFF, False) for b in all_buttons()}
        self._button_cbs: List[Callable[[ButtonEvent], None]] = []
        self._conn_cbs: List[Callable[[bool], None]] = []
        self._held: set = set()
        self._threads: List[threading.Thread] = []

    # ------------------------------------------------------------------ public API
    def on_button(self, cb: Callable[[ButtonEvent], None]) -> None:
        self._button_cbs.append(cb)

    def on_connection(self, cb: Callable[[bool], None]) -> None:
        self._conn_cbs.append(cb)

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def held(self) -> set:
        return set(self._held)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for target in (self._reader_loop, self._writer_loop):
            t = threading.Thread(target=target, daemon=True, name=target.__name__)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        if self.connected:
            self._send(bytes([CC, 0, 0]))  # reset: all LEDs off
            time.sleep(0.15)  # let the writer thread flush
        self._running = False
        self._close()

    def led(self, button: Button) -> Tuple[Color, bool]:
        return self._leds[button]

    def leds(self) -> Dict[Button, Tuple[Color, bool]]:
        return dict(self._leds)

    def set_led(self, button: Button, color: Color, flash: bool = False) -> None:
        """Set one LED. flash=True makes it blink using the hardware flash timer."""
        if color.off:
            flash = False
        with self._lock:
            if self._leds.get(button) == (color, flash):
                return
            self._leds[button] = (color, flash)
        self._send(self._led_msg(button, color.velocity(FLAGS_FLASH if flash else FLAGS_NORMAL)))

    def set_leds(self, updates: Dict[Button, Tuple[Color, bool]], force: bool = False) -> None:
        """Set many LEDs; automatically uses rapid update when it is cheaper."""
        with self._lock:
            changed = {b: v for b, v in updates.items()
                       if force or self._leds.get(b) != v}
            self._leds.update(changed)
        if not changed:
            return
        if len(changed) > 24 and self._layout == "xy":
            self._send(self._frame_bytes(self._leds))
        else:
            self._send(b"".join(
                self._led_msg(b, c.velocity(FLAGS_FLASH if f and not c.off else FLAGS_NORMAL))
                for b, (c, f) in changed.items()))

    def clear(self) -> None:
        self.set_leds({b: (OFF, False) for b in all_buttons()})

    def redraw(self) -> None:
        """Re-send the full shadow state (rapid update)."""
        if self._layout == "xy":
            self._send(self._frame_bytes(self._leds))
        else:
            self.set_leds(dict(self._leds), force=True)

    def set_brightness(self, numerator: int, denominator: int) -> None:
        """LED duty cycle numerator/denominator: 1..16 / 3..18 (default 1/5)."""
        n = max(1, min(16, int(numerator)))
        d = max(3, min(18, int(denominator)))
        self._brightness = (n, d)
        self._send(self._brightness_msg())

    def set_layout(self, layout: str) -> None:
        """'xy' (default) or 'drum' (drum-rack note mapping). Transparent to callers."""
        if layout not in ("xy", "drum"):
            raise ValueError(layout)
        self._layout = layout
        self._send(bytes([CC, 0, 1 if layout == "xy" else 2]))
        self.redraw()

    def test_leds(self, level: int = 3, duration: float = 1.0) -> None:
        """Hardware all-LED test: level 1=low, 2=medium, 3=full. Restores state after."""
        level = max(1, min(3, level))
        self._send(bytes([CC, 0, 0x7C + level]))
        def restore():
            time.sleep(duration)
            self._init_device_state()
        threading.Thread(target=restore, daemon=True).start()

    def animate(self, frames: Iterable[Dict[Button, Color]], fps: float = 12.0) -> None:
        """
        Play frames tear-free using the hardware double buffer: each frame is
        written to the hidden buffer and then swapped in. Restores state after.
        """
        if self._layout != "xy":
            return
        with self._anim_lock:
            update = 0
            self._send(bytes([CC, 0, 0x20 | 1]))  # display buffer 1, write buffer 0
            for frame in frames:
                full = {b: (frame.get(b, OFF), False) for b in all_buttons()}
                ctl = bytes([CC, 0, 0x20 | (4 if update == 1 else 0) | (update ^ 1)])
                self._send(self._frame_bytes(full, FLAGS_BUFFER, prefix=ctl))
                # show the buffer we just wrote, write into the other one
                self._send(bytes([CC, 0, 0x20 | (4 if update == 0 else 0) | update]))
                update ^= 1
                time.sleep(1.0 / fps)
            self._init_device_state()

    # ------------------------------------------------------------------ protocol helpers
    def _note_for(self, b: Button) -> Tuple[int, int]:
        """(status, data1) for a button in the current layout."""
        if b.kind == "top":
            return CC, 104 + b.x
        if self._layout == "drum":
            if b.kind == "side":
                return NOTE_ON, 100 + (7 - b.y)
            base = 36 if b.x < 4 else 68
            return NOTE_ON, base + (7 - b.y) * 4 + (b.x % 4)
        return NOTE_ON, 16 * b.y + (8 if b.kind == "side" else b.x)

    def _button_for(self, status: int, data1: int) -> Optional[Button]:
        if status == CC:
            return Button("top", data1 - 104, -1) if 104 <= data1 <= 111 else None
        if self._layout == "drum":
            if 100 <= data1 <= 107:
                return Button("side", 8, 7 - (data1 - 100))
            if 36 <= data1 <= 99:
                half, idx = (0, data1 - 36) if data1 < 68 else (4, data1 - 68)
                return Button("pad", half + idx % 4, 7 - idx // 4)
            return None
        y, x = divmod(data1, 16)
        if y > 7 or x > 8:
            return None
        return Button("side", 8, y) if x == 8 else Button("pad", x, y)

    def _led_msg(self, b: Button, velocity: int) -> bytes:
        status, d1 = self._note_for(b)
        return bytes([status, d1, velocity])

    def _frame_bytes(self, leds: Dict[Button, Tuple[Color, bool]], flags: Optional[int] = None,
                     prefix: bytes = bytes([CC, 0, 0x28])) -> bytes:
        # A non-rapid message first resets the rapid-update cursor to the first LED.
        out = bytearray(prefix) + bytearray([RAPID])
        for b in all_buttons():
            c, f = leds.get(b, (OFF, False))
            fl = flags if flags is not None else (FLAGS_FLASH if f and not c.off else FLAGS_NORMAL)
            out.append(c.velocity(fl))
        return bytes(out)

    def _brightness_msg(self) -> bytes:
        n, d = self._brightness
        if n < 9:
            return bytes([CC, 0x1E, 16 * (n - 1) + (d - 3)])
        return bytes([CC, 0x1F, 16 * (n - 9) + (d - 3)])

    def _init_device_state(self) -> None:
        self._send(bytes([CC, 0, 0]))                              # reset
        self._send(bytes([CC, 0, 1 if self._layout == "xy" else 2]))  # layout
        self._send(self._brightness_msg())
        self._send(bytes([CC, 0, 0x28]))                           # auto flash mode
        self.redraw()

    # ------------------------------------------------------------------ USB plumbing
    def _send(self, data: bytes) -> None:
        if self.connected:
            self._out.put(data)

    def _backend(self):
        import usb.backend.libusb1 as libusb1
        candidates = [self._libusb_path, os.environ.get("LIBUSB_PATH"),
                      "/opt/homebrew/lib/libusb-1.0.dylib", "/usr/local/lib/libusb-1.0.dylib"]
        for c in candidates:
            if c and os.path.exists(c):
                return libusb1.get_backend(find_library=lambda _x, c=c: c)
        return libusb1.get_backend()

    def _open(self) -> bool:
        dev = usb.core.find(idVendor=VID, idProduct=PID, backend=self._backend())
        if dev is None:
            return False
        try:
            try:
                if dev.is_kernel_driver_active(0):
                    dev.detach_kernel_driver(0)
            except (NotImplementedError, usb.core.USBError):
                pass
            dev.set_configuration()
            usb.util.claim_interface(dev, 0)
        except usb.core.USBError:
            usb.util.dispose_resources(dev)
            return False
        self._dev = dev
        while not self._out.empty():
            self._out.get_nowait()
        self._held.clear()
        self._connected.set()
        self._init_device_state()
        for cb in self._conn_cbs:
            cb(True)
        return True

    def _close(self) -> None:
        was = self._connected.is_set()
        self._connected.clear()
        dev, self._dev = self._dev, None
        if dev is not None:
            try:
                usb.util.release_interface(dev, 0)
            except Exception:
                pass
            usb.util.dispose_resources(dev)
        if was:
            for cb in self._conn_cbs:
                cb(False)

    def _reader_loop(self) -> None:
        status, data = 0, []
        while self._running:
            if not self.connected:
                if not self._open():
                    if not self.auto_reconnect:
                        self._running = False
                        return
                    time.sleep(1.0)
                    continue
                status, data = 0, []
            try:
                packet = bytes(self._dev.read(EP_IN, 8, timeout=100))
            except usb.core.USBTimeoutError:
                continue
            except (usb.core.USBError, AttributeError):
                self._close()
                continue
            # MIDI stream parser with running status (the device relies on it)
            for byte in packet:
                if byte & 0x80:
                    status, data = byte, []
                    continue
                data.append(byte)
                if len(data) == 2:
                    self._dispatch(status, data[0], data[1])
                    data = []

    def _dispatch(self, status: int, d1: int, d2: int) -> None:
        kind = status & 0xF0
        if kind not in (NOTE_ON, NOTE_OFF, CC):
            return
        b = self._button_for(kind if kind != NOTE_OFF else NOTE_ON, d1)
        if b is None:
            return
        pressed = kind != NOTE_OFF and d2 > 0
        (self._held.add if pressed else self._held.discard)(b)
        ev = ButtonEvent(b, pressed, time.time())
        for cb in self._button_cbs:
            try:
                cb(ev)
            except Exception as e:  # never let a callback kill the USB thread
                print(f"[lpdriver] callback error: {e!r}")

    def _writer_loop(self) -> None:
        while self._running:
            try:
                chunk = self._out.get(timeout=0.2)
            except queue.Empty:
                continue
            buf = bytearray(chunk)
            while len(buf) < 512:  # coalesce pending writes into one transfer
                try:
                    buf += self._out.get_nowait()
                except queue.Empty:
                    break
            dev = self._dev
            if dev is None:
                continue
            try:
                dev.write(EP_OUT, bytes(buf), timeout=2000)
            except (usb.core.USBError, AttributeError):
                pass  # reader thread notices disconnects and handles reconnection
