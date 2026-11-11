"""
python -m lpdriver monitor   # print button events, light pads while pressed
python -m lpdriver demo      # colour palette, flashing, double-buffered animation
python -m lpdriver test      # hardware LED test (low/medium/full)
"""
import math
import sys
import time

from . import Launchpad, Button, Color, all_buttons


def wait_connected(lp: Launchpad, timeout: float = 5.0) -> None:
    t = time.time()
    while not lp.connected and time.time() - t < timeout:
        time.sleep(0.05)
    if not lp.connected:
        sys.exit("Launchpad (USB 1235:000e) not found. Is it plugged in?")


def rainbow_frames(n: int = 24):
    for i in range(n):
        frame = {}
        for b in all_buttons():
            x, y = (b.x, b.y + 1)
            v = math.sin((x + y) * 0.6 - i * 0.5)
            frame[b] = Color(round(1.5 + 1.5 * v), round(1.5 - 1.5 * v))
        yield frame


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "monitor"
    lp = Launchpad()

    if cmd == "monitor":
        def on(ev):
            print(f"{ev.button.id:7s} {'DOWN' if ev.pressed else 'up'}", flush=True)
            if ev.button.kind == "top":
                lp.set_led(ev.button, Color(3, 3) if ev.pressed else Color())
            else:
                lp.set_led(ev.button, Color(3, 1) if ev.pressed else Color())
        lp.on_button(on)
        lp.start(); wait_connected(lp)
        print("Connected. Press buttons (Ctrl+C to quit).")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    elif cmd == "demo":
        lp.start(); wait_connected(lp)
        print("16-colour palette (red increases to the right, green downward)")
        lp.set_leds({Button("pad", x, y): (Color(x % 4, y % 4), False) for x in range(8) for y in range(8)})
        time.sleep(2)
        print("Hardware flashing on the right half")
        lp.set_leds({Button("pad", x, y): (Color(x % 4, y % 4), x >= 4) for x in range(8) for y in range(8)})
        time.sleep(3)
        print("Brightness sweep (duty cycle)")
        for n, d in [(1, 18), (1, 9), (1, 5), (1, 3), (1, 2), (1, 5)]:
            lp.set_brightness(n, d); time.sleep(0.6)
        print("Double-buffered animation")
        lp.animate(rainbow_frames(40), fps=12)
        lp.clear(); time.sleep(0.2)
    elif cmd == "test":
        lp.start(); wait_connected(lp)
        for lvl in (1, 2, 3):
            print("LED test level", lvl)
            lp.test_leds(lvl, duration=0.8); time.sleep(1.0)
    else:
        print(__doc__); return
    lp.stop()


if __name__ == "__main__":
    main(sys.argv)
