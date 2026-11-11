# lpdriver — open-source driver + soundboard for the original Novation Launchpad

The first-generation Launchpad (2009, USB `1235:000e`) is **not** USB-MIDI class compliant
and Novation's legacy driver no longer runs on modern macOS. `lpdriver` talks to it
directly over libusb from user space, with no kernel extension needed.

## Setup (macOS)

```sh
brew install libusb
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python run.py          # opens the soundboard GUI at http://127.0.0.1:8765
```

Driver-only tools:

```sh
.venv/bin/python -m lpdriver monitor   # print button events
.venv/bin/python -m lpdriver demo      # palette, flashing, brightness, animation
.venv/bin/python -m lpdriver test      # hardware LED test
```

Linux works too (install `libusb-1.0`; you may need a udev rule granting access to `1235:000e`).

## Soundboard

* **8 pages × 72 sounds**: the top row selects the page; the grid and right column hold sounds.
* Per button: sound file, label, mode (one-shot / hold / toggle), loop, volume,
  and **idle / pressed / playing LED colours** (all 16 hardware colours), plus optional
  hardware flashing while playing. Any button can also be *Stop all* or *Stop page*.
* Top-row LEDs: amber = current page, green = page has sounds, flashing green = sounds playing there.
* GUI: live mirror of the LEDs, drag-and-drop audio onto pads, uploads, preview,
  "press on hardware to select" editing, master volume, LED brightness, LED test, animation.
* Sounds live in `sounds/`, settings in `config.json`. Hot-plugging is supported.

## Instruments

Any page can be switched (dropdown under the pad in the GUI) between **Soundboard**, **Piano / synth**
and **Drum kit**. By default page 7 is a piano and page 8 a drum kit, so the top-row buttons switch
between the soundboard and the instruments. All instrument sounds are synthesized in
`soundboard/synth.py`, with no sample files needed.

* **Piano / synth**: six sounds (E-Piano, Pluck, Supersaw lead, Bass, Pad, Organ), on side buttons 3–8;
  side buttons 1–2 shift the octave. Two grid layouts:
  *Piano keys* (pairs of rows = white keys + black keys above, 4 octaves) or
  *Scale grid* (only notes in the chosen key/scale, each row a 4th higher; 10 scales).
  The root note is highlighted; pressing a note lights every pad that plays that note.
  Notes sustain while held and use per-sound release times.
* **Drum kit**: 16 big pads (2×2 buttons each) with kick, sub kick, snare, clap, closed/open hat
  (closed chokes open), shaker, ride, 3 toms, crash, rim, cowbell, conga and zap. Three synthesized kits
  (808-, 909- and electro-style) on side buttons 1–3; side 8 stops everything.
* Top-row LEDs show each page's type: orange = piano, red = drums, green = soundboard with sounds.

## Protocol notes (as observed on hardware)

| Thing | Detail |
|---|---|
| USB | Vendor class `0xFF`, interface 0, EP `0x81` interrupt IN, EP `0x02` interrupt OUT, 8-byte packets, 10 ms interval |
| Payload | Raw MIDI byte stream **with running status**; one packet may hold several events (e.g. `34 7f 35 7f`) |
| Grid / right column | Note On `90 key vel`, X-Y layout key = `16*row + col` (col 8 = right column); vel `7f` press, `00` release |
| Top row | CC `B0 68..6F` (104–111) |
| LED colour | velocity = `16*green + red + flags`, red/green 0–3; flags `0x0C` steady, `0x08` flash, `0x00` write hidden buffer only |
| Reset | `B0 00 00` |
| Layout | `B0 00 01` X-Y, `B0 00 02` drum rack |
| Double buffering / flash | `B0 00 (0x20 + 16·copy + 8·flash + 4·update + display)`; `B0 00 28` = automatic flashing |
| Brightness | duty cycle n/d: `B0 1E 16(n-1)+(d-3)` for n<9, `B0 1F 16(n-9)+(d-3)` otherwise |
| LED test | `B0 00 7D/7E/7F` (low/medium/full) |
| Rapid update | `92 v1 v2 …` fills 64 grid → 8 right → 8 top LEDs in order; any other message resets the cursor |

## License

MIT
