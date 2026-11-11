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

## Setup (Linux)

```sh
linux/install.sh                 # libusb + PortAudio + libsndfile, udev rule, Python venv
.venv/bin/python run.py          # GUI at http://127.0.0.1:8765
```

* `linux/install.sh --service` also starts it at login (systemd user service),
  `--desktop` adds an application-menu launcher.
* Only the USB permission? `sudo linux/install-udev.sh` installs
  `linux/50-novation-launchpad.rules` and applies it to a plugged-in Launchpad (no replug needed).
  Without it the driver prints a hint instead of failing silently.
* Headless box / Raspberry Pi: `run.py --host 0.0.0.0 --no-browser`, then open
  `http://<machine-ip>:8765` from any computer or phone on the network (there is no login, so only do this on a trusted network).

## Where the sound plays

The header selector chooses **Host machine** (the computer running `run.py`), **This browser**
(the server streams its mix to the GUI page, about 60–100 ms of latency) or **Both**. Default: host on macOS,
browser on Linux. The browser needs one click before it may play sound. Choose a specific
host sound card with `LAUNCHPAD_AUDIO_DEVICE=<name or index>` (list them with `python -m sounddevice`).

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

Any page can be switched (dropdown under the pad in the GUI) between **Soundboard**, **Piano / synth**,
**Drum kit**, **Step sequencer** and **Looper**. By default page 5 is a sequencer and page 6 a looper. By default page 7 is a piano and page 8 a drum kit, so the top-row buttons switch
between the soundboard and the instruments. All instrument sounds are synthesized in
`soundboard/synth.py`, with no sample files needed.

* **Piano / synth**: 15 sounds (E-Piano, Pluck, Supersaw lead, Bass, Pad, Organ, FM bell, Strings, Brass,
  Acid bass, Marimba, Karplus-Strong guitar, Vox choir, Chiptune, 808 sub bass). Any six of them sit on
  side buttons 3–8 (choose which in the GUI); side buttons 1–2 shift the octave. Two grid layouts:
  *Piano keys* (pairs of rows = white keys + black keys above, 4 octaves) or
  *Scale grid* (only notes in the chosen key/scale, each row a 4th higher; 10 scales).
  The root note is highlighted; pressing a note lights every pad that plays that note.
  Notes sustain while held and use per-sound release times.
* **Drum kit**: 16 big pads (2×2 buttons each) with kick, sub kick, snare, clap, closed/open hat
  (closed chokes open), shaker, ride, 3 toms, crash, rim, cowbell, conga and zap. Five synthesized kits
  (808, 909, electro, lo-fi, industrial) on side buttons 1–5; side 8 stops everything.
* **Step sequencer**: 8 rows (pick the drum for each) × 8/16/32 sixteenth-note steps, swing.
  Side buttons: 1–4 step block, 5 play/stop, 6 next kit, 7 length, 8 clear. Every sequencer page plays
  along while the clock runs, and the patterns are saved.
* **Looper**: 8 loops locked to the tempo. Arm one, then play any other page; what you play is recorded
  (1/2/4/8 bars, optional 16th-note quantize), then repeats. Each row is a loop: pads 1–6
  record / mute / unmute, 7 overdub, 8 clear. Side: play, metronome, tempo −/+, tap tempo, loop length,
  quantize, clear all. In the GUI: undo overdub, and *Save as sound* bounces a loop to a WAV in
  `sounds/` so you can put it on a soundboard pad. Loops live in memory (not saved).
* **Tempo bar** (GUI): play/stop (or space), BPM, tap tempo, metronome. All timing is scheduled
  ahead on the audio sample clock, so it stays tight even though it's Python.

## Effects

Every soundboard pad and every instrument page has an **Effects** panel: pitch (±24 semitones) and fine tune,
distortion, bit-crusher, low-pass tone, echo (with time), reverb and reverse. On synths, pitch transposes;
on samples and drums it works like a sampler (higher = faster). Processed sounds are cached.
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
