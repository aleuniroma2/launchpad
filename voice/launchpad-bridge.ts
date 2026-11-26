/**
 * Launchpad bridge for CHIMERA (copied into src/ by voice/build.sh; nothing else of CHIMERA is changed
 * except one import line in src/main.tsx).
 *
 * The Launchpad soundboard server streams pad events on /api/voice (Server-Sent Events). This module
 * performs them with CHIMERA's own store and controls, exactly as its on-screen controls would, so
 * the page stays in sync and everything CHIMERA can do remains clickable on screen too.
 * It only runs when the page is served from /voice/ (or opened with ?launchpad).
 */
import { initChimera } from "./wasm/chimera.ts";
import type { Mode } from "./ui/store.ts";

// CHIMERA's rule: nothing may touch the engine before the wasm is loaded, so the store (and what
// imports it) is loaded dynamically, after initChimera() has resolved (see boot()).
let useStore!: typeof import("./ui/store.ts").useStore;
let stopSinging!: typeof import("./ui/singing.ts").stopSinging;
let stopSpeaking!: typeof import("./ui/speech.ts").stopSpeaking;
let PRESETS!: typeof import("./ui/body.ts").PRESETS;
let nearestBody!: typeof import("./ui/body.ts").nearestBody;
let VOWEL_TARGETS!: typeof import("./wasm/engine.ts").VOWEL_TARGETS;

type Ev = { cmd: string; [k: string]: any };

/** The voice encoders (VoiceControls.tsx): where each lives and its range. norm 0..1 → value. */
const PARAMS: Record<string, { to: "source" | "artic"; key: string; min: number; max: number; log?: boolean; invert?: boolean }> = {
  pitch: { to: "source", key: "f0", min: 40, max: 1100, log: true },
  breath: { to: "source", key: "pressure", min: 2, max: 22 },
  pressed: { to: "source", key: "tension", min: 0, max: 1 },
  vibrato: { to: "source", key: "vibratoDepth", min: 0, max: 120 },
  bright: { to: "source", key: "brightness", min: 0, max: 1 },
  breathy: { to: "source", key: "breathiness", min: 0, max: 1 },
  rough: { to: "source", key: "roughness", min: 0, max: 1 },
  fry: { to: "source", key: "fry", min: 0, max: 1 },
  glide: { to: "source", key: "glide", min: 0.005, max: 0.6, log: true },
  rounding: { to: "artic", key: "lipWidth", min: 0, max: 1, invert: true }, // shown as 1 - lipWidth
  lipsopen: { to: "artic", key: "lipAp", min: -0.3, max: 2.2 },
  protrude: { to: "artic", key: "lipProt", min: -0.1, max: 1.4 },
  jaw: { to: "artic", key: "jaw", min: 0, max: 2.4 },
  root: { to: "artic", key: "root", min: -0.9, max: 0.9 },
  tipcurl: { to: "artic", key: "ttCurl", min: 0, max: 1 },
  velum: { to: "artic", key: "velum", min: -0.25, max: 1.2 },
  larynx: { to: "artic", key: "larynx", min: -1.5, max: 1.5 },
  twang: { to: "artic", key: "epilarynx", min: 0, max: 1 },
  falsefolds: { to: "artic", key: "ventricular", min: 0, max: 1 },
};

const st = () => useStore.getState();

function setNative(el: HTMLInputElement | HTMLSelectElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value")!.set!;
  setter.call(el, value);
  el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true }));
}
const clickText = (re: RegExp) => {
  for (const b of Array.from(document.querySelectorAll("button"))) {
    if (re.test((b.textContent ?? "").trim()) || re.test(b.getAttribute("aria-label") ?? "")) {
      (b as HTMLButtonElement).click();
      return true;
    }
  }
  return false;
};
const primary = () => (document.querySelector(".action-bar .act-primary") as HTMLButtonElement | null)?.click();

function stopAll() {
  const s = st();
  for (const n of [...s.heldNotes]) s.noteOff(n);
  s.setHold(false);
  s.setBeatPlaying(false);
  stopSinging();
  stopSpeaking();
}

function perform(e: Ev) {
  const s = st();
  switch (e.cmd) {
    case "hello": {
      const v = e.state ?? {};
      if (v.mode && v.mode !== s.mode) s.setMode(v.mode as Mode);
      if (v.body && PRESETS[v.body as keyof typeof PRESETS]) s.setBody(PRESETS[v.body as keyof typeof PRESETS]);
      if (v.creature) s.setCreature(v.creature);
      break;
    }
    case "note_on": void s.start(); s.noteOn(e.midi, e.vel ?? 96); break;
    case "note_off": s.noteOff(e.midi); break;
    case "mode": for (const n of [...s.heldNotes]) s.noteOff(n); s.setMode(e.mode as Mode); break;
    case "body": s.setBody(PRESETS[e.name as keyof typeof PRESETS]); break;
    case "breath": if (e.on) { if (!s.hold) void s.toggleBreath(); } else s.setHold(false); break;
    case "whisper": s.setSource({ voiced: !e.on }); break;
    case "vowel": {
      const t = (VOWEL_TARGETS as any)[nearestBody(s.body)]?.[e.ipa]; // calibrated per body, then vowel
      if (t) s.setArtic(t.artic, true);
      break;
    }
    case "primary": void s.start(); primary(); break;
    case "tune": {
      const sel = document.querySelector('select[aria-label="Tune"]') as HTMLSelectElement | null;
      if (sel) setNative(sel, e.id);
      break;
    }
    case "fromstart": (document.querySelector(".from-start") as HTMLButtonElement | null)?.click(); break;
    case "ipa": clickText(/Text is IPA/i); break;
    case "lang": clickText(new RegExp(e.lang === "TR" ? "^(TR|Turkish)$" : "^(EN|English)$", "i")); break;
    case "rate": {
      const r = document.querySelector(".act-slider input[type=range]") as HTMLInputElement | null;
      if (r) {
        const lo = +r.min, hi = +r.max;
        setNative(r, String(Math.min(hi, Math.max(lo, +r.value + e.step * 0.1 * (hi - lo)))));
      }
      break;
    }
    case "example": {
      const sel = document.querySelector('select[aria-label="Examples"]') as HTMLSelectElement | null;
      const opt = sel && Array.from(sel.options).filter((o) => !o.disabled)[e.index];
      if (sel && opt) setNative(sel, opt.value);
      break;
    }
    case "creature": s.setCreature(e.id); break;
    case "call": void s.start(); s.playCall(e.id); break;
    case "keycall": s.setKeyCall(e.id ?? null); break;
    case "beat": s.beatHit(e.id); break;
    case "preset": s.loadBeatPreset(e.name); break;
    case "beat_play": s.setBeatPlaying(!!e.on); break;
    case "tempo": s.setBeatTempo({ bpm: Math.min(240, Math.max(50, s.beat.bpm + e.delta)) }); break;
    case "mic": s.setMic(!!e.on); break;
    case "stop": stopAll(); break;
    case "param": {
      const p = PARAMS[e.id];
      if (!p) break;
      const n = Math.min(1, Math.max(0, e.norm));
      const v = p.log ? Math.exp(Math.log(p.min) + n * (Math.log(p.max) - Math.log(p.min))) : p.min + n * (p.max - p.min);
      if (p.to === "source") s.setSource({ [p.key]: v } as any);
      else s.setArtic({ [p.key]: p.invert ? 1 - v : v } as any);
      break;
    }
  }
}

function banner() {
  const el = document.createElement("button");
  el.textContent = "Click once to enable the Launchpad voice (browsers need a click before playing sound)";
  el.style.cssText = "position:fixed;z-index:99999;left:50%;top:8px;transform:translateX(-50%);padding:10px 16px;background:#f5a623;color:#000;border:0;border-radius:6px;font:600 14px system-ui;cursor:pointer";
  el.onclick = () => void st().start();
  document.body.appendChild(el);
  const off = useStore.subscribe((s) => { if (s.started) { el.remove(); off(); } });
}

function connect() {
  const es = new EventSource("/api/voice");
  es.onmessage = (m) => { try { perform(JSON.parse(m.data)); } catch (err) { console.error("launchpad bridge", err); } };
  es.onerror = () => { es.close(); setTimeout(connect, 1500); };
}

async function boot() {
  await initChimera();
  ({ useStore } = await import("./ui/store.ts"));
  ({ stopSinging } = await import("./ui/singing.ts"));
  ({ stopSpeaking } = await import("./ui/speech.ts"));
  ({ PRESETS, nearestBody } = await import("./ui/body.ts"));
  ({ VOWEL_TARGETS } = await import("./wasm/engine.ts"));
  (window as any).__chimera = useStore; // for tests and debugging
  banner();
  connect();
}

if (location.pathname.startsWith("/voice") || new URLSearchParams(location.search).has("launchpad")) {
  void boot();
}
