#!/usr/bin/env bash
# Build the CHIMERA web app (jamesondh/chimera-vocal-synth, MIT) with the Launchpad bridge into voice/dist,
# which the soundboard serves at /voice/. Needs: git, Rust (+ `rustup target add wasm32-unknown-unknown`), Bun.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PIN=f795b84a6f451aae10c38882be1fbcbc93b3cdd8                       # CHIMERA commit this bridge was written against
SRC="$HERE/chimera-src"

if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/jamesondh/chimera-vocal-synth "$SRC"
fi
git -C "$SRC" fetch --quiet origin "$PIN" 2>/dev/null || git -C "$SRC" fetch --quiet origin
git -C "$SRC" checkout --quiet --force "$PIN"
git -C "$SRC" clean -fdq -e node_modules -e target -e src/wasm/chimera.wasm

cp "$HERE/launchpad-bridge.ts" "$SRC/src/launchpad-bridge.ts"
grep -q launchpad-bridge "$SRC/src/main.tsx" || printf '\nvoid import("./launchpad-bridge.ts"); // Launchpad bridge: loads after the wasm engine\n' >> "$SRC/src/main.tsx"

cd "$SRC"
bun install --frozen-lockfile
bun run wasm
bunx tsc --noEmit
bunx vite build --base /voice/
rm -rf "$HERE/dist"
cp -r dist "$HERE/dist"
echo "Built CHIMERA with the Launchpad bridge -> $HERE/dist  (served at /voice/)"
