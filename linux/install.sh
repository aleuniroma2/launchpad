#!/usr/bin/env bash
# One-shot Linux setup for the Launchpad soundboard.
#   linux/install.sh              system packages + udev rule + Python venv
#   linux/install.sh --service    ...and also start it automatically at login (systemd --user)
#   linux/install.sh --desktop    ...and also add an application-menu launcher
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WANT_SERVICE=0; WANT_DESKTOP=0
for a in "$@"; do
    case "$a" in
        --service) WANT_SERVICE=1 ;;
        --desktop) WANT_DESKTOP=1 ;;
        -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
        *) echo "unknown option $a"; exit 1 ;;
    esac
done
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "==> System packages (libusb, PortAudio, libsndfile, Python venv)"
if command -v apt-get >/dev/null; then
    $SUDO apt-get install -y libusb-1.0-0 libportaudio2 libsndfile1 python3-venv
elif command -v dnf >/dev/null; then
    $SUDO dnf install -y libusb1 portaudio libsndfile python3
elif command -v pacman >/dev/null; then
    $SUDO pacman -S --needed --noconfirm libusb portaudio libsndfile python
elif command -v zypper >/dev/null; then
    $SUDO zypper install -y libusb-1_0-0 libportaudio2 libsndfile1 python3
else
    echo "    unknown package manager: install libusb-1.0, PortAudio and libsndfile yourself"
fi

echo "==> udev rule"
$SUDO "$ROOT/linux/install-udev.sh"

echo "==> Python environment in $ROOT/.venv"
if command -v uv >/dev/null; then
    uv venv "$ROOT/.venv" && uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"
else
    python3 -m venv "$ROOT/.venv"
    "$ROOT/.venv/bin/pip" install -q --upgrade pip
    "$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt"
fi

if [ $WANT_SERVICE = 1 ]; then
    echo "==> systemd user service"
    mkdir -p ~/.config/systemd/user
    sed "s#@ROOT@#$ROOT#g" "$ROOT/linux/launchpad-soundboard.service" > ~/.config/systemd/user/launchpad-soundboard.service
    systemctl --user daemon-reload
    systemctl --user enable --now launchpad-soundboard.service
    echo "    running: systemctl --user status launchpad-soundboard   (GUI on http://127.0.0.1:8765)"
fi
if [ $WANT_DESKTOP = 1 ]; then
    echo "==> desktop launcher"
    mkdir -p ~/.local/share/applications
    sed "s#@ROOT@#$ROOT#g" "$ROOT/linux/launchpad-soundboard.desktop" > ~/.local/share/applications/launchpad-soundboard.desktop
fi
echo
echo "Done.  Start it with:  $ROOT/.venv/bin/python $ROOT/run.py"
