#!/bin/sh
# Install the udev rule that gives normal users access to the Launchpad, and apply it
# to an already plugged-in device (no replug needed).   Usage:  sudo linux/install-udev.sh
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
if [ "$(id -u)" -ne 0 ]; then
    exec sudo "$0" "$@"
fi
install -m 0644 "$DIR/50-novation-launchpad.rules" /etc/udev/rules.d/50-novation-launchpad.rules
udevadm control --reload-rules
udevadm trigger --subsystem-match=usb --attr-match=idVendor=1235 --action=add || true
udevadm settle || true
echo "udev rule installed: /etc/udev/rules.d/50-novation-launchpad.rules"
