#!/usr/bin/env python3
"""Start the Launchpad soundboard GUI:  python run.py [--port 8765] [--no-browser] [--host 127.0.0.1]"""
import argparse
from soundboard.app import serve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach the GUI from other machines")
    a = ap.parse_args()
    serve(a.port, not a.no_browser, a.host)
