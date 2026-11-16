"""Small command-line entrypoint for the local setup app."""

from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn

from .app import create_app
from .config import VisionStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Demeter Vision Setup")
    parser.add_argument("--data-dir", help="Where datasets, jobs, and releases are stored")
    parser.add_argument("--mode", choices=("trainer", "runtime"), default="trainer",
                        help="Trainer on the NVIDIA workstation; runtime on the Admin PC")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = create_app(VisionStore(args.data_dir), mode=args.mode)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port))
    app.state.uvicorn_server = server
    server.run()
