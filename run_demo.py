"""SEVA-Net demo launcher.

    python run_demo.py                 # start the full pipeline + dashboard
    python run_demo.py --port 8080
    python run_demo.py --config custom.yaml

Opens the cloud dashboard on http://127.0.0.1:8000 with the edge node, the
emulated backhaul and the scripted degradation timeline all running.
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from threading import Timer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from seva.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the SEVA-Net prototype")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    host = args.host or cfg.get_path("cloud.host", "127.0.0.1")
    port = args.port or int(cfg.get_path("cloud.port", 8000))

    import uvicorn
    from seva.cloud.server import create_app

    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"
    print("=" * 62)
    print("  SEVA-Net - Dynamic Network-Aware Semantic Encoding")
    print("=" * 62)
    print(f"  Dashboard : {url}")
    print(f"  Config    : {args.config or 'config.yaml'}")
    print("  Ctrl-C to stop")
    print("=" * 62)

    if not args.no_browser:
        Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(args.config), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
