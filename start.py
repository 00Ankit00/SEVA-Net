"""One-command SEVA-Net launcher: install deps, fetch assets, start the dashboard.

    python start.py                    # first run installs + downloads, then opens the app
    python start.py --skip-fetch       # don't try to download assets
    python start.py --port 8080        # any run_demo.py option is passed through

Already-installed packages and already-downloaded assets are skipped, so later
runs go straight to the server.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Import names of the packages the pipeline cannot run without.
CORE_MODULES = ["numpy", "cv2", "yaml", "fastapi", "uvicorn", "websockets"]


def ensure_requirements() -> None:
    missing = [m for m in CORE_MODULES if importlib.util.find_spec(m) is None]
    if not missing:
        return
    print(f"Installing requirements (missing: {', '.join(missing)}) …")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "-r", str(ROOT / "requirements.txt")])


def main() -> None:
    args = sys.argv[1:]
    skip_fetch = "--skip-fetch" in args
    if skip_fetch:
        args.remove("--skip-fetch")

    ensure_requirements()

    if not skip_fetch:
        import fetch_assets
        print("=" * 62)
        print("  Fetching assets")
        print("=" * 62)
        fetch_assets.main()
        print()

    import run_demo
    sys.argv = [sys.argv[0], *args]
    run_demo.main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSEVA-Net stopped.")
