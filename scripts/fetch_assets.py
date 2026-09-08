"""Fetch the two optional assets: YOLOv8n weights and a sample street clip.

Both are optional — without them SEVA-Net falls back to the synthetic camera
and the simulated detector, and the pipeline still runs end to end.

    python scripts/fetch_assets.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"

WEIGHTS_URL = ("https://github.com/ultralytics/assets/releases/download/"
               "v8.3.0/yolov8n.pt")

# Short, permissively-licensed clips with people and vehicles in frame.
VIDEO_URLS = [
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
]


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already present: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  downloading {url.split('/')[-1]} …")
        req = urllib.request.Request(url, headers={"User-Agent": "seva-net/0.1"})
        with urllib.request.urlopen(req, timeout=90) as r, open(dest, "wb") as fh:
            fh.write(r.read())
        print(f"  saved {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  failed ({exc.__class__.__name__}: {exc})")
        dest.unlink(missing_ok=True)
        return False


def main() -> int:
    print("YOLOv8n weights")
    weights_ok = download(WEIGHTS_URL, ROOT / "yolov8n.pt")

    print("\nSample camera footage")
    video_ok = any(download(u, VIDEO_DIR / u.split("/")[-1]) for u in VIDEO_URLS)

    print("\nSummary")
    print(f"  detector : {'YOLOv8n' if weights_ok else 'simulated fallback'}")
    print(f"  camera   : {'recorded clip' if video_ok else 'synthetic fallback'}")
    if not (weights_ok and video_ok):
        print("\n  Missing assets are fine — the pipeline runs either way.")
        print("  To supply your own footage, drop any .mp4 into data/videos/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
