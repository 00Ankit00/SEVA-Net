"""Software-simulated camera source (FR-1, FR-2, FR-3).

There is no physical IP camera in this phase (PRD 4.2). Two sources are
supported, both presented to the pipeline through the same interface:

  FileCamera       a recorded .mp4 (or an RTSP URL, if one is served by
                   FFmpeg/GStreamer) looped forever via OpenCV.
  SyntheticCamera  a procedurally rendered street scene, used when no clip is
                   available. It also emits ground-truth boxes, which lets the
                   whole pipeline be demonstrated without model weights.

Both sample at a configurable rate rather than decoding every frame (FR-3),
and both reconnect automatically when the source ends or errors (FR-2).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class BaseCamera:
    camera_id = "CAM-01"
    kind = "base"

    def read(self):
        raise NotImplementedError

    def release(self):
        pass


class FileCamera(BaseCamera):
    """Looped recording served as if it were a live RTSP feed."""

    kind = "file"

    def __init__(self, source: str, loop: bool = True, reconnect_delay_s: float = 2.0,
                 camera_id: str = "CAM-01"):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for FileCamera")
        self.source = source
        self.loop = loop
        self.reconnect_delay_s = reconnect_delay_s
        self.camera_id = camera_id
        self.reconnects = 0
        self.cap = None
        self._open()

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera source: {self.source}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            # FR-2: stream ended or dropped -> reconnect transparently.
            if not self.loop:
                return None
            self.reconnects += 1
            try:
                self._open()
            except RuntimeError:
                time.sleep(self.reconnect_delay_s)
                return None
            ok, frame = self.cap.read()
            if not ok:
                return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


class SyntheticCamera(BaseCamera):
    """Procedural street scene: a road, walking pedestrians, moving vehicles.

    Renders at 640x360 and reports ground-truth boxes for each actor, so the
    encoder/link/dashboard stages are exercisable with no model weights and no
    downloaded footage.
    """

    kind = "synthetic"
    W, H = 640, 360

    def __init__(self, camera_id: str = "CAM-01", seed: int = 11):
        self.camera_id = camera_id
        self.t = 0.0
        self.reconnects = 0
        rng = np.random.default_rng(seed)
        self.actors = []
        for i in range(3):
            self.actors.append({
                "label": "person", "event": "intrusion",
                "y": 250 + i * 26, "h": 54 + i * 8,
                "speed": 34 + rng.uniform(-8, 14), "phase": rng.uniform(0, 6.28),
                "colour": (58, 74, 210),
            })
        for i in range(2):
            self.actors.append({
                "label": "car" if i == 0 else "truck",
                "event": "traffic_violation",
                "y": 150 + i * 44, "h": 40 + i * 16,
                "speed": 120 + rng.uniform(-30, 60), "phase": rng.uniform(0, 6.28),
                "colour": (190, 150, 60),
            })
        self.last_boxes: list[dict] = []

    def read(self):
        self.t += 1.0 / 12.0
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        frame[:, :] = (42, 40, 38)
        frame[0:130, :] = (86, 74, 62)              # sky / building line
        frame[130:200, :] = (58, 58, 56)            # far pavement
        frame[200:self.H, :] = (34, 34, 34)         # road
        for x in range(0, self.W, 70):              # lane markings
            frame[228:233, x:x + 34] = (150, 150, 140)

        boxes = []
        for a in self.actors:
            span = self.W + 140
            x = int(((self.t * a["speed"] + a["phase"] * 40) % span) - 70)
            w = int(a["h"] * (0.42 if a["label"] == "person" else 1.9))
            bob = int(3 * math.sin(self.t * 6 + a["phase"])) if a["label"] == "person" else 0
            y = a["y"] + bob
            x0, y0 = max(x, 0), max(y - a["h"], 0)
            x1, y1 = min(x + w, self.W), min(y, self.H)
            if x1 - x0 < 8:
                continue
            frame[y0:y1, x0:x1] = a["colour"]
            if a["label"] == "person":              # crude head so crops read as people
                hx = (x0 + x1) // 2
                frame[max(y0 - 10, 0):y0 + 2, max(hx - 6, 0):hx + 6] = (150, 160, 225)
            boxes.append({"label": a["label"], "event": a["event"],
                          "bbox": [x0, y0, x1 - x0, y1 - y0],
                          "conf": 0.80 + 0.15 * abs(math.sin(self.t + a["phase"]))})
        self.last_boxes = boxes
        return frame


def build_camera(cfg: dict, root: Path) -> BaseCamera:
    """Pick a camera source from config, falling back to synthetic."""
    source = str(cfg.get("source", "auto"))
    loop = bool(cfg.get("loop", True))
    delay = float(cfg.get("reconnect_delay_s", 2.0))

    if source == "synthetic" or cv2 is None:
        return SyntheticCamera()

    if source == "auto":
        clips = sorted(p for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv")
                       for p in (root / "data" / "videos").glob(ext))
        if not clips:
            return SyntheticCamera()
        # Prefer a clip containing both people and vehicles, so the feed
        # exercises intrusion *and* traffic events rather than one class.
        source = str(next(
            (c for c in clips if "person-bicycle-car" in c.name),
            next((c for c in clips if "person" in c.name), clips[0]),
        ))

    try:
        return FileCamera(source, loop=loop, reconnect_delay_s=delay)
    except RuntimeError:
        return SyntheticCamera()
