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
import sys
import threading
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
    live = False
    sample_fps: float | None = None   # None -> camera.sample_fps from config

    def read(self):
        raise NotImplementedError

    def release(self):
        pass


class FileCamera(BaseCamera):
    """Looped recording served as if it were a live RTSP feed."""

    kind = "file"

    def __init__(self, source: str, loop: bool = True, reconnect_delay_s: float = 2.0,
                 camera_id: str = "CAM-01", start_offset_s: float = 0.0):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for FileCamera")
        self.source = source
        self.loop = loop
        self.reconnect_delay_s = reconnect_delay_s
        self.camera_id = camera_id
        self.start_offset_s = float(start_offset_s)
        self.reconnects = 0
        self.cap = None
        self._open()

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera source: {self.source}")
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if frame_count > 0 and fps > 0 and self.start_offset_s:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.start_offset_s * fps) % int(frame_count))

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


class WebcamCamera(BaseCamera):
    """A local capture device (the laptop camera) acting as a live edge camera.

    A background thread keeps only the newest frame, so detection never works
    on a stale buffered frame and the local preview stays smooth. The device is
    opened lazily and reopened after a drop; stop() releases it entirely so the
    camera light goes off.
    """

    kind = "webcam"
    live = True

    def __init__(self, device: int = 0, camera_id: str = "CAM-05", *,
                 width: int = 640, height: int = 480, sample_fps: float = 6.0,
                 reconnect_delay_s: float = 2.0, label: str = "Laptop camera",
                 enabled: bool = True):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for WebcamCamera")
        self.device = device
        self.camera_id = camera_id
        self.label = label
        self.W, self.H = int(width), int(height)
        self.sample_fps = float(sample_fps)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self.reconnects = 0
        self.status = "off"
        self.status_detail = ""
        self.last_detections: list[dict] = []
        self.frame_seq = 0
        self.frame_ts = 0.0
        self._frame = None
        self._lock = threading.Lock()
        self._enabled = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._capture_loop,
                                        name=f"seva-webcam-{camera_id}", daemon=True)
        if enabled:
            self._enabled.set()
            self.status = "starting"
        self._thread.start()

    # -- control -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

    def start(self) -> None:
        if not self._enabled.is_set():
            self.status, self.status_detail = "starting", ""
            self._enabled.set()

    def stop(self) -> None:
        self._enabled.clear()
        with self._lock:
            self._frame = None
            self.last_detections = []

    def release(self):
        self._closed = True
        self.stop()
        self._thread.join(timeout=3.0)

    # -- capture thread ----------------------------------------------------
    def _open(self):
        # DirectShow opens in ~1 s on Windows; the default MSMF backend can take ~20 s.
        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.device, api)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _capture_loop(self):
        cap = None
        while not self._closed:
            if not self._enabled.wait(timeout=0.25):
                if cap is not None:
                    cap.release()
                    cap = None
                    self.status = "off"
                continue
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.status = "unavailable"
                    self.status_detail = (f"cannot open camera device {self.device} "
                                          f"(in use by another app, or blocked in privacy settings)")
                    time.sleep(self.reconnect_delay_s)
                    continue
                self.status, self.status_detail = "live", ""
            ok, frame = cap.read()
            if not ok or frame is None:
                # Drop -> release and reopen on the next pass (FR-2).
                cap.release()
                cap = None
                self.reconnects += 1
                self.status, self.status_detail = "unavailable", "camera stream dropped; reconnecting"
                time.sleep(self.reconnect_delay_s)
                continue
            with self._lock:
                if not self._enabled.is_set():
                    continue
                self._frame = frame
                self.frame_seq += 1
                self.frame_ts = time.time()
        if cap is not None:
            cap.release()
        self.status = "off"

    # -- pipeline interface ------------------------------------------------
    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def latest(self):
        """(frame, sequence number, detections) for the local preview."""
        with self._lock:
            return self._frame, self.frame_seq, list(self.last_detections)


def build_live_camera(cfg: dict, default_id: str) -> WebcamCamera | None:
    if not cfg.get("enabled", False) or cv2 is None:
        return None
    return WebcamCamera(
        device=int(cfg.get("device", 0)),
        camera_id=str(cfg.get("camera_id") or default_id),
        width=int(cfg.get("width", 640)), height=int(cfg.get("height", 480)),
        sample_fps=float(cfg.get("sample_fps", 6.0)),
        reconnect_delay_s=float(cfg.get("reconnect_delay_s", 2.0)),
        label=str(cfg.get("label", "Laptop camera")),
        enabled=bool(cfg.get("start_on", True)),
    )


def build_camera(cfg: dict, root: Path, index: int = 0) -> BaseCamera:
    """Pick a camera source from config, falling back to synthetic."""
    source = str(cfg.get("source", "auto"))
    loop = bool(cfg.get("loop", True))
    delay = float(cfg.get("reconnect_delay_s", 2.0))
    camera_id = f"CAM-{index + 1:02d}"
    seed = int(cfg.get("seed", 11)) + index
    offset = index * float(cfg.get("reuse_offset_s", 7.0))

    if source == "synthetic" or cv2 is None:
        return SyntheticCamera(camera_id, seed)

    if source == "auto":
        clips = sorted(p for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv")
                       for p in (root / "data" / "videos").glob(ext))
        if not clips:
            return SyntheticCamera(camera_id, seed)
        # Prefer a clip containing both people and vehicles, so the feed
        # exercises intrusion *and* traffic events rather than one class.
        source = str(next(
            (c for c in clips if "person-bicycle-car" in c.name),
            next((c for c in clips if "person" in c.name), clips[0]),
        ))
        # Preserve the preferred legacy source for camera one, then use other
        # clips before cycling back to an already-used source with an offset.
        ordered = [Path(source)] + [clip for clip in clips if str(clip) != source]
        source = str(ordered[index % len(ordered)])
        offset = (index // len(ordered)) * float(cfg.get("reuse_offset_s", 7.0))

    if source not in ("auto", "synthetic"):
        path = Path(source)
        if not path.is_absolute() and (root / path).exists():
            source = str(root / path)

    try:
        return FileCamera(source, loop=loop, reconnect_delay_s=delay,
                          camera_id=camera_id, start_offset_s=offset)
    except RuntimeError:
        return SyntheticCamera(camera_id, seed)


def build_cameras(cfg: dict, root: Path) -> list[BaseCamera]:
    count = int(cfg.get("count", 1))
    if count < 1:
        raise ValueError("camera.count must be at least one")
    return [build_camera(cfg, root, index) for index in range(count)]
