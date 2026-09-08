"""Edge object detection and ROI cropping (FR-4, FR-5, sections 10.1 / 10.4).

YOLOv8n runs locally inside the edge stage; only its output ever reaches the
network. If ultralytics/weights are unavailable the SimulatedDetector takes
over, consuming the synthetic camera's ground truth so every downstream stage
(encoder, link, dashboard, KPIs) stays demonstrable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# Detections are mapped to the PRD's use cases (section 6).
EVENT_FOR_LABEL = {
    "person": "intrusion",
    "bicycle": "intrusion",
    "car": "traffic_violation",
    "motorcycle": "traffic_violation",
    "bus": "traffic_violation",
    "truck": "traffic_violation",
}


@dataclass
class DetectorInfo:
    backend: str
    detail: str


class BaseDetector:
    info = DetectorInfo("base", "")

    def detect(self, frame, camera) -> list[dict]:
        raise NotImplementedError


class YoloDetector(BaseDetector):
    """YOLOv8n via ultralytics (FR-4)."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35,
                 imgsz: int = 480, classes: list[str] | None = None):
        from ultralytics import YOLO  # imported lazily so the fallback works

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names
        wanted = set(classes or list(EVENT_FOR_LABEL))
        self.keep_ids = {i for i, n in self.names.items() if n in wanted}
        self.info = DetectorInfo("yolov8n", f"{weights} @ {imgsz}px, conf>{conf}")
        self.last_infer_ms = 0.0

    def detect(self, frame, camera=None) -> list[dict]:
        t0 = time.perf_counter()
        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                     verbose=False)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        out: list[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.keep_ids:
                    continue
                label = self.names[cls_id]
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                out.append({
                    "label": label,
                    "event": EVENT_FOR_LABEL.get(label, "object"),
                    "conf": float(box.conf[0]),
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                })
        return out


class SimulatedDetector(BaseDetector):
    """Ground-truth passthrough for the synthetic camera."""

    def __init__(self, conf: float = 0.35):
        self.conf = conf
        self.info = DetectorInfo("simulated",
                                 "synthetic ground truth (YOLOv8n unavailable)")
        self.last_infer_ms = 0.0

    def detect(self, frame, camera=None) -> list[dict]:
        t0 = time.perf_counter()
        boxes = list(getattr(camera, "last_boxes", []) or [])
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return [b for b in boxes if b["conf"] >= self.conf]


def resolve_weights(weights: str) -> str:
    """Anchor a relative weights path to the project root.

    Without this, launching from any other working directory makes ultralytics
    treat the name as missing and silently re-download the checkpoint.
    """
    p = Path(weights)
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p.resolve())
    candidate = ROOT / p
    return str(candidate) if candidate.exists() else weights


def build_detector(cfg: dict) -> BaseDetector:
    backend = str(cfg.get("backend", "auto"))
    conf = float(cfg.get("conf_threshold", 0.35))

    if backend in ("auto", "yolo"):
        try:
            return YoloDetector(
                weights=resolve_weights(cfg.get("weights", "yolov8n.pt")),
                conf=conf,
                imgsz=int(cfg.get("imgsz", 480)),
                classes=cfg.get("classes"),
            )
        except Exception as exc:
            if backend == "yolo":
                raise
            print(f"[detector] YOLOv8n unavailable ({exc.__class__.__name__}: {exc}); "
                  f"falling back to simulated detector")
    return SimulatedDetector(conf=conf)


# -- ROI cropping (FR-5, section 10.4) -------------------------------------
def crop_roi(frame, bbox, max_px: int, quality: int, pad: float = 0.12):
    """Crop around a bounding box and JPEG-compress it. Returns bytes or None."""
    if cv2 is None or frame is None:
        return None
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    px, py = int(bw * pad), int(bh * pad)
    x0, y0 = max(x - px, 0), max(y - py, 0)
    x1, y1 = min(x + bw + px, w), min(y + bh + py, h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    scale = min(max_px / max(rh, rw), 1.0)
    if scale < 1.0:
        roi = cv2.resize(roi, (max(int(rw * scale), 1), max(int(rh * scale), 1)),
                         interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


def encode_full_frame(frame, quality: int) -> int:
    """Size in bytes of the full frame as JPEG — the streaming baseline (NFR-4)."""
    if cv2 is None or frame is None:
        return 0
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return int(buf.nbytes) if ok else 0
