"""DNA-SE: Dynamic Network-Aware Semantic Encoding.

The project's core contribution. Two responsibilities:

  1. Mode decision (FR-8, FR-9, section 10.2) — smooth noisy QoS telemetry with
     an EWMA, then apply hysteresis-bound thresholds so transient noise cannot
     flap the mode.

  2. Payload construction (FR-11, FR-12, FR-13, section 9) — emit JSON plus a
     JPEG ROI thumbnail in HIGH, richer frame evidence in RICH, and JSON alone
     in LOW. Every payload carries its camera identity, mode and timestamp.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

MODE_HIGH = "high"
MODE_LOW = "low"
MODE_RICH = "rich"


@dataclass
class ModeDecision:
    mode: str
    changed: bool
    ewma_bw_mbps: float
    ewma_lat_ms: float
    reason: str
    ts: float


class EWMA:
    """EWMA_t = alpha * x_t + (1 - alpha) * EWMA_{t-1}"""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return self.value

    def get(self, default: float = 0.0) -> float:
        return self.value if self.value is not None else default


class SemanticEncoder:
    def __init__(self, cfg: dict, payload_cfg: dict):
        self.validate(cfg)
        self.cfg = dict(cfg)
        self.payload_cfg = dict(payload_cfg)
        self.bw = EWMA(cfg.get("ewma_alpha", 0.3))
        self.lat = EWMA(cfg.get("ewma_alpha", 0.3))
        self.mode = cfg.get("start_mode", MODE_HIGH)
        self.last_switch_ts = time.time()
        self.transitions: list[dict] = []

    # -- tunables (FR-10: retunable at runtime, no code change) -------------
    def retune(self, **kwargs) -> None:
        candidate = dict(self.cfg)
        for key, value in kwargs.items():
            if key in self.thresholds and value is not None:
                candidate[key] = float(value)
        self.validate(candidate)
        self.cfg = candidate
        self.bw.alpha = float(candidate.get("ewma_alpha", 0.3))
        self.lat.alpha = self.bw.alpha

    @staticmethod
    def validate(cfg: dict) -> None:
        import math
        for key, value in cfg.items():
            if key != "start_mode" and not math.isfinite(float(value)):
                raise ValueError(f"{key} must be finite")
        if not 0 < cfg.get("ewma_alpha", 0.3) <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if not (0 < cfg.get("bw_switch_down_mbps", 1.2)
                < cfg.get("bw_switch_up_mbps", 2.0)
                < cfg.get("rich_bw_switch_down_mbps", 10.0)
                < cfg.get("rich_bw_switch_up_mbps", 12.0)):
            raise ValueError("bandwidth bands must satisfy LOW down < LOW up < RICH down < RICH up")
        if not (0 < cfg.get("rich_lat_switch_up_ms", 80.0)
                < cfg.get("rich_lat_switch_down_ms", 110.0)
                < cfg.get("lat_switch_up_ms", 120.0)
                < cfg.get("lat_switch_down_ms", 180.0)):
            raise ValueError("latency bands must satisfy RICH up < RICH down < LOW up < LOW down")
        if cfg.get("min_hold_s", 1.0) < 0:
            raise ValueError("min_hold_s must be nonnegative")
        if cfg.get("start_mode", "high") not in (MODE_LOW, MODE_HIGH, MODE_RICH):
            raise ValueError("invalid start_mode")

    @property
    def thresholds(self) -> dict:
        c = self.cfg
        return {
            "bw_switch_down_mbps": c.get("bw_switch_down_mbps", 1.2),
            "bw_switch_up_mbps": c.get("bw_switch_up_mbps", 2.0),
            "lat_switch_down_ms": c.get("lat_switch_down_ms", 180.0),
            "lat_switch_up_ms": c.get("lat_switch_up_ms", 120.0),
            "ewma_alpha": c.get("ewma_alpha", 0.3),
            "min_hold_s": c.get("min_hold_s", 1.0),
            "rich_bw_switch_up_mbps": c.get("rich_bw_switch_up_mbps", 12.0),
            "rich_bw_switch_down_mbps": c.get("rich_bw_switch_down_mbps", 10.0),
            "rich_lat_switch_up_ms": c.get("rich_lat_switch_up_ms", 80.0),
            "rich_lat_switch_down_ms": c.get("rich_lat_switch_down_ms", 110.0),
        }

    # -- mode decision (FR-8 / FR-9) ---------------------------------------
    def update(self, sample) -> ModeDecision:
        bw = self.bw.update(sample.bandwidth_mbps)
        lat = self.lat.update(sample.latency_ms)
        t = self.thresholds
        now = float(sample.ts)

        target, reason = self.mode, "within hysteresis band"

        # Collapse takes priority over the RICH band, including RICH -> LOW.
        if self.mode != MODE_LOW and (bw < t["bw_switch_down_mbps"]
                                      or lat > t["lat_switch_down_ms"]):
            target = MODE_LOW
            reason = (f"EWMA bw {bw:.2f} < {t['bw_switch_down_mbps']:.2f} Mbps"
                      if bw < t["bw_switch_down_mbps"] else
                      f"EWMA latency {lat:.0f} > {t['lat_switch_down_ms']:.0f} ms")
        elif self.mode == MODE_RICH:
            if bw < t["rich_bw_switch_down_mbps"] or lat > t["rich_lat_switch_down_ms"]:
                target = MODE_HIGH
                reason = (f"RICH headroom lost: EWMA {bw:.2f} Mbps / {lat:.0f} ms; "
                          f"exit below {t['rich_bw_switch_down_mbps']:.2f} Mbps "
                          f"or above {t['rich_lat_switch_down_ms']:.0f} ms")
        elif self.mode == MODE_HIGH:
            if bw > t["rich_bw_switch_up_mbps"] and lat < t["rich_lat_switch_up_ms"]:
                target = MODE_RICH
                reason = (f"RICH headroom: EWMA bw {bw:.2f} > "
                          f"{t['rich_bw_switch_up_mbps']:.2f} Mbps and latency "
                          f"{lat:.0f} < {t['rich_lat_switch_up_ms']:.0f} ms")
        elif bw > t["bw_switch_up_mbps"] and lat < t["lat_switch_up_ms"]:
            target = MODE_HIGH
            reason = (f"EWMA bw {bw:.2f} > {t['bw_switch_up_mbps']:.2f} Mbps "
                      f"and latency {lat:.0f} < {t['lat_switch_up_ms']:.0f} ms")

        changed = False
        if target != self.mode:
            held = now - self.last_switch_ts
            if self.last_switch_ts and held < t["min_hold_s"]:
                reason = f"switch suppressed: dwell {held:.1f}s < {t['min_hold_s']:.1f}s"
            else:
                self.transitions.append({
                    "ts": now, "from": self.mode, "to": target, "reason": reason,
                    "ewma_bw_mbps": bw, "ewma_lat_ms": lat,
                })
                self.mode = target
                self.last_switch_ts = now
                changed = True

        return ModeDecision(self.mode, changed, bw, lat, reason, now)

    # -- payload construction (section 9) ----------------------------------
    def build_payload(self, detection: dict, camera_id: str,
                      thumbnail_jpeg: bytes | None, *,
                      detections: list[dict] | None = None,
                      keyframe_jpeg: bytes | None = None,
                      edge_stats: dict | None = None,
                      mode: str | None = None) -> tuple[dict, bytes]:
        """Return (payload dict, encoded wire bytes) for one detection."""
        mode = mode or self.mode
        ts_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

        payload = {
            "mode": mode,                      # FR-13
            "camera": camera_id,
            "camera_id": camera_id,                # additive multi-camera extension
            "event": detection["event"],
            "label": detection["label"],
            "conf": round(float(detection["conf"]), 3),
            "bbox": [int(v) for v in detection["bbox"]],
            "ts": ts_iso,
            "detect_ts": detection.get("detect_ts", time.time()),
        }

        if mode in (MODE_HIGH, MODE_RICH) and thumbnail_jpeg:
            # FR-11: JSON metadata + JPEG-compressed ROI crop.
            payload["thumbnail"] = base64.b64encode(thumbnail_jpeg).decode("ascii")
        if mode == MODE_RICH:
            objects = detections or [detection]
            payload["detections"] = [
                {"class": d["label"], "conf": round(float(d["conf"]), 3),
                 "bbox": [int(v) for v in d["bbox"]]} for d in objects
            ]
            counts: dict[str, int] = {}
            for d in objects:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            payload["class_counts"] = counts
            payload["edge_stats"] = edge_stats or {}
            if keyframe_jpeg:
                payload["keyframe"] = base64.b64encode(keyframe_jpeg).decode("ascii")
        # FR-12: in LOW mode the image is dropped entirely — no key at all.

        wire = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return payload, wire
