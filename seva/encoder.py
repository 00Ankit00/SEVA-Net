"""DNA-SE: Dynamic Network-Aware Semantic Encoding.

The project's core contribution. Two responsibilities:

  1. Mode decision (FR-8, FR-9, section 10.2) — smooth noisy QoS telemetry with
     an EWMA, then apply hysteresis-bound thresholds so transient noise cannot
     flap the mode.

  2. Payload construction (FR-11, FR-12, FR-13, section 9) — emit JSON plus a
     JPEG ROI thumbnail in HIGH mode, JSON alone in LOW mode, every payload
     stamped with its mode and timestamp.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

MODE_HIGH = "high"
MODE_LOW = "low"


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
        self.cfg = dict(cfg)
        self.payload_cfg = dict(payload_cfg)
        self.bw = EWMA(cfg.get("ewma_alpha", 0.3))
        self.lat = EWMA(cfg.get("ewma_alpha", 0.3))
        self.mode = cfg.get("start_mode", MODE_HIGH)
        self.last_switch_ts = 0.0
        self.transitions: list[dict] = []
        self._pending_reason = "initial mode"

    # -- tunables (FR-10: retunable at runtime, no code change) -------------
    def retune(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k in self.cfg and v is not None:
                self.cfg[k] = float(v) if k != "start_mode" else v
        self.bw.alpha = float(self.cfg.get("ewma_alpha", 0.3))
        self.lat.alpha = float(self.cfg.get("ewma_alpha", 0.3))

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
        }

    # -- mode decision (FR-8 / FR-9) ---------------------------------------
    def update(self, sample) -> ModeDecision:
        bw = self.bw.update(sample.bandwidth_mbps)
        lat = self.lat.update(sample.latency_ms)
        t = self.thresholds
        now = time.time()

        target, reason = self.mode, "within hysteresis band"

        if self.mode == MODE_HIGH:
            # Either signal degrading past the *lower* edge forces a downgrade.
            if bw < t["bw_switch_down_mbps"]:
                target = MODE_LOW
                reason = f"EWMA bw {bw:.2f} < {t['bw_switch_down_mbps']:.2f} Mbps"
            elif lat > t["lat_switch_down_ms"]:
                target = MODE_LOW
                reason = f"EWMA latency {lat:.0f} > {t['lat_switch_down_ms']:.0f} ms"
        else:
            # Both signals must clear the *upper* edge before upgrading again.
            if bw > t["bw_switch_up_mbps"] and lat < t["lat_switch_up_ms"]:
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
                      thumbnail_jpeg: bytes | None) -> tuple[dict, bytes]:
        """Return (payload dict, encoded wire bytes) for one detection."""
        ts_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

        payload = {
            "mode": self.mode,                      # FR-13
            "camera": camera_id,
            "event": detection["event"],
            "label": detection["label"],
            "conf": round(float(detection["conf"]), 3),
            "bbox": [int(v) for v in detection["bbox"]],
            "ts": ts_iso,
            "detect_ts": detection.get("detect_ts", time.time()),
        }

        if self.mode == MODE_HIGH and thumbnail_jpeg:
            # FR-11: JSON metadata + JPEG-compressed ROI crop.
            payload["thumbnail"] = base64.b64encode(thumbnail_jpeg).decode("ascii")
        # FR-12: in LOW mode the image is dropped entirely — no key at all.

        wire = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return payload, wire
