"""Operator alerts.

AlertManager      bandwidth alerts derived from hysteresis-protected encoder decisions.
DetectionAlerter  object alerts derived from detection events the cloud received.
"""
from __future__ import annotations

DETECTION_TITLES = {"person": "Person", "bicycle": "Bicycle", "motorcycle": "Motorbike",
                    "car": "Car", "bus": "Bus", "truck": "Truck"}


class DetectionAlerter:
    """Raise one alert when a class *enters* a watched camera's view.

    A class re-arms only after it has been absent for `rearm_s`, and never
    alerts more often than `cooldown_s`, so someone standing in front of the
    camera produces one alert, not one per frame.
    """

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.last_seen: dict[tuple[str, str], float] = {}
        self.last_alert: dict[tuple[str, str], float] = {}

    def watches(self, camera_id: str, live_ids: set[str]) -> bool:
        cameras = self.cfg.get("cameras", "live")
        if cameras == "all":
            return True
        if cameras == "live":
            return camera_id in live_ids
        return camera_id in set(cameras or [])

    def decide(self, payload: dict, live_ids: set[str]) -> dict | None:
        if not self.cfg.get("enabled", True):
            return None
        camera_id = payload.get("camera_id") or payload.get("camera")
        label = payload.get("label")
        levels = self.cfg.get("levels", {}) or {}
        if label not in levels or not self.watches(camera_id, live_ids):
            return None
        if float(payload.get("conf", 0)) < float(self.cfg.get("min_conf", 0.5)):
            return None

        ts = float(payload.get("detect_ts", 0))
        key = (camera_id, label)
        previous = self.last_seen.get(key)
        self.last_seen[key] = max(ts, previous or ts)
        entered = previous is None or ts - previous >= float(self.cfg.get("rearm_s", 5.0))
        cooled = ts - self.last_alert.get(key, float("-inf")) >= float(self.cfg.get("cooldown_s", 20.0))
        if not (entered and cooled):
            return None
        self.last_alert[key] = ts

        title = DETECTION_TITLES.get(label, label.title())
        alert = {
            "type": "alert", "kind": "detection", "level": levels[label],
            "camera_id": camera_id, "label": label, "event": payload.get("event"),
            "conf": payload.get("conf"), "mode": payload.get("mode"),
            "message": f"{title} detected on {camera_id}",
            "reason": f"{label} {float(payload.get('conf', 0)):.0%} confidence · "
                      f"{payload.get('event', 'object').replace('_', ' ')}",
            "ts": ts,
        }
        # The ROI only exists if the link mode let it through (HIGH/RICH).
        if payload.get("thumbnail"):
            alert["thumbnail"] = payload["thumbnail"]
        return alert


class AlertManager:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.warning_armed = True

    def decide(self, decision, previous_mode: str, thresholds: dict) -> list[dict]:
        bw = decision.ewma_bw_mbps
        edge = thresholds["bw_switch_down_mbps"]
        ceiling = edge * (1.0 + float(self.cfg.get("warning_margin", 0.15)))
        if bw > ceiling:
            self.warning_armed = True
        if not self.cfg.get("enabled", True):
            return []

        level, message = None, None
        if decision.changed:
            if decision.mode == "low":
                level = "critical"
                message = "Low bandwidth: images suppressed, metadata-only alerts active"
            elif previous_mode == "low" or decision.mode == "rich":
                level = "info"
                message = f"Link recovered: {decision.mode.upper()} transmission active"
            else:
                level = "warning"
                message = "Link headroom reduced: HIGH transmission active"
        elif (self.cfg.get("warning_enabled", True) and decision.mode == "high"
              and edge <= bw <= ceiling and self.warning_armed):
            self.warning_armed = False
            level = "warning"
            message = "Bandwidth approaching the metadata-only threshold"
        if level is None:
            return []
        return [{
            "type": "alert", "kind": "bandwidth", "level": level, "from": previous_mode,
            "to": decision.mode, "message": message, "reason": decision.reason,
            "ewma_bw": round(bw, 3), "ewma_lat": round(decision.ewma_lat_ms, 1),
            "ts": decision.ts,
        }]
