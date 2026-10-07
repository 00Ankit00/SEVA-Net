"""Operator alerts derived from hysteresis-protected encoder decisions."""
from __future__ import annotations


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
            "type": "alert", "level": level, "from": previous_mode,
            "to": decision.mode, "message": message, "reason": decision.reason,
            "ewma_bw": round(bw, 3), "ewma_lat": round(decision.ewma_lat_ms, 1),
            "ts": decision.ts,
        }]
