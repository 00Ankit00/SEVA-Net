"""KPI collection (PRD section 13).

Tracks the five success metrics plus the supporting counters the report needs:

  bandwidth saved        bytes actually transmitted vs. a continuous
                         full-frame streaming baseline
  mode-transition latency  injected link event -> resulting mode change
  mode-flap rate         transitions per hour
  end-to-end alert latency  detection -> dashboard receipt
  detection throughput   frames and objects processed, inference time
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from statistics import mean, median


class MetricsCollector:
    def __init__(self, window: int = 600):
        self.started = time.time()

        self.frames = 0
        self.detections = 0
        self.payloads_sent = 0
        self.payloads_delivered = 0
        self.payloads_dropped = 0

        self.bytes_sent = 0
        self.bytes_sent_high = 0
        self.bytes_sent_low = 0
        self.baseline_bytes = 0          # full-frame streaming equivalent

        self.infer_ms = deque(maxlen=window)
        self.e2e_ms = deque(maxlen=window)
        self.transition_latency_ms: list[float] = []
        self.transitions: list[dict] = []

        self._pending_link_event: dict | None = None
        self.qos_history = deque(maxlen=window)

    # -- ingest ------------------------------------------------------------
    def record_frame(self, n_detections: int, infer_ms: float, baseline_bytes: int):
        self.frames += 1
        self.detections += n_detections
        self.infer_ms.append(infer_ms)
        self.baseline_bytes += baseline_bytes

    def record_payload(self, nbytes: int, mode: str, delivered: bool):
        self.payloads_sent += 1
        self.bytes_sent += nbytes
        if mode == "high":
            self.bytes_sent_high += nbytes
        else:
            self.bytes_sent_low += nbytes
        if delivered:
            self.payloads_delivered += 1
        else:
            self.payloads_dropped += 1

    def record_e2e(self, detect_ts: float, received_ts: float):
        self.e2e_ms.append(max(received_ts - detect_ts, 0.0) * 1000.0)

    def record_qos(self, sample, ewma_bw: float, ewma_lat: float, mode: str):
        self.qos_history.append({
            "ts": sample.ts,
            "bw": round(sample.bandwidth_mbps, 3),
            "lat": round(sample.latency_ms, 1),
            "ewma_bw": round(ewma_bw, 3),
            "ewma_lat": round(ewma_lat, 1),
            "mode": mode,
        })

    def record_link_event(self, profile: dict, ts: float):
        """Arm a stopwatch: the next mode change is attributed to this event."""
        self._pending_link_event = {"ts": ts, "profile": profile}

    def record_transition(self, transition: dict):
        entry = dict(transition)
        if self._pending_link_event:
            latency_ms = (transition["ts"] - self._pending_link_event["ts"]) * 1000.0
            # Only credit a transition that plausibly followed the injection.
            if 0 <= latency_ms <= 30_000:
                entry["triggered_by"] = self._pending_link_event["profile"].get("label")
                entry["transition_latency_ms"] = round(latency_ms, 1)
                self.transition_latency_ms.append(latency_ms)
            self._pending_link_event = None
        self.transitions.append(entry)

    # -- derived -----------------------------------------------------------
    @property
    def elapsed_s(self) -> float:
        return max(time.time() - self.started, 1e-6)

    def bandwidth_saved_pct(self) -> float:
        if self.baseline_bytes <= 0:
            return 0.0
        return max(0.0, (1.0 - self.bytes_sent / self.baseline_bytes) * 100.0)

    def flap_rate_per_hour(self) -> float:
        return len(self.transitions) / (self.elapsed_s / 3600.0)

    def summary(self) -> dict:
        return {
            "elapsed_s": round(self.elapsed_s, 1),
            "frames": self.frames,
            "detections": self.detections,
            "payloads_sent": self.payloads_sent,
            "payloads_delivered": self.payloads_delivered,
            "payloads_dropped": self.payloads_dropped,
            "bytes_sent": self.bytes_sent,
            "bytes_sent_high": self.bytes_sent_high,
            "bytes_sent_low": self.bytes_sent_low,
            "baseline_bytes": self.baseline_bytes,
            "bandwidth_saved_pct": round(self.bandwidth_saved_pct(), 2),
            "avg_payload_bytes": round(self.bytes_sent / self.payloads_sent, 1)
                                 if self.payloads_sent else 0,
            "avg_infer_ms": round(mean(self.infer_ms), 2) if self.infer_ms else 0,
            "avg_e2e_ms": round(mean(self.e2e_ms), 1) if self.e2e_ms else 0,
            "median_e2e_ms": round(median(self.e2e_ms), 1) if self.e2e_ms else 0,
            "mode_transitions": len(self.transitions),
            "flap_rate_per_hour": round(self.flap_rate_per_hour(), 2),
            "avg_transition_latency_ms": round(mean(self.transition_latency_ms), 1)
                                         if self.transition_latency_ms else None,
            "transition_latency_samples": len(self.transition_latency_ms),
        }

    def export(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kpis": self.summary(),
            "transitions": self.transitions,
            "qos_history": list(self.qos_history),
        }
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return p
