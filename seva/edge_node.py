"""Edge node - orchestrates PRD stages 1-5.

  Stage 1  Data collection   camera.read()
  Stage 2  Edge processing   detector.detect() + ROI crop
  Stage 3  QoS telemetry     QoSProbe (concurrent task)
  Stage 4  Semantic encoder  SemanticEncoder.update() -> mode
  Stage 5  Dynamic output    payload pushed across the emulated link

Stages 2 and 3 run concurrently, as the PRD specifies; stage 4 fuses their
outputs before stage 5 transmits.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from .camera import build_camera
from .detector import build_detector, crop_roi, encode_full_frame
from .encoder import SemanticEncoder, MODE_HIGH
from .link import EmulatedLink, LinkProfile, ScenarioRunner
from .metrics import MetricsCollector
from .probe import QoSProbe, QoSSample


class EdgeNode:
    def __init__(self, cfg, root: Path, on_event=None, on_telemetry=None):
        self.cfg = cfg
        self.root = root
        self.on_event = on_event
        self.on_telemetry = on_telemetry

        ini = cfg.get_path("link.initial", {})
        self.link = EmulatedLink(LinkProfile(
            bw_mbps=float(ini.get("bw_mbps", 8.0)),
            delay_ms=float(ini.get("delay_ms", 20.0)),
            jitter_ms=float(ini.get("jitter_ms", 3.0)),
            loss_pct=float(ini.get("loss_pct", 0.0)),
            label=str(ini.get("label", "initial")),
        ))
        self.scenario = ScenarioRunner(
            self.link,
            cfg.get_path("link.scenario", []) or [],
            loop=bool(cfg.get_path("link.loop_scenario", True)),
        )

        self.metrics = MetricsCollector()
        self.scenario.on_change = self._on_link_change

        self.probe = QoSProbe(
            self.link,
            probe_bytes=int(cfg.get_path("qos.probe_bytes", 24000)),
            ping_bytes=int(cfg.get_path("qos.ping_bytes", 64)),
        )
        self.encoder = SemanticEncoder(cfg.get("encoder", {}), cfg.get("payload", {}))

        self.camera = None
        self.detector = None
        self.latest_sample: QoSSample | None = None
        self.running = False
        self.paused = False
        self._tasks: list[asyncio.Task] = []
        self.status_note = "initialising"

    # -- lifecycle ---------------------------------------------------------
    def prepare(self) -> None:
        self.camera = build_camera(self.cfg.get("camera", {}), self.root)
        self.detector = build_detector(self.cfg.get("detector", {}))
        self.status_note = (f"camera={self.camera.kind} "
                            f"detector={self.detector.info.backend}")

    async def start(self) -> None:
        if self.running:
            return
        if self.camera is None:
            await asyncio.get_running_loop().run_in_executor(None, self.prepare)
        self.running = True
        self.metrics.started = time.time()
        self.scenario.reset()
        self._tasks = [
            asyncio.create_task(self._telemetry_loop(), name="seva-qos"),
            asyncio.create_task(self._vision_loop(), name="seva-vision"),
            asyncio.create_task(self._scenario_loop(), name="seva-scenario"),
        ]

    async def stop(self) -> None:
        self.running = False
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._tasks = []
        if self.camera:
            self.camera.release()

    # -- stage 3: QoS telemetry -------------------------------------------
    def _on_link_change(self, profile: LinkProfile, ts: float) -> None:
        self.metrics.record_link_event(profile.as_dict(), ts)

    async def _scenario_loop(self) -> None:
        while self.running:
            self.scenario.tick()
            await asyncio.sleep(0.25)

    def _handle_sample(self, sample: QoSSample) -> None:
        self.latest_sample = sample
        decision = self.encoder.update(sample)
        self.metrics.record_qos(sample, decision.ewma_bw_mbps,
                                decision.ewma_lat_ms, decision.mode)
        if decision.changed:
            self.metrics.record_transition(self.encoder.transitions[-1])
        if self.on_telemetry:
            self.on_telemetry(self.telemetry_snapshot(sample, decision))

    async def _telemetry_loop(self) -> None:
        interval = float(self.cfg.get_path("qos.probe_interval_s", 0.8))
        await self.probe.run(interval, self._handle_sample)

    # -- stages 1, 2, 5 ----------------------------------------------------
    async def _vision_loop(self) -> None:
        loop = asyncio.get_running_loop()
        fps = max(float(self.cfg.get_path("camera.sample_fps", 4.0)), 0.1)
        period = 1.0 / fps
        pcfg = self.cfg.get("payload", {})
        max_objs = int(pcfg.get("max_objects_per_frame", 4))
        thumb_px = int(pcfg.get("thumbnail_max_px", 160))
        thumb_q = int(pcfg.get("thumbnail_jpeg_quality", 55))
        base_q = int(pcfg.get("baseline_jpeg_quality", 85))

        while self.running:
            cycle_start = time.perf_counter()
            if self.paused:
                await asyncio.sleep(period)
                continue

            # Stages 1-2 run off the event loop: decode and inference block.
            frame = await loop.run_in_executor(None, self.camera.read)
            if frame is None:
                await asyncio.sleep(period)
                continue

            detections = await loop.run_in_executor(
                None, self.detector.detect, frame, self.camera)
            baseline = await loop.run_in_executor(
                None, encode_full_frame, frame, base_q)

            self.metrics.record_frame(len(detections),
                                      getattr(self.detector, "last_infer_ms", 0.0),
                                      baseline)

            # Strongest detections first, so a capped budget keeps best evidence.
            detections.sort(key=lambda d: d["conf"], reverse=True)
            for det in detections[:max_objs]:
                det["detect_ts"] = time.time()
                thumb = None
                if self.encoder.mode == MODE_HIGH:
                    thumb = await loop.run_in_executor(
                        None, crop_roi, frame, det["bbox"], thumb_px, thumb_q)
                payload, wire = self.encoder.build_payload(
                    det, self.camera.camera_id, thumb)
                # Stage 5 must not stall stages 1-2 while the link drains.
                asyncio.create_task(self._transmit(payload, wire))

            elapsed = time.perf_counter() - cycle_start
            await asyncio.sleep(max(period - elapsed, 0.0))

    async def _transmit(self, payload: dict, wire: bytes) -> None:
        try:
            result = await self.link.transmit(len(wire))
            self.metrics.record_payload(len(wire), payload["mode"], result.delivered)
            if result.delivered:
                received = time.time()
                self.metrics.record_e2e(payload.get("detect_ts", received), received)
                if self.on_event:
                    await self.on_event(payload, {
                        "bytes": len(wire),
                        "link_delay_ms": round(result.total_delay_s * 1000.0, 1),
                        "received_ts": received,
                    })
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    # -- introspection -----------------------------------------------------
    def telemetry_snapshot(self, sample=None, decision=None) -> dict:
        sample = sample or self.latest_sample
        return {
            "type": "telemetry",
            "ts": time.time(),
            "mode": self.encoder.mode,
            "reason": decision.reason if decision else "",
            "changed": bool(decision.changed) if decision else False,
            "sample": {
                "bw_mbps": round(sample.bandwidth_mbps, 3) if sample else 0,
                "lat_ms": round(sample.latency_ms, 1) if sample else 0,
                "loss_pct": round(sample.loss_pct, 2) if sample else 0,
            },
            "ewma": {
                "bw_mbps": round(self.encoder.bw.get(), 3),
                "lat_ms": round(self.encoder.lat.get(), 1),
            },
            "thresholds": self.encoder.thresholds,
            "link": self.link.snapshot(),
            "kpis": self.metrics.summary(),
            "pipeline": {
                "camera": getattr(self.camera, "kind", "?"),
                "detector": self.detector.info.backend if self.detector else "?",
                "detector_detail": self.detector.info.detail if self.detector else "",
                "reconnects": getattr(self.camera, "reconnects", 0),
                "paused": self.paused,
                "scenario_enabled": self.scenario.enabled,
            },
        }
