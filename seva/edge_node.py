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

from .camera import build_cameras
from .alerts import AlertManager
from .detector import build_detector, crop_roi, encode_full_frame, encode_keyframe
from .encoder import SemanticEncoder, MODE_HIGH, MODE_RICH
from .link import EmulatedLink, LinkProfile, ScenarioRunner
from .metrics import MetricsCollector
from .probe import QoSProbe, QoSSample


class EdgeNode:
    def __init__(self, cfg, root: Path, on_event=None, on_telemetry=None, on_alert=None):
        self.cfg = cfg
        self.root = root
        self.on_event = on_event
        self.on_telemetry = on_telemetry
        self.on_alert = on_alert
        self.alerts = AlertManager(cfg.get("alerts", {}))

        ini = cfg.get_path("link.initial", {})
        self.link = EmulatedLink(LinkProfile(
            bw_mbps=float(ini.get("bw_mbps", 8.0)),
            delay_ms=float(ini.get("delay_ms", 20.0)),
            jitter_ms=float(ini.get("jitter_ms", 3.0)),
            loss_pct=float(ini.get("loss_pct", 0.0)),
            label=str(ini.get("label", "initial")),
        ), burst_seconds=float(cfg.get_path("link.burst_seconds", 0.05)))
        self.scenario = ScenarioRunner(
            self.link,
            cfg.get_path("link.scenario", []) or [],
            loop=bool(cfg.get_path("link.loop_scenario", True)),
        )

        self.metrics = MetricsCollector(start_mode=cfg.get_path("encoder.start_mode", "high"))
        self.scenario.on_change = self._on_link_change

        self.probe = QoSProbe(
            self.link,
            probe_bytes=int(cfg.get_path("qos.probe_bytes", 24000)),
            ping_bytes=int(cfg.get_path("qos.ping_bytes", 64)),
        )
        self.encoder = SemanticEncoder(cfg.get("encoder", {}), cfg.get("payload", {}))

        self.cameras = []
        self.detectors = []
        self.camera_state: dict[str, dict] = {}
        self._transmissions: set[asyncio.Task] = set()
        self.camera = None
        self.detector = None
        self.latest_sample: QoSSample | None = None
        self.running = False
        self.paused = False
        self._tasks: list[asyncio.Task] = []
        self.status_note = "initialising"

    # -- lifecycle ---------------------------------------------------------
    def prepare(self) -> None:
        self.cameras = build_cameras(self.cfg.get("camera", {}), self.root)
        self.detectors = [build_detector(self.cfg.get("detector", {})) for _ in self.cameras]
        self.camera, self.detector = self.cameras[0], self.detectors[0]
        for camera, detector in zip(self.cameras, self.detectors):
            self.camera_state[camera.camera_id] = {
                "camera_id": camera.camera_id, "kind": camera.kind,
                "detector": detector.info.backend, "frame_index": 0,
                "fps": 0.0, "width": getattr(camera, "W", 640),
                "height": getattr(camera, "H", 360), "last_event_ts": None,
            }
        self.status_note = (f"{len(self.cameras)} camera(s)={self.camera.kind} "
                            f"detector={self.detector.info.backend}; shared link/encoder")

    async def start(self) -> None:
        if self.running:
            return
        if self.camera is None:
            await asyncio.get_running_loop().run_in_executor(None, self.prepare)
        self.running = True
        self.metrics.started = self.metrics.mode_since = time.time()
        self.scenario.reset()
        self._tasks = [
            asyncio.create_task(self._telemetry_loop(), name="seva-qos"),
            asyncio.create_task(self._scenario_loop(), name="seva-scenario"),
        ]
        self._tasks.extend(asyncio.create_task(self._vision_loop(camera, detector),
                                             name=f"seva-vision-{camera.camera_id}")
                           for camera, detector in zip(self.cameras, self.detectors))

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
        if self._transmissions:
            _, pending = await asyncio.wait(self._transmissions,
                timeout=float(self.cfg.get_path("payload.shutdown_drain_s", 3.0)))
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._transmissions.clear()
        for camera in self.cameras:
            camera.release()
        self.metrics.finish()

    # -- stage 3: QoS telemetry -------------------------------------------
    def _on_link_change(self, profile: LinkProfile, ts: float) -> None:
        self.metrics.record_link_event(profile.as_dict(), ts)

    async def _scenario_loop(self) -> None:
        while self.running:
            self.scenario.tick()
            await asyncio.sleep(0.25)

    def _handle_sample(self, sample: QoSSample) -> None:
        self.latest_sample = sample
        previous_mode = self.encoder.mode
        decision = self.encoder.update(sample)
        self.metrics.record_qos(sample, decision.ewma_bw_mbps,
                                decision.ewma_lat_ms, decision.mode)
        if decision.changed:
            self.metrics.record_transition(self.encoder.transitions[-1])
        for alert in self.alerts.decide(decision, previous_mode, self.encoder.thresholds):
            if self.on_alert:
                self.on_alert(alert)
        if self.on_telemetry:
            self.on_telemetry(self.telemetry_snapshot(sample, decision))

    async def _telemetry_loop(self) -> None:
        interval = float(self.cfg.get_path("qos.probe_interval_s", 0.8))
        await self.probe.run(interval, self._handle_sample)

    # -- stages 1, 2, 5 ----------------------------------------------------
    async def _vision_loop(self, camera, detector) -> None:
        loop = asyncio.get_running_loop()
        fps = max(float(self.cfg.get_path("camera.sample_fps", 4.0)), 0.1)
        period = 1.0 / fps
        pcfg = self.encoder.payload_cfg
        max_objs = int(pcfg.get("max_objects_per_frame", 4))
        thumb_px = int(pcfg.get("thumbnail_max_px", 160))
        thumb_q = int(pcfg.get("thumbnail_jpeg_quality", 55))
        base_q = int(pcfg.get("baseline_jpeg_quality", 85))
        frame_index, rich_events = 0, 0
        previous_payload_mode = self.encoder.mode
        last_frame_at = None

        while self.running:
            cycle_start = time.perf_counter()
            if self.paused:
                await asyncio.sleep(period)
                continue

            pcfg = self.encoder.payload_cfg
            max_objs = int(pcfg.get("max_objects_per_frame", 4))
            # Stages 1-2 run off the event loop: decode and inference block.
            frame = await loop.run_in_executor(None, camera.read)
            if frame is None:
                await asyncio.sleep(period)
                continue

            detections = await loop.run_in_executor(
                None, detector.detect, frame, camera)
            baseline = await loop.run_in_executor(
                None, encode_full_frame, frame, base_q)

            self.metrics.record_frame(len(detections),
                                      getattr(detector, "last_infer_ms", 0.0),
                                      baseline, camera.camera_id)

            frame_index += 1
            frame_at = time.perf_counter()
            current_fps = 1.0 / (frame_at - last_frame_at) if last_frame_at else fps
            last_frame_at = frame_at
            camera_status = self.camera_state[camera.camera_id]
            camera_status.update(frame_index=frame_index, fps=round(current_fps, 2),
                                 width=int(frame.shape[1]), height=int(frame.shape[0]))
            detect_ts = time.time()
            # Strongest detections first, so a capped budget keeps best evidence.
            detections.sort(key=lambda d: d["conf"], reverse=True)
            for det in detections[:max_objs]:
                det["detect_ts"] = detect_ts
                thumb = None
                mode = self.encoder.mode
                if mode != previous_payload_mode:
                    rich_events = 0
                    previous_payload_mode = mode
                rich = mode == MODE_RICH
                thumb_px = int(pcfg.get("rich_thumbnail_max_px", 240) if rich else pcfg.get("thumbnail_max_px", 160))
                thumb_q = int(pcfg.get("rich_thumbnail_jpeg_quality", 80) if rich else pcfg.get("thumbnail_jpeg_quality", 55))
                if mode in (MODE_HIGH, MODE_RICH):
                    thumb = await loop.run_in_executor(
                        None, crop_roi, frame, det["bbox"], thumb_px, thumb_q)
                keyframe = None
                if rich:
                    cadence = max(1, int(pcfg.get("keyframe_every_n_events", 8)))
                    if rich_events % cadence == 0:
                        keyframe = await loop.run_in_executor(
                            None, encode_keyframe, frame, detections,
                            int(pcfg.get("keyframe_max_px", 640)),
                            int(pcfg.get("keyframe_jpeg_quality", 65)))
                    rich_events += 1
                payload, wire = self.encoder.build_payload(
                    det, camera.camera_id, thumb, mode=mode,
                    detections=detections, keyframe_jpeg=keyframe,
                    edge_stats={"inference_ms": round(detector.last_infer_ms, 3),
                                "frame_index": frame_index, "camera_fps": round(current_fps, 2)})
                # Stage 5 must not stall stages 1-2 while the link drains.
                if len(self._transmissions) >= int(pcfg.get("max_pending_transmissions", 256)):
                    self.metrics.record_queue_drop(camera.camera_id)
                    continue
                task = asyncio.create_task(self._transmit(payload, wire))
                self._transmissions.add(task)
                task.add_done_callback(self._transmissions.discard)

            elapsed = time.perf_counter() - cycle_start
            await asyncio.sleep(max(period - elapsed, 0.0))

    async def _transmit(self, payload: dict, wire: bytes) -> None:
        try:
            result = await self.link.transmit(len(wire))
            self.metrics.record_payload(len(wire), payload["mode"], result.delivered,
                                        payload["camera_id"])
            if result.delivered:
                received = time.time()
                self.camera_state[payload["camera_id"]]["last_event_ts"] = received
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
            "alerts_config": dict(self.alerts.cfg),
            "payload_config": dict(self.encoder.payload_cfg),
            "dashboard_config": self.cfg.get("dashboard", {}),
            "link": self.link.snapshot(),
            "kpis": self.metrics.summary(),
            "pipeline": {
                "camera": getattr(self.camera, "kind", "?"),
                "cameras": list(self.camera_state.values()),
                "camera_count": len(self.cameras),
                "pending_transmissions": len(self._transmissions),
                "detector": self.detector.info.backend if self.detector else "?",
                "detector_detail": self.detector.info.detail if self.detector else "",
                "reconnects": getattr(self.camera, "reconnects", 0),
                "paused": self.paused,
                "scenario_enabled": self.scenario.enabled,
            },
        }
