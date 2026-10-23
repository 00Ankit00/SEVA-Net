"""Cloud dashboard container (FR-14, FR-15, FR-16).

Receives payloads from the edge node across the emulated link, keeps a rolling
event history, streams everything to connected browsers over a WebSocket, and
exposes control endpoints so the link can be degraded live during a demo.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..alerts import DetectionAlerter
from ..config import load_config
from ..detector import encode_placeholder, encode_preview
from ..edge_node import EdgeNode
from ..link import LinkProfile

ROOT = Path(__file__).resolve().parent.parent.parent
STATIC = Path(__file__).resolve().parent / "static"

class Hub:
    """Fan-out to every connected dashboard."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def register(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def drop(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        if not self.clients:
            return
        text = json.dumps(message)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.drop(ws)


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path)
    presets = {name: LinkProfile(**profile)
               for name, profile in cfg.get_path("link.presets", {}).items()}
    app = FastAPI(title="SEVA-Net Cloud Dashboard", version="0.1.0")

    hub = Hub()
    events: deque[dict] = deque(maxlen=int(cfg.get_path("cloud.max_events", 400)))
    alerts: deque[dict] = deque(maxlen=int(cfg.get_path("alerts.max_history", 200)))
    state: dict = {"node": None, "loop": None, "last_telemetry": None}
    detection_alerts = DetectionAlerter(cfg.get("detection_alerts", {}))

    # -- edge -> cloud ingest ---------------------------------------------
    async def on_event(payload: dict, meta: dict) -> None:
        record = {
            "type": "event",
            "payload": payload,
            "meta": meta,
            "e2e_ms": round((meta["received_ts"] - payload.get("detect_ts",
                             meta["received_ts"])) * 1000.0, 1),
        }
        events.append(record)          # FR-16: historical log
        await hub.broadcast(record)    # FR-14: near-real-time display
        live_ids = {node.live_camera.camera_id} if node.live_camera else set()
        alert = detection_alerts.decide(payload, live_ids)
        if alert is not None:
            alerts.append(alert)
            await hub.broadcast(alert)

    def on_telemetry(snapshot: dict) -> None:
        state["last_telemetry"] = snapshot
        loop = state.get("loop")
        if loop is not None:
            loop.create_task(hub.broadcast(snapshot))

    def on_alert(alert: dict) -> None:
        alerts.append(alert)
        loop = state.get("loop")
        if loop is not None:
            loop.create_task(hub.broadcast(alert))

    node = EdgeNode(cfg, ROOT, on_event=on_event, on_telemetry=on_telemetry,
                    on_alert=on_alert)
    app.state.node = node
    state["node"] = node

    # -- lifecycle ---------------------------------------------------------
    @app.on_event("startup")
    async def _startup() -> None:
        state["loop"] = asyncio.get_running_loop()
        await asyncio.get_running_loop().run_in_executor(None, node.prepare)
        await node.start()
        print(f"[seva] pipeline running - {node.status_note}")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await node.stop()

    # -- dashboard ---------------------------------------------------------
    @app.get("/")
    async def index() -> FileResponse:
        # no-cache: browsers must revalidate, so a stale dashboard is never shown
        return FileResponse(STATIC / "index.html",
                            headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.register(ws)
        try:
            # Prime the new client with current state and recent history.
            await ws.send_text(json.dumps({
                "type": "snapshot",
                "telemetry": state["last_telemetry"] or node.telemetry_snapshot(),
                "events": list(events)[-int(cfg.get_path("dashboard.max_snapshot_events", 200)):],
                "alerts": list(alerts),
                "presets": {k: v.as_dict() for k, v in presets.items()},
            }))
            while True:
                await ws.receive_text()   # keepalive; controls go over REST
        except WebSocketDisconnect:
            hub.drop(ws)
        except Exception:
            hub.drop(ws)

    # -- REST: state -------------------------------------------------------
    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse(state["last_telemetry"] or node.telemetry_snapshot())

    @app.get("/api/events")
    async def api_events(limit: int = 100) -> JSONResponse:
        return JSONResponse({"events": list(events)[-limit:]})

    @app.get("/api/kpis")
    async def api_kpis() -> JSONResponse:
        return JSONResponse(node.metrics.summary())

    @app.get("/api/alerts")
    async def api_alerts(limit: int = 100) -> JSONResponse:
        return JSONResponse({"alerts": list(alerts)[-max(1, min(limit, alerts.maxlen)):]})

    @app.post("/api/alerts/tune")
    async def tune_alerts(body: dict) -> JSONResponse:
        allowed = {"enabled", "warning_enabled", "warning_margin", "auto_dismiss_s",
                   "sound_enabled", "flash_duration_s", "max_toasts"}
        candidate = dict(node.alerts.cfg)
        candidate.update({k: v for k, v in body.items() if k in allowed})
        try:
            for key in ("enabled", "warning_enabled", "sound_enabled"):
                if not isinstance(candidate.get(key, False), bool):
                    raise ValueError(f"{key} must be boolean")
            for key in ("warning_margin", "auto_dismiss_s", "flash_duration_s"):
                candidate[key] = float(candidate[key])
                if not math.isfinite(candidate[key]):
                    raise ValueError(f"{key} must be finite")
            if not 0 <= float(candidate["warning_margin"]) <= 1:
                raise ValueError("warning_margin must be between 0 and 1")
            if not 1 <= int(candidate["max_toasts"]) <= 3 or int(candidate["max_toasts"]) != float(candidate["max_toasts"]):
                raise ValueError("max_toasts must be between 1 and 3")
            candidate["max_toasts"] = int(candidate["max_toasts"])
            if float(candidate["auto_dismiss_s"]) <= 0 or float(candidate["flash_duration_s"]) < 0:
                raise ValueError("invalid toast or flash duration")
        except (ValueError, TypeError, OverflowError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        node.alerts.cfg = candidate
        return JSONResponse({"ok": True, "alerts_config": candidate})

    # -- live camera -------------------------------------------------------
    @app.get("/api/live/stream")
    async def live_stream() -> StreamingResponse:
        """MJPEG preview of the live camera with detection boxes drawn on.

        This is the operator's local view of the edge camera. Detection events
        from the same camera still cross the shared link like every other one.
        """
        live_cfg = cfg.get("live_camera", {})
        period = 1.0 / max(float(live_cfg.get("preview_fps", 12)), 1.0)
        quality = int(live_cfg.get("preview_jpeg_quality", 70))
        placeholders = {"off": "CAMERA OFF", "starting": "CONNECTING TO CAMERA...",
                        "unavailable": "CAMERA UNAVAILABLE"}

        async def frames():
            loop = asyncio.get_running_loop()
            last_seq = -1
            while True:
                cam = node.live_camera
                frame, seq, detections = cam.latest() if cam else (None, -1, [])
                if frame is not None and seq != last_seq:
                    jpeg = await loop.run_in_executor(None, encode_preview, frame,
                                                      detections, quality)
                    last_seq = seq
                elif frame is None:
                    text = placeholders.get(cam.status if cam else "off", "NO SIGNAL")
                    jpeg = await loop.run_in_executor(None, encode_placeholder, text,
                                                      getattr(cam, "W", 640), getattr(cam, "H", 480))
                    last_seq = -1
                    await asyncio.sleep(0.5)
                else:
                    await asyncio.sleep(period / 2)
                    continue
                if jpeg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(period)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.post("/api/live/{action}")
    async def live_control(action: str) -> JSONResponse:
        cam = node.live_camera
        if cam is None:
            return JSONResponse({"error": "live camera is disabled in config.yaml"},
                                status_code=404)
        if action == "start":
            cam.start()
        elif action == "stop":
            cam.stop()
        else:
            return JSONResponse({"error": "action must be start or stop"}, status_code=400)
        return JSONResponse({"ok": True, "live": node.live_status()})

    # -- REST: demo controls ----------------------------------------------
    @app.post("/api/link/preset/{name}")
    async def set_preset(name: str) -> JSONResponse:
        profile = presets.get(name)
        if profile is None:
            return JSONResponse({"error": f"unknown preset {name}"}, status_code=404)
        # A manual override takes the link off the scripted timeline.
        node.scenario.enabled = False
        node.link.apply(profile)
        node.metrics.record_link_event(profile.as_dict(), time.time())
        return JSONResponse({"ok": True, "applied": profile.as_dict(),
                             "scenario_enabled": False})

    @app.post("/api/link/custom")
    async def set_custom(body: dict) -> JSONResponse:
        profile = LinkProfile(
            bw_mbps=float(body.get("bw_mbps", node.link.profile.bw_mbps)),
            delay_ms=float(body.get("delay_ms", node.link.profile.delay_ms)),
            jitter_ms=float(body.get("jitter_ms", node.link.profile.jitter_ms)),
            loss_pct=float(body.get("loss_pct", node.link.profile.loss_pct)),
            label=str(body.get("label", "manual")),
        )
        node.scenario.enabled = False
        node.link.apply(profile)
        node.metrics.record_link_event(profile.as_dict(), time.time())
        return JSONResponse({"ok": True, "applied": profile.as_dict()})

    @app.post("/api/scenario/{action}")
    async def scenario(action: str) -> JSONResponse:
        if action == "start":
            node.scenario.enabled = True
            node.scenario.reset()
        elif action == "stop":
            node.scenario.enabled = False
        else:
            return JSONResponse({"error": "action must be start or stop"},
                                status_code=400)
        return JSONResponse({"ok": True, "scenario_enabled": node.scenario.enabled})

    @app.post("/api/encoder/tune")
    async def tune(body: dict) -> JSONResponse:
        """FR-10 / NFR-6: retune thresholds live, no restart."""
        try:
            node.encoder.retune(**{k: v for k, v in body.items() if v is not None})
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "thresholds": node.encoder.thresholds})

    @app.post("/api/payload/tune")
    async def tune_payload(body: dict) -> JSONResponse:
        candidate = dict(node.encoder.payload_cfg)
        allowed = {"thumbnail_max_px", "thumbnail_jpeg_quality", "rich_thumbnail_max_px",
                   "rich_thumbnail_jpeg_quality", "keyframe_max_px", "keyframe_jpeg_quality",
                   "keyframe_every_n_events", "max_objects_per_frame", "max_pending_transmissions"}
        try:
            for key, value in body.items():
                if key not in allowed:
                    continue
                number = int(value)
                if number != float(value) or number < 1:
                    raise ValueError(f"{key} must be a positive integer")
                if key.endswith("quality") and number > 100:
                    raise ValueError("JPEG quality must be <= 100")
                candidate[key] = number
        except (ValueError, TypeError, OverflowError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        node.encoder.payload_cfg = candidate
        return JSONResponse({"ok": True, "payload_config": candidate})

    @app.post("/api/pipeline/{action}")
    async def pipeline(action: str) -> JSONResponse:
        if action == "pause":
            node.paused = True
        elif action == "resume":
            node.paused = False
        elif action == "reset-metrics":
            node.metrics.__init__(start_mode=node.encoder.mode)
        else:
            return JSONResponse({"error": "unknown action"}, status_code=400)
        return JSONResponse({"ok": True, "paused": node.paused})

    @app.post("/api/report")
    async def report() -> JSONResponse:
        out = ROOT / cfg.get_path("run.results_dir", "results") / \
            f"seva_kpi_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path = node.metrics.export(out)
        return JSONResponse({"ok": True, "path": str(path),
                             "kpis": node.metrics.summary()})

    return app


app = create_app()
