"""Cloud dashboard container (FR-14, FR-15, FR-16).

Receives payloads from the edge node across the emulated link, keeps a rolling
event history, streams everything to connected browsers over a WebSocket, and
exposes control endpoints so the link can be degraded live during a demo.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import load_config
from ..edge_node import EdgeNode
from ..link import LinkProfile

ROOT = Path(__file__).resolve().parent.parent.parent
STATIC = Path(__file__).resolve().parent / "static"

# Named link conditions the dashboard buttons can apply on demand.
PRESETS = {
    "healthy":  LinkProfile(8.0, 18, 4, 0.0, "Healthy backhaul"),
    "peak":     LinkProfile(2.6, 45, 10, 0.3, "Peak-hour load"),
    "degraded": LinkProfile(0.9, 140, 30, 1.5, "Degraded uplink"),
    "collapse": LinkProfile(0.35, 220, 45, 4.0, "Congestion collapse"),
    "noisy":    LinkProfile(1.6, 130, 60, 1.5, "Noisy band (flap test)"),
}


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
    app = FastAPI(title="SEVA-Net Cloud Dashboard", version="0.1.0")

    hub = Hub()
    events: deque[dict] = deque(maxlen=int(cfg.get_path("cloud.max_events", 400)))
    state: dict = {"node": None, "loop": None, "last_telemetry": None}

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

    def on_telemetry(snapshot: dict) -> None:
        state["last_telemetry"] = snapshot
        loop = state.get("loop")
        if loop is not None:
            # Called from the pipeline; hand the send back to the event loop.
            asyncio.run_coroutine_threadsafe(hub.broadcast(snapshot), loop)

    node = EdgeNode(cfg, ROOT, on_event=on_event, on_telemetry=on_telemetry)
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
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.register(ws)
        try:
            # Prime the new client with current state and recent history.
            await ws.send_text(json.dumps({
                "type": "snapshot",
                "telemetry": state["last_telemetry"] or node.telemetry_snapshot(),
                "events": list(events)[-60:],
                "presets": {k: v.as_dict() for k, v in PRESETS.items()},
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

    # -- REST: demo controls ----------------------------------------------
    @app.post("/api/link/preset/{name}")
    async def set_preset(name: str) -> JSONResponse:
        profile = PRESETS.get(name)
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
        node.encoder.retune(**{k: v for k, v in body.items() if v is not None})
        return JSONResponse({"ok": True, "thresholds": node.encoder.thresholds})

    @app.post("/api/pipeline/{action}")
    async def pipeline(action: str) -> JSONResponse:
        if action == "pause":
            node.paused = True
        elif action == "resume":
            node.paused = False
        elif action == "reset-metrics":
            node.metrics.__init__()
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
