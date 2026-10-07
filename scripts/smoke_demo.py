"""Exercise the actual demo launcher, HTTP controls, and WebSocket stream.

    python scripts/smoke_demo.py
"""
from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    process = subprocess.Popen([sys.executable, str(ROOT / "run_demo.py"),
                                "--no-browser", "--port", str(port)], cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        with urllib.request.urlopen(urllib.request.Request(base + path, data=data,
                headers={"Content-Type": "application/json"}), timeout=3) as response:
            return json.load(response)

    async def wait_mode(mode):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            status = await asyncio.to_thread(request, "/api/status")
            if status["mode"] == mode:
                return status
            await asyncio.sleep(.2)
        raise AssertionError(f"never reached {mode}: {status}")

    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                await asyncio.to_thread(request, "/api/status")
                break
            except (urllib.error.URLError, TimeoutError):
                if process.poll() is not None:
                    raise RuntimeError("demo exited during startup")
                await asyncio.sleep(.2)
        else:
            raise RuntimeError("demo did not start")

        async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=8_000_000) as ws:
            snapshot = json.loads(await asyncio.wait_for(ws.recv(), 5))
            assert snapshot["type"] == "snapshot"
            assert len(snapshot["telemetry"]["pipeline"]["cameras"]) == 4
            seen = set()
            async def consume():
                async for message in ws:
                    seen.add(json.loads(message)["type"])
            receiver = asyncio.create_task(consume())
            await asyncio.to_thread(request, "/api/encoder/tune", {"ewma_alpha": 1, "min_hold_s": .2})
            try:
                await asyncio.to_thread(request, "/api/encoder/tune", {"rich_bw_switch_up_mbps": -1})
                raise AssertionError("invalid threshold accepted")
            except urllib.error.HTTPError as error:
                assert error.code == 400
            await asyncio.to_thread(request, "/api/payload/tune", {"keyframe_every_n_events": 4})
            await asyncio.to_thread(request, "/api/alerts/tune", {"auto_dismiss_s": 2})
            for body in ({"auto_dismiss_s": float("nan")}, {"max_toasts": 2.5}):
                try:
                    await asyncio.to_thread(request, "/api/alerts/tune", body)
                    raise AssertionError("invalid alert setting accepted")
                except urllib.error.HTTPError as error:
                    assert error.code == 400
            for preset, mode in (("excellent", "rich"), ("healthy", "high"),
                                 ("collapse", "low"), ("healthy", "high")):
                await asyncio.to_thread(request, "/api/link/preset/" + preset, {})
                await wait_mode(mode)
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    events = (await asyncio.to_thread(request, "/api/events", None))["events"]
                    matching = [e["payload"] for e in events if e["payload"]["mode"] == mode]
                    if matching and (mode != "rich" or any("keyframe" in p for p in matching)):
                        break
                    await asyncio.sleep(.2)
                assert matching, f"no delivered {mode} events"
                if mode == "low":
                    assert all("thumbnail" not in p and "keyframe" not in p for p in matching)
                if mode == "rich":
                    assert any("keyframe" in p for p in matching)
            alerts = (await asyncio.to_thread(request, "/api/alerts"))["alerts"]
            assert any(a["level"] == "critical" and a["to"] == "low" for a in alerts)
            assert any(a["level"] == "info" and a["from"] == "low" for a in alerts)
            assert {"alert", "event"}.issubset(seen)
            kpis = await asyncio.to_thread(request, "/api/kpis")
            assert len(kpis["per_camera"]) == 4
            print("Demo smoke passed: four cameras; Excellent/RICH, Healthy/HIGH, Collapse/LOW,")
            print("recovery; JPEG delivery/suppression; REST tuning/history; distinct WS alerts/events.")
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
