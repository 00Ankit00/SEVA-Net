"""Headless trial harness — produces the numbers for the review report.

Runs the full pipeline with no dashboard for a fixed duration, replaying the
scripted degradation timeline, then writes a KPI JSON and prints a summary
table. This is what generates the figures quoted in Reviews 2-4.

    python scripts/run_trial.py --duration 120
    python scripts/run_trial.py --duration 180 --tag hysteresis-wide
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from seva.config import load_config       # noqa: E402
from seva.edge_node import EdgeNode       # noqa: E402


async def trial(cfg, duration_s: float, tag: str) -> dict:
    node = EdgeNode(cfg, ROOT)
    await asyncio.get_running_loop().run_in_executor(None, node.prepare)
    print(f"[trial] pipeline: {node.status_note}")
    print(f"[trial] running for {duration_s:.0f}s …")

    await node.start()
    deadline = time.time() + duration_s
    last_mode = node.encoder.mode
    while time.time() < deadline:
        await asyncio.sleep(1.0)
        remaining = deadline - time.time()
        if node.encoder.mode != last_mode:
            last_mode = node.encoder.mode
            print(f"  [{duration_s - remaining:6.1f}s] MODE -> {last_mode.upper()}"
                  f"   ({node.link.profile.label})")
        if int(remaining) % 15 == 0 and remaining > 1:
            k = node.metrics.summary()
            print(f"  [{duration_s - remaining:6.1f}s] mode={node.encoder.mode:<4} "
                  f"saved={k['bandwidth_saved_pct']:5.1f}%  "
                  f"frames={k['frames']:<4} payloads={k['payloads_sent']}")
            await asyncio.sleep(1.0)

    await node.stop()

    out = ROOT / cfg.get_path("run.results_dir", "results") / \
        f"trial_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    node.metrics.export(out)
    return {"kpis": node.metrics.summary(), "path": out,
            "transitions": node.metrics.transitions}


def print_report(result: dict) -> None:
    k = result["kpis"]
    rows = [
        ("Bandwidth saved vs full-frame streaming", f"{k['bandwidth_saved_pct']:.2f} %"),
        ("Total bytes transmitted", f"{k['bytes_sent']:,} B"),
        ("  in HIGH mode", f"{k['bytes_sent_high']:,} B"),
        ("  in LOW mode", f"{k['bytes_sent_low']:,} B"),
        ("Full-frame streaming baseline", f"{k['baseline_bytes']:,} B"),
        ("Average payload size", f"{k['avg_payload_bytes']:.0f} B"),
        ("", ""),
        ("Mode transitions", str(k["mode_transitions"])),
        ("Mode-flap rate", f"{k['flap_rate_per_hour']:.2f} / hour"),
        ("Avg mode-transition latency",
         f"{k['avg_transition_latency_ms']:.0f} ms" if k["avg_transition_latency_ms"]
         else "n/a"),
        ("", ""),
        ("Median end-to-end alert latency", f"{k['median_e2e_ms']:.0f} ms"),
        ("Avg edge inference per frame", f"{k['avg_infer_ms']:.2f} ms"),
        ("Frames processed", str(k["frames"])),
        ("Objects detected", str(k["detections"])),
        ("Payloads delivered / lost",
         f"{k['payloads_delivered']} / {k['payloads_dropped']}"),
    ]
    width = max(len(a) for a, _ in rows)
    print("\n" + "=" * (width + 24))
    print("  SEVA-Net trial results (PRD section 13 KPIs)")
    print("=" * (width + 24))
    for label, value in rows:
        print(f"  {label:<{width}}   {value}" if label else "")
    print("=" * (width + 24))

    if result["transitions"]:
        print("\n  Mode transitions")
        print("  " + "-" * (width + 20))
        t0 = result["transitions"][0]["ts"]
        for tr in result["transitions"]:
            lat = tr.get("transition_latency_ms")
            lat_s = f"  (+{lat:.0f} ms after link event: {tr.get('triggered_by')})" \
                    if lat is not None else ""
            print(f"  t+{tr['ts']-t0:6.1f}s  {tr['from']:>4} -> {tr['to']:<4} "
                  f"| {tr['reason']}{lat_s}")
    print(f"\n  Report written to {result['path']}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    cfg = load_config(args.config)
    result = asyncio.run(trial(cfg, args.duration, args.tag))
    print_report(result)


if __name__ == "__main__":
    main()
