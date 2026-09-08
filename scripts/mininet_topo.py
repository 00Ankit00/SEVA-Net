"""Mininet topology for SEVA-Net (Linux only).

The laptop demo uses seva/link.py, an in-process emulation of this topology,
because Mininet needs Linux network namespaces and `tc` and therefore cannot
run on Windows or macOS. This script is the real thing, for running the same
trials on a Linux host or VM:

    sudo python3 scripts/mininet_topo.py --duration 160

Topology

    h_edge ---- s1 ---- h_cloud
             (shaped link: bw / delay / jitter / loss)

The scenario steps below are read from config.yaml, so a trial scripted for
the emulator reproduces here without edits. iPerf3 and ping run between the
two hosts to supply genuine QoS telemetry, replacing the emulated probe.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from mininet.net import Mininet
    from mininet.node import OVSKernelSwitch
    from mininet.link import TCLink
    from mininet.log import setLogLevel, info
except ImportError:
    print("Mininet is not installed, or this is not Linux.\n"
          "Mininet requires Linux network namespaces and `tc`.\n\n"
          "On Windows/macOS use the in-process emulation instead:\n"
          "    python run_demo.py\n"
          "    python scripts/run_trial.py --duration 160\n\n"
          "On Linux:  sudo apt install mininet iperf3\n")
    sys.exit(1)

from seva.config import load_config  # noqa: E402


def build(cfg):
    net = Mininet(switch=OVSKernelSwitch, link=TCLink, controller=None)

    h_edge = net.addHost("h_edge", ip="10.0.0.1/24")
    h_cloud = net.addHost("h_cloud", ip="10.0.0.2/24")
    s1 = net.addSwitch("s1", failMode="standalone")

    ini = cfg.get_path("link.initial", {})
    net.addLink(h_edge, s1, cls=TCLink,
                bw=float(ini.get("bw_mbps", 8.0)),
                delay=f"{float(ini.get('delay_ms', 20))}ms",
                jitter=f"{float(ini.get('jitter_ms', 3))}ms",
                loss=float(ini.get("loss_pct", 0.0)))
    net.addLink(s1, h_cloud, cls=TCLink, bw=1000, delay="1ms")
    return net, h_edge, h_cloud


def reshape(net, step):
    """Retune the shaped link, the way the emulator's ScenarioRunner does."""
    link = net.get("h_edge").intfList()[1].link
    intf = link.intf1
    intf.config(bw=float(step.get("bw_mbps", 8.0)),
                delay=f"{float(step.get('delay_ms', 20))}ms",
                jitter=f"{float(step.get('jitter_ms', 3))}ms",
                loss=float(step.get("loss_pct", 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=160.0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    setLogLevel("info")
    cfg = load_config(args.config)
    net, h_edge, h_cloud = build(cfg)
    net.start()

    info("*** connectivity check\n")
    net.ping([h_edge, h_cloud])

    info("*** starting iperf3 server + cloud dashboard on h_cloud\n")
    h_cloud.cmd("iperf3 -s -D")
    h_cloud.cmd(f"cd {ROOT} && python3 run_demo.py --host 0.0.0.0 "
                f"--port {args.port} --no-browser > /tmp/seva_cloud.log 2>&1 &")
    time.sleep(6)

    info(f"*** dashboard at http://10.0.0.2:{args.port}\n")
    info("*** replaying scenario\n")

    steps = sorted(cfg.get_path("link.scenario", []) or [],
                   key=lambda s: s.get("at_s", 0))
    t0 = time.time()
    idx = 0
    while time.time() - t0 < args.duration:
        elapsed = time.time() - t0
        while idx < len(steps) and elapsed >= steps[idx]["at_s"]:
            step = steps[idx]
            info(f"    t+{elapsed:6.1f}s  {step.get('label')} "
                 f"({step.get('bw_mbps')} Mbps, {step.get('delay_ms')} ms, "
                 f"{step.get('loss_pct')}% loss)\n")
            reshape(net, step)
            idx += 1
        # Real telemetry, replacing the emulated probe.
        bw = h_edge.cmd("iperf3 -c 10.0.0.2 -t 1 -J 2>/dev/null | "
                        "grep -m1 bits_per_second | head -1")
        rtt = h_edge.cmd("ping -c 1 -W 1 10.0.0.2 | tail -1")
        info(f"    iperf3: {bw.strip()[:70]} | ping: {rtt.strip()[:60]}\n")
        time.sleep(4)

    info("*** stopping\n")
    h_cloud.cmd("pkill -f run_demo.py; pkill iperf3")
    net.stop()


if __name__ == "__main__":
    main()
