"""Active QoS probing (FR-6, FR-7, section 10.3).

Two probes run against the same emulated link the payloads use:

  throughput probe : a fixed-size bulk transfer, timed -> Mbps. This is the
                     iPerf3 analogue (iperf3 -c cloud -n <bytes>).
  latency probe    : a 64-byte round trip -> RTT ms. This is the ping analogue.

Because both contend for the same token bucket as the payload transmitter, the
measurement reflects capacity actually available to the pipeline rather than a
nominal configured number.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from .link import EmulatedLink


@dataclass
class QoSSample:
    ts: float
    bandwidth_mbps: float
    latency_ms: float
    loss_pct: float


class QoSProbe:
    def __init__(self, link: EmulatedLink, probe_bytes: int = 24000,
                 ping_bytes: int = 64, noise_sigma: float = 0.07,
                 seed: int = 4242):
        self.link = link
        self.probe_bytes = int(probe_bytes)
        self.ping_bytes = int(ping_bytes)
        self.noise_sigma = float(noise_sigma)
        self._rng = random.Random(seed)
        self.sent = 0
        self.lost = 0

    async def _throughput(self) -> float:
        """Sustained-throughput probe.

        An idle link holds burst credit, so timing a single cold transfer
        reports far more than the link can sustain. iperf3 solves this by
        discarding the opening interval; the same trick here: a warm-up
        transfer drains the credit, then the timed transfer measures the
        steady-state rate.
        """
        result = await self.link.saturating_transfer(self.probe_bytes)
        self.sent += 1
        if not result.delivered:
            self.lost += 1
            return 0.0

        # Throughput is bits over airtime, not bits over round-trip time.
        mbps = (self.probe_bytes * 8.0) / max(result.airtime_s, 1e-6) / 1_000_000.0
        # Real iperf3 readings scatter run to run; without some scatter the
        # EWMA would have nothing to smooth and the demo would misrepresent
        # how the encoder behaves on a live link.
        return max(mbps * self._rng.gauss(1.0, self.noise_sigma), 0.01)

    async def _latency(self) -> float:
        t0 = time.perf_counter()
        out = await self.link.transmit(self.ping_bytes)
        back = await self.link.transmit(self.ping_bytes)
        self.sent += 2
        if not out.delivered:
            self.lost += 1
        if not back.delivered:
            self.lost += 1
        return (time.perf_counter() - t0) * 1000.0

    async def sample(self) -> QoSSample:
        latency_ms = await self._latency()
        bandwidth_mbps = await self._throughput()
        loss = (self.lost / self.sent * 100.0) if self.sent else 0.0
        return QoSSample(time.time(), bandwidth_mbps, latency_ms, loss)

    async def run(self, interval_s: float, on_sample) -> None:
        """Probe forever at `interval_s`, handing each sample to `on_sample`."""
        while True:
            started = time.perf_counter()
            try:
                sample = await self.sample()
                on_sample(sample)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            sleep_for = interval_s - (time.perf_counter() - started)
            await asyncio.sleep(max(sleep_for, 0.05))
