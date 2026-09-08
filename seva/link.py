"""Emulated camera -> cloud backhaul link.

The PRD specifies Mininet for link emulation. Mininet depends on Linux network
namespaces and `tc`, so it cannot run on Windows or macOS. This module is a
drop-in stand-in exposing the same four knobs Mininet's TCLink exposes
(bandwidth, delay, jitter, loss) with the same semantics:

  * bandwidth  -> token bucket, so oversized payloads queue instead of vanishing
  * delay      -> fixed one-way propagation delay
  * jitter     -> uniform random variation around that delay
  * loss       -> Bernoulli per-transmission drop

`scripts/mininet_topo.py` builds the equivalent real topology for Linux hosts;
`LinkProfile` values map 1:1 onto its TCLink parameters, so a trial scripted
here reproduces on Mininet without edits.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, asdict, field


@dataclass
class LinkProfile:
    bw_mbps: float = 8.0
    delay_ms: float = 20.0
    jitter_ms: float = 3.0
    loss_pct: float = 0.0
    label: str = "unnamed"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transmission:
    """Outcome of pushing `nbytes` across the link.

    `airtime_s` is the time the bits actually occupied the link: for an idle
    link it equals nbytes/rate (pure serialization), and it grows above that
    when the bucket is already drained by earlier traffic (queueing).
    """
    delivered: bool
    nbytes: int
    airtime_s: float = 0.0
    propagation_s: float = 0.0

    @property
    def total_delay_s(self) -> float:
        return self.airtime_s + self.propagation_s


class EmulatedLink:
    """Token-bucket rate limiter with delay, jitter and loss.

    A single asyncio lock serialises the bucket so concurrent senders (the
    payload transmitter and the QoS prober) contend for the same capacity —
    which is what makes the probe an honest measurement of what the payload
    path will actually get.
    """

    # Bucket may hold at most this many seconds' worth of credit, so an idle
    # link cannot grant an unrealistic instantaneous burst.
    BURST_SECONDS = 0.05

    def __init__(self, profile: LinkProfile, rng: random.Random | None = None):
        self.profile = profile
        self._rng = rng or random.Random(1337)
        self._lock = asyncio.Lock()
        self._tokens = 0.0
        self._last_refill = time.perf_counter()
        self.total_bytes_offered = 0
        self.total_bytes_delivered = 0
        self.total_drops = 0
        self.profile_changed_at = time.time()

    # -- configuration -----------------------------------------------------
    def apply(self, profile: LinkProfile) -> None:
        """Change link conditions, as `tc qdisc change` would in Mininet."""
        self.profile = profile
        self.profile_changed_at = time.time()
        self._tokens = 0.0
        self._last_refill = time.perf_counter()

    @property
    def rate_bytes_s(self) -> float:
        return max(self.profile.bw_mbps, 0.001) * 1_000_000.0 / 8.0

    # -- transport ---------------------------------------------------------
    def _refill(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_refill
        self._last_refill = now
        rate = self.rate_bytes_s
        self._tokens = min(self._tokens + elapsed * rate, rate * self.BURST_SECONDS)

    async def transmit(self, nbytes: int) -> Transmission:
        """Send `nbytes` across the link, sleeping for the real elapsed time."""
        self.total_bytes_offered += nbytes

        if self.profile.loss_pct > 0 and self._rng.random() * 100.0 < self.profile.loss_pct:
            self.total_drops += 1
            # A dropped packet still consumed link time before it was lost.
            await asyncio.sleep(self.profile.delay_ms / 1000.0)
            return Transmission(delivered=False, nbytes=nbytes,
                                propagation_s=self.profile.delay_ms / 1000.0)

        async with self._lock:
            self._refill()
            rate = self.rate_bytes_s
            # Waiting for enough credit *is* the time the link needs to clock
            # these bytes out; charging serialization again would double-count.
            airtime = max(nbytes - self._tokens, 0.0) / rate
            self._tokens = max(self._tokens - nbytes, 0.0)

        jitter = self._rng.uniform(-self.profile.jitter_ms, self.profile.jitter_ms)
        propagation = max(self.profile.delay_ms + jitter, 0.0) / 1000.0

        await asyncio.sleep(airtime + propagation)

        self.total_bytes_delivered += nbytes
        return Transmission(True, nbytes, airtime, propagation)

    async def saturating_transfer(self, nbytes: int) -> Transmission:
        """Transfer used by the QoS prober, not by payload traffic.

        A single short transfer across an idle link rides on accumulated burst
        credit and reports far more than the link can sustain. A real iperf3
        run does not: it saturates the link long enough that the shaper is the
        binding constraint. Draining the bucket inside the lock reproduces that
        steady state, so the probe reports achievable rather than burst rate.

        Holding the lock for the transfer also means payload traffic queues
        behind the probe, as it would behind a real measurement stream.
        """
        self.total_bytes_offered += nbytes

        if self.profile.loss_pct > 0 and self._rng.random() * 100.0 < self.profile.loss_pct:
            self.total_drops += 1
            await asyncio.sleep(self.profile.delay_ms / 1000.0)
            return Transmission(False, nbytes,
                                propagation_s=self.profile.delay_ms / 1000.0)

        async with self._lock:
            rate = self.rate_bytes_s
            self._tokens = 0.0          # steady state: no burst credit left
            self._last_refill = time.perf_counter()
            airtime = nbytes / rate
            jitter = self._rng.uniform(-self.profile.jitter_ms, self.profile.jitter_ms)
            propagation = max(self.profile.delay_ms + jitter, 0.0) / 1000.0
            await asyncio.sleep(airtime + propagation)

        self.total_bytes_delivered += nbytes
        return Transmission(True, nbytes, airtime, propagation)

    def snapshot(self) -> dict:
        return {
            **self.profile.as_dict(),
            "bytes_offered": self.total_bytes_offered,
            "bytes_delivered": self.total_bytes_delivered,
            "drops": self.total_drops,
        }


class ScenarioRunner:
    """Replays a scripted degradation timeline onto a link.

    Mirrors what a Mininet trial script does between `net.start()` and
    `net.stop()`: wait, retune the link, wait, retune again.
    """

    def __init__(self, link: EmulatedLink, steps: list[dict], loop: bool = True):
        self.link = link
        self.steps = sorted(steps, key=lambda s: s.get("at_s", 0))
        self.loop = loop
        self.enabled = True
        self.started_at = time.time()
        self._idx = 0
        self.last_event: dict | None = None
        self.on_change = None  # callable(profile, wall_ts)

    def reset(self) -> None:
        self.started_at = time.time()
        self._idx = 0

    def _profile_from(self, step: dict) -> LinkProfile:
        return LinkProfile(
            bw_mbps=float(step.get("bw_mbps", 8.0)),
            delay_ms=float(step.get("delay_ms", 20.0)),
            jitter_ms=float(step.get("jitter_ms", 3.0)),
            loss_pct=float(step.get("loss_pct", 0.0)),
            label=str(step.get("label", "step")),
        )

    def tick(self) -> LinkProfile | None:
        """Apply any step whose scheduled time has arrived. Returns it, or None."""
        if not self.enabled or not self.steps:
            return None

        elapsed = time.time() - self.started_at
        applied = None
        while self._idx < len(self.steps) and elapsed >= self.steps[self._idx]["at_s"]:
            profile = self._profile_from(self.steps[self._idx])
            self.link.apply(profile)
            applied = profile
            self._idx += 1

        if applied is not None:
            self.last_event = {"ts": time.time(), "profile": applied.as_dict()}
            if self.on_change:
                self.on_change(applied, self.last_event["ts"])

        if self._idx >= len(self.steps) and self.loop:
            span = self.steps[-1]["at_s"] + 20
            if elapsed >= span:
                self.reset()
        return applied
