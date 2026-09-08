"""Flap-rate comparison (NFR-3).

Drives the same noisy-but-stable bandwidth trace through three decision
strategies and counts mode oscillations. This isolates the contribution of
EWMA smoothing and of the hysteresis band, which is the cleanest evidence
that DNA-SE does what section 10.2 claims.

    python scripts/flap_comparison.py
    python scripts/flap_comparison.py --samples 800 --sigma 0.9
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from seva.config import load_config      # noqa: E402
from seva.encoder import SemanticEncoder, EWMA  # noqa: E402
from seva.probe import QoSSample          # noqa: E402


def make_trace(n: int, centre: float, sigma: float, seed: int) -> list[float]:
    """Noisy but stationary bandwidth — the condition NFR-3 is defined over."""
    rng = random.Random(seed)
    return [max(rng.gauss(centre, sigma), 0.05) for _ in range(n)]


def count_naive(trace, threshold):
    """Single threshold, no smoothing, no hysteresis."""
    mode, flips = "high", 0
    for bw in trace:
        target = "high" if bw > threshold else "low"
        if target != mode:
            flips += 1
            mode = target
    return flips


def count_ewma_only(trace, threshold, alpha):
    """Smoothing but a single threshold — isolates the EWMA contribution."""
    ewma, mode, flips = EWMA(alpha), "high", 0
    for bw in trace:
        v = ewma.update(bw)
        target = "high" if v > threshold else "low"
        if target != mode:
            flips += 1
            mode = target
    return flips


def count_dna_se(trace, cfg):
    """Full DNA-SE: EWMA plus the hysteresis band."""
    enc = SemanticEncoder(cfg["encoder"], cfg["payload"])
    enc.cfg["min_hold_s"] = 0.0     # isolate hysteresis from the dwell timer
    now = time.time()
    for bw in trace:
        enc.update(QoSSample(now, bw, 30.0, 0.0))
    return len(enc.transitions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    enc_cfg = cfg["encoder"]
    down = float(enc_cfg["bw_switch_down_mbps"])
    up = float(enc_cfg["bw_switch_up_mbps"])
    alpha = float(enc_cfg["ewma_alpha"])
    centre = (down + up) / 2.0          # worst case: noise straddles the band

    trace = make_trace(args.samples, centre, args.sigma, args.seed)

    rows = [
        ("Naive single threshold (no smoothing, no hysteresis)",
         count_naive(trace, centre)),
        (f"EWMA only (alpha={alpha}), single threshold",
         count_ewma_only(trace, centre, alpha)),
        (f"DNA-SE: EWMA + hysteresis band {down:.2f}-{up:.2f} Mbps",
         count_dna_se(trace, cfg)),
    ]

    width = max(len(a) for a, _ in rows)
    print()
    print("  Flap-rate comparison (NFR-3)")
    print(f"  {args.samples} samples, noise sigma={args.sigma} Mbps "
          f"centred on {centre:.2f} Mbps (mid-band)")
    print("  " + "-" * (width + 16))
    for label, flips in rows:
        print(f"  {label:<{width}}  {flips:>5}")
    print("  " + "-" * (width + 16))

    naive, dnase = rows[0][1], rows[2][1]
    if dnase:
        print(f"  DNA-SE reduces mode oscillation {naive/dnase:.1f}x "
              f"vs. naive thresholding.")
    else:
        print("  DNA-SE eliminated flapping entirely on this trace.")
    print()


if __name__ == "__main__":
    main()
