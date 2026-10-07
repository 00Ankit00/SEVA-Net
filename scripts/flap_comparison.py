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
import json
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

    def compare(values):
        # The naive and EWMA traces use HIGH/RICH when testing the upper band.
        # Only the number of changes matters for these stationary comparisons.
        local = load_config(args.config)
        local["encoder"]["min_hold_s"] = 0.0
        encoder = SemanticEncoder(local["encoder"], local["payload"])
        for i, bw in enumerate(values):
            encoder.update(QoSSample(time.time() + i, bw, 30.0, 0.0))
        by_edge = {"low_high": 0, "high_rich": 0}
        for change in encoder.transitions:
            if "low" in (change["from"], change["to"]):
                by_edge["low_high"] += 1
            if "rich" in (change["from"], change["to"]):
                by_edge["high_rich"] += 1
        def ladder_changes(smoothing=False):
            ewma, mode, changes = EWMA(alpha), "high", 0
            for value in values:
                bw = ewma.update(value) if smoothing else value
                target = "rich" if bw > rich_centre else "high" if bw > centre else "low"
                if target != mode:
                    changes += 1
                    mode = target
            return changes
        return {"naive": ladder_changes(), "ewma_only": ladder_changes(True),
                "dna_se": len(encoder.transitions), "changes_by_edge": by_edge}

    rich_down = float(enc_cfg["rich_bw_switch_down_mbps"])
    rich_up = float(enc_cfg["rich_bw_switch_up_mbps"])
    rich_centre = (rich_down + rich_up) / 2
    rich_trace = make_trace(args.samples, rich_centre, args.sigma, args.seed)
    low_result = compare(trace)
    rich_result = compare(rich_trace)
    total = {key: low_result[key] + rich_result[key]
             for key in ("naive", "ewma_only", "dna_se")}
    print(f"\nFlap comparison: {args.samples} samples per band, sigma={args.sigma}, seed={args.seed}")
    print("Dwell disabled here to isolate smoothing/hysteresis; production dwell stays enabled.")
    print(f"{'Band / ladder':<30} {'Naive':>8} {'EWMA':>8} {'DNA-SE':>8}")
    for name, result in ((f"LOW/HIGH {down:g}-{up:g}", low_result),
                         (f"HIGH/RICH {rich_down:g}-{rich_up:g}", rich_result),
                         ("Three-tier total (two traces)", total)):
        print(f"{name:<30} {result['naive']:>8} {result['ewma_only']:>8} {result['dna_se']:>8}")
    out = ROOT / cfg.get_path("run.results_dir", "results") / "flap_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"samples_per_band": args.samples, "sigma": args.sigma,
                              "seed": args.seed, "dwell_disabled": True,
                              "low_high": low_result, "high_rich": rich_result,
                              "three_tier_total": total}, indent=2), encoding="utf-8")
    print(f"Report: {out}\n")


if __name__ == "__main__":
    main()
