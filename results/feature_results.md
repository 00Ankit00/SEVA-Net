# Three-feature verification results

Measured on 7 October 2026 on Windows with the in-process emulator. No footage,
YOLO weights, or ultralytics were available. Current results use independent
synthetic cameras and simulated detections; they are not YOLO performance claims.
Original HEAD and final full trials were run on the same machine. Background
verification jobs also ran during measurement, so timings include normal host
scheduling variation. New default timeline adds Excellent at 145 seconds.

| KPI | Historical README: real footage, one camera | Original code rerun: synthetic, one camera | New: synthetic, four cameras |
|---|---|---|---|
| Bandwidth saved | 95.6% | 61.95% | 53.46% |
| Bytes sent | ~599 KB | 2,251,990 B | 11,160,703 B |
| Streaming baseline | ~13.7 MB | 5,917,968 B | 23,980,156 B |
| Avg payload, all modes | Not listed | 921.8 B | 1148.5 B |
| Avg HIGH payload | ~1.7 KB | Not recorded by mode | 1612.5 B |
| Avg LOW payload | ~170 B | Not recorded by mode | 186.9 B |
| Avg RICH payload | Not applicable | Not applicable | 3688.3 B |
| Time LOW / HIGH / RICH | Not recorded | Not recorded | 72.09 s / 74.41 s / 14.00 s |
| Transition to LOW | 1.6 s | 1.458 s | 5.673 s |
| Recovery to HIGH | 3.1 s | 2.561 s | 2.840 s |
| Transition to RICH | Not applicable | Not applicable | 1.335 s |
| Mode transitions | 2 | 2 | 3 |
| Median e2e, trailing window | 34 ms | 27.1 ms | 14.6 ms |
| Average inference | 24.6 ms (YOLO) | 0.0 ms (simulated) | 0.0 ms (simulated) |
| Delivered / link-loss drops | 350 / 4 | 2423 / 20 | 9596 / 122 |
| Queue overflow drops | No bounded queue | No bounded queue | 141 |

Savings decreased by 8.49 percentage
points against the remeasured synthetic baseline. Degradation and recovery to HIGH
were slower. Absolute bytes/losses increased with four cameras, corrected shared
serialization and RICH evidence. Queue overflow dropped 141 payloads
before transmission; these bytes are excluded from bytes sent and reported
separately from the 122 shaped-link losses. No 95% claim is made
for this configuration. End-to-end/inference statistics use the existing trailing
600-sample windows, which cover different time spans for one versus four cameras. Savings includes payload
base64/JSON overhead but excludes active QoS probe traffic, as in the original
collector; it does not claim total backhaul traffic savings.

## RICH congestion check

A 40-second Excellent-only run with four cameras and one keyframe per eight RICH
events per camera produced one HIGH→RICH transition, no subsequent transitions,
38.57 s in RICH, zero queue drops, and zero network losses.
Average RICH payload: 3796.6 B. Sent
9,429,526 B versus baseline 6,098,569 B, giving
-54.62% savings. RICH is stable at this cadence but consumes
more than the full-frame JPEG baseline for these simple synthetic scenes.

## Flap comparison

400 samples per band, sigma=0.8 Mbps, seed=7; dwell disabled solely to isolate
EWMA/hysteresis, with production dwell covered by regression tests.

| Trace | Original naive / EWMA / DNA-SE | New naive / EWMA / DNA-SE |
|---|---|---|
| LOW/HIGH | 190 / 100 / 19 | 190 / 100 / 19 |
| HIGH/RICH | Not applicable | 189 / 97 / 0 |
| Two traces / three-tier total (800 samples) | Not applicable | 379 / 197 / 19 |

Planned transitions are still counted by the legacy transitions-per-hour KPI;
three planned transitions are not three self-induced flaps.

## Per-camera final breakdown

| Camera | Frames | Delivered | Link loss | Queue overflow | Bytes sent | Saved |
|---|---:|---:|---:|---:|---:|---:|
| CAM-01 | 644 | 2416 | 28 | 31 | 2,860,971 | 52.30% |
| CAM-02 | 644 | 2402 | 35 | 42 | 2,866,338 | 52.37% |
| CAM-03 | 645 | 2412 | 33 | 36 | 2,775,632 | 53.71% |
| CAM-04 | 644 | 2366 | 26 | 32 | 2,657,762 | 55.47% |

## Verification and artifacts

- Feature 2: 5-second headless trial and actual run_demo HTTP/history checks passed.
- Feature 1: 5-second headless trial; actual demo Excellent→RICH delivered JPEG
  keyframes, all detections, counts, and edge stats.
- Feature 3: 5-second four-camera headless trial and full demo HTTP/WS smoke passed.
- Twelve focused regression tests pass; a single-camera fallback trial also passed.
- Linux Mininet was not executed. The original Windows unsupported-platform
  diagnostic and script compilation were checked.
- Dashboard verified at laptop width: RICH views, HIGH ROI views, LOW outlines,
  critical/recovery toasts, history, layouts, tile expansion, and stale styling.

Raw reports (ignored by Git, preserved in the workspace):

- `baseline_synthetic_160.json`
- `trial_final_four_camera_20261007_095805.json`
- `trial_rich_stability_20261007_095606.json`
- `flap_comparison.json`
