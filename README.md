# SEVA-Net

**Semantic Relevance-Driven Edge Video Analytics for Bandwidth-Adaptive Smart City Surveillance**

Team HAN24 · Guide: Dr. S. Yoganand · Computer Networks & IoT

A working prototype of **DNA-SE** (Dynamic Network-Aware Semantic Encoding): YOLOv8n
runs at the edge and the system transmits only what the network can currently
sustain — compact JSON in LOW, a JPEG ROI thumbnail plus JSON in HIGH, and annotated
keyframes with all detection metadata in RICH. The decision is driven by an EWMA-smoothed, hysteresis-bound
reading of live link telemetry, so the stream degrades gracefully instead of failing.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_assets.py     # YOLOv8n weights + sample footage (optional)
python run_demo.py                 # dashboard opens at http://127.0.0.1:8000
```

Both fetched assets are optional. Without them the pipeline falls back to a
synthetic camera and a simulated detector and still runs end to end.

To produce report numbers with no dashboard:

```bash
python scripts/run_trial.py --duration 160 --tag review1
```

---

## What the demo shows

The dashboard drives the whole story in one screen:

| Panel | What to point at |
|---|---|
| **Camera monitoring wall** | CAM-01 through CAM-04 share one encoder and backhaul; grid, 1+3, single view, and click-to-expand |
| **Transmission mode** | LOW (amber), HIGH (green), RICH (purple) |
| **DNA-SE decision state** | Both EWMA hysteresis bands, RTT, dwell and the exact decision reason |
| **QoS telemetry & mode timeline** | Measured and smoothed bandwidth with both bands |
| **Live event feed** | Detection events remain separate from bandwidth alerts; RICH adds keyframes, detection tables, class counts and edge stats |
| **Alert history** | Encoder-driven warnings, critical degradation and recovery notifications |
| **KPIs** | Aggregate traffic and a per-camera breakdown; bytes and time by mode |
| **Network and encoder controls** | Presets, link sliders and live threshold tuning |

**Suggested demo sequence**

1. Run `python run_demo.py`. Four camera tiles start in **HIGH** on Healthy 8 Mbps.
2. Click **Excellent 20 Mbps**. After EWMA and dwell allow recovery, the mode becomes
   **RICH**. Annotated keyframes appear on all tiles, and the event feed shows every
   relevant detection, class counts, inference time, frame index and measured FPS.
3. Click **Healthy 8 Mbps**. The encoder returns to **HIGH** and the wall displays ROI
   thumbnails centred on dark frames. The downgrade from RICH produces a warning.
4. Click **Collapse**. The shared link queues traffic, then the encoder enters **LOW**.
   A red, persistent pop-up says images are suppressed. Tiles immediately switch to
   metadata outlines; detection events continue arriving as the queue drains.
5. Click **Healthy** to recover. An **INFO** toast announces HIGH. Click **Excellent**
   to recover further to RICH and get another INFO toast.
6. Try **Peak load** and **Noisy**, and adjust the encoder tuning controls. A wider
   band reduces oscillation; all transitions still respect minimum dwell.
7. Try the three wall layouts, click a camera tile to expand/return, and toggle sound.
   Sound starts muted. A user click enables the browser's Web Audio playback.

**Operator alerts**

Alerts use encoder decisions rather than raw probe samples. Entering LOW emits
CRITICAL; recovery to HIGH/RICH emits INFO; RICH→HIGH emits WARNING. An optional
pre-warning fires once while HIGH bandwidth is within 15% above the LOW bandwidth
edge, and re-arms only after bandwidth rises outside that margin. Repeated LOW
samples do not create repeated critical alerts.

CRITICAL toasts stay until dismissed. Others expire after `alerts.auto_dismiss_s`.
At most three are visible; further alerts queue until a slot opens. The header
briefly flashes red on CRITICAL. History is bounded and available from `/api/alerts`.

**Four-camera monitoring**

`camera.count: 4` enables CAM-01–CAM-04. Auto source selection uses distinct video
files first, then cycles through them with `camera.reuse_offset_s` offsets. Each
synthetic camera uses a different seed. Each camera has its own detector/task;
all cameras share one QoS probe, one DNA-SE decision and one emulated link.
Set `camera.count: 1` for the original camera/task topology and source selection.
Camera count/source changes require a restart.

RICH tiles display the latest **received** keyframe, HIGH tiles the latest received
ROI, and LOW tiles draw received bbox metadata on a dark canvas. Images never
bypass the link. Camera dimensions/FPS in telemetry contain no image data. Wall
clocks show operator time; last-event ages and `NO SIGNAL` use received-event times.
`dashboard.no_signal_s` controls staleness. No signal means no recent detection
event, rather than proof that the camera has disconnected.

---

## Measured results

The original README reported **95.6% saved**, ~1.7 KB HIGH / ~170 B LOW,
1.6 s degradation / 3.1 s recovery, 34 ms median end-to-end latency,
24.6 ms YOLOv8n inference, and 350 delivered / 4 lost. Those were a
**single camera with real footage and YOLOv8n**. They remain historical results;
this checkout contains no video assets or weights and ultralytics is unavailable.
Fresh measurements therefore use synthetic scenes and simulated detection.

The comparison below reran original `HEAD` and the new pipeline on the same
Windows machine for 160 seconds. Original code used its original single-camera
scenario. The new default adds Excellent at 145 s so RICH is exercised. These are
emulated results, with changed camera count, traffic shaping, and scenario.

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

The full trial's transitions were HIGH→LOW, LOW→HIGH, HIGH→RICH. There were no
additional RICH/HIGH reversals. The legacy `flap_rate_per_hour` KPI counts **all mode
changes**, including planned scenario changes; it is not a count of spurious flaps.
Inference is simulated and cannot be compared with the historical YOLO timing.
End-to-end and inference summaries retain the existing trailing-600-sample window.
Savings uses detection-payload wire bytes (including base64 overhead) against
sampled full-frame JPEG bytes; active probe overhead is excluded, as in the
original collector. It is not total backhaul traffic savings.

**RICH congestion check:** a separate 40-second run held Excellent at 20 Mbps /
10 ms with four synthetic cameras. It entered RICH once, stayed there for 38.57 s,
and recorded zero payload losses and zero queue-overflow drops. One keyframe per
8 RICH events per camera was stable. It transmitted 9.43 MB against a 6.10 MB
baseline: **−54.62% savings**. Rich ROI evidence and repeated frame metadata cost
more than full-frame JPEGs of this very simple synthetic scene. RICH sacrifices
bandwidth savings for detail; the dashboard deliberately shows negative savings.
No new 95% savings claim is made.

**Flap comparison**, 400 stationary noisy samples per band, sigma 0.8 Mbps, seed 7:

| Trace | Original naive / EWMA / DNA-SE | New naive / EWMA / DNA-SE |
|---|---|---|
| LOW/HIGH, 1.2–2.0 Mbps | 190 / 100 / 19 | 190 / 100 / 19 |
| HIGH/RICH, 10–12 Mbps | Not applicable | 189 / 97 / 0 |
| Three-tier total, two traces (800 samples) | Not applicable | 379 / 197 / 19 |

Dwell is disabled **only** in this comparison to isolate smoothing and hysteresis;
production dwell remains enabled and is separately tested. The totals cover two
traces, so do not compare 379 directly with the original 400-sample count of 190.

Reproduce with:

```bash
python scripts/run_trial.py --duration 160 --tag review-three-tier
python scripts/run_trial.py --duration 40 --preset excellent --tag rich-stability
python scripts/run_trial.py --duration 5 --cameras 1 --tag single-camera
python scripts/flap_comparison.py
python -m unittest discover -s tests -v
python scripts/smoke_demo.py
```

Reports include configuration, actual camera/detector backend, per-mode payload
averages, mode durations, per-camera counters and RICH transition latency. Raw
trial JSON files are saved under `results/` and remain ignored by Git as before;
`results/feature_results.md` records the reviewed numbers and report filenames.

---

## Architecture

The five PRD stages map one-to-one onto modules:

| # | Stage | Module | Requirements |
|---|---|---|---|
| 1 | Data collection | [`seva/camera.py`](seva/camera.py) | FR-1, FR-2, FR-3 |
| 2 | Edge processing | [`seva/detector.py`](seva/detector.py) | FR-4, FR-5 |
| 3 | QoS telemetry | [`seva/probe.py`](seva/probe.py) | FR-6, FR-7 |
| 4 | Semantic encoder | [`seva/encoder.py`](seva/encoder.py) | FR-8, FR-9, FR-10 |
| 5 | Dynamic output | [`seva/edge_node.py`](seva/edge_node.py) | FR-11, FR-12, FR-13 |
| — | Cloud dashboard | [`seva/cloud/server.py`](seva/cloud/server.py) | FR-14, FR-15, FR-16 |
| — | Network emulation | [`seva/link.py`](seva/link.py) | — |
| — | KPI collection | [`seva/metrics.py`](seva/metrics.py) | §13 |

Stages 2 and 3 run as concurrent asyncio tasks; stage 4 fuses their outputs before
stage 5 transmits.

### Payload contract (PRD §9, additive camera identity extension)

Existing fields, values and compact JSON serialization remain intact. As approved,
`camera_id` is added in all modes while retaining the legacy `camera` field. This
is an additive extension, **not literal byte-for-byte compatibility**. LOW has no
`thumbnail`, `keyframe`, detection list, class counts or edge stats keys at all.

LOW example (the actual implementation has always retained label/bbox/detect_ts):

```json
{"mode":"low","camera":"CAM-01","camera_id":"CAM-01","event":"intrusion","label":"person","conf":0.92,"bbox":[10,20,30,40],"ts":"2026-10-07T04:00:00Z","detect_ts":1791345600.0}
```

HIGH adds `thumbnail`, containing a base64 JPEG ROI. RICH includes the same event
fields plus a higher-quality thumbnail, `detections` (all relevant detector outputs,
including those beyond the event cap), `class_counts`, and `edge_stats` with
`inference_ms`, `frame_index`, and measured sampled `camera_fps`. Selected RICH
events also include `keyframe`, a downscaled JPEG with boxes and labels drawn at
the edge. `payload.keyframe_every_n_events` caps the cadence independently for
each camera; the first RICH event after a mode change includes a keyframe.
Keyframe and ROI dimensions/quality are configurable and live-tunable.

### Mode-switch logic (PRD §10.2)

`EWMA_t = α·x_t + (1−α)·EWMA_{t−1}` over bandwidth and RTT:

| Transition | Condition |
|---|---|
| HIGH/RICH → LOW | Bandwidth < 1.20 Mbps **or** RTT > 180 ms |
| LOW → HIGH | Bandwidth > 2.00 Mbps **and** RTT < 120 ms |
| HIGH → RICH | Bandwidth > 12.00 Mbps **and** RTT < 80 ms |
| RICH → HIGH | Bandwidth < 10.00 Mbps **or** RTT > 110 ms |

Degradation takes priority; severe collapse can go directly RICH→LOW. Recovery
climbs LOW→HIGH→RICH, with minimum dwell protecting **each** step. Strict threshold
comparisons leave exact boundary values inside the band. The default dwell is 1 s,
including the initial mode. Both bands and dwell can be tuned without restarting.

The approved RICH bandwidth defaults are 12/10 Mbps rather than the initially
suggested 5/4: the latter would also classify Healthy 8 Mbps as RICH. Presets
configure link conditions and never force or override a mode decision.

---

## A note on Mininet

The PRD specifies Mininet for link emulation. **Mininet requires Linux network
namespaces and `tc`, so it cannot run on Windows or macOS**, where this prototype
was built.

[`seva/link.py`](seva/link.py) is therefore a faithful in-process stand-in exposing
the same four knobs as Mininet's `TCLink`, with the same semantics:

- **bandwidth** → token bucket, so oversized payloads queue rather than vanish
- **delay** → fixed one-way propagation
- **jitter** → uniform variation around that delay
- **loss** → Bernoulli per-transmission drop

[`scripts/mininet_topo.py`](scripts/mininet_topo.py) builds the equivalent real
topology for a Linux host, reading the same scenario steps from `config.yaml` and printing genuine `iperf3`/`ping` measurements. It launches the dashboard, whose
edge/cloud pipeline still uses the in-process emulator. Those printed probes are
not wired into the encoder and camera payloads do not traverse the Mininet hosts.
Use it to inspect the topology; do not present emulator KPIs as physical Mininet
payload-path measurements:

```bash
sudo python3 scripts/mininet_topo.py --duration 160    # Linux only
```

Both paths are honest about what they are. KPIs from the emulator should be
reported as emulation-based, exactly as PRD §15 requires.

### How the probe measures honestly

A single short transfer across an idle token bucket rides on accumulated burst
credit and reports far more throughput than the link can sustain — an early version
of this code read 20 Mbps on a 7 Mbps link. Real `iperf3` avoids this by saturating
the link and discarding the opening interval. `EmulatedLink.saturating_transfer()`
reproduces that steady state, and the probe follows the idle nominal rate with
configured ±7% noise. Under contention its measured
airtime includes time queued behind the camera payloads, reducing available
throughput. Payloads now reserve serialization time until it finishes, so all
cameras truly share capacity. Propagation does not occupy serialization capacity.
A Windows elapsed-time guard prevents coarse asyncio timers from finishing an
emulated delay early. Dropped payloads still consume link capacity.

The shared pending-transmission queue is bounded by
`payload.max_pending_transmissions`; overflow counts are reported separately from
network-loss drops and never credited as transmitted bytes. Shutdown drains or
cancels tracked tasks rather than leaving orphaned transmissions.

---

## Configuration

Everything tunable lives in [`config.yaml`](config.yaml) (FR-10, NFR-6) — EWMA α,
both hysteresis bands, dwell, camera count/seeds/offsets, alert behavior,
keyframe cadence and quality, thumbnail quality, sampling rate, probe interval, and
the full degradation timeline. Encoder thresholds can also be retuned live from the
dashboard or via `POST /api/encoder/tune`, with no restart. Invalid band ordering
is rejected atomically with HTTP 400. `POST /api/payload/tune` changes image sizes,
JPEG qualities, event caps, pending queue limits and keyframe cadence;
`POST /api/alerts/tune` changes alert toggles, warning margin and toast behavior.
History capacities and camera topology are startup settings.

To use your own footage, drop any `.mp4` into `data/videos/`.

## Docker (NFR-5)

```bash
docker compose -f docker/docker-compose.yml up --build      # dashboard
docker compose -f docker/docker-compose.yml --profile trial up --build   # + headless trial
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Dashboard |
| `WS /ws` | Live telemetry + separate detection-event and bandwidth-alert streams |
| `GET /api/status` · `/api/kpis` · `/api/events` | Current state, KPIs, history |
| `POST /api/link/preset/{excellent\|healthy\|peak\|degraded\|collapse\|noisy}` | Inject link conditions |
| `POST /api/link/custom` | Arbitrary bw / delay / jitter / loss |
| `POST /api/scenario/{start\|stop}` | Scripted timeline on/off |
| `POST /api/encoder/tune` | Retune α, both mode bands and dwell live |
| `GET /api/alerts` | Bounded bandwidth-alert history; optional `limit` |
| `POST /api/alerts/tune` | Alert enable/pre-warning toggles, margin, toast lifetime/cap, sound default and flash duration |
| `POST /api/payload/tune` | ROI/keyframe sizes and JPEG quality, keyframe cadence, event cap and pending queue limit |
| `POST /api/pipeline/{pause\|resume\|reset-metrics}` | Existing pipeline controls; reset retains current mode |
| `POST /api/report` | Write a KPI JSON into `results/` |

---

## Status against the PRD

**Complete:** all 16 functional requirements, the §9 payload contract, the §10
algorithms, all five §13 KPIs, and the Review 1 milestone list — with the Review 2
adaptive logic and the Review 3 dashboard already working.

**Deliberately not built** (PRD §4.2 scopes these out): physical hardware,
multi-camera fusion, dashboard authentication, model training from scratch, field
deployment, mobile clients.

**Known limitations**, worth stating plainly at review:

- Detection accuracy (mAP) is quoted from the pretrained YOLOv8n COCO checkpoint;
  this prototype does not run a COCO/Cityscapes evaluation pass, so mAP is the one
  §13 KPI not measured here.
- Link behaviour is emulated, not measured on a real cellular or Wi-Fi network.
- Multi-camera monitoring now works over one shared link. Cross-camera identity
  tracking and fusion remain out of scope.


## Files changed for the three features

| File | Why |
|---|---|
| `config.yaml` | Commented presets, RICH band/evidence, alerts, camera count/offsets/seeds, queue and wall controls |
| `seva/alerts.py` | Encoder-decision alerts and once-per-approach warning state |
| `seva/encoder.py` | Three-tier hysteresis/dwell, validated tuning, camera identity and RICH payload construction |
| `seva/detector.py` | Downscaled annotated JPEGs; automatic fallback without silently downloading weights |
| `seva/camera.py` | Independent camera identities, seeds, video selection and reuse offsets |
| `seva/edge_node.py` | Concurrent camera tasks sharing one encoder/link, alert callback, keyframe cadence and bounded/drained transmissions |
| `seva/link.py` | Real serialization contention, queue-aware probes, dropped-byte shaping and elapsed-time guard |
| `seva/probe.py` | Document queue-inclusive throughput measurement |
| `seva/metrics.py` | Per-mode averages/time/RICH latency, per-camera accounting, queue drops and honest negative savings |
| `seva/cloud/server.py` | Separate alert WS/history, configurable presets and validated live image/alert/encoder controls |
| `seva/cloud/static/index.html` | CCTV wall/layouts/expansion, metadata canvas, RICH detail, toasts/sound/history and extended KPIs |
| `scripts/run_trial.py` | New metrics and backend/configuration provenance in reports |
| `scripts/flap_comparison.py` | Both ladder bands and comparable three-tier strategies; JSON results |
| `scripts/smoke_demo.py` | Actual launcher, HTTP and WebSocket scenario regression check |
| `tests/test_pipeline.py` | Payload contract, transitions/dwell, warnings, cameras, annotation, contention and accounting tests |
| `results/feature_results.md` | Fresh measured comparison, stability evidence and artifact paths |
| `results/dashboard-rich.png` · `results/dashboard-low.png` | Screenshots of delivered keyframes and metadata-only LOW with a critical pop-up |
| `results/*.json` (ignored) | Raw stage checks, original-code baseline, final 160-second trial, stability and flap evidence |
| `README.md` | Feature descriptions, API/demo updates, contracts and measurement limitations |

`run_demo.py`, `scripts/fetch_assets.py`, and `scripts/mininet_topo.py` keep their
existing entry points. Each implementation stage passed a demo/HTTP check and a
short headless trial. The final demo smoke check covers all three tiers and
four-camera delivery. Mininet's Linux runtime is unavailable on this Windows host;
its existing platform diagnostic and Python compilation were checked.
