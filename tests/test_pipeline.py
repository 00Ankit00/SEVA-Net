"""Contract and behavior checks; run with python -m unittest discover -s tests."""
from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from seva.alerts import AlertManager
from seva.camera import build_cameras
from seva.config import ROOT, load_config
from seva.detector import encode_keyframe
from seva.edge_node import EdgeNode
from seva.encoder import SemanticEncoder
from seva.link import EmulatedLink, LinkProfile
from seva.metrics import MetricsCollector
from seva.probe import QoSSample


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.cfg["encoder"].update(ewma_alpha=1.0, min_hold_s=1.0)
        with patch("seva.encoder.time.time", return_value=100):
            self.encoder = SemanticEncoder(self.cfg["encoder"], self.cfg["payload"])

    def decide(self, ts, bw, latency):
        return self.encoder.update(QoSSample(ts, bw, latency, 0))

    def test_every_transition_has_dwell_and_hysteresis(self):
        self.assertFalse(self.decide(100.5, 20, 20).changed)
        self.assertEqual(self.decide(101, 20, 20).mode, "rich")
        self.assertEqual(self.decide(102, 11, 90).mode, "rich")
        self.assertFalse(self.decide(101.5, 0.5, 400).changed)
        self.assertEqual(self.decide(103, 0.5, 400).mode, "low")
        self.assertEqual(self.decide(104, 1.6, 100).mode, "low")
        self.assertEqual(self.decide(105, 20, 20).mode, "high")
        self.assertFalse(self.decide(105.5, 20, 20).changed)
        self.assertEqual(self.decide(106, 20, 20).mode, "rich")
        self.assertEqual(self.decide(107, 8, 40).mode, "high")
        self.assertEqual(self.decide(108, 8, 40).mode, "high")

    def test_either_signal_degrades_both_required_to_recover(self):
        self.assertEqual(self.decide(101, 8, 200).mode, "low")
        self.assertEqual(self.decide(102, 20, 150).mode, "low")
        self.assertEqual(self.decide(103, 1.5, 30).mode, "low")
        self.assertEqual(self.decide(104, 8, 30).mode, "high")
        self.assertEqual(self.decide(105, 20, 90).mode, "high")
        self.assertEqual(self.decide(106, 20, 30).mode, "rich")
        self.assertEqual(self.decide(107, 20, 115).mode, "high")

    def test_low_wire_contract_only_adds_camera_id_and_omits_images(self):
        detection = {"event": "intrusion", "label": "person", "conf": .9237,
                     "bbox": [1, 2, 30, 40], "detect_ts": 50.0}
        payload, wire = self.encoder.build_payload(detection, "CAM-01", b"jpeg",
            mode="low", keyframe_jpeg=b"keyframe", detections=[detection], edge_stats={"fps": 4})
        self.assertEqual(list(payload), ["mode", "camera", "camera_id", "event", "label",
                                         "conf", "bbox", "ts", "detect_ts"])
        self.assertEqual(json.loads(wire), payload)
        self.assertNotIn("thumbnail", payload)
        self.assertNotIn("keyframe", payload)
        self.assertNotIn("detections", payload)
        # Original field order, values and compact serialization are preserved
        # after removing the approved additive camera_id field.
        original = dict(payload)
        original.pop("camera_id")
        expected = (f'{{"mode":"low","camera":"CAM-01","event":"intrusion",'
                    f'"label":"person","conf":0.924,"bbox":[1,2,30,40],'
                    f'"ts":"{payload["ts"]}","detect_ts":50.0}}').encode()
        self.assertEqual(json.dumps(original, separators=(",", ":")).encode(), expected)

    def test_rich_all_detections_and_images_are_encoded_in_wire(self):
        detections = [{"event": "intrusion", "label": "person", "conf": .9,
                       "bbox": [20, 30, 50, 60]} for _ in range(6)]
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        jpeg = encode_keyframe(frame, detections, 320, 65)
        payload, wire = self.encoder.build_payload(detections[0], "CAM-04", b"roi",
            mode="rich", keyframe_jpeg=jpeg, detections=detections,
            edge_stats={"inference_ms": 2, "frame_index": 8, "camera_fps": 4})
        received = json.loads(wire)
        self.assertEqual(len(received["detections"]), 6)
        self.assertEqual(received["class_counts"], {"person": 6})
        image = cv2.imdecode(np.frombuffer(base64.b64decode(received["keyframe"]), np.uint8), 1)
        self.assertEqual(image.shape[:2], (180, 320))
        self.assertGreater(np.count_nonzero(image), 0)
        self.assertEqual(payload, received)

    def test_retune_rejects_invalid_bands_atomically(self):
        before = self.encoder.thresholds
        with self.assertRaises(ValueError):
            self.encoder.retune(bw_switch_up_mbps=.5)
        self.assertEqual(before, self.encoder.thresholds)
        with self.assertRaises(ValueError):
            self.encoder.retune(ewma_alpha=float("nan"))

    def test_alert_warning_rearms_only_after_leaving_margin(self):
        manager = AlertManager(self.cfg["alerts"])
        def alerts(ts, bw, latency=30):
            old = self.encoder.mode
            return manager.decide(self.decide(ts, bw, latency), old, self.encoder.thresholds)
        self.assertEqual(alerts(102, 1.35)[0]["level"], "warning")
        self.assertEqual(alerts(103, 1.30), [])
        self.assertEqual(alerts(104, 1.5), [])
        self.assertEqual(alerts(105, 1.35)[0]["level"], "warning")
        self.assertEqual(alerts(106, .5)[0]["level"], "critical")
        self.assertEqual(alerts(107, .5), [])
        self.assertEqual(alerts(108, 8)[0]["level"], "info")
        manager.cfg["enabled"] = False
        self.assertEqual(alerts(110, .5), [])


class CameraTests(unittest.TestCase):
    def test_unique_synthetic_views_and_single_camera(self):
        cameras = build_cameras({"source": "synthetic", "count": 4}, ROOT)
        self.assertEqual([c.camera_id for c in cameras], [f"CAM-{i:02d}" for i in range(1, 5)])
        frames = [camera.read() for camera in cameras]
        self.assertTrue(all(not np.array_equal(frames[0], frame) for frame in frames[1:]))
        self.assertEqual(len(build_cameras({"source": "auto", "count": 1}, Path("missing"))), 1)

    def test_video_reuse_offsets(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            videos = root / "data" / "videos"
            videos.mkdir(parents=True)
            for name in ("a", "b"):
                writer = cv2.VideoWriter(str(videos / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
                for i in range(20):
                    writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
                writer.release()
            cameras = build_cameras({"source": "auto", "count": 4, "reuse_offset_s": .5}, root)
            try:
                self.assertEqual(cameras[0].source, cameras[2].source)
                self.assertNotEqual(cameras[0].source, cameras[1].source)
                self.assertGreater(cameras[2].start_offset_s, 0)
                self.assertFalse(np.array_equal(cameras[0].read(), cameras[2].read()))
            finally:
                for camera in cameras:
                    camera.release()


class AsyncPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_link_preserves_delay_under_busy_event_loop(self):
        link = EmulatedLink(LinkProfile(20, 10, 0, 0))
        async def busy():
            for _ in range(1000):
                await asyncio.sleep(0)
        started = time.perf_counter()
        result, _ = await asyncio.gather(link.transmit(1), busy())
        self.assertGreaterEqual(time.perf_counter() - started, .01)
        self.assertTrue(result.delivered)

    async def test_link_serializes_contenders_and_probe_sees_queue(self):
        link = EmulatedLink(LinkProfile(.8, 0, 0, 0))
        started = time.perf_counter()
        first = asyncio.create_task(link.transmit(10_000))
        await asyncio.sleep(.01)
        probe = asyncio.create_task(link.saturating_transfer(10_000))
        one, two = await asyncio.gather(first, probe)
        self.assertGreaterEqual(time.perf_counter() - started, .19)
        self.assertGreater(two.airtime_s, .17)
        self.assertEqual(link.total_bytes_delivered, 20_000)

    async def test_four_cameras_only_deliver_after_link_success(self):
        cfg = load_config()
        cfg["camera"].update(source="synthetic", count=4)
        cfg["detector"]["backend"] = "simulated"
        received = []
        async def on_event(payload, meta):
            received.append(payload)
        node = EdgeNode(cfg, ROOT, on_event=on_event)
        node.scenario.enabled = False
        await node.start()
        await asyncio.sleep(.8)
        await node.stop()
        self.assertEqual({p["camera_id"] for p in received}, {f"CAM-{i:02d}" for i in range(1, 5)})
        summary = node.metrics.summary()
        self.assertEqual(sum(c["frames"] for c in summary["per_camera"].values()), summary["frames"])
        self.assertEqual(sum(c["bytes_sent"] for c in summary["per_camera"].values()), summary["bytes_sent"])
        self.assertFalse(node._transmissions)
        # A fully lossy link cannot expose any of the locally generated images.
        cfg["link"]["initial"]["loss_pct"] = 100
        received.clear()
        node = EdgeNode(cfg, ROOT, on_event=on_event)
        node.scenario.enabled = False
        await node.start()
        await asyncio.sleep(.4)
        await node.stop()
        self.assertEqual(received, [])
        self.assertGreater(node.metrics.payloads_dropped, 0)


class MetricsTests(unittest.TestCase):
    def test_mode_time_and_per_camera_totals(self):
        with patch("seva.metrics.time.time", return_value=100):
            metrics = MetricsCollector()
        metrics.record_transition({"ts": 103, "from": "high", "to": "rich"})
        metrics.record_transition({"ts": 107, "from": "rich", "to": "low"})
        metrics.record_frame(2, 5, 1000, "CAM-01")
        metrics.record_payload(600, "rich", True, "CAM-01")
        metrics.record_frame(1, 3, 500, "CAM-02")
        metrics.record_payload(200, "low", False, "CAM-02")
        with patch("seva.metrics.time.time", return_value=110):
            metrics.finish()
        summary = metrics.summary()
        self.assertEqual(summary["mode_time_s"], {"high": 3, "rich": 4, "low": 3})
        self.assertEqual(summary["avg_payload_bytes_by_mode"]["rich"], 600)
        self.assertEqual(summary["bytes_sent"], 800)
        self.assertEqual(summary["payloads_dropped"], 1)


if __name__ == "__main__":
    unittest.main()
