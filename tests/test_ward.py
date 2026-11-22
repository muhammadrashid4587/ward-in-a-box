"""Tests for ward-in-a-box."""

import http.client
import io
import json
import os
import sys
import threading
import unittest
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ward.chaos import ChaosEvent
from ward.cli import load_scenario, main
from ward.server import make_handler
from ward.vitals import PROFILES
from ward.ward import Ward

HERE = os.path.dirname(__file__)


def serve_in_thread(ward):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ward))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def get(port, path, timeout=3):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


class TestVitals(unittest.TestCase):
    def test_vitals_stay_in_plausible_ranges(self):
        for profile, ranges in PROFILES.items():
            ward = Ward(beds=1, profiles=[profile], seed=7)
            for _ in range(200):
                snap = ward.tick()
                reading = snap["beds"][0]
                self.assertTrue(reading["online"])
                for key, (lo, hi) in ranges.items():
                    self.assertGreaterEqual(reading[key], lo, (profile, key))
                    self.assertLessEqual(reading[key], hi, (profile, key))

    def test_alarms_fire_out_of_band(self):
        from ward.vitals import alarms_for

        self.assertIn("tachycardia", alarms_for({"heart_rate": 150, "spo2": 98, "systolic": 120, "diastolic": 80, "temperature": 37.0, "resp_rate": 16}))
        self.assertIn("low_spo2", alarms_for({"heart_rate": 70, "spo2": 85, "systolic": 120, "diastolic": 80, "temperature": 37.0, "resp_rate": 16}))
        self.assertEqual(alarms_for({"heart_rate": 70, "spo2": 98, "systolic": 120, "diastolic": 80, "temperature": 37.0, "resp_rate": 16}), [])


class TestChaos(unittest.TestCase):
    def test_dropout_stops_a_monitors_stream(self):
        ward = Ward(beds=4, chaos=[ChaosEvent("dropout", at=1, duration=5, bed=1)], seed=1)
        ward.tick()  # t=1: dropout starts
        snap = ward.tick()  # t=2
        self.assertFalse(snap["beds"][1]["online"])
        self.assertTrue(snap["beds"][0]["online"])
        for _ in range(5):  # t=7: dropout over
            snap = ward.tick()
        self.assertTrue(snap["beds"][1]["online"])

    def test_alarm_storm_produces_alarms(self):
        ward = Ward(
            beds=6,
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, beds=[0, 1, 2])],
            seed=2,
        )
        ward.tick()  # t=1: storm starts
        alarmed = 0
        for _ in range(6):
            snap = ward.tick()
            alarmed = sum(1 for b in snap["beds"] if b.get("alarms"))
        self.assertGreaterEqual(alarmed, 3, "storm beds should be alarming")

    def test_stale_reading_is_frozen(self):
        ward = Ward(beds=2, chaos=[ChaosEvent("stale", at=1, duration=5, bed=0)], seed=3)
        ward.tick()  # t=1: stale starts
        first = ward.tick()["beds"][0]  # t=2, stale copy
        second = ward.tick()["beds"][0]  # t=3, stale copy
        self.assertTrue(first["stale"])
        self.assertTrue(second["stale"])
        for key in ("heart_rate", "spo2", "systolic", "diastolic", "temperature", "resp_rate"):
            self.assertEqual(first[key], second[key])
        # Bed 1 keeps moving.
        self.assertFalse(ward.tick()["beds"][1].get("stale"))

    def test_partition_blocks_snapshot(self):
        ward = Ward(beds=2, chaos=[ChaosEvent("partition", at=1, duration=5)], seed=4)
        ward.tick()  # t=1: partitioned
        self.assertTrue(ward.partitioned)
        server, port = serve_in_thread(ward)
        try:
            status, _ = get(port, "/snapshot")
            self.assertEqual(status, 503)
            for _ in range(5):  # t=6: partition over
                ward.tick()
            self.assertFalse(ward.partitioned)
            status, body = get(port, "/snapshot")
            self.assertEqual(status, 200)
            self.assertEqual(len(json.loads(body)["beds"]), 2)
        finally:
            server.shutdown()
            server.server_close()


class TestHttp(unittest.TestCase):
    def setUp(self):
        self.ward = Ward(beds=4, rate_hz=20.0, seed=5)
        self.server, self.port = serve_in_thread(self.ward)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_health_and_snapshot_serve_valid_json(self):
        status, body = get(self.port, "/health")
        self.assertEqual(status, 200)
        health = json.loads(body)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["beds"], 4)

        status, body = get(self.port, "/snapshot")
        self.assertEqual(status, 200)
        snap = json.loads(body)
        self.assertEqual(len(snap["beds"]), 4)
        self.assertTrue(all("heart_rate" in b for b in snap["beds"] if b["online"]))

    def test_stream_serves_json_lines(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            conn.request("GET", "/stream")
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            for _ in range(3):
                line = resp.readline()
                self.assertTrue(line, "stream should keep producing lines")
                snap = json.loads(line)
                self.assertEqual(len(snap["beds"]), 4)
        finally:
            conn.close()

    def test_unknown_path_is_404(self):
        status, _ = get(self.port, "/nope")
        self.assertEqual(status, 404)


class TestScenario(unittest.TestCase):
    def test_example_scenario_loads(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "example.json"))
        self.assertEqual(cfg["beds"], 12)
        self.assertEqual(len(cfg["chaos"]), 4)
        ward = Ward(beds=cfg["beds"], rate_hz=cfg["rate_hz"], profiles=cfg["profiles"], chaos=cfg["chaos"], seed=9)
        snap = ward.tick()
        self.assertEqual(len(snap["beds"]), 12)

    def test_bad_profile_rejected(self):
        path = os.path.join(HERE, "..", "scenarios", "example.json")
        cfg = load_scenario(path)
        with self.assertRaises(ValueError):
            Ward(beds=2, profiles=["not-a-profile"])

    def test_unknown_chaos_type_rejected(self):
        with self.assertRaises(ValueError):
            ChaosEvent("meteor", at=1, duration=1)

    def test_cli_once_prints_a_snapshot(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--beds", "2", "--once"])
        self.assertEqual(rc, 0)
        snap = json.loads(buf.getvalue())
        self.assertEqual(len(snap["beds"]), 2)


if __name__ == "__main__":
    unittest.main()
