"""Tests for ward-in-a-box."""

import http.client
import io
import json
import os
import random
import sys
import threading
import unittest
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ward.chaos import ChaosEvent
from ward.cli import load_scenario, main
from ward.monitor import Monitor
from ward.server import make_handler
from ward.timeline import Timeline, TimelinePhase
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


class TestAlarmHistory(unittest.TestCase):
    # Pin the signs to known in-band values so the storm drags are exact.
    # Each tick of an active storm moves a vital 35% toward its alarm
    # target, so a vital starting at the top of its profile range trips
    # its alarm on a known tick, and ending the storm clamps the signs
    # back into band (via step()) so the alarm clears on the next tick.
    SIGNS = {
        "heart_rate": 88.0,
        "spo2": 99.0,
        "systolic": 117.0,
        "diastolic": 75.0,
        "temperature": 36.8,
        "resp_rate": 15.0,
    }

    def test_history_records_raise_and_clear(self):
        m = Monitor(0, "stable", rng=random.Random(1))
        m.signs.update(self.SIGNS)
        m.alarm_until = 10.0
        reading = m.tick(1.0)
        self.assertEqual(reading["alarms"], [])
        self.assertEqual(len(m.alarm_history), 0)
        # t=2: heart rate 88 -> 121.2, the only vital out of band.
        reading = m.tick(2.0)
        self.assertEqual(reading["alarms"], ["tachycardia"])
        self.assertEqual(len(m.alarm_history), 1)
        event = m.alarm_history[0]
        self.assertEqual(event["alarm"], "tachycardia")
        self.assertEqual(event["raised_t"], 2.0)
        self.assertIsNone(event["cleared_t"])
        self.assertEqual(m.total_raised, 1)

        # Storm over: the next step clamps the signs back into band,
        # so the same event is closed and no new event is raised.
        m.alarm_until = None
        reading = m.tick(3.0)
        self.assertEqual(reading["alarms"], [])
        self.assertEqual(m.alarm_history[0]["cleared_t"], 3.0)
        self.assertEqual(m.total_raised, 1)

        # A second, different alarm raises its own event.
        m.alarm_until = 10.0
        m.signs["spo2"] = 90.0
        reading = m.tick(4.0)
        self.assertEqual(reading["alarms"], ["low_spo2"])
        self.assertEqual(m.total_raised, 2)
        self.assertEqual(m.alarm_history[1]["alarm"], "low_spo2")
        self.assertEqual(m.alarm_history[1]["raised_t"], 4.0)

    def test_alarm_rate_counts_trailing_window(self):
        m = Monitor(0, "stable", rng=random.Random(2))
        m.signs.update(self.SIGNS)
        m.signs["heart_rate"] = 130.0  # out of band already; storm keeps it there
        m.alarm_until = 100.0
        m.tick(1.0)  # tachycardia raised at t=1
        m.alarm_until = None
        m.tick(2.0)  # cleared
        m.signs["spo2"] = 88.0
        m.alarm_until = 100.0
        m.tick(10.0)  # low_spo2 raised at t=10
        m.alarm_until = None
        m.tick(11.0)  # cleared
        m.signs["temperature"] = 39.0
        m.alarm_until = 100.0
        m.tick(70.0)  # fever raised at t=70
        self.assertEqual(m.alarm_rate(70.0), 2.0)  # raises at t=10, t=70
        self.assertEqual(m.alarm_rate(70.0, window_s=120.0), 1.5)  # all three
        self.assertEqual(m.alarm_rate(5.0), 1.0)  # only the t=1 raise
        with self.assertRaises(ValueError):
            m.alarm_rate(70.0, window_s=0)


class TestWardAlarms(unittest.TestCase):
    def test_ward_alarm_summary_after_storm(self):
        ward = Ward(
            beds=6,
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, beds=[0, 1, 2])],
            seed=2,
        )
        for _ in range(7):
            ward.tick()
        summary = ward.alarm_summary()
        self.assertEqual(summary["t"], ward.elapsed)
        self.assertEqual(len(summary["beds"]), 6)
        self.assertGreaterEqual(summary["total_raised"], 3)
        self.assertGreater(summary["rate_per_min"], 0)
        for bed in summary["beds"]:
            self.assertIn("history", bed)
            self.assertIn("active", bed)
            self.assertIn("rate_per_min", bed)
            self.assertEqual(bed["total_raised"], len(bed["history"]))
        storm = summary["beds"][0]
        self.assertGreater(storm["total_raised"], 0)
        self.assertTrue(storm["active"])


class TestAlarmsHttp(unittest.TestCase):
    def test_alarms_endpoint_serves_json(self):
        ward = Ward(beds=4, rate_hz=20.0, seed=5)
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/alarms")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(len(data["beds"]), 4)
            self.assertIn("total_raised", data)
            self.assertIn("rate_per_min", data)
        finally:
            server.shutdown()
            server.server_close()

    def test_alarms_endpoint_is_503_while_partitioned(self):
        ward = Ward(beds=2, chaos=[ChaosEvent("partition", at=1, duration=5)], seed=4)
        ward.tick()  # t=1: partitioned
        server, port = serve_in_thread(ward)
        try:
            status, _ = get(port, "/alarms")
            self.assertEqual(status, 503)
        finally:
            server.shutdown()
            server.server_close()


class TestTimeline(unittest.TestCase):
    def test_occupied_at_ramps_linearly(self):
        tl = Timeline([
            TimelinePhase(at=0, beds=4, ramp_s=10),
            TimelinePhase(at=30, beds=8, ramp_s=20),
        ])
        self.assertEqual(tl.occupied_at(0), 0)
        self.assertEqual(tl.occupied_at(5), 2)
        self.assertEqual(tl.occupied_at(10), 4)
        self.assertEqual(tl.occupied_at(20), 4)
        self.assertEqual(tl.occupied_at(40), 6)
        self.assertEqual(tl.occupied_at(50), 8)
        self.assertEqual(tl.occupied_at(999), 8)

    def test_instant_phase_admits_at_once(self):
        tl = Timeline([TimelinePhase(at=5, beds=3, ramp_s=0)])
        self.assertEqual(tl.occupied_at(4.9), 0)
        self.assertEqual(tl.occupied_at(5), 3)

    def test_admission_time_spreads_beds_over_ramp(self):
        tl = Timeline([TimelinePhase(at=0, beds=4, ramp_s=10)])
        self.assertEqual(tl.admission_time(0), 2.5)
        self.assertEqual(tl.admission_time(1), 5.0)
        self.assertEqual(tl.admission_time(2), 7.5)
        self.assertEqual(tl.admission_time(3), 10.0)
        self.assertIsNone(tl.admission_time(4))

    def test_admission_time_across_phases(self):
        tl = Timeline([
            TimelinePhase(at=0, beds=2, ramp_s=2),
            TimelinePhase(at=5, beds=4, ramp_s=0),
        ])
        self.assertEqual(tl.admission_time(0), 1.0)
        self.assertEqual(tl.admission_time(1), 2.0)
        self.assertEqual(tl.admission_time(2), 5.0)
        self.assertEqual(tl.admission_time(3), 5.0)

    def test_decreasing_targets_rejected(self):
        with self.assertRaises(ValueError):
            Timeline([TimelinePhase(0, 4), TimelinePhase(10, 2)])

    def test_bad_phase_values_rejected(self):
        with self.assertRaises(ValueError):
            TimelinePhase(at=-1, beds=2)
        with self.assertRaises(ValueError):
            TimelinePhase(at=0, beds=-1)
        with self.assertRaises(ValueError):
            TimelinePhase(at=0, beds=2, ramp_s=-1)

    def test_phases_sort_by_start_time(self):
        tl = Timeline([TimelinePhase(30, 8, 20), TimelinePhase(0, 4, 10)])
        self.assertEqual(tl.occupied_at(5), 2)
        self.assertEqual(tl.max_beds, 8)


class TestTimelineWard(unittest.TestCase):
    def test_beds_come_online_as_timeline_admits_them(self):
        ward = Ward(
            beds=4,
            timeline=Timeline([
                TimelinePhase(at=0, beds=2, ramp_s=2),
                TimelinePhase(at=5, beds=4, ramp_s=0),
            ]),
            seed=11,
        )
        snap = ward.tick()  # t=1: bed 0 admitted at t=1
        self.assertTrue(snap["beds"][0]["online"])
        self.assertFalse(snap["beds"][1]["online"])
        self.assertFalse(snap["beds"][1]["admitted"])
        snap = ward.tick()  # t=2: bed 1 admitted at t=2
        self.assertTrue(snap["beds"][1]["online"])
        self.assertFalse(snap["beds"][2]["online"])
        for _ in range(3):  # t=5: instant phase admits beds 2 and 3
            snap = ward.tick()
        self.assertTrue(all(b["online"] for b in snap["beds"]))

    def test_timeline_target_above_bed_count_rejected(self):
        with self.assertRaises(ValueError):
            Ward(beds=2, timeline=Timeline([TimelinePhase(0, 3)]))

    def test_ramp_up_scenario_loads_and_fills(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "ramp-up.json"))
        self.assertEqual(cfg["beds"], 12)
        self.assertEqual(len(cfg["timeline"].phases), 3)
        ward = Ward(beds=cfg["beds"], rate_hz=cfg["rate_hz"], profiles=cfg["profiles"], chaos=cfg["chaos"], timeline=cfg["timeline"], seed=12)
        snap = ward.tick()  # t=1: no bed admitted yet (first at t=2.5)
        self.assertFalse(any(b["online"] for b in snap["beds"]))
        for _ in range(69):  # t=70: last ramp ends at t=70
            snap = ward.tick()
        self.assertTrue(all(b["online"] for b in snap["beds"]))

    def test_cli_once_with_ramp_up_scenario(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main([os.path.join(HERE, "..", "scenarios", "ramp-up.json"), "--once"])
        self.assertEqual(rc, 0)
        snap = json.loads(buf.getvalue())
        self.assertEqual(len(snap["beds"]), 12)
        self.assertFalse(any(b["online"] for b in snap["beds"]))


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
