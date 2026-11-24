"""Tests for ward-in-a-box."""

import base64
import hashlib
import http.client
import io
import json
import os
import random
import socket
import struct
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ward.alert_hooks import (
    TrendAlertHooks,
    validate_boarding_full_hooks,
    validate_escalation_hooks,
    validate_trend_alert_hooks,
)
from ward.chaos import ChaosEvent
from ward.cli import build_ward, load_scenario, main
from ward.discharge import DischargePlanner
from ward.monitor import Monitor
from ward.readiness import DischargeReadiness
from ward.recorder import Recorder, iter_ticks, summarize
from ward.server import make_handler
from ward.staffing import Staffing
from ward.timeline import Timeline, TimelinePhase
from ward.triage import TriageQueue
from ward.news2 import (
    score_consciousness,
    score_signs,
    score_spo2,
    trend_for,
    validate_consciousness,
    validate_consciousness_dynamics,
    validate_news2_trend,
)
from ward.vitals import PROFILES, effective_ranges, validate_acuity
from ward.ward import Ward
from ward.websocket import accept_key, encode_frame, read_frame

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


def capture_server():
    """A stub HTTP server that records POST bodies.

    Returns ``(server, port, received)``: ``received`` is a list of
    ``(path, content_type, body_bytes)`` in arrival order. The
    caller shuts the server down.
    """
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received.append(
                (
                    self.path,
                    self.headers.get("Content-Type"),
                    self.rfile.read(length),
                )
            )
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1], received


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


class TestWebSocketFraming(unittest.TestCase):
    def test_accept_key_matches_rfc6455_example(self):
        # The worked example from RFC 6455, section 1.3.
        self.assertEqual(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
        )

    def test_encode_read_frame_round_trip(self):
        for payload in (b"hello", b"x" * 200, b"y" * 70000):
            buf = io.BytesIO(encode_frame(payload))
            fin, opcode, out = read_frame(buf)
            self.assertTrue(fin)
            self.assertEqual(opcode, 1)
            self.assertEqual(out, payload)

    def test_read_frame_unmasks_client_payload(self):
        mask = b"\x37\xfa\x21\x3d"
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(b"Hello"))
        frame = bytes([0x81, 0x80 | 5]) + mask + masked
        fin, opcode, out = read_frame(io.BytesIO(frame))
        self.assertTrue(fin)
        self.assertEqual(opcode, 1)
        self.assertEqual(out, b"Hello")

    def test_read_frame_empty_on_clean_eof(self):
        self.assertEqual(read_frame(io.BytesIO(b"")), (False, None, b""))


class WsClient:
    """Minimal WebSocket client for tests (masked frames, like a browser)."""

    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.buf = b""
        self.key = base64.b64encode(os.urandom(16)).decode()
        req = (
            "GET /ws HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n" % self.key
        )
        self.sock.sendall(req.encode())
        head = self._read_until(b"\r\n\r\n")
        lines = head.decode().split("\r\n")
        self.status = int(lines[0].split()[1])
        self.headers = {}
        for line in lines[1:]:
            if line:
                k, v = line.split(": ", 1)
                self.headers[k.lower()] = v

    def _read_until(self, sep):
        while sep not in self.buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("EOF waiting for %r" % sep)
            self.buf += chunk
        out, self.buf = self.buf.split(sep, 1)
        return out + sep

    def _read_exact(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("EOF while reading frame")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def read_frame(self):
        head = self._read_exact(2)
        opcode = head[0] & 0x0F
        length = head[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self._read_exact(8))[0]
        payload = self._read_exact(length) if length else b""
        return opcode, payload

    def read_until_opcode(self, opcode, limit=20):
        for _ in range(limit):
            got, payload = self.read_frame()
            if got == opcode:
                return payload
        raise AssertionError("no frame with opcode %d in %d frames" % (opcode, limit))

    def send_frame(self, payload, opcode=0x1, fin=True):
        if isinstance(payload, str):
            payload = payload.encode()
        mask = os.urandom(4)
        first = (0x80 if fin else 0x00) | opcode
        n = len(payload)
        if n < 126:
            header = bytes([first, 0x80 | n])
        elif n < 65536:
            header = bytes([first, 0x80 | 126]) + struct.pack(">H", n)
        else:
            header = bytes([first, 0x80 | 127]) + struct.pack(">Q", n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def close(self):
        self.sock.close()


class TestWebSocketEndpoint(unittest.TestCase):
    def setUp(self):
        self.ward = Ward(beds=4, rate_hz=20.0, seed=5)
        self.server, self.port = serve_in_thread(self.ward)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_handshake_returns_101_and_correct_accept(self):
        ws = WsClient(self.port)
        try:
            self.assertEqual(ws.status, 101)
            self.assertEqual(ws.headers.get("upgrade"), "websocket")
            self.assertEqual(
                ws.headers.get("sec-websocket-accept"), accept_key(ws.key)
            )
        finally:
            ws.close()

    def test_streams_tick_frames_with_beds_and_alarms(self):
        ws = WsClient(self.port)
        try:
            first = json.loads(ws.read_until_opcode(1))
            second = json.loads(ws.read_until_opcode(1))
            for msg in (first, second):
                self.assertEqual(len(msg["beds"]), 4)
                self.assertTrue(all("heart_rate" in b for b in msg["beds"] if b["online"]))
                self.assertIn("total_raised", msg["alarms"])
                self.assertIn("rate_per_min", msg["alarms"])
                self.assertEqual(len(msg["alarms"]["beds"]), 4)
            self.assertGreater(second["t"], first["t"])
        finally:
            ws.close()

    def test_ping_gets_pong(self):
        ws = WsClient(self.port)
        try:
            ws.send_frame("hello", opcode=0x9)
            self.assertEqual(ws.read_until_opcode(0xA), b"hello")
        finally:
            ws.close()

    def test_close_handshake(self):
        ws = WsClient(self.port)
        ws.send_frame(b"", opcode=0x8)
        # A tick frame may land before the close echo; skip past it.
        ws.read_until_opcode(0x8)
        # Server closes the socket after replying to close.
        with self.assertRaises(ConnectionError):
            ws.read_frame()
        ws.close()

    def test_fragmented_client_message_is_accepted(self):
        ws = WsClient(self.port)
        try:
            ws.send_frame("hel", opcode=0x1, fin=False)
            ws.send_frame("lo", opcode=0x0, fin=True)
            # Stream keeps ticking after the fragmented message.
            msg = json.loads(ws.read_until_opcode(1))
            self.assertEqual(len(msg["beds"]), 4)
        finally:
            ws.close()

    def test_ws_without_upgrade_headers_is_400(self):
        status, body = get(self.port, "/ws")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "expected a WebSocket upgrade")

    def test_ws_is_503_while_partitioned(self):
        ward = Ward(beds=2, chaos=[ChaosEvent("partition", at=1, duration=5)], seed=4)
        ward.tick()  # t=1: partitioned
        server, port = serve_in_thread(ward)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            try:
                conn.request(
                    "GET",
                    "/ws",
                    headers={
                        "Upgrade": "websocket",
                        "Connection": "Upgrade",
                        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
                        "Sec-WebSocket-Version": "13",
                    },
                )
                resp = conn.getresponse()
                self.assertEqual(resp.status, 503)
            finally:
                conn.close()
        finally:
            server.shutdown()
            server.server_close()


class TestRecording(unittest.TestCase):
    def _recorded(self, ward, ticks, seed=1):
        path = os.path.join(tempfile.mkdtemp(), "run.jsonl")
        with Recorder(path, ward.beds, ward.rate_hz, seed=seed) as rec:
            ward.recorder = rec
            snaps = [ward.tick() for _ in range(ticks)]
        return path, snaps

    def test_recorder_writes_header_and_ticks(self):
        ward = Ward(beds=2, seed=1)
        path, snaps = self._recorded(ward, 3)
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(lines[0]["type"], "header")
        self.assertEqual(lines[0]["beds"], 2)
        self.assertEqual(lines[0]["rate_hz"], 1.0)
        self.assertEqual(lines[0]["seed"], 1)
        self.assertEqual(len(lines) - 1, 3)
        for line, snap in zip(lines[1:], snaps):
            self.assertEqual(line["type"], "tick")
            self.assertEqual(line["t"], snap["t"])
            self.assertEqual(len(line["beds"]), 2)

    def test_recorder_counts_ticks(self):
        ward = Ward(beds=2, seed=1)
        path, _ = self._recorded(ward, 4)
        with open(path) as f:
            self.assertEqual(sum(1 for _ in f), 5)  # header + 4 ticks

    def test_summary_counts_ticks_duration_and_readings(self):
        ward = Ward(beds=2, seed=1)
        path, _ = self._recorded(ward, 5)
        s = summarize(path)
        self.assertEqual(s["ticks"], 5)
        self.assertEqual(s["first_t"], 1.0)
        self.assertEqual(s["last_t"], 5.0)
        self.assertEqual(s["duration_s"], 4.0)
        self.assertEqual(s["readings_per_bed"], {0: 5, 1: 5})
        self.assertEqual(s["offline_ticks_per_bed"], {})
        self.assertEqual(s["alarms_raised"], 0)

    def test_summary_counts_offline_ticks_and_alarms(self):
        ward = Ward(
            beds=2,
            seed=7,
            chaos=[
                ChaosEvent("dropout", at=1, duration=3, bed=0),
                ChaosEvent("alarm_storm", at=1, duration=5, beds=[1]),
            ],
        )
        path, _ = self._recorded(ward, 5, seed=7)
        s = summarize(path)
        self.assertEqual(s["ticks"], 5)
        self.assertEqual(s["offline_ticks_per_bed"], {0: 3})
        self.assertEqual(s["readings_per_bed"], {0: 2, 1: 5})
        self.assertGreaterEqual(s["alarms_raised"], 1)

    def test_summary_of_empty_recording(self):
        path = os.path.join(tempfile.mkdtemp(), "run.jsonl")
        Recorder(path, 2, 1.0, seed=1).close()
        s = summarize(path)
        self.assertEqual(s["ticks"], 0)
        self.assertIsNone(s["first_t"])
        self.assertIsNone(s["duration_s"])
        self.assertEqual(s["beds"], 2)

    def test_iter_ticks_replays_snapshots(self):
        ward = Ward(beds=2, seed=1)
        path, snaps = self._recorded(ward, 3)
        replayed = list(iter_ticks(path))
        self.assertEqual(len(replayed), 3)
        for replayed_snap, snap in zip(replayed, snaps):
            self.assertEqual(replayed_snap["t"], snap["t"])
            self.assertEqual(replayed_snap["beds"], snap["beds"])

    def test_cli_record_once_writes_file(self):
        path = os.path.join(tempfile.mkdtemp(), "run.jsonl")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--beds", "2", "--once", "--record", path])
        self.assertEqual(rc, 0)
        snap = json.loads(buf.getvalue())
        self.assertEqual(len(snap["beds"]), 2)
        s = summarize(path)
        self.assertEqual(s["ticks"], 1)
        self.assertEqual(s["beds"], 2)

    def test_cli_summary_prints_json(self):
        ward = Ward(beds=2, seed=1)
        path, _ = self._recorded(ward, 3)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--summary", path])
        self.assertEqual(rc, 0)
        s = json.loads(buf.getvalue())
        self.assertEqual(s["ticks"], 3)
        self.assertEqual(s["readings_per_bed"], {"0": 3, "1": 3})

    def test_cli_replay_prints_tick_lines(self):
        ward = Ward(beds=2, seed=1)
        path, snaps = self._recorded(ward, 2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--replay", path])
        self.assertEqual(rc, 0)
        lines = [json.loads(line) for line in buf.getvalue().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["t"], snaps[0]["t"])
        self.assertNotIn("type", lines[0])


class TestAcuity(unittest.TestCase):
    def test_effective_ranges_widen_with_acuity(self):
        lo1, hi1 = effective_ranges("stable", 1)["heart_rate"]
        lo3, hi3 = effective_ranges("stable", 3)["heart_rate"]
        lo4, hi4 = effective_ranges("stable", 4)["heart_rate"]
        self.assertEqual((lo3, hi3), PROFILES["stable"]["heart_rate"])
        self.assertLess(lo1, lo3)
        self.assertGreater(hi1, hi3)
        self.assertGreater(lo4, lo3)
        self.assertLess(hi4, hi3)

    def test_effective_ranges_stay_inside_plausible_envelope(self):
        for acuity in (1, 2, 3, 4):
            for profile in PROFILES:
                ranges = effective_ranges(profile, acuity)
                self.assertGreaterEqual(ranges["spo2"][1], 88.0)
                self.assertLessEqual(ranges["spo2"][1], 100.0)
                self.assertGreaterEqual(ranges["heart_rate"][0], 30.0)

    def test_validate_acuity_rejects_bad_values(self):
        for bad in (0, 5, -1, "1", 2.5, None, True):
            with self.assertRaises(ValueError):
                validate_acuity(bad)

    def test_monitor_defaults_to_acuity_3_and_reports_it(self):
        m = Monitor(0, "stable", rng=random.Random(1))
        self.assertEqual(m.acuity, 3)
        reading = m.tick(1.0)
        self.assertEqual(reading["acuity"], 3)
        self.assertEqual(m.alarm_summary(1.0)["acuity"], 3)

    def test_monitor_rejects_bad_acuity(self):
        with self.assertRaises(ValueError):
            Monitor(0, "stable", rng=random.Random(1), acuity=9)

    def test_critical_patient_swings_more_than_low_acuity(self):
        means = {}
        for acuity in (1, 4):
            m = Monitor(0, "stable", rng=random.Random(11), acuity=acuity)
            prev = None
            total = 0.0
            n = 0
            for t in range(1, 501):
                r = m.tick(float(t))
                if prev is not None:
                    total += abs(r["heart_rate"] - prev)
                    n += 1
                prev = r["heart_rate"]
            means[acuity] = total / n
        self.assertGreater(means[1], means[4] * 2)

    def test_critical_patient_raises_more_alarms(self):
        raised = {}
        for acuity in (1, 4):
            m = Monitor(0, "hypoxic", rng=random.Random(7), acuity=acuity)
            for t in range(1, 601):
                m.tick(float(t))
            raised[acuity] = m.total_raised
        self.assertGreater(raised[1], raised[4] * 2)

    def test_ward_acuities(self):
        ward = Ward(beds=2, profiles=["stable", "stable"], acuities=[1, 4], seed=1)
        self.assertEqual([m.acuity for m in ward.monitors], [1, 4])
        snap = ward.tick()
        self.assertEqual([r["acuity"] for r in snap["beds"]], [1, 4])

    def test_scenario_acuity(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "acuity.json"))
        self.assertEqual(cfg["acuities"], [1, 2, 3, 4])
        ward = build_ward(cfg, seed=42)
        snap = ward.tick()
        self.assertEqual(snap["beds"][0]["acuity"], 1)
        self.assertEqual(snap["beds"][0]["profile"], "tachycardic")
        self.assertEqual(snap["beds"][3]["acuity"], 4)

    def test_scenario_rejects_bad_acuity(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"beds": 1, "patients": [{"profile": "stable", "acuity": 0}]}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_scenario(path)
        finally:
            os.unlink(path)


class TestStaffing(unittest.TestCase):
    def test_shift_is_day_midday_and_night_after_19(self):
        s = Staffing(start_hour=8)
        self.assertEqual(s.shift_at(0), "day")        # 08:00
        self.assertEqual(s.shift_at(36000), "day")    # 18:00
        self.assertEqual(s.shift_at(39600), "night")  # 19:00
        self.assertEqual(s.shift_at(79200), "night")  # 06:00 next day
        self.assertEqual(s.shift_at(82800), "day")    # 07:00 next day
        self.assertEqual(s.shift_at(86400), "day")    # 08:00 next day

    def test_on_shift_counts_follow_the_roster(self):
        s = Staffing(
            day={"nurse": 3, "doctor": 1, "tech": 1},
            night={"nurse": 1, "doctor": 1, "tech": 0},
            start_hour=8,
        )
        self.assertEqual(s.on_shift(0)["nurse"], 3)
        self.assertEqual(s.on_shift(39600)["nurse"], 1)
        self.assertEqual(s.on_shift(39600)["tech"], 0)

    def test_handover_is_the_first_fifteen_minutes_of_a_shift(self):
        s = Staffing(start_hour=7)
        self.assertTrue(s.in_handover(0))       # shift just started
        self.assertTrue(s.in_handover(899))
        self.assertFalse(s.in_handover(900))
        self.assertTrue(s.in_handover(43200))   # 19:00, night shift starts
        self.assertFalse(s.in_handover(44100))

    def test_load_factor_is_one_when_adequate_and_grows_with_overload(self):
        s = Staffing(day={"nurse": 3}, night={"nurse": 3}, start_hour=8)
        self.assertEqual(s.load_factor(12, 0), 1.0)   # 4 per nurse
        self.assertEqual(s.load_factor(6, 0), 1.0)    # 2 per nurse
        s = Staffing(day={"nurse": 2}, night={"nurse": 2}, start_hour=8)
        self.assertEqual(s.load_factor(12, 0), 1.5)   # 6 per nurse
        s = Staffing(day={"nurse": 1}, night={"nurse": 1}, start_hour=8)
        self.assertEqual(s.load_factor(12, 0), 3.0)   # 12 per nurse

    def test_load_factor_rises_during_handover(self):
        s = Staffing(day={"nurse": 3}, night={"nurse": 3}, start_hour=7)
        self.assertEqual(s.load_factor(12, 0), 1.5)
        self.assertEqual(s.load_factor(12, 900), 1.0)

    def test_no_nurse_on_shift_is_not_adequate(self):
        s = Staffing(day={"nurse": 0}, night={"nurse": 0}, start_hour=8)
        self.assertIsNone(s.patients_per_nurse(12, 0))
        self.assertEqual(s.load_factor(12, 0), 4.0)
        self.assertFalse(s.summary(0, 12)["adequate"])

    def test_staffing_rejects_bad_roster(self):
        with self.assertRaises(ValueError):
            Staffing(day={"nurse": -1})
        with self.assertRaises(ValueError):
            Staffing(day={"surgeon": 1})
        with self.assertRaises(ValueError):
            Staffing(day={"nurse": 1.5})
        with self.assertRaises(ValueError):
            Staffing(start_hour=24)

    def _storm_ward(self, nurses, ticks=400):
        staffing = {
            "day": {"nurse": nurses, "doctor": 1, "tech": 1},
            "night": {"nurse": nurses, "doctor": 1, "tech": 0},
            "start_hour": 8,
        }
        chaos = [ChaosEvent("alarm_storm", 0, 600, beds=list(range(12)))]
        ward = Ward(beds=12, rate_hz=1.0, chaos=chaos, seed=42, staffing=staffing)
        for _ in range(ticks):
            ward.tick()
        return ward

    def test_understaffing_measurably_worsens_time_to_ack(self):
        full = self._storm_ward(3).staffing_summary()
        thin = self._storm_ward(1).staffing_summary()
        # Same seeded vitals, so both wards raised the same alarms.
        self.assertEqual(full["alarms"]["raised"], thin["alarms"]["raised"])
        self.assertGreater(thin["alarms"]["raised"], 0)
        self.assertEqual(full["alarms"]["acked"], full["alarms"]["raised"])
        self.assertEqual(thin["alarms"]["acked"], thin["alarms"]["raised"])
        self.assertEqual(full["alarms"]["mean_time_to_ack_s"], 30.0)
        self.assertEqual(thin["alarms"]["mean_time_to_ack_s"], 90.0)
        self.assertFalse(full["adequate"] is False)
        self.assertFalse(thin["adequate"])

    def test_understaffing_delays_vitals_checks(self):
        full = self._storm_ward(3).staffing_summary()
        thin = self._storm_ward(1).staffing_summary()
        self.assertEqual(full["checks"]["delayed"], 0)
        self.assertGreater(thin["checks"]["delayed"], 0)
        self.assertLess(thin["checks"]["performed"], full["checks"]["performed"])

    def test_snapshot_and_ward_summary_carry_staffing(self):
        ward = self._storm_ward(3, ticks=10)
        snap = ward.tick()
        self.assertIn("staffing", snap)
        self.assertEqual(snap["staffing"]["shift"], "day")
        self.assertEqual(snap["staffing"]["occupied_beds"], 12)
        summary = ward.staffing_summary()
        self.assertEqual(summary["alarms"]["raised"], snap["staffing"]["alarms"]["raised"])

    def test_staffing_endpoint_serves_json_and_is_503_while_partitioned(self):
        ward = Ward(beds=2, seed=1)
        ward.tick()
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/staffing")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["occupied_beds"], 2)
            self.assertIn("mean_time_to_ack_s", data["alarms"])
            ward.partitioned = True
            status, _ = get(port, "/staffing")
            self.assertEqual(status, 503)
        finally:
            server.shutdown()
            server.server_close()

    def test_scenario_staffing(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "staffing.json"))
        self.assertEqual(cfg["staffing"].roster["night"]["nurse"], 1)
        ward = build_ward(cfg, seed=42)
        snap = ward.tick()
        self.assertEqual(snap["staffing"]["on_shift"]["nurse"], 3)

    def test_scenario_rejects_bad_staffing(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"beds": 1, "staffing": {"day": {"nurse": -2}}}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_scenario(path)
        finally:
            os.unlink(path)


class TestTurnover(unittest.TestCase):
    def _ward(self, beds=2, cleaning_s=10.0, **kw):
        ward = Ward(beds=beds, cleaning_s=cleaning_s, seed=42, **kw)
        ward.tick()
        return ward

    def test_discharge_moves_bed_to_cleaning_and_stops_readings(self):
        ward = self._ward()
        ward.discharge(0)
        snap = ward.tick()
        bed = snap["beds"][0]
        self.assertFalse(bed["online"])
        self.assertFalse(bed["admitted"])
        self.assertEqual(bed["state"], "cleaning")
        self.assertTrue(snap["beds"][1]["online"])
        summary = snap["turnover"]
        self.assertEqual(summary["occupied"], 1)
        self.assertEqual(summary["cleaning"], 1)
        self.assertEqual(summary["available"], 0)
        self.assertEqual(summary["beds"][0]["state"], "cleaning")
        self.assertEqual(summary["beds"][0]["cleaning_until"], 11.0)
        self.assertEqual(summary["total_turnovers"], 1)

    def test_discharged_bed_is_not_immediately_readmittable(self):
        ward = self._ward()
        ward.discharge(0)
        with self.assertRaises(ValueError):
            ward.admit(0)
        # Still cleaning after 5 of the 10 turnover seconds.
        for _ in range(5):
            ward.tick()
        with self.assertRaises(ValueError):
            ward.admit(0)

    def test_bed_becomes_available_after_turnover(self):
        ward = self._ward()
        ward.discharge(0)
        for _ in range(10):
            ward.tick()
        # Discharged at t=1, cleaning_s 10: available at t=11.
        snap = ward.tick()
        self.assertEqual(snap["beds"][0]["state"], "available")
        ward.admit(0, profile="febrile")
        snap = ward.tick()
        reading = snap["beds"][0]
        self.assertTrue(reading["online"])
        self.assertEqual(reading["profile"], "febrile")
        self.assertEqual(reading["seq"], 1)

    def test_new_admission_gets_fresh_vitals(self):
        ward = self._ward()
        first = ward.tick()["beds"][0]
        ward.discharge(0)
        for _ in range(11):
            ward.tick()
        ward.admit(0, profile="stable")
        reading = ward.tick()["beds"][0]
        self.assertEqual(reading["seq"], 1)
        self.assertNotEqual(reading["heart_rate"], first["heart_rate"])

    def test_turnover_summary_tracks_counts_and_cleaning_time(self):
        ward = self._ward()
        ward.discharge(0)
        for _ in range(11):
            ward.tick()
        ward.admit(0)
        ward.tick()
        ward.discharge(0)
        summary = ward.turnover_summary()
        self.assertEqual(summary["beds"][0]["turnovers"], 2)
        self.assertEqual(summary["beds"][0]["total_cleaning_s"], 20.0)
        self.assertEqual(summary["total_turnovers"], 2)
        self.assertEqual(summary["total_cleaning_s"], 20.0)

    def test_discharge_of_unoccupied_bed_raises(self):
        ward = self._ward()
        ward.discharge(0)
        with self.assertRaises(ValueError):
            ward.discharge(0)
        with self.assertRaises(ValueError):
            ward.discharge(99)

    def test_admit_of_occupied_bed_raises(self):
        ward = self._ward()
        with self.assertRaises(ValueError):
            ward.admit(0)

    def test_timeline_does_not_readmit_a_cleaning_bed(self):
        timeline = Timeline([TimelinePhase(0, 2, ramp_s=0)])
        ward = self._ward(timeline=timeline)
        ward.discharge(0)
        for _ in range(5):
            ward.tick()
        # Timeline admission time (t=0) has passed, but the bed is
        # still cleaning, so it stays offline.
        snap = ward.tick()
        self.assertFalse(snap["beds"][0]["online"])
        self.assertEqual(snap["beds"][0]["state"], "cleaning")
        for _ in range(5):
            ward.tick()
        # Turnover done at t=11: the timeline refills the bed.
        snap = ward.tick()
        self.assertTrue(snap["beds"][0]["online"])

    def test_discharge_chaos_event_fires_once(self):
        ward = self._ward(
            chaos=[ChaosEvent("discharge", 5, bed=0)],
        )
        for _ in range(3):
            ward.tick()
        self.assertEqual(ward.turnover_summary()["total_turnovers"], 0)
        ward.tick()  # t=5: discharge fires
        self.assertEqual(ward.turnover_summary()["total_turnovers"], 1)
        for _ in range(20):
            ward.tick()
        self.assertEqual(ward.turnover_summary()["total_turnovers"], 1)

    def test_discharge_event_without_duration_is_valid(self):
        ev = ChaosEvent("discharge", 5, bed=0)
        self.assertEqual(ev.duration, 0)
        with self.assertRaises(ValueError):
            ChaosEvent("dropout", 5, bed=0)
        with self.assertRaises(ValueError):
            ChaosEvent("dropout", 5, 0, bed=0)

    def test_scenario_turnover(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "turnover.json"))
        self.assertEqual(cfg["cleaning_s"], 10.0)
        ward = build_ward(cfg, seed=42)
        for _ in range(5):
            ward.tick()
        snap = ward.tick()  # t=6: bed 1 discharged at t=5, cleaning
        self.assertEqual(snap["beds"][1]["state"], "cleaning")
        self.assertTrue(snap["beds"][2]["online"])
        for _ in range(9):
            ward.tick()
        snap = ward.tick()  # t=16: bed 1 cleaned (t=15) and refilled
        self.assertTrue(snap["beds"][1]["online"])
        self.assertEqual(snap["beds"][2]["state"], "cleaning")
        ward.tick()
        snap = ward.tick()  # t=18: bed 2 cleaned and refilled too
        self.assertTrue(snap["beds"][1]["online"])
        self.assertTrue(snap["beds"][2]["online"])

    def test_scenario_rejects_bad_cleaning_s(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"beds": 1, "cleaning_s": 0}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_scenario(path)
        finally:
            os.unlink(path)

    def test_turnover_endpoint_serves_json_and_is_503_while_partitioned(self):
        ward = Ward(beds=2, cleaning_s=10.0, seed=1)
        ward.tick()
        ward.discharge(0)
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/turnover")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["occupied"], 1)
            self.assertEqual(data["cleaning"], 1)
            self.assertEqual(data["beds"][0]["state"], "cleaning")
            ward.partitioned = True
            status, _ = get(port, "/turnover")
            self.assertEqual(status, 503)
        finally:
            server.shutdown()
            server.server_close()


class TestMeds(unittest.TestCase):
    def test_validate_med_rejects_bad_values(self):
        from ward.meds import validate_med

        for bad in (
            {"name": "", "every_s": 3600},
            {"name": "insulin"},
            {"name": "insulin", "every_s": 0},
            {"name": "insulin", "every_s": -60},
            {"name": "insulin", "every_s": 3600, "first_at": -1},
            {"name": "insulin", "every_s": 3600, "grace_s": 0},
            "insulin",
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_med(bad)
        self.assertEqual(
            validate_med({"name": "insulin", "every_s": 3600}),
            {"name": "insulin", "every_s": 3600.0, "first_at": 0.0, "grace_s": 600.0},
        )

    def test_interaction_warnings(self):
        from ward.meds import MedSchedule

        sched = MedSchedule(0, [
            {"name": "warfarin", "every_s": 7200},
            {"name": "Aspirin", "every_s": 7200},
        ])
        self.assertEqual(
            sched.warnings,
            [{"meds": ["aspirin", "warfarin"], "warning": "bleeding risk"}],
        )
        plain = MedSchedule(1, [{"name": "metoprolol", "every_s": 3600}])
        self.assertEqual(plain.warnings, [])

    def test_dose_given_on_time_with_adequate_staffing(self):
        ward = Ward(
            beds=1,
            meds=[[{"name": "metoprolol", "every_s": 3600, "first_at": 0}]],
            seed=42,
        )
        for _ in range(40):
            ward.tick()
        summary = ward.meds_summary()
        self.assertEqual(summary["total_given"], 1)
        self.assertEqual(summary["total_late"], 0)
        self.assertEqual(summary["total_missed"], 0)
        self.assertEqual(summary["beds"][0]["history"][0]["status"], "given")
        self.assertEqual(summary["beds"][0]["history"][0]["delay_s"], 30.0)

    def test_dose_is_late_when_understaffed(self):
        # 64 beds, one nurse: load factor 16, so a dose takes 480 s
        # to administer — past the 300 s late mark but inside the
        # 600 s grace window, so it is given, late.
        ward = Ward(
            beds=64,
            staffing=Staffing(day={"nurse": 1, "doctor": 1, "tech": 0}),
            meds=[[{"name": "metoprolol", "every_s": 3600}]] + [[] for _ in range(63)],
            seed=42,
        )
        for _ in range(500):
            ward.tick()
        summary = ward.meds_summary()
        self.assertEqual(summary["total_given"], 1)
        self.assertEqual(summary["total_late"], 1)
        self.assertEqual(summary["beds"][0]["history"][0]["delay_s"], 480.0)

    def test_dose_missed_when_delay_exceeds_grace(self):
        from ward.meds import MedSchedule

        sched = MedSchedule(0, [
            {"name": "insulin", "every_s": 3600, "first_at": 0, "grace_s": 60},
        ])
        sched.tick(0, 4.0)  # no nurse on shift: 120 s delay, grace 60 s
        self.assertEqual(sched.missed, 0)
        sched.tick(60, 4.0)
        self.assertEqual(sched.missed, 1)
        self.assertEqual(sched.missed_active, {"insulin"})
        self.assertEqual(sched.history[0]["status"], "missed")
        # Next dose, given on time, clears the missed-dose alarm.
        sched.tick(3600, 1.0)
        sched.tick(3630, 1.0)
        self.assertEqual(sched.given, 1)
        self.assertEqual(sched.missed_active, set())

    def test_missed_dose_raises_an_alarm_in_the_reading(self):
        ward = Ward(
            beds=1,
            staffing=Staffing(day={"nurse": 0, "doctor": 1, "tech": 0}),
            meds=[[{"name": "insulin", "every_s": 3600, "first_at": 0, "grace_s": 60}]],
            seed=42,
        )
        for _ in range(59):
            ward.tick()
        snap = ward.tick()  # t=60: grace expires, dose missed
        self.assertIn("missed_dose", snap["beds"][0]["alarms"])
        self.assertEqual(ward.meds_summary()["total_missed"], 1)
        history = ward.monitors[0].alarm_history
        self.assertTrue(any(e["alarm"] == "missed_dose" for e in history))

    def test_snapshot_carries_meds(self):
        ward = Ward(
            beds=1,
            meds=[[{"name": "metoprolol", "every_s": 3600}]],
            seed=42,
        )
        snap = ward.tick()
        self.assertIn("meds", snap)
        self.assertEqual(snap["meds"]["beds"][0]["meds"], ["metoprolol"])

    def test_meds_endpoint_serves_json_and_is_503_while_partitioned(self):
        ward = Ward(
            beds=2,
            meds=[[{"name": "metoprolol", "every_s": 3600}], []],
            seed=1,
        )
        ward.tick()
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/meds")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["beds"][0]["meds"], ["metoprolol"])
            self.assertEqual(data["beds"][1]["meds"], [])
            ward.partitioned = True
            status, _ = get(port, "/meds")
            self.assertEqual(status, 503)
        finally:
            server.shutdown()
            server.server_close()

    def test_discharge_resets_meds(self):
        ward = Ward(
            beds=1,
            cleaning_s=10.0,
            meds=[[{"name": "metoprolol", "every_s": 3600}]],
            seed=42,
        )
        for _ in range(40):
            ward.tick()
        self.assertEqual(ward.meds_summary()["total_given"], 1)
        ward.discharge(0)
        summary = ward.meds_summary()
        self.assertEqual(summary["total_given"], 0)
        self.assertEqual(summary["beds"][0]["history"], [])

    def test_scenario_meds(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "meds.json"))
        ward = build_ward(cfg, seed=42)
        snap = ward.tick()
        self.assertEqual(
            snap["meds"]["beds"][0]["warnings"],
            [{"meds": ["aspirin", "warfarin"], "warning": "bleeding risk"}],
        )
        self.assertEqual(
            snap["meds"]["beds"][3]["warnings"],
            [{"meds": ["lisinopril", "spironolactone"], "warning": "hyperkalaemia risk"}],
        )
        # Day shift (18:00-19:00) is adequately staffed: the 0 s doses
        # are given. After 19:00 one nurse covers 12 beds, so the
        # time-critical insulin (60 s grace) is missed at night while
        # the 600 s grace doses are still given, late or on time.
        for _ in range(3699):
            ward.tick()
        summary = ward.meds_summary()
        self.assertGreater(summary["total_given"], 0)
        self.assertEqual(summary["total_missed"], 1)
        self.assertEqual(summary["beds"][1]["doses"]["missed"], 1)

    def test_stockout_event_requires_a_medication(self):
        from ward.chaos import ChaosEvent

        for bad in (None, "", "   ", 42):
            with self.assertRaises(ValueError, msg=repr(bad)):
                ChaosEvent("stockout", 5, 60, med=bad)
        ev = ChaosEvent("stockout", 5, 60, med="Insulin")
        self.assertEqual(ev.med, "insulin")
        with self.assertRaises(ValueError):
            ChaosEvent("stockout", 5, 0, med="insulin")

    def test_dose_skipped_during_stockout(self):
        from ward.meds import MedSchedule

        sched = MedSchedule(0, [
            {"name": "insulin", "every_s": 3600, "first_at": 0},
        ])
        sched.tick(0, 1.0, {"insulin"})
        self.assertEqual(sched.stockout, 1)
        self.assertEqual(sched.given, 0)
        self.assertEqual(sched.history[0]["status"], "stockout")
        self.assertIsNone(sched.history[0]["admin_t"])
        self.assertEqual(sched.missed_active, {"insulin"})
        self.assertEqual(sched.stockout_active, {"insulin"})
        # Next dose, after the stockout ends, is given and clears the flag.
        sched.tick(3600, 1.0)
        self.assertEqual(sched.stockout_active, set())
        sched.tick(3630, 1.0)
        self.assertEqual(sched.given, 1)
        self.assertEqual(sched.missed_active, set())
        self.assertEqual(sched.history[1]["status"], "given")

    def test_stockout_raises_missed_dose_alarm_and_counts(self):
        from ward.chaos import ChaosEvent

        ward = Ward(
            beds=1,
            meds=[[{"name": "insulin", "every_s": 3600, "first_at": 0}]],
            chaos=[ChaosEvent("stockout", 0, 60, bed=0, med="insulin")],
            seed=42,
        )
        snap = ward.tick()  # t=1: insulin due at 0, pharmacy is out
        self.assertIn("missed_dose", snap["beds"][0]["alarms"])
        self.assertEqual(snap["meds"]["beds"][0]["stockouts"], ["insulin"])
        self.assertEqual(ward.meds_summary()["total_stockout"], 1)
        history = ward.monitors[0].alarm_history
        self.assertTrue(any(e["alarm"] == "missed_dose" for e in history))
        for _ in range(60):
            ward.tick()  # stockout over at t=60
        self.assertEqual(
            ward.meds_summary()["beds"][0]["stockouts"], []
        )

    def test_stockout_only_affects_that_medication(self):
        from ward.chaos import ChaosEvent

        ward = Ward(
            beds=2,
            meds=[
                [{"name": "metoprolol", "every_s": 3600, "first_at": 0}],
                [{"name": "insulin", "every_s": 3600, "first_at": 0}],
            ],
            chaos=[ChaosEvent("stockout", 0, 3600, med="insulin")],
            seed=42,
        )
        for _ in range(40):
            ward.tick()
        summary = ward.meds_summary()
        self.assertEqual(summary["beds"][0]["doses"]["given"], 1)
        self.assertEqual(summary["beds"][0]["doses"]["stockout"], 0)
        self.assertEqual(summary["beds"][1]["doses"]["given"], 0)
        self.assertEqual(summary["beds"][1]["doses"]["stockout"], 1)
        self.assertEqual(summary["total_stockout"], 1)

    def test_scenario_stockout(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "stockout.json"))
        ward = build_ward(cfg, seed=42)
        for _ in range(3700):
            ward.tick()
        summary = ward.meds_summary()
        # Insulin due at 0 is given; due at 1200 and 2400 fall inside
        # the 5-3005 s stockout and are skipped; due at 3600 is given.
        self.assertEqual(summary["beds"][1]["doses"]["stockout"], 2)
        self.assertEqual(summary["beds"][1]["doses"]["given"], 2)
        self.assertEqual(summary["total_stockout"], 2)
        self.assertEqual(summary["total_given"], 4)
        self.assertEqual(
            [h["status"] for h in summary["beds"][1]["history"]],
            ["given", "stockout", "stockout", "given"],
        )

    def test_scenario_rejects_bad_meds(self):
        for patients in (
            [{"meds": [{"name": "insulin", "every_s": 0}]}],
            [{"meds": [{"name": "insulin", "every_s": 3600},
                       {"name": "Insulin", "every_s": 1800}]}],
        ):
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump({"beds": 1, "patients": patients}, f)
                path = f.name
            try:
                with self.assertRaises(ValueError):
                    load_scenario(path)
            finally:
                os.unlink(path)
class TestTriage(unittest.TestCase):
    def _ward(self, beds=2, cleaning_s=10.0, **kw):
        ward = Ward(beds=beds, cleaning_s=cleaning_s, seed=42, **kw)
        ward.tick()
        return ward

    def test_arrival_waits_while_ward_is_full(self):
        ward = self._ward()
        self.assertTrue(ward.triage_arrive(2, "febrile"))
        snap = ward.tick()
        self.assertEqual(snap["triage"]["waiting_total"], 1)
        self.assertEqual(snap["triage"]["waiting"][0]["acuity"], 2)
        self.assertEqual(snap["triage"]["waiting"][0]["profile"], "febrile")
        self.assertEqual(snap["triage"]["admitted"], 0)
        self.assertEqual(snap["triage"]["diverted"], 0)

    def test_patient_admitted_when_bed_frees_up(self):
        ward = self._ward()
        ward.triage_arrive(2, "febrile")  # arrives at t=1
        ward.discharge(0)
        for _ in range(9):
            ward.tick()
        # Discharged at t=1, cleaning 10 s: bed 0 free at t=11.
        snap = ward.tick()
        self.assertTrue(snap["beds"][0]["online"])
        self.assertEqual(snap["beds"][0]["profile"], "febrile")
        self.assertEqual(snap["beds"][0]["acuity"], 2)
        self.assertEqual(snap["beds"][0]["seq"], 1)
        self.assertEqual(snap["triage"]["waiting_total"], 0)
        self.assertEqual(snap["triage"]["admitted"], 1)
        self.assertEqual(snap["triage"]["mean_wait_s"], 10.0)
        self.assertEqual(snap["triage"]["max_wait_s"], 10.0)

    def test_critical_patient_jumps_the_queue(self):
        ward = self._ward()
        ward.triage_arrive(4, "stable")  # t=1
        ward.tick()  # t=2
        ward.triage_arrive(3, "febrile")  # t=2
        ward.tick()  # t=3
        ward.triage_arrive(1, "hypoxic")  # t=3, critical, arrives last
        ward.discharge(0)
        for _ in range(10):
            ward.tick()
        snap = ward.tick()  # t=14: bed 0 free (discharged t=3, +10)
        self.assertEqual(snap["beds"][0]["acuity"], 1)
        self.assertEqual(snap["beds"][0]["profile"], "hypoxic")
        # The critical patient waited 10 s; the others still wait,
        # acuity 3 ahead of acuity 4.
        waiting = snap["triage"]["waiting"]
        self.assertEqual([w["acuity"] for w in waiting], [3, 4])
        self.assertEqual(snap["triage"]["mean_wait_s"], 10.0)

    def test_fifo_within_one_acuity(self):
        ward = self._ward()
        ward.triage_arrive(2, "stable")  # t=1
        ward.tick()  # t=2
        ward.triage_arrive(2, "febrile")  # t=2
        ward.discharge(0)
        ward.discharge(1)
        for _ in range(10):
            ward.tick()
        snap = ward.tick()  # t=13: both beds free (discharged t=2, +10)
        self.assertEqual(snap["beds"][0]["profile"], "stable")
        self.assertEqual(snap["beds"][1]["profile"], "febrile")
        self.assertEqual(snap["triage"]["admitted"], 2)
        # Waits 11 s and 10 s: admitted at t=12, arrived at t=1 and t=2.
        self.assertEqual(snap["triage"]["mean_wait_s"], 10.5)
        self.assertEqual(snap["triage"]["max_wait_s"], 11.0)

    def test_queue_full_diverts_arrivals(self):
        ward = self._ward(triage={"capacity": 2})
        self.assertTrue(ward.triage_arrive(2))
        self.assertTrue(ward.triage_arrive(3))
        self.assertFalse(ward.triage_arrive(1))  # diverted
        self.assertFalse(ward.triage_arrive(4))  # diverted
        summary = ward.triage_summary()
        self.assertEqual(summary["waiting_total"], 2)
        self.assertEqual(summary["diverted"], 2)
        self.assertEqual(summary["waiting_by_acuity"], {"1": 0, "2": 1, "3": 1, "4": 0})
        # A freed bed admits one patient, making room for the next arrival.
        ward.discharge(0)
        for _ in range(11):
            ward.tick()
        self.assertTrue(ward.triage_arrive(1))
        self.assertEqual(ward.triage_summary()["waiting_total"], 2)

    def test_wait_metrics_track_longest_current_wait(self):
        ward = self._ward()
        ward.triage_arrive(3)  # t=1
        ward.tick()  # t=2
        ward.tick()  # t=3
        summary = ward.triage_summary()
        self.assertEqual(summary["longest_wait_s"], 2.0)
        self.assertEqual(summary["mean_wait_s"], 0.0)
        self.assertEqual(summary["max_wait_s"], 0.0)

    def test_wait_alarm_raised_at_threshold(self):
        ward = self._ward(triage={"wait_thresholds": {"1": 10}})
        ward.triage_arrive(1, "hypoxic")  # t=1
        for _ in range(9):
            ward.tick()
        # t=10: waited 9 s, under the 10 s threshold, no alarm yet.
        summary = ward.triage_summary()
        self.assertEqual(summary["wait_alarms_active"], 0)
        self.assertEqual(summary["wait_alarms_raised"], 0)
        self.assertFalse(summary["waiting"][0]["wait_alarm"])
        snap = ward.tick()  # t=11: waited 10 s, alarm raises at t=11
        self.assertEqual(snap["triage"]["wait_alarms_active"], 1)
        self.assertEqual(snap["triage"]["wait_alarms_raised"], 1)
        waiting = snap["triage"]["waiting"][0]
        self.assertTrue(waiting["wait_alarm"])
        self.assertEqual(waiting["wait_alarm_t"], 11.0)
        self.assertEqual(waiting["wait_threshold_s"], 10.0)
        history = snap["triage"]["wait_alarm_history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["acuity"], 1)
        self.assertEqual(history[0]["raised_t"], 11.0)
        self.assertIsNone(history[0]["cleared_t"])
        # The alarm raises once: a later tick does not re-raise it.
        ward.tick()  # t=12
        summary = ward.triage_summary()
        self.assertEqual(summary["wait_alarms_raised"], 1)
        self.assertEqual(len(summary["wait_alarm_history"]), 1)

    def test_wait_alarm_clears_on_admission(self):
        ward = self._ward(triage={"wait_thresholds": {"1": 5}})
        ward.triage_arrive(1, "hypoxic")  # t=1
        for _ in range(5):
            ward.tick()
        # t=6: waited 5 s, alarm raised at t=6.
        summary = ward.triage_summary()
        self.assertEqual(summary["wait_alarms_active"], 1)
        ward.discharge(0)  # t=6, cleaning 10 s: bed 0 free at t=16
        for _ in range(10):
            ward.tick()
        snap = ward.tick()  # t=17: patient admitted at t=16
        self.assertEqual(snap["triage"]["admitted"], 1)
        self.assertEqual(snap["triage"]["wait_alarms_active"], 0)
        self.assertEqual(snap["triage"]["wait_alarms_raised"], 1)
        history = snap["triage"]["wait_alarm_history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["raised_t"], 6.0)
        self.assertEqual(history[0]["cleared_t"], 16.0)

    def test_wait_thresholds_differ_by_acuity(self):
        # Defaults: acuity 1 gets 300 s, acuity 4 gets 3600 s.
        ward = self._ward()
        ward.triage_arrive(1, "hypoxic")  # t=1
        ward.triage_arrive(4, "stable")  # t=1
        for _ in range(300):
            ward.tick()
        # t=301: the critical patient has waited 300 s (alarm), the
        # low-acuity patient has waited 300 s of 3600 s (no alarm).
        summary = ward.triage_summary()
        self.assertEqual(
            summary["wait_thresholds_s"],
            {"1": 300.0, "2": 900.0, "3": 1800.0, "4": 3600.0},
        )
        self.assertEqual(summary["wait_alarms_active"], 1)
        waiting = {w["acuity"]: w for w in summary["waiting"]}
        self.assertTrue(waiting[1]["wait_alarm"])
        self.assertFalse(waiting[4]["wait_alarm"])

    def test_wait_threshold_validation(self):
        from ward.triage import TriageQueue

        with self.assertRaises(ValueError):
            TriageQueue(wait_thresholds={"5": 10})
        with self.assertRaises(ValueError):
            TriageQueue(wait_thresholds={"1": 0})
        with self.assertRaises(ValueError):
            TriageQueue(wait_thresholds={"1": -5})
        with self.assertRaises(ValueError):
            TriageQueue(wait_thresholds={"1": "soon"})
        with self.assertRaises(ValueError):
            TriageQueue(wait_thresholds="soon")
        # A partial override keeps the defaults for the rest.
        queue = TriageQueue(wait_thresholds={"1": 60, 4: 120})
        self.assertEqual(queue.wait_thresholds[1], 60.0)
        self.assertEqual(queue.wait_thresholds[2], 900.0)
        self.assertEqual(queue.wait_thresholds[4], 120.0)

    def test_triage_does_not_take_a_cleaning_bed(self):
        ward = self._ward()
        ward.triage_arrive(1, "hypoxic")
        ward.discharge(0)
        for _ in range(5):
            ward.tick()
        # Still cleaning at t=7: patient waits, bed stays offline.
        snap = ward.tick()
        self.assertFalse(snap["beds"][0]["online"])
        self.assertEqual(snap["beds"][0]["state"], "cleaning")
        self.assertEqual(snap["triage"]["waiting_total"], 1)
        self.assertEqual(snap["triage"]["admitted"], 0)

    def test_triage_does_not_steal_timeline_beds(self):
        timeline = Timeline([TimelinePhase(0, 1, ramp_s=0)])
        ward = self._ward(beds=2, timeline=timeline)
        # Only bed 0 is admitted by the timeline; bed 1 is unadmitted
        # and "available", but it belongs to the timeline, not triage.
        ward.triage_arrive(1, "hypoxic")
        for _ in range(5):
            ward.tick()
        snap = ward.tick()
        self.assertFalse(snap["beds"][1]["online"])
        self.assertEqual(snap["triage"]["waiting_total"], 1)
        self.assertEqual(snap["triage"]["admitted"], 0)

    def test_scheduled_arrivals_and_priority_admission(self):
        ward = self._ward(
            beds=2,
            triage={
                "capacity": 4,
                "arrivals": [
                    {"at": 0, "acuity": 3, "profile": "stable"},
                    {"at": 1, "acuity": 1, "profile": "hypoxic"},
                ],
            },
        )
        ward.discharge(0)
        for _ in range(10):
            ward.tick()
        snap = ward.tick()  # t=12: bed 0 free, critical patient admitted
        self.assertEqual(snap["beds"][0]["acuity"], 1)
        self.assertEqual(snap["triage"]["admitted"], 1)
        self.assertEqual(snap["triage"]["waiting_total"], 1)
        self.assertEqual(snap["triage"]["waiting"][0]["acuity"], 3)

    def test_triage_validation(self):
        from ward.triage import TriageQueue

        with self.assertRaises(ValueError):
            TriageQueue(capacity=0)
        with self.assertRaises(ValueError):
            TriageQueue(arrivals=[{"at": -1, "acuity": 2}])
        with self.assertRaises(ValueError):
            TriageQueue(arrivals=[{"at": 0, "acuity": 5}])
        with self.assertRaises(ValueError):
            TriageQueue(arrivals=[{"at": 0, "acuity": 2, "profile": "nope"}])
        with self.assertRaises(ValueError):
            TriageQueue(arrivals=[{"acuity": 2}])
        ward = self._ward()
        with self.assertRaises(ValueError):
            ward.triage_arrive(0)
        with self.assertRaises(ValueError):
            ward.triage_arrive(2, "nope")
        with self.assertRaises(ValueError):
            Ward(beds=2, triage="nope")

    def test_triage_endpoint_serves_json_and_is_503_while_partitioned(self):
        ward = Ward(beds=2, cleaning_s=10.0, seed=1)
        ward.tick()
        ward.triage_arrive(2, "febrile")
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/triage")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["waiting_total"], 1)
            self.assertEqual(data["capacity"], 8)
            self.assertEqual(data["waiting"][0]["acuity"], 2)
            ward.partitioned = True
            status, _ = get(port, "/triage")
            self.assertEqual(status, 503)
        finally:
            server.shutdown()
            server.server_close()

    def test_triage_in_snapshot(self):
        ward = Ward(beds=2, cleaning_s=10.0, seed=1)
        ward.tick()
        ward.triage_arrive(1, "hypoxic")
        snap = ward.tick()
        self.assertIn("triage", snap)
        self.assertEqual(snap["triage"]["waiting_total"], 1)

    def test_scenario_triage(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "triage.json"))
        self.assertEqual(cfg["triage"].capacity, 5)
        ward = build_ward(cfg, seed=42)
        for _ in range(15):
            ward.tick()
        snap = ward.tick()  # t=16: bed 1 freed at t=15, critical patient in
        self.assertTrue(snap["beds"][1]["online"])
        self.assertEqual(snap["beds"][1]["acuity"], 1)
        self.assertEqual(snap["beds"][1]["profile"], "hypoxic")
        self.assertEqual(snap["triage"]["admitted"], 1)
        self.assertEqual(snap["triage"]["mean_wait_s"], 15.0)
        self.assertEqual(snap["triage"]["waiting_total"], 4)
        for _ in range(2):
            ward.tick()
        snap = ward.tick()  # t=19: bed 2 freed at t=18, second critical in
        self.assertTrue(snap["beds"][2]["online"])
        self.assertEqual(snap["beds"][2]["acuity"], 1)
        self.assertEqual(snap["triage"]["admitted"], 2)
        self.assertEqual(snap["triage"]["mean_wait_s"], 12.0)
        self.assertEqual(snap["triage"]["max_wait_s"], 15.0)
        self.assertEqual(snap["triage"]["waiting_total"], 3)

    def test_scenario_wait_alarm(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "wait-alarm.json"))
        self.assertEqual(cfg["triage"].wait_thresholds[1], 5.0)
        self.assertEqual(cfg["triage"].wait_thresholds[4], 20.0)
        ward = build_ward(cfg, seed=42)
        for _ in range(5):
            ward.tick()
        # t=5: the critical patient (arrived t=0) has waited 5 s and
        # raised a wait-time alarm; the acuity-4 patient (arrived
        # t=2) has waited 3 s of 20 s, no alarm yet.
        summary = ward.triage_summary()
        self.assertEqual(summary["wait_alarms_active"], 1)
        self.assertEqual(summary["wait_alarms_raised"], 1)
        self.assertEqual(summary["wait_alarm_history"][0]["raised_t"], 5.0)
        self.assertEqual(summary["wait_alarm_history"][0]["acuity"], 1)
        for _ in range(17):
            ward.tick()
        # t=22: the acuity-4 patient has waited 20 s and raised one too.
        summary = ward.triage_summary()
        self.assertEqual(summary["wait_alarms_active"], 2)
        self.assertEqual(summary["wait_alarms_raised"], 2)
        self.assertEqual(summary["wait_alarm_history"][1]["raised_t"], 22.0)
        self.assertEqual(summary["wait_alarm_history"][1]["acuity"], 4)

    def test_scenario_rejects_bad_triage(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"beds": 1, "triage": {"capacity": 0}}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_scenario(path)
        finally:
            os.unlink(path)


class TestBoarding(unittest.TestCase):
    def _ward(self, beds=2, cleaning_s=10.0, **kw):
        ward = Ward(beds=beds, cleaning_s=cleaning_s, seed=42, **kw)
        ward.tick()
        return ward

    def _full_ward(self):
        # Queue of 1, boarding of 1, both occupied: one patient
        # waiting (t=1), one boarding (t=2).
        ward = self._ward(triage={"capacity": 1, "boarding_capacity": 1})
        self.assertTrue(ward.triage_arrive(3, "stable"))  # t=1, queue
        ward.tick()  # t=2
        self.assertTrue(ward.triage_arrive(3, "febrile"))  # t=2, boards
        return ward

    def test_boarding_takes_overflow_when_queue_full(self):
        ward = self._full_ward()
        summary = ward.triage_summary()
        self.assertEqual(summary["waiting_total"], 1)
        self.assertEqual(summary["boarding_total"], 1)
        self.assertEqual(summary["boarded"], 1)
        self.assertEqual(summary["boarding_capacity"], 1)
        self.assertEqual(summary["diverted"], 0)
        self.assertEqual(summary["boarding"][0]["profile"], "febrile")
        self.assertEqual(summary["boarding"][0]["arrival_t"], 2.0)
        self.assertEqual(summary["longest_boarding_wait_s"], 0.0)

    def test_boarding_full_diverts_arrivals(self):
        ward = self._full_ward()
        self.assertFalse(ward.triage_arrive(1, "hypoxic"))  # diverted
        self.assertFalse(ward.triage_arrive(4))  # diverted
        summary = ward.triage_summary()
        self.assertEqual(summary["waiting_total"], 1)
        self.assertEqual(summary["boarding_total"], 1)
        self.assertEqual(summary["diverted"], 2)

    def test_boarding_off_by_default(self):
        ward = self._ward(triage={"capacity": 1})
        self.assertEqual(ward.triage.boarding_capacity, 0)
        self.assertTrue(ward.triage_arrive(3))
        self.assertFalse(ward.triage_arrive(3))  # diverted, no boarding
        self.assertEqual(ward.triage_summary()["boarding_total"], 0)
        self.assertEqual(ward.triage_summary()["diverted"], 1)

    def test_boarded_patient_admitted_before_queued_patient(self):
        ward = self._full_ward()
        ward.discharge(0)  # t=2, bed 0 free at t=12
        for _ in range(10):
            ward.tick()
        snap = ward.tick()  # t=12: the boarded patient takes bed 0,
        # even though the queued patient arrived first.
        self.assertEqual(snap["beds"][0]["profile"], "febrile")
        self.assertEqual(snap["triage"]["boarding_total"], 0)
        self.assertEqual(snap["triage"]["waiting_total"], 1)
        self.assertEqual(snap["triage"]["admitted"], 1)
        self.assertEqual(snap["triage"]["mean_boarding_wait_s"], 10.0)
        self.assertEqual(snap["triage"]["max_boarding_wait_s"], 10.0)
        self.assertEqual(snap["triage"]["mean_wait_s"], 0.0)
        ward.discharge(1)  # t=13, bed 1 free at t=23
        for _ in range(10):
            ward.tick()
        snap = ward.tick()  # t=24: the queued patient takes bed 1.
        self.assertEqual(snap["beds"][1]["profile"], "stable")
        self.assertEqual(snap["triage"]["waiting_total"], 0)
        self.assertEqual(snap["triage"]["admitted"], 2)
        self.assertEqual(snap["triage"]["mean_wait_s"], 22.0)

    def test_boarded_patient_raises_wait_alarm(self):
        ward = self._ward(
            triage={
                "capacity": 1,
                "boarding_capacity": 1,
                "wait_thresholds": {"3": 5},
            }
        )
        ward.triage_arrive(3, "stable")  # t=1, queue
        ward.tick()  # t=2
        ward.triage_arrive(3, "febrile")  # t=2, boards
        for _ in range(4):
            ward.tick()
        # t=6: the queued patient (arrived t=1) has waited 5 s and
        # raised an alarm; the boarded one (arrived t=2) has not.
        summary = ward.triage_summary()
        self.assertEqual(summary["wait_alarms_active"], 1)
        self.assertFalse(summary["boarding"][0]["wait_alarm"])
        snap = ward.tick()  # t=7: the boarded patient waited 5 s too.
        self.assertEqual(snap["triage"]["wait_alarms_active"], 2)
        self.assertEqual(snap["triage"]["wait_alarms_raised"], 2)
        self.assertTrue(snap["triage"]["boarding"][0]["wait_alarm"])
        self.assertEqual(snap["triage"]["boarding"][0]["wait_alarm_t"], 7.0)
        history = snap["triage"]["wait_alarm_history"]
        self.assertEqual(len(history), 2)
        self.assertFalse(history[0]["boarding"])
        self.assertTrue(history[1]["boarding"])
        self.assertEqual(history[1]["raised_t"], 7.0)
        # The alarm raises once: a later tick does not re-raise it.
        ward.tick()  # t=8
        self.assertEqual(ward.triage_summary()["wait_alarms_raised"], 2)

    def test_boarded_wait_alarm_clears_on_admission(self):
        ward = self._ward(
            triage={
                "capacity": 1,
                "boarding_capacity": 1,
                "wait_thresholds": {"3": 5},
            }
        )
        ward.triage_arrive(3, "stable")  # t=1, queue
        ward.tick()  # t=2
        ward.triage_arrive(3, "febrile")  # t=2, boards
        for _ in range(5):
            ward.tick()
        # t=7: the boarded patient's alarm has raised.
        self.assertEqual(ward.triage_summary()["wait_alarms_active"], 2)
        ward.discharge(0)  # t=7, bed 0 free at t=17
        for _ in range(10):
            ward.tick()
        snap = ward.tick()  # t=17: boarded patient admitted, alarm clears.
        self.assertEqual(snap["triage"]["wait_alarms_active"], 1)
        history = snap["triage"]["wait_alarm_history"]
        boarded = [e for e in history if e["boarding"]]
        self.assertEqual(len(boarded), 1)
        self.assertEqual(boarded[0]["cleared_t"], 17.0)

    def test_boarding_wait_metrics(self):
        ward = self._full_ward()
        ward.tick()  # t=3
        ward.tick()  # t=4
        summary = ward.triage_summary()
        self.assertEqual(summary["longest_boarding_wait_s"], 2.0)
        self.assertEqual(summary["mean_boarding_wait_s"], 0.0)
        self.assertEqual(summary["max_boarding_wait_s"], 0.0)

    def test_boarding_capacity_validation(self):
        from ward.triage import TriageQueue

        self.assertEqual(TriageQueue().boarding_capacity, 0)
        for bad in (-1, 1.5, "1", True):
            with self.assertRaises(ValueError):
                TriageQueue(boarding_capacity=bad)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {"beds": 1, "triage": {"boarding_capacity": -1}}, f
            )
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_scenario(path)
        finally:
            os.unlink(path)

    def test_arrivals_view_carries_boarding_and_diverting(self):
        ward = self._ward(triage={"capacity": 1, "boarding_capacity": 1})
        view = ward.arrivals_summary()
        self.assertEqual(view["boarding_total"], 0)
        self.assertEqual(view["boarding_capacity"], 1)
        self.assertEqual(view["boarded"], 0)
        self.assertFalse(view["diverting"])
        ward.triage_arrive(3)  # t=1, queue full, boarding still open
        view = ward.arrivals_summary()
        self.assertFalse(view["diverting"])
        ward.tick()  # t=2
        ward.triage_arrive(3)  # t=2, boards: queue and boarding full now
        view = ward.arrivals_summary()
        self.assertTrue(view["diverting"])
        self.assertEqual(view["boarding_total"], 1)
        self.assertEqual(view["boarded"], 1)
        self.assertEqual(view["diverted"], 0)
        ward.triage_arrive(1)  # t=2, diverted
        view = ward.arrivals_summary()
        self.assertEqual(view["diverted"], 1)
        self.assertEqual(view["diverted_last_hour"], 1)

    def test_boarding_in_snapshot_and_arrivals_endpoint(self):
        ward = self._full_ward()
        snap = ward.tick()  # t=3
        self.assertEqual(snap["triage"]["boarding_total"], 1)
        self.assertEqual(snap["triage"]["boarded"], 1)
        self.assertEqual(snap["arrivals"]["boarding_total"], 1)
        self.assertEqual(snap["arrivals"]["boarded"], 1)
        self.assertTrue(snap["arrivals"]["diverting"])
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/arrivals")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["boarding_total"], 1)
            self.assertEqual(data["boarding_capacity"], 1)
        finally:
            server.shutdown()
            server.server_close()

    def test_scenario_boarding(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "boarding.json"))
        self.assertEqual(cfg["triage"].capacity, 1)
        self.assertEqual(cfg["triage"].boarding_capacity, 1)
        ward = build_ward(cfg, seed=42)
        ward.tick()  # t=1: first arrival queues, second boards
        ward.tick()  # t=2: third arrival (critical) diverted
        summary = ward.triage_summary()
        self.assertEqual(summary["waiting_total"], 1)
        self.assertEqual(summary["boarding_total"], 1)
        self.assertEqual(summary["boarded"], 1)
        self.assertEqual(summary["diverted"], 1)
        for _ in range(13):
            ward.tick()
        snap = ward.tick()  # t=16: bed 0 freed at t=15, boarded patient in
        self.assertEqual(snap["beds"][0]["profile"], "febrile")
        self.assertEqual(snap["triage"]["admitted"], 1)
        self.assertEqual(snap["triage"]["mean_boarding_wait_s"], 14.0)
        for _ in range(2):
            ward.tick()
        snap = ward.tick()  # t=19: bed 1 freed at t=18, queued patient in
        self.assertEqual(snap["beds"][1]["profile"], "stable")
        self.assertEqual(snap["triage"]["admitted"], 2)
        self.assertEqual(snap["triage"]["mean_wait_s"], 18.0)
        self.assertEqual(snap["triage"]["waiting_total"], 0)
        self.assertEqual(snap["triage"]["boarding_total"], 0)


class TestBoardingFullAlarm(unittest.TestCase):
    def _ward(self, beds=2, cleaning_s=10.0, **kw):
        ward = Ward(beds=beds, cleaning_s=cleaning_s, seed=42, **kw)
        ward.tick()
        return ward

    def _full_ward(self):
        # Queue of 1, boarding of 1, both occupied: one patient
        # waiting (t=1), one boarding (t=2).
        ward = self._ward(triage={"capacity": 1, "boarding_capacity": 1})
        self.assertTrue(ward.triage_arrive(3, "stable"))  # t=1, queue
        ward.tick()  # t=2
        self.assertTrue(ward.triage_arrive(3, "febrile"))  # t=2, boards
        return ward

    def test_alarm_raises_when_boarding_full_and_arrival_diverts(self):
        ward = self._full_ward()
        summary = ward.triage_summary()
        self.assertEqual(summary["boarding_full_alarms_raised"], 0)
        self.assertFalse(summary["boarding_full_alarm_active"])
        self.assertEqual(summary["boarding_full_alarm_history"], [])
        self.assertFalse(ward.triage_arrive(1, "hypoxic"))  # t=2, diverted
        summary = ward.triage_summary()
        self.assertEqual(summary["boarding_full_alarms_raised"], 1)
        self.assertTrue(summary["boarding_full_alarm_active"])
        self.assertEqual(summary["boarding_full_diverted"], 1)
        self.assertEqual(
            summary["boarding_full_alarm_history"],
            [
                {
                    "raised_t": 2.0,
                    "cleared_t": None,
                    "diverted": 1,
                    "hook_retries": 0,
                    "hook_final_status": "not_invoked",
                    "hook_invoked": False,
                    "hook_errors": [],
                }
            ],
        )

    def test_further_diversions_count_against_active_episode(self):
        ward = self._full_ward()
        ward.triage_arrive(1, "hypoxic")  # t=2, diverted: raises
        ward.triage_arrive(4)  # t=2, diverted: same episode
        ward.triage_arrive(2)  # t=2, diverted: same episode
        summary = ward.triage_summary()
        self.assertEqual(summary["boarding_full_alarms_raised"], 1)
        self.assertTrue(summary["boarding_full_alarm_active"])
        self.assertEqual(summary["boarding_full_diverted"], 3)
        self.assertEqual(
            summary["boarding_full_alarm_history"],
            [
                {
                    "raised_t": 2.0,
                    "cleared_t": None,
                    "diverted": 3,
                    "hook_retries": 0,
                    "hook_final_status": "not_invoked",
                    "hook_invoked": False,
                    "hook_errors": [],
                }
            ],
        )
        self.assertEqual(summary["diverted"], 3)

    def test_alarm_clears_when_boarded_patient_admitted(self):
        ward = self._full_ward()
        ward.triage_arrive(1, "hypoxic")  # t=2, diverted: raises
        ward.discharge(0)  # t=2, bed 0 free at t=12 (cleaning is 10 s)
        for _ in range(9):
            ward.tick()
        # t=11: the bed is still cleaning, the alarm is still active.
        self.assertTrue(ward.triage_summary()["boarding_full_alarm_active"])
        snap = ward.tick()  # t=12: bed 0 free, boarded patient admitted.
        self.assertEqual(snap["beds"][0]["profile"], "febrile")
        summary = snap["triage"]
        self.assertEqual(summary["boarding_full_alarms_raised"], 1)
        self.assertFalse(summary["boarding_full_alarm_active"])
        self.assertEqual(summary["boarding_full_diverted"], 1)
        self.assertEqual(
            summary["boarding_full_alarm_history"],
            [
                {
                    "raised_t": 2.0,
                    "cleared_t": 12.0,
                    "diverted": 1,
                    "hook_retries": 0,
                    "hook_final_status": "not_invoked",
                    "hook_invoked": False,
                    "hook_errors": [],
                }
            ],
        )

    def test_second_episode_raises_after_boarding_refills(self):
        ward = self._full_ward()
        ward.triage_arrive(1, "hypoxic")  # t=2, diverted: first episode
        ward.discharge(0)  # t=2, bed 0 free at t=12 (cleaning is 10 s)
        for _ in range(10):
            ward.tick()
        # t=12: boarded patient admitted, first episode cleared.
        summary = ward.triage_summary()
        self.assertFalse(summary["boarding_full_alarm_active"])
        self.assertEqual(summary["boarding_full_alarms_raised"], 1)
        # The queue is still full (the stable patient waits), so the
        # next arrival boards, and the one after is diverted: a
        # second episode raises at t=12.
        self.assertTrue(ward.triage_arrive(3))  # boards
        self.assertFalse(ward.triage_arrive(4))  # diverted
        summary = ward.triage_summary()
        self.assertEqual(summary["boarding_full_alarms_raised"], 2)
        self.assertTrue(summary["boarding_full_alarm_active"])
        self.assertEqual(summary["boarding_full_diverted"], 2)
        self.assertEqual(
            summary["boarding_full_alarm_history"],
            [
                {
                    "raised_t": 2.0,
                    "cleared_t": 12.0,
                    "diverted": 1,
                    "hook_retries": 0,
                    "hook_final_status": "not_invoked",
                    "hook_invoked": False,
                    "hook_errors": [],
                },
                {
                    "raised_t": 12.0,
                    "cleared_t": None,
                    "diverted": 1,
                    "hook_retries": 0,
                    "hook_final_status": "not_invoked",
                    "hook_invoked": False,
                    "hook_errors": [],
                },
            ],
        )

    def test_no_alarm_when_boarding_off(self):
        ward = self._ward(triage={"capacity": 1})
        self.assertTrue(ward.triage_arrive(3))  # t=1, queue
        self.assertFalse(ward.triage_arrive(3))  # t=1, diverted, no boarding
        self.assertFalse(ward.triage_arrive(1))  # t=1, diverted again
        summary = ward.triage_summary()
        self.assertEqual(summary["diverted"], 2)
        self.assertEqual(summary["boarding_full_alarms_raised"], 0)
        self.assertFalse(summary["boarding_full_alarm_active"])
        self.assertEqual(summary["boarding_full_alarm_history"], [])
        self.assertEqual(summary["boarding_full_diverted"], 0)

    def test_arrivals_view_carries_boarding_full_alarm(self):
        ward = self._full_ward()
        view = ward.arrivals_summary()
        self.assertFalse(view["boarding_full_alarm"])
        self.assertEqual(view["boarding_full_alarms_raised"], 0)
        ward.triage_arrive(1)  # t=2, diverted: raises
        view = ward.arrivals_summary()
        self.assertTrue(view["boarding_full_alarm"])
        self.assertEqual(view["boarding_full_alarms_raised"], 1)
        ward.discharge(0)  # t=2, bed 0 free at t=12 (cleaning is 10 s)
        for _ in range(10):
            ward.tick()
        # t=12: boarded patient admitted, alarm cleared.
        view = ward.arrivals_summary()
        self.assertFalse(view["boarding_full_alarm"])
        self.assertEqual(view["boarding_full_alarms_raised"], 1)

    def test_alarm_in_snapshot_and_arrivals_endpoint(self):
        ward = self._full_ward()
        ward.triage_arrive(1)  # t=2, diverted: raises
        snap = ward.tick()  # t=3
        self.assertTrue(snap["triage"]["boarding_full_alarm_active"])
        self.assertEqual(snap["triage"]["boarding_full_alarms_raised"], 1)
        self.assertTrue(snap["arrivals"]["boarding_full_alarm"])
        self.assertEqual(snap["arrivals"]["boarding_full_alarms_raised"], 1)
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/arrivals")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertTrue(data["boarding_full_alarm"])
            self.assertEqual(data["boarding_full_alarms_raised"], 1)
            status, body = get(port, "/triage")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertTrue(data["boarding_full_alarm_active"])
            self.assertEqual(
                data["boarding_full_alarm_history"][0]["raised_t"], 2.0
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_scenario_boarding_full(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "boarding-full.json")
        )
        self.assertEqual(cfg["triage"].capacity, 1)
        self.assertEqual(cfg["triage"].boarding_capacity, 1)
        ward = build_ward(cfg, seed=42)
        ward.tick()  # t=1: first arrival queues, second boards
        ward.tick()  # t=2: third arrival (critical) diverted: raises
        summary = ward.triage_summary()
        self.assertEqual(summary["boarding_full_alarms_raised"], 1)
        self.assertTrue(summary["boarding_full_alarm_active"])
        self.assertEqual(
            summary["boarding_full_alarm_history"],
            [
                {
                    "raised_t": 2.0,
                    "cleared_t": None,
                    "diverted": 1,
                    "hook_retries": 0,
                    "hook_final_status": "not_invoked",
                    "hook_invoked": False,
                    "hook_errors": [],
                }
            ],
        )
        ward.tick()  # t=3: fourth arrival diverted: same episode
        summary = ward.triage_summary()
        self.assertEqual(summary["boarding_full_alarms_raised"], 1)
        self.assertEqual(summary["boarding_full_diverted"], 2)
        self.assertTrue(ward.arrivals_summary()["boarding_full_alarm"])
        for _ in range(12):
            ward.tick()
        snap = ward.tick()  # t=16
        # Bed 0 freed at t=15 and the boarded patient took it, so the
        # alarm cleared at t=15 with two diversions in the episode.
        self.assertEqual(snap["beds"][0]["profile"], "febrile")
        summary = snap["triage"]
        self.assertFalse(summary["boarding_full_alarm_active"])
        self.assertEqual(
            summary["boarding_full_alarm_history"],
            [
                {
                    "raised_t": 2.0,
                    "cleared_t": 15.0,
                    "diverted": 2,
                    "hook_retries": 0,
                    "hook_final_status": "not_invoked",
                    "hook_invoked": False,
                    "hook_errors": [],
                }
            ],
        )
        self.assertFalse(snap["arrivals"]["boarding_full_alarm"])
        for _ in range(2):
            ward.tick()
        snap = ward.tick()  # t=19: bed 1 freed at t=18, queued patient in
        self.assertEqual(snap["beds"][1]["profile"], "stable")
        self.assertEqual(snap["triage"]["admitted"], 2)
        self.assertEqual(snap["triage"]["mean_boarding_wait_s"], 14.0)
        self.assertEqual(snap["triage"]["mean_wait_s"], 18.0)

    def test_recorder_summary_carries_boarding_full_alarm(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "boarding-full.json")
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(20):
                    ward.tick()
            summary = summarize(path)
            self.assertEqual(
                summary["triage"]["boarding_full_alarms_raised"], 1
            )
            self.assertFalse(
                summary["triage"]["boarding_full_alarm_active"]
            )
            self.assertEqual(summary["triage"]["boarding_full_diverted"], 2)
            self.assertEqual(summary["triage"]["diverted"], 2)
        finally:
            os.unlink(path)


class TestDischargePlanning(unittest.TestCase):
    def _ward(self, beds=2, cleaning_s=10.0, **kw):
        ward = Ward(beds=beds, cleaning_s=cleaning_s, seed=42, **kw)
        ward.tick()
        return ward

    def test_planned_discharge_fires_at_expected_time(self):
        ward = self._ward(
            discharge_planning={
                "lead_time_s": 5,
                "orders": [{"bed": 0, "at": 0}],
            }
        )
        for _ in range(3):
            ward.tick()
        self.assertEqual(ward.turnover_summary()["occupied"], 2)
        snap = ward.tick()  # t=5: expected discharge
        self.assertEqual(snap["beds"][0]["state"], "cleaning")
        self.assertFalse(snap["beds"][0]["online"])
        summary = ward.discharge_summary()
        self.assertEqual(summary["planned"], 1)
        self.assertEqual(summary["discharged"], 1)
        self.assertEqual(summary["delayed"], 0)
        order = summary["orders"][0]
        self.assertEqual(order["status"], "discharged")
        self.assertEqual(order["ordered_t"], 0.0)
        self.assertEqual(order["expected_t"], 5.0)
        self.assertEqual(order["actual_t"], 5.0)
        self.assertEqual(order["delay_s"], 0.0)

    def test_discharge_delayed_when_cleaning_backed_up(self):
        ward = self._ward(
            cleaning_capacity=1,
            discharge_planning={
                "lead_time_s": 5,
                "orders": [{"bed": 0, "at": 0}, {"bed": 1, "at": 0}],
            },
        )
        for _ in range(4):
            ward.tick()
        snap = ward.tick()  # t=5: bed 0 cleans, bed 1 queues
        self.assertEqual(snap["beds"][0]["state"], "cleaning")
        self.assertTrue(snap["beds"][1]["online"])
        self.assertEqual(snap["turnover"]["beds"][1]["state"], "occupied")
        self.assertTrue(snap["turnover"]["beds"][1]["awaiting_cleaning"])
        self.assertEqual(snap["discharge"]["awaiting_cleaning"], 1)
        self.assertEqual(snap["discharge"]["pending"], 1)
        for _ in range(9):
            ward.tick()
        snap = ward.tick()  # t=15: bed 0 done, bed 1 starts cleaning
        self.assertEqual(snap["beds"][1]["state"], "cleaning")
        summary = ward.discharge_summary()
        self.assertEqual(summary["discharged"], 2)
        self.assertEqual(summary["delayed"], 1)
        self.assertEqual(summary["total_delay_s"], 10.0)
        self.assertEqual(summary["mean_delay_s"], 10.0)
        self.assertEqual(summary["orders"][1]["actual_t"], 15.0)
        self.assertEqual(summary["orders"][1]["delay_s"], 10.0)
        self.assertEqual(
            ward.turnover_summary()["total_discharge_delay_s"], 10.0
        )

    def test_queued_patient_still_monitored(self):
        ward = self._ward(
            cleaning_capacity=1,
            discharge_planning={
                "lead_time_s": 5,
                "orders": [{"bed": 0, "at": 0}, {"bed": 1, "at": 0}],
            },
        )
        for _ in range(5):
            ward.tick()
        # Bed 1 is awaiting a cleaning slot but still in the bed:
        # it keeps emitting readings while it waits.
        snap = ward.tick()
        reading = snap["beds"][1]
        self.assertTrue(reading["online"])
        self.assertIn("heart_rate", reading)
        self.assertEqual(snap["turnover"]["beds"][1]["state"], "occupied")
        self.assertTrue(snap["turnover"]["beds"][1]["awaiting_cleaning"])

    def test_order_cancelled_when_patient_discharged_early(self):
        ward = self._ward(
            chaos=[ChaosEvent("discharge", 3, bed=0)],
            discharge_planning={
                "lead_time_s": 5,
                "orders": [{"bed": 0, "at": 0}],
            },
        )
        for _ in range(5):
            ward.tick()
        summary = ward.discharge_summary()
        self.assertEqual(summary["orders"][0]["status"], "cancelled")
        self.assertEqual(summary["cancelled"], 1)
        self.assertEqual(summary["discharged"], 0)

    def test_live_order_uses_default_lead_time(self):
        ward = self._ward(discharge_planning={"lead_time_s": 5})
        order = ward.discharge_planner.place_order(1, ward.elapsed)
        self.assertEqual(order["expected_t"], 6.0)
        for _ in range(5):
            ward.tick()
        self.assertEqual(ward.turnover_summary()["beds"][1]["state"], "cleaning")

    def test_discharge_planning_validation(self):
        with self.assertRaises(ValueError):
            DischargePlanner(lead_time_s=0)
        with self.assertRaises(ValueError):
            DischargePlanner(orders=[{"bed": 0}])
        with self.assertRaises(ValueError):
            DischargePlanner(orders=[{"bed": -1, "at": 0}])
        with self.assertRaises(ValueError):
            DischargePlanner(orders=[{"bed": 0, "at": -1}])
        with self.assertRaises(ValueError):
            DischargePlanner(orders=[{"bed": 0, "at": 0, "lead_time_s": 0}])
        with self.assertRaises(ValueError):
            DischargePlanner(orders=[{"bed": 0, "at": 0, "lead_time_s": "soon"}])
        with self.assertRaises(ValueError):
            self._ward(
                discharge_planning={
                    "orders": [{"bed": 99, "at": 0}],
                }
            )
        with self.assertRaises(ValueError):
            Ward(beds=2, discharge_planning="tomorrow")

    def test_cleaning_capacity_validation(self):
        with self.assertRaises(ValueError):
            Ward(beds=2, cleaning_capacity=0)
        with self.assertRaises(ValueError):
            Ward(beds=2, cleaning_capacity=True)
        with self.assertRaises(ValueError):
            Ward(beds=2, cleaning_capacity=1.5)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"beds": 2, "cleaning_capacity": 0}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_scenario(path)
        finally:
            os.unlink(path)

    def test_chaos_discharge_queues_when_cleaning_full(self):
        ward = self._ward(
            cleaning_capacity=1,
            chaos=[
                ChaosEvent("discharge", 5, bed=0),
                ChaosEvent("discharge", 5, bed=1),
            ],
        )
        for _ in range(5):
            ward.tick()
        snap = ward.tick()  # t=6: bed 0 cleaning, bed 1 still in bed
        self.assertEqual(snap["beds"][0]["state"], "cleaning")
        self.assertTrue(snap["turnover"]["beds"][1]["awaiting_cleaning"])
        for _ in range(9):
            ward.tick()
        snap = ward.tick()  # t=15: bed 1 starts cleaning, 10 s late
        self.assertEqual(snap["beds"][1]["state"], "cleaning")
        self.assertEqual(
            ward.turnover_summary()["total_discharge_delay_s"], 10.0
        )

    def test_discharge_in_snapshot_and_endpoint(self):
        ward = self._ward(
            discharge_planning={
                "lead_time_s": 5,
                "orders": [{"bed": 0, "at": 0}],
            }
        )
        snap = ward.tick()
        self.assertIn("discharge", snap)
        self.assertEqual(snap["discharge"]["planned"], 1)
        self.assertEqual(snap["discharge"]["cleaning_capacity"], None)
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/discharge")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["planned"], 1)
            self.assertEqual(data["lead_time_s"], 5.0)
            ward.partitioned = True
            status, _ = get(port, "/discharge")
            self.assertEqual(status, 503)
        finally:
            server.shutdown()
            server.server_close()

    def test_ws_frame_carries_discharge(self):
        ward = Ward(
            beds=2,
            rate_hz=20.0,
            seed=5,
            discharge_planning={
                "lead_time_s": 5,
                "orders": [{"bed": 0, "at": 0}],
            },
        )
        server, port = serve_in_thread(ward)
        try:
            ws = WsClient(port)
            try:
                msg = json.loads(ws.read_until_opcode(1))
                self.assertIn("discharge", msg)
                self.assertEqual(msg["discharge"]["planned"], 1)
            finally:
                ws.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_scenario_discharge(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "discharge.json")
        )
        self.assertEqual(cfg["cleaning_capacity"], 1)
        ward = build_ward(cfg, seed=42)
        for _ in range(40):
            ward.tick()
        summary = ward.discharge_summary()
        self.assertEqual(summary["planned"], 3)
        self.assertEqual(summary["discharged"], 3)
        self.assertEqual(summary["delayed"], 2)
        self.assertEqual(summary["cancelled"], 0)
        self.assertEqual(summary["total_delay_s"], 18.0)
        self.assertEqual(summary["mean_delay_s"], 9.0)
        orders = summary["orders"]
        self.assertEqual(orders[0]["actual_t"], 5.0)
        self.assertEqual(orders[0]["delay_s"], 0.0)
        self.assertEqual(orders[1]["actual_t"], 15.0)
        self.assertEqual(orders[1]["delay_s"], 10.0)
        self.assertEqual(orders[2]["actual_t"], 25.0)
        self.assertEqual(orders[2]["delay_s"], 8.0)

    def test_recorder_summary_carries_discharge(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "discharge.json")
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            summary = summarize(path)
            self.assertEqual(summary["discharge"]["planned"], 3)
            self.assertEqual(summary["discharge"]["discharged"], 3)
            self.assertEqual(summary["discharge"]["delayed"], 2)
            self.assertEqual(summary["discharge"]["total_delay_s"], 18.0)
            self.assertEqual(summary["discharge"]["mean_delay_s"], 9.0)
        finally:
            os.unlink(path)


class TestDischargeReadiness(unittest.TestCase):
    def test_stable_patient_becomes_ready_once_window_elapses(self):
        ward = Ward(
            beds=1, profiles=["stable"],
            discharge_readiness={"stable_window_s": 5}, seed=42,
        )
        for _ in range(4):
            snap = ward.tick()
        self.assertFalse(snap["beds"][0]["discharge_ready"])
        snap = ward.tick()
        self.assertTrue(snap["beds"][0]["discharge_ready"])
        summary = ward.readiness_summary()
        self.assertEqual(summary["ready"], 1)
        self.assertEqual(summary["not_ready"], 0)
        self.assertEqual(summary["stable_window_s"], 5.0)
        self.assertEqual(summary["beds"][0]["stable_s"], 5.0)
        self.assertTrue(summary["beds"][0]["orders_complete"])

    def test_alarming_patient_is_never_ready(self):
        # Febrile patients run hot enough to trip the fever alarm
        # most ticks; with a 300 s window they never stay clear.
        ward = Ward(
            beds=1, profiles=["febrile"],
            discharge_readiness={"stable_window_s": 300}, seed=42,
        )
        for _ in range(40):
            snap = ward.tick()
        self.assertFalse(snap["beds"][0]["discharge_ready"])
        self.assertEqual(ward.readiness_summary()["ready"], 0)

    def test_pending_dose_blocks_readiness_until_given(self):
        ward = Ward(
            beds=1, profiles=["stable"],
            meds=[[{"name": "metoprolol", "every_s": 3600,
                    "first_at": 0, "grace_s": 600}]],
            discharge_readiness={"stable_window_s": 5}, seed=42,
        )
        for _ in range(29):
            snap = ward.tick()
        # The dose due at t=0 is still waiting to be given (30 s
        # with a full roster), so the orders are incomplete.
        self.assertFalse(snap["beds"][0]["discharge_ready"])
        self.assertFalse(
            ward.readiness_summary()["beds"][0]["orders_complete"]
        )
        snap = ward.tick()  # t=30: dose given, orders complete.
        self.assertTrue(snap["beds"][0]["discharge_ready"])

    def test_missed_dose_blocks_readiness_until_next_dose(self):
        # A 1 s grace window means the dose due at t=0 misses at
        # once (administration would take 30 s); the orders stay
        # incomplete until the next dose, 3600 s later.
        ward = Ward(
            beds=1, profiles=["stable"],
            meds=[[{"name": "insulin", "every_s": 3600,
                    "first_at": 0, "grace_s": 1}]],
            discharge_readiness={"stable_window_s": 5}, seed=42,
        )
        for _ in range(40):
            snap = ward.tick()
        self.assertFalse(snap["beds"][0]["discharge_ready"])
        detail = ward.readiness_summary()["beds"][0]
        self.assertFalse(detail["orders_complete"])
        self.assertEqual(detail["stable_s"], 0.0)

    def test_alarm_storm_resets_the_stability_clock(self):
        ward = Ward(
            beds=1, profiles=["stable"],
            discharge_readiness={"stable_window_s": 5},
            chaos=[ChaosEvent("alarm_storm", 10, 5, bed=0)],
            seed=42,
        )
        for _ in range(10):
            snap = ward.tick()
        self.assertTrue(snap["beds"][0]["discharge_ready"])
        for _ in range(5):
            snap = ward.tick()  # t=11..15: storm drags vitals out.
        self.assertFalse(snap["beds"][0]["discharge_ready"])
        # The storm ends at t=15 and the alarms clear at once; the
        # patient is ready again once 5 s of quiet vitals have passed.
        for _ in range(4):
            snap = ward.tick()
        self.assertFalse(snap["beds"][0]["discharge_ready"])
        snap = ward.tick()
        self.assertTrue(snap["beds"][0]["discharge_ready"])

    def test_unadmitted_bed_is_not_ready(self):
        ward = Ward(
            beds=2, profiles=["stable", "stable"],
            timeline=Timeline([TimelinePhase(0, 1, ramp_s=0)]),
            discharge_readiness={"stable_window_s": 5}, seed=42,
        )
        for _ in range(10):
            snap = ward.tick()
        detail = ward.readiness_summary()["beds"]
        self.assertTrue(detail[0]["ready"])
        self.assertFalse(detail[1]["ready"])
        self.assertFalse(detail[1]["admitted"])

    def test_readiness_validation(self):
        for bad in (0, -5, "300", None, True):
            with self.assertRaises(ValueError):
                DischargeReadiness(stable_window_s=bad)
        with self.assertRaises(TypeError):
            DischargeReadiness(stable_window_s=5, bogus=1)
        with self.assertRaises(ValueError):
            Ward(beds=1, discharge_readiness="5")

    def test_readiness_in_snapshot_endpoint_and_ws_frame(self):
        ward = Ward(
            beds=1, profiles=["stable"],
            discharge_readiness={"stable_window_s": 5}, seed=42,
        )
        for _ in range(10):
            ward.tick()
        snap = ward.snapshot()
        self.assertTrue(snap["beds"][0]["discharge_ready"])
        self.assertEqual(snap["discharge_readiness"]["ready"], 1)
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/discharge-readiness")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["ready"], 1)
            ward.partitioned = True
            status, _ = get(port, "/discharge-readiness")
            self.assertEqual(status, 503)
            ward.partitioned = False
        finally:
            server.shutdown()
            server.server_close()

    def test_ws_frame_carries_readiness(self):
        ward = Ward(
            beds=1,
            profiles=["stable"],
            rate_hz=20.0,
            discharge_readiness={"stable_window_s": 5},
            seed=42,
        )
        server, port = serve_in_thread(ward)
        try:
            ws = WsClient(port)
            try:
                msg = json.loads(ws.read_until_opcode(1))
                self.assertIn("discharge_readiness", msg)
                self.assertEqual(msg["discharge_readiness"]["ready"], 0)
            finally:
                ws.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_scenario_discharge_ready(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "discharge-ready.json")
        )
        self.assertEqual(cfg["discharge_readiness"].stable_window_s, 5.0)
        ward = build_ward(cfg, seed=42)
        for _ in range(40):
            snap = ward.tick()
        flags = [b["discharge_ready"] for b in snap["beds"]]
        # Bed 0 stable with no orders: ready from t=5. Bed 1's dose
        # lands at t=30: ready from then. Bed 2's insulin missed at
        # t=1 (1 s grace): orders incomplete, never ready. Bed 3's
        # fever comes and goes; at t=40 it has been quiet for 6 s.
        self.assertEqual(flags, [True, True, False, True])
        summary = ward.readiness_summary()
        self.assertEqual(summary["ready"], 3)
        self.assertEqual(summary["not_ready"], 1)

    def test_recorder_summary_carries_discharge_readiness(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "discharge-ready.json")
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            summary = summarize(path)
            self.assertEqual(summary["discharge_readiness"]["ready"], 3)
            self.assertEqual(
                summary["discharge_readiness"]["not_ready"], 1
            )
            self.assertEqual(summary["discharge_readiness"]["occupied"], 4)
            self.assertEqual(
                summary["discharge_readiness"]["stable_window_s"], 5.0
            )
        finally:
            os.unlink(path)


class TestAutoDischargeOrders(unittest.TestCase):
    def _ward(self, beds=1, cleaning_s=10.0, **kw):
        return Ward(
            beds=beds, profiles=["stable"] * beds,
            cleaning_s=cleaning_s, seed=42, **kw
        )

    def test_auto_order_raised_when_bed_becomes_ready(self):
        ward = self._ward(
            discharge_readiness={"stable_window_s": 5, "auto_order": True},
            discharge_planning={"lead_time_s": 5},
        )
        for _ in range(4):
            snap = ward.tick()
        self.assertEqual(ward.discharge_summary()["planned"], 0)
        snap = ward.tick()  # t=5: ready, so an order is raised.
        self.assertTrue(snap["beds"][0]["discharge_ready"])
        summary = ward.discharge_summary()
        self.assertEqual(summary["planned"], 1)
        self.assertEqual(summary["auto_planned"], 1)
        order = summary["orders"][0]
        self.assertEqual(order["bed"], 0)
        self.assertEqual(order["source"], "auto")
        self.assertEqual(order["ordered_t"], 5.0)
        self.assertEqual(order["expected_t"], 10.0)
        self.assertEqual(order["status"], "planned")
        self.assertEqual(snap["discharge"]["orders"][0]["source"], "auto")
        self.assertTrue(
            snap["discharge_readiness"]["beds"][0]["auto_ordered"]
        )
        for _ in range(5):
            snap = ward.tick()  # t=10: the order fires, bed cleans.
        self.assertEqual(snap["beds"][0]["state"], "cleaning")
        self.assertEqual(ward.discharge_summary()["discharged"], 1)

    def test_auto_order_off_by_default(self):
        ward = self._ward(
            discharge_readiness={"stable_window_s": 5},
            discharge_planning={"lead_time_s": 5},
        )
        for _ in range(10):
            snap = ward.tick()
        self.assertTrue(snap["beds"][0]["discharge_ready"])
        summary = ward.discharge_summary()
        self.assertEqual(summary["planned"], 0)
        self.assertEqual(summary["auto_planned"], 0)
        self.assertFalse(
            ward.readiness_summary()["beds"][0]["auto_ordered"]
        )

    def test_auto_order_not_raised_twice_after_readiness_returns(self):
        ward = self._ward(
            discharge_readiness={"stable_window_s": 5, "auto_order": True},
            chaos=[ChaosEvent("alarm_storm", 10, 5, bed=0)],
        )
        for _ in range(5):
            ward.tick()
        # Ready at t=5: one auto order, expected an hour out.
        self.assertEqual(ward.discharge_summary()["planned"], 1)
        for _ in range(15):
            snap = ward.tick()
        # The storm (t=10..15) took readiness away; it is back by
        # t=20, but the live order is still in flight, so no second
        # order is raised.
        self.assertTrue(snap["beds"][0]["discharge_ready"])
        summary = ward.discharge_summary()
        self.assertEqual(summary["planned"], 1)
        self.assertEqual(summary["auto_planned"], 1)
        self.assertEqual(summary["orders"][0]["ordered_t"], 5.0)

    def test_auto_order_skipped_when_order_already_exists(self):
        ward = self._ward(
            discharge_readiness={"stable_window_s": 5, "auto_order": True},
            discharge_planning={
                "lead_time_s": 3600,
                "orders": [{"bed": 0, "at": 0}],
            },
        )
        for _ in range(10):
            snap = ward.tick()
        self.assertTrue(snap["beds"][0]["discharge_ready"])
        summary = ward.discharge_summary()
        self.assertEqual(summary["planned"], 1)
        self.assertEqual(summary["auto_planned"], 0)
        self.assertEqual(summary["orders"][0]["source"], "scheduled")

    def test_new_patient_gets_fresh_auto_order(self):
        ward = self._ward(
            discharge_readiness={"stable_window_s": 5, "auto_order": True},
            discharge_planning={"lead_time_s": 5},
        )
        for _ in range(10):
            ward.tick()  # t=10: first patient discharged, bed cleans.
        self.assertEqual(ward.discharge_summary()["discharged"], 1)
        for _ in range(10):
            ward.tick()  # t=20: cleaning done, bed available.
        self.assertEqual(
            ward.turnover_summary()["beds"][0]["state"], "available"
        )
        ward.admit(0)
        for _ in range(5):
            ward.tick()  # t=25: the new patient is ready.
        summary = ward.discharge_summary()
        self.assertEqual(summary["planned"], 2)
        self.assertEqual(summary["auto_planned"], 2)
        self.assertEqual(
            [o["ordered_t"] for o in summary["orders"]], [5.0, 25.0]
        )
        self.assertTrue(
            all(o["source"] == "auto" for o in summary["orders"])
        )

    def test_auto_order_validation(self):
        with self.assertRaises(ValueError):
            DischargeReadiness(stable_window_s=5, auto_order="yes")
        with self.assertRaises(ValueError):
            DischargeReadiness(stable_window_s=5, auto_order=1)
        with self.assertRaises(ValueError):
            DischargePlanner().place_order(0, 0.0, source="surprise")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {"beds": 1, "discharge_readiness": {"auto_order": "yes"}},
                f,
            )
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_scenario(path)
        finally:
            os.unlink(path)

    def test_auto_order_in_snapshot_and_endpoint(self):
        ward = self._ward(
            discharge_readiness={"stable_window_s": 5, "auto_order": True},
            discharge_planning={"lead_time_s": 5},
        )
        for _ in range(5):
            ward.tick()
        snap = ward.snapshot()
        self.assertEqual(snap["discharge"]["auto_planned"], 1)
        self.assertEqual(snap["discharge"]["orders"][0]["source"], "auto")
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/discharge")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["auto_planned"], 1)
            self.assertEqual(data["orders"][0]["source"], "auto")
            ward.partitioned = True
            status, _ = get(port, "/discharge")
            self.assertEqual(status, 503)
        finally:
            server.shutdown()
            server.server_close()

    def test_scenario_discharge_auto_order(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "discharge-auto-order.json")
        )
        self.assertTrue(cfg["discharge_readiness"].auto_order)
        ward = build_ward(cfg, seed=42)
        for _ in range(40):
            ward.tick()
        summary = ward.discharge_summary()
        self.assertEqual(summary["planned"], 3)
        self.assertEqual(summary["auto_planned"], 3)
        self.assertEqual(summary["discharged"], 3)
        self.assertEqual(summary["delayed"], 0)
        self.assertEqual(summary["cancelled"], 0)
        orders = summary["orders"]
        # Beds 0 and 3 are quiet from admission: ready at 5 s, gone
        # at 10 s. Bed 1's dose lands at 30 s: ready then, gone at
        # 35 s. Bed 2's insulin missed, so it never flags ready and
        # never gets an order.
        self.assertEqual(
            [
                (o["bed"], o["ordered_t"], o["expected_t"], o["status"],
                 o["source"])
                for o in orders
            ],
            [
                (0, 5.0, 10.0, "discharged", "auto"),
                (3, 5.0, 10.0, "discharged", "auto"),
                (1, 30.0, 35.0, "discharged", "auto"),
            ],
        )

    def test_recorder_summary_carries_auto_orders(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "discharge-auto-order.json")
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            summary = summarize(path)
            self.assertEqual(summary["discharge"]["planned"], 3)
            self.assertEqual(summary["discharge"]["auto_planned"], 3)
            self.assertEqual(summary["discharge"]["discharged"], 3)
            self.assertEqual(summary["discharge"]["total_delay_s"], 0.0)
        finally:
            os.unlink(path)


class TestAutoOrderLeadTime(unittest.TestCase):
    def _ward(self, beds=1, cleaning_s=10.0, **kw):
        return Ward(
            beds=beds, profiles=["stable"] * beds,
            cleaning_s=cleaning_s, seed=42, **kw
        )

    def test_override_shortens_auto_order_lead_time(self):
        # The planner default is 5 s, but the auto-order override
        # is 2 s: the order is expected 2 s after it is raised, and
        # the patient leaves then, not 5 s later.
        ward = self._ward(
            discharge_readiness={
                "stable_window_s": 5,
                "auto_order": True,
                "auto_order_lead_time_s": 2,
            },
            discharge_planning={"lead_time_s": 5},
        )
        for _ in range(5):
            snap = ward.tick()  # t=5: ready, auto order raised.
        order = ward.discharge_summary()["orders"][0]
        self.assertEqual(order["source"], "auto")
        self.assertEqual(order["ordered_t"], 5.0)
        self.assertEqual(order["expected_t"], 7.0)
        self.assertEqual(
            snap["discharge_readiness"]["auto_order_lead_time_s"], 2.0
        )
        ward.tick()  # t=6: still in bed, the order has not fired.
        self.assertEqual(ward.discharge_summary()["discharged"], 0)
        snap = ward.tick()  # t=7: the override fires, bed cleans.
        self.assertEqual(snap["beds"][0]["state"], "cleaning")
        self.assertEqual(ward.discharge_summary()["discharged"], 1)

    def test_override_does_not_touch_manual_or_scheduled_orders(self):
        # A scheduled order still uses the planner's default (or
        # its own per-order lead time); only auto orders take the
        # override.
        ward = self._ward(
            discharge_readiness={
                "stable_window_s": 5,
                "auto_order": True,
                "auto_order_lead_time_s": 2,
            },
            discharge_planning={
                "lead_time_s": 5,
                "orders": [{"bed": 0, "at": 0}],
            },
        )
        for _ in range(5):
            ward.tick()
        summary = ward.discharge_summary()
        self.assertEqual(summary["planned"], 1)
        self.assertEqual(summary["auto_planned"], 0)
        order = summary["orders"][0]
        self.assertEqual(order["source"], "scheduled")
        self.assertEqual(order["expected_t"], 5.0)

    def test_no_override_uses_planner_default(self):
        ward = self._ward(
            discharge_readiness={"stable_window_s": 5, "auto_order": True},
            discharge_planning={"lead_time_s": 5},
        )
        for _ in range(5):
            ward.tick()
        order = ward.discharge_summary()["orders"][0]
        self.assertEqual(order["expected_t"], 10.0)
        self.assertIsNone(
            ward.readiness_summary()["auto_order_lead_time_s"]
        )

    def test_override_validation(self):
        for bad in (0, -1, True, "2"):
            with self.assertRaises(ValueError):
                DischargeReadiness(
                    stable_window_s=5,
                    auto_order=True,
                    auto_order_lead_time_s=bad,
                )
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "beds": 1,
                    "discharge_readiness": {
                        "auto_order": True,
                        "auto_order_lead_time_s": 0,
                    },
                },
                f,
            )
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_scenario(path)
        finally:
            os.unlink(path)

    def test_scenario_discharge_auto_order_lead_time(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios",
                "discharge-auto-order-lead-time.json",
            )
        )
        self.assertTrue(cfg["discharge_readiness"].auto_order)
        self.assertEqual(
            cfg["discharge_readiness"].auto_order_lead_time_s, 2.0
        )
        self.assertEqual(cfg["discharge_planning"].lead_time_s, 5.0)
        ward = build_ward(cfg, seed=42)
        for _ in range(40):
            ward.tick()
        summary = ward.discharge_summary()
        self.assertEqual(summary["planned"], 3)
        self.assertEqual(summary["auto_planned"], 3)
        self.assertEqual(summary["discharged"], 3)
        # Beds 0 and 3 are quiet from admission: ready at 5 s, gone
        # at 7 s on the 2 s override, not at 10 s on the planner's
        # 5 s default. Bed 1's dose lands at 30 s: ready then, gone
        # at 32 s. Bed 2's insulin missed, so it never flags ready
        # and never gets an order.
        self.assertEqual(
            [
                (o["bed"], o["ordered_t"], o["expected_t"], o["status"],
                 o["source"])
                for o in summary["orders"]
            ],
            [
                (0, 5.0, 7.0, "discharged", "auto"),
                (3, 5.0, 7.0, "discharged", "auto"),
                (1, 30.0, 32.0, "discharged", "auto"),
            ],
        )
        # The paperwork timing follows the override too: ready to
        # discharge is the 2 s lead time, not the planner default.
        self.assertEqual(
            summary["timing"]["mean_ready_to_discharge_s"], 2.0
        )
        self.assertEqual(
            summary["timing"]["max_ready_to_discharge_s"], 2.0
        )

    def test_recorder_summary_carries_override(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios",
                "discharge-auto-order-lead-time.json",
            )
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            summary = summarize(path)
            self.assertEqual(
                summary["discharge_readiness"]["auto_order_lead_time_s"],
                2.0,
            )
            self.assertEqual(summary["discharge"]["planned"], 3)
            self.assertEqual(summary["discharge"]["auto_planned"], 3)
            self.assertEqual(summary["discharge"]["discharged"], 3)
        finally:
            os.unlink(path)


class TestDischargeTiming(unittest.TestCase):
    def _ward(self, beds=1, cleaning_s=10.0, **kw):
        return Ward(
            beds=beds, profiles=["stable"] * beds,
            cleaning_s=cleaning_s, seed=42, **kw
        )

    def test_auto_order_timing_measured(self):
        ward = self._ward(
            discharge_readiness={"stable_window_s": 5, "auto_order": True},
            discharge_planning={"lead_time_s": 5},
        )
        for _ in range(5):
            ward.tick()  # t=5: ready, so an auto order is raised.
        order = ward.discharge_summary()["orders"][0]
        self.assertEqual(order["ready_t"], 5.0)
        self.assertEqual(order["ready_to_order_s"], 0.0)
        self.assertIsNone(order["ready_to_discharge_s"])
        for _ in range(5):
            snap = ward.tick()  # t=10: the order fires, bed cleans.
        order = ward.discharge_summary()["orders"][0]
        self.assertEqual(order["status"], "discharged")
        self.assertEqual(order["ready_to_discharge_s"], 5.0)
        timing = ward.discharge_summary()["timing"]
        self.assertEqual(timing["measured"], 1)
        self.assertEqual(timing["mean_ready_to_order_s"], 0.0)
        self.assertEqual(timing["mean_ready_to_discharge_s"], 5.0)
        self.assertEqual(timing["max_ready_to_discharge_s"], 5.0)
        self.assertEqual(snap["discharge"]["timing"], timing)

    def test_order_placed_before_ready_has_no_ready_to_order(self):
        ward = self._ward(
            discharge_readiness={"stable_window_s": 5},
            discharge_planning={
                "lead_time_s": 10,
                "orders": [{"bed": 0, "at": 0}],
            },
        )
        for _ in range(10):
            ward.tick()  # t=10: order placed at 0, ready at 5.
        order = ward.discharge_summary()["orders"][0]
        self.assertEqual(order["status"], "discharged")
        self.assertEqual(order["ready_t"], 5.0)
        self.assertIsNone(order["ready_to_order_s"])
        self.assertEqual(order["ready_to_discharge_s"], 5.0)
        timing = ward.discharge_summary()["timing"]
        self.assertEqual(timing["measured"], 1)
        self.assertEqual(timing["mean_ready_to_order_s"], 0.0)
        self.assertEqual(timing["mean_ready_to_discharge_s"], 5.0)

    def test_delayed_discharge_timing_includes_cleaning_delay(self):
        ward = self._ward(
            beds=2,
            cleaning_capacity=1,
            discharge_readiness={"stable_window_s": 5, "auto_order": True},
            discharge_planning={"lead_time_s": 5},
        )
        for _ in range(20):
            ward.tick()
        # Both beds ready at 5 s with orders expected at 10 s; bed 0
        # leaves at 10 s, bed 1 waits for the one cleaning slot and
        # leaves at 20 s, so its ready-to-discharge time is 15 s.
        summary = ward.discharge_summary()
        self.assertEqual(summary["discharged"], 2)
        self.assertEqual(summary["delayed"], 1)
        orders = sorted(summary["orders"], key=lambda o: o["bed"])
        self.assertEqual(orders[0]["ready_to_discharge_s"], 5.0)
        self.assertEqual(orders[1]["ready_to_discharge_s"], 15.0)
        self.assertEqual(orders[1]["delay_s"], 10.0)
        timing = summary["timing"]
        self.assertEqual(timing["measured"], 2)
        self.assertEqual(timing["mean_ready_to_order_s"], 0.0)
        self.assertEqual(timing["mean_ready_to_discharge_s"], 10.0)
        self.assertEqual(timing["max_ready_to_discharge_s"], 15.0)

    def test_never_ready_order_has_no_timing(self):
        ward = self._ward(
            discharge_planning={
                "lead_time_s": 5,
                "orders": [{"bed": 0, "at": 0}],
            },
        )
        for _ in range(5):
            ward.tick()  # t=5: discharged, long before the 300 s window.
        order = ward.discharge_summary()["orders"][0]
        self.assertEqual(order["status"], "discharged")
        self.assertIsNone(order["ready_t"])
        self.assertIsNone(order["ready_to_order_s"])
        self.assertIsNone(order["ready_to_discharge_s"])
        timing = ward.discharge_summary()["timing"]
        self.assertEqual(timing["measured"], 0)
        self.assertEqual(timing["mean_ready_to_order_s"], 0.0)
        self.assertEqual(timing["mean_ready_to_discharge_s"], 0.0)
        self.assertEqual(timing["max_ready_to_discharge_s"], 0.0)

    def test_scenario_auto_order_timing(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "discharge-auto-order.json")
        )
        ward = build_ward(cfg, seed=42)
        for _ in range(40):
            ward.tick()
        summary = ward.discharge_summary()
        # Beds 0 and 3 ready at 5 s and gone at 10 s, bed 1 ready at
        # 30 s and gone at 35 s: every order placed the tick its bed
        # flagged ready, every patient out 5 s after flagging.
        self.assertEqual(
            [(o["bed"], o["ready_t"], o["ready_to_order_s"],
              o["ready_to_discharge_s"]) for o in summary["orders"]],
            [
                (0, 5.0, 0.0, 5.0),
                (3, 5.0, 0.0, 5.0),
                (1, 30.0, 0.0, 5.0),
            ],
        )
        self.assertEqual(
            summary["timing"],
            {
                "measured": 3,
                "mean_ready_to_order_s": 0.0,
                "mean_ready_to_discharge_s": 5.0,
                "max_ready_to_discharge_s": 5.0,
            },
        )

    def test_recorder_summary_carries_discharge_timing(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "discharge-auto-order.json")
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            summary = summarize(path)
            self.assertEqual(
                summary["discharge"]["timing"],
                {
                    "measured": 3,
                    "mean_ready_to_order_s": 0.0,
                    "mean_ready_to_discharge_s": 5.0,
                    "max_ready_to_discharge_s": 5.0,
                },
            )
        finally:
            os.unlink(path)


class TestPoissonArrivals(unittest.TestCase):
    def test_flat_rate_generates_near_expected_count(self):
        # 600 one-second ticks at 3600/hr is one arrival per tick on
        # average; the count should land near 600 (sd ~24).
        queue = TriageQueue(poisson={"rate_per_hour": 3600}, seed=42)
        for i in range(600):
            queue.process_poisson_arrivals(i + 1, 1.0, 8)
        self.assertEqual(queue.poisson.generated, 595)
        self.assertGreater(queue.poisson.generated, 500)
        self.assertLess(queue.poisson.generated, 700)
        # Most arrivals were diverted: the default queue holds 8.
        self.assertEqual(queue.diverted, 587)
        self.assertEqual(queue.summary(600)["arrivals_generated"], 595)

    def test_arrivals_by_hour_buckets_by_clock_hour(self):
        queue = TriageQueue(capacity=2)
        queue.arrive(0, 3, "stable", hour=8)
        queue.arrive(10, 2, "hypoxic", hour=9)
        queue.arrive(20, 4, "stable", hour=8)
        # The second hour-8 arrival was diverted (capacity 2), but it
        # still showed up at the ED, so it still counts toward hour 8.
        self.assertEqual(queue.diverted, 1)
        self.assertEqual(
            queue.summary(20)["arrivals_by_hour"], {"8": 2, "9": 1}
        )
        # Hours wrap past midnight.
        queue.arrive(30, 3, "stable", hour=26)
        self.assertEqual(
            queue.summary(30)["arrivals_by_hour"], {"2": 1, "8": 2, "9": 1}
        )
        # Without an hour, an arrival is not bucketed at all.
        plain = TriageQueue(capacity=2)
        plain.arrive(0, 3, "stable")
        self.assertEqual(plain.summary(0)["arrivals_by_hour"], {})

    def test_arrivals_by_hour_spans_midnight(self):
        # A ward starting at 23:00 with a flat 3600/hr walk-in rate,
        # run for just over an hour, sees arrivals in both hour 23
        # and hour 0 — and the per-hour counts add up to the total.
        ward = Ward(
            beds=2,
            seed=42,
            staffing={"start_hour": 23},
            triage={"poisson": {"rate_per_hour": 3600}},
        )
        for _ in range(3700):
            ward.tick()
        summary = ward.triage_summary()
        by_hour = summary["arrivals_by_hour"]
        self.assertIn("23", by_hour)
        self.assertIn("0", by_hour)
        self.assertEqual(sum(by_hour.values()), summary["arrivals_generated"])
        self.assertGreater(sum(by_hour.values()), 3000)

    def test_hourly_rates_vary_by_hour(self):
        rates = [30] * 6 + [60, 90] + [180] * 4 + [150] * 5 + [120] * 2 + [90] * 3 + [60] * 2
        model_queue = TriageQueue(poisson={"hourly_rates": rates}, seed=42)
        model = model_queue.poisson
        self.assertEqual(model.rate_at(2), 30.0)
        self.assertEqual(model.rate_at(10), 180.0)
        self.assertEqual(model.rate_at(22), 60.0)
        # Hours wrap past midnight.
        self.assertEqual(model.rate_at(26), 30.0)
        self.assertEqual(model.rate_at(8.5), 180.0)
        # A flat rate reads the same at every hour.
        flat = TriageQueue(poisson={"rate_per_hour": 120}, seed=42).poisson
        self.assertEqual(flat.rate_at(3), 120.0)
        self.assertEqual(flat.rate_at(15), 120.0)

    def test_arrival_model_validation(self):
        with self.assertRaises(ValueError):
            TriageQueue(poisson={"rate_per_hour": 0}, seed=42)
        with self.assertRaises(ValueError):
            TriageQueue(poisson={"rate_per_hour": -5}, seed=42)
        with self.assertRaises(ValueError):
            TriageQueue(poisson={}, seed=42)
        with self.assertRaises(ValueError):
            TriageQueue(
                poisson={"rate_per_hour": 60, "hourly_rates": [60] * 24},
                seed=42,
            )
        with self.assertRaises(ValueError):
            TriageQueue(poisson={"hourly_rates": [60] * 23}, seed=42)
        with self.assertRaises(ValueError):
            TriageQueue(
                poisson={"hourly_rates": [60] * 23 + [-1]}, seed=42
            )
        with self.assertRaises(ValueError):
            TriageQueue(
                poisson={"rate_per_hour": 60, "acuity_weights": {5: 1.0}},
                seed=42,
            )
        with self.assertRaises(ValueError):
            TriageQueue(
                poisson={"rate_per_hour": 60, "profile_weights": {"sick": 1.0}},
                seed=42,
            )
        with self.assertRaises(ValueError):
            TriageQueue(
                poisson={"rate_per_hour": 60, "acuity_weights": {1: 0}},
                seed=42,
            )
        with self.assertRaises(ValueError):
            TriageQueue(poisson="fast", seed=42)

    def test_generated_arrivals_have_valid_acuity_and_profile(self):
        queue = TriageQueue(poisson={"rate_per_hour": 36000}, seed=42)
        arrivals = queue.process_poisson_arrivals(1, 3600.0, 8)
        self.assertGreater(len(arrivals), 0)
        for arrival in arrivals:
            self.assertIn(arrival["acuity"], (1, 2, 3, 4))
            self.assertIn(arrival["profile"], PROFILES)
            self.assertEqual(arrival["arrival_t"], 1.0)

    def test_seeded_runs_are_reproducible(self):
        def run():
            ward = Ward(
                beds=2,
                cleaning_s=10.0,
                seed=42,
                triage={"capacity": 8, "poisson": {"rate_per_hour": 3600}},
            )
            for _ in range(20):
                ward.tick()
            return ward.triage_summary()

        first, second = run(), run()
        self.assertEqual(first, second)
        self.assertEqual(first["arrivals_generated"], 25)

    def test_poisson_arrivals_feed_queue_and_get_admitted(self):
        ward = Ward(
            beds=2,
            cleaning_s=10.0,
            seed=42,
            triage={"capacity": 8, "poisson": {"rate_per_hour": 3600}},
        )
        ward.tick()  # t=1
        ward.discharge(0)
        for _ in range(14):
            snap = ward.tick()
        # Discharged at t=1, cleaning 10 s: bed 0 free at t=11, and
        # the highest-priority walk-in takes it with fresh vitals.
        summary = ward.triage_summary()
        self.assertEqual(summary["arrivals_generated"], 14)
        self.assertEqual(summary["admitted"], 1)
        self.assertTrue(snap["beds"][0]["online"])
        self.assertEqual(snap["beds"][0]["profile"], "febrile")
        self.assertEqual(snap["beds"][0]["acuity"], 2)
        # Admitted at t=11, so five readings by t=15: fresh vitals,
        # not the discharged patient's.
        self.assertEqual(snap["beds"][0]["seq"], 5)

    def test_triage_summary_carries_arrival_rate(self):
        ward = Ward(
            beds=2,
            seed=42,
            triage={"poisson": {"rate_per_hour": 120}},
        )
        # Before the first tick there is no current rate yet.
        summary = ward.triage_summary()
        self.assertIsNone(summary["arrival_rate_per_hour"])
        self.assertEqual(summary["arrivals_generated"], 0)
        self.assertIsNone(summary["arrival_hourly_rates"])
        ward.tick()
        summary = ward.triage_summary()
        self.assertEqual(summary["arrival_rate_per_hour"], 120.0)
        # And a ward with no arrival model reports no rate at all.
        plain = Ward(beds=2, seed=42)
        plain.tick()
        plain_summary = plain.triage_summary()
        self.assertIsNone(plain_summary["arrival_rate_per_hour"])
        self.assertEqual(plain_summary["arrivals_generated"], 0)

    def test_scenario_arrivals(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "arrivals.json"))
        self.assertEqual(cfg["triage"].poisson.rate_at(8), 180.0)
        self.assertEqual(cfg["triage"].poisson.rate_at(2), 30.0)
        ward = build_ward(cfg, seed=42)
        for _ in range(60):
            ward.tick()
        # 180/hr over 60 s is 3 arrivals on average; two take the
        # beds freed by the discharges at 5 s and 8 s, one still waits.
        summary = ward.triage_summary()
        self.assertEqual(summary["arrivals_generated"], 3)
        self.assertEqual(summary["admitted"], 2)
        self.assertEqual(summary["diverted"], 0)
        self.assertEqual(summary["waiting_total"], 1)
        self.assertEqual(summary["arrival_rate_per_hour"], 180.0)
        self.assertEqual(len(summary["arrival_hourly_rates"]), 24)
        # All 3 walk-ins arrived in hour 8 (the run starts at 08:00).
        self.assertEqual(summary["arrivals_by_hour"], {"8": 3})

    def test_recorder_summary_carries_arrivals(self):
        cfg = load_scenario(os.path.join(HERE, "..", "scenarios", "arrivals.json"))
        ward = build_ward(cfg, seed=42)
        path = os.path.join(tempfile.mkdtemp(), "run.jsonl")
        with Recorder(path, ward.beds, ward.rate_hz, seed=42) as rec:
            ward.recorder = rec
            for _ in range(60):
                ward.tick()
        s = summarize(path)
        self.assertEqual(s["triage"]["admitted"], 2)
        self.assertEqual(s["triage"]["diverted"], 0)
        self.assertEqual(s["triage"]["arrivals_generated"], 3)
        self.assertEqual(s["triage"]["arrival_rate_per_hour"], 180.0)
        # The run summary carries the per-hour arrivals and the
        # configured hourly rates, so the load shape is visible.
        self.assertEqual(s["triage"]["arrivals_by_hour"], {"8": 3})
        self.assertEqual(len(s["triage"]["arrival_hourly_rates"]), 24)
        self.assertEqual(s["triage"]["arrival_hourly_rates"][8], 180.0)

    def test_scenario_rejects_bad_poisson(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {"beds": 1, "triage": {"poisson": {"rate_per_hour": 0}}}, f
            )
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_scenario(path)
        finally:
            os.unlink(path)

    def test_arrivals_view_counts_rate_by_hour_and_diversion(self):
        queue = TriageQueue(capacity=2)
        queue.arrive(0, 3, "stable", hour=8)
        queue.arrive(10, 2, "hypoxic", hour=9)
        queue.arrive(20, 4, "stable", hour=8)
        view = queue.arrivals_view(20)
        # No Poisson model, so there is no current rate.
        self.assertIsNone(view["arrival_rate_per_hour"])
        self.assertEqual(view["t"], 20)
        self.assertEqual(view["arrivals_last_hour"], 3)
        self.assertEqual(view["arrivals_total"], 3)
        self.assertEqual(view["arrivals_by_hour"], {"8": 2, "9": 1})
        self.assertEqual(view["waiting_total"], 2)
        self.assertEqual(view["capacity"], 2)
        self.assertEqual(view["admitted"], 0)
        # The third arrival was diverted (capacity 2) — and the
        # queue is still full, so the ward is still diverting.
        self.assertEqual(view["diverted"], 1)
        self.assertEqual(view["diverted_last_hour"], 1)
        self.assertTrue(view["diverting"])

    def test_arrivals_view_last_hour_window(self):
        queue = TriageQueue(capacity=8)
        queue.arrive(0, 3, "stable", hour=8)
        queue.arrive(3700, 3, "stable", hour=9)
        view = queue.arrivals_view(3700)
        # The window is (now - 3600, now]: the arrival at t=0 is
        # exactly an hour old, so it falls outside it.
        self.assertEqual(view["arrivals_last_hour"], 1)
        self.assertEqual(view["arrivals_total"], 2)
        self.assertEqual(view["diverted_last_hour"], 0)
        self.assertFalse(view["diverting"])

    def test_arrivals_view_carries_current_poisson_rate(self):
        queue = TriageQueue(poisson={"rate_per_hour": 120}, seed=42)
        self.assertIsNone(queue.arrivals_view(0)["arrival_rate_per_hour"])
        queue.process_poisson_arrivals(1, 1.0, 8)
        self.assertEqual(
            queue.arrivals_view(1)["arrival_rate_per_hour"], 120.0
        )

    def test_arrivals_in_snapshot_endpoint_and_updates_live(self):
        ward = Ward(
            beds=2,
            seed=42,
            triage={"poisson": {"rate_per_hour": 3600}},
        )
        for _ in range(10):
            ward.tick()
        first = ward.tick()["arrivals"]
        self.assertEqual(first["arrival_rate_per_hour"], 3600.0)
        self.assertEqual(
            first["arrivals_last_hour"], first["arrivals_total"]
        )
        for _ in range(590):
            ward.tick()
        snap = ward.tick()
        view = snap["arrivals"]
        # A seeded 600-tick run at 3600/hr is ~600 arrivals, all
        # inside the first ward hour, and the count grew as ticks
        # happened rather than staying at the first snapshot's.
        self.assertGreater(view["arrivals_last_hour"], first["arrivals_last_hour"])
        self.assertEqual(view["arrivals_last_hour"], view["arrivals_total"])
        self.assertGreater(view["arrivals_last_hour"], 500)
        self.assertLess(view["arrivals_last_hour"], 700)
        self.assertEqual(view["arrivals_by_hour"], {"8": view["arrivals_total"]})
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/arrivals")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(
                data["arrivals_last_hour"], view["arrivals_last_hour"]
            )
            self.assertEqual(data["diverted"], view["diverted"])
            self.assertEqual(data["diverting"], view["diverting"])
            ward.partitioned = True
            status, _ = get(port, "/arrivals")
            self.assertEqual(status, 503)
        finally:
            server.shutdown()
            server.server_close()

    def test_ws_frame_carries_arrivals_and_updates(self):
        # A fast walk-in rate, so the arrivals view in successive
        # frames grows visibly as the ward ticks.
        ward = Ward(
            beds=2,
            rate_hz=20.0,
            seed=42,
            triage={"poisson": {"rate_per_hour": 360000}},
        )
        server, port = serve_in_thread(ward)
        try:
            ws = WsClient(port)
            try:
                first = json.loads(ws.read_until_opcode(1))
                self.assertIn("arrivals", first)
                self.assertEqual(
                    first["arrivals"]["arrival_rate_per_hour"], 360000.0
                )
                later = first
                for _ in range(10):
                    later = json.loads(ws.read_until_opcode(1))
                self.assertGreater(
                    later["arrivals"]["arrivals_last_hour"],
                    first["arrivals"]["arrivals_last_hour"],
                )
                self.assertEqual(
                    later["arrivals"]["arrivals_last_hour"],
                    later["arrivals"]["arrivals_total"],
                )
            finally:
                ws.close()
        finally:
            server.shutdown()
            server.server_close()


class TestNEWS2(unittest.TestCase):
    def _signs(self, **over):
        signs = {
            "heart_rate": 70,
            "spo2": 98,
            "systolic": 120,
            "diastolic": 80,
            "temperature": 37.0,
            "resp_rate": 16,
        }
        signs.update(over)
        return signs

    def test_component_boundaries(self):
        # Respiratory rate: <=8: 3, 9-11: 1, 12-20: 0, 21-24: 2, >=25: 3.
        for rr, want in ((8, 3), (9, 1), (11, 1), (12, 0), (20, 0), (21, 2), (24, 2), (25, 3)):
            self.assertEqual(score_signs(self._signs(resp_rate=rr))["components"]["resp_rate"], want, rr)
        # SpO2 Scale 1: <=91: 3, 92-93: 2, 94-95: 1, >=96: 0.
        for spo2, want in ((91, 3), (92, 2), (93, 2), (94, 1), (95, 1), (96, 0)):
            self.assertEqual(score_signs(self._signs(spo2=spo2))["components"]["spo2"], want, spo2)
        # Systolic: <=90: 3, 91-100: 2, 101-110: 1, 111-219: 0, >=220: 3.
        for sys, want in ((90, 3), (91, 2), (100, 2), (101, 1), (110, 1), (111, 0), (219, 0), (220, 3)):
            self.assertEqual(score_signs(self._signs(systolic=sys))["components"]["systolic"], want, sys)
        # Heart rate: <=40: 3, 41-50: 1, 51-90: 0, 91-110: 1, 111-130: 2, >=131: 3.
        for hr, want in ((40, 3), (41, 1), (50, 1), (51, 0), (90, 0), (91, 1), (110, 1), (111, 2), (130, 2), (131, 3)):
            self.assertEqual(score_signs(self._signs(heart_rate=hr))["components"]["heart_rate"], want, hr)
        # Temperature: <=35.0: 3, 35.1-36.0: 1, 36.1-38.0: 0, 38.1-39.0: 1, >=39.1: 2.
        for temp, want in ((35.0, 3), (35.1, 1), (36.0, 1), (36.1, 0), (38.0, 0), (38.1, 1), (39.0, 1), (39.1, 2)):
            self.assertEqual(score_signs(self._signs(temperature=temp))["components"]["temperature"], want, temp)

    def test_aggregate_risk_and_red_score(self):
        normal = score_signs(self._signs())
        self.assertEqual(normal["score"], 0)
        self.assertEqual(normal["risk"], "low")
        self.assertFalse(normal["red_score"])
        self.assertIn("routine monitoring", normal["response"])
        # One parameter scoring 3 is a red score: medium risk even
        # when the aggregate is low.
        red = score_signs(self._signs(spo2=90))
        self.assertEqual(red["score"], 3)
        self.assertTrue(red["red_score"])
        self.assertEqual(red["risk"], "medium")
        self.assertIn("ward team", red["response"])
        # 5-6 without a red score is medium.
        medium = score_signs(self._signs(spo2=93, resp_rate=22, heart_rate=95))
        self.assertEqual(medium["score"], 5)
        self.assertFalse(medium["red_score"])
        self.assertEqual(medium["risk"], "medium")
        # 7 or more is high.
        high = score_signs(self._signs(spo2=90, resp_rate=22, heart_rate=95, temperature=38.5))
        self.assertEqual(high["score"], 7)
        self.assertEqual(high["risk"], "high")
        self.assertIn("critical care", high["response"])

    def test_reading_carries_news2(self):
        ward = Ward(beds=2, profiles=["stable", "hypoxic"], seed=42)
        snap = ward.tick()
        stable = snap["beds"][0]["news2"]
        self.assertEqual(stable["risk"], "low")
        self.assertFalse(stable["red_score"])
        hypoxic = snap["beds"][1]["news2"]
        self.assertEqual(hypoxic["risk"], "medium")
        self.assertTrue(hypoxic["red_score"])
        self.assertEqual(hypoxic["components"]["spo2"], 3)

    def test_news2_does_not_touch_the_alarm_list(self):
        # The score is a parallel track: a hypoxic patient's alarms
        # are only the ones alarms_for raises, never a NEWS2 alarm.
        known = {
            "tachycardia", "bradycardia", "low_spo2", "hypertension",
            "hypotension", "fever", "hypothermia", "tachypnea",
            "missed_dose",
        }
        ward = Ward(beds=1, profiles=["hypoxic"], seed=7)
        for _ in range(600):
            snap = ward.tick()
            self.assertTrue(set(snap["beds"][0]["alarms"]) <= known)

    def test_storm_pushes_patient_high_then_episode_clears(self):
        ward = Ward(
            beds=1,
            profiles=["febrile"],
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, bed=0)],
            seed=3,
        )
        snap = ward.tick()  # t=1: storm starts
        self.assertEqual(snap["beds"][0]["news2"]["risk"], "high")
        self.assertGreaterEqual(snap["beds"][0]["news2"]["score"], 7)
        for _ in range(39):
            ward.tick()
        m = ward.monitors[0]
        self.assertEqual(m.escalations_raised, 2)
        first = m.escalation_history[0]
        self.assertEqual(first["raised_t"], 1.0)
        self.assertEqual(first["cleared_t"], 14.0)
        self.assertEqual(first["peak_score"], 11)
        self.assertEqual(first["peak_risk"], "high")
        self.assertIsNone(m._escalation)

    def test_hypoxic_patient_stays_escalated(self):
        ward = Ward(beds=1, profiles=["hypoxic"], seed=3)
        for _ in range(5):
            snap = ward.tick()
        self.assertTrue(snap["beds"][0]["news2"]["red_score"])
        m = ward.monitors[0]
        self.assertIsNotNone(m._escalation)
        self.assertEqual(m._escalation["raised_t"], 1.0)
        # The red SpO2 score alone is 3 points, so the peak is at
        # least that; the episode raised at medium risk.
        self.assertGreaterEqual(m._escalation["peak_score"], 3)
        self.assertEqual(m._escalation["peak_risk"], "medium")

    def test_discharge_closes_open_episode(self):
        ward = Ward(beds=1, profiles=["hypoxic"], seed=3)
        for _ in range(5):
            ward.tick()
        self.assertIsNotNone(ward.monitors[0]._escalation)
        ward.discharge(0)
        m = ward.monitors[0]
        self.assertIsNone(m._escalation)
        self.assertEqual(m.escalation_history[0]["cleared_t"], 5.0)

    def test_no_score_while_offline_or_unadmitted(self):
        ward = Ward(
            beds=1,
            profiles=["hypoxic"],
            chaos=[ChaosEvent("dropout", at=1, duration=5, bed=0)],
            seed=3,
        )
        ward.tick()
        ward.tick()
        self.assertIsNone(
            ward.monitors[0].news2_detail(ward.elapsed)["news2"]
        )
        for _ in range(5):
            ward.tick()
        self.assertIsNotNone(
            ward.monitors[0].news2_detail(ward.elapsed)["news2"]
        )
        ward2 = Ward(
            beds=2,
            profiles=["stable", "stable"],
            timeline=Timeline([TimelinePhase(0, 1, ramp_s=0)]),
            seed=3,
        )
        ward2.tick()
        detail = ward2.monitors[1].news2_detail(ward2.elapsed)
        self.assertFalse(detail["admitted"])
        self.assertIsNone(detail["news2"])
        self.assertFalse(detail["escalated"])

    def test_news2_in_snapshot_endpoint_and_ws_frame(self):
        ward = Ward(beds=2, profiles=["stable", "hypoxic"], seed=42)
        ward.tick()
        snap = ward.snapshot()
        self.assertIn("early_warning", snap)
        self.assertEqual(snap["early_warning"]["medium_risk"], 1)
        self.assertEqual(snap["early_warning"]["low_risk"], 1)
        self.assertEqual(snap["early_warning"]["escalated"], 1)
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/news2")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["escalations_raised"], 1)
            self.assertEqual(data["beds"][1]["news2"]["risk"], "medium")
            ward.partitioned = True
            status, _ = get(port, "/news2")
            self.assertEqual(status, 503)
            ward.partitioned = False
        finally:
            server.shutdown()
            server.server_close()
        ward2 = Ward(beds=1, profiles=["hypoxic"], rate_hz=20.0, seed=42)
        server, port = serve_in_thread(ward2)
        try:
            ws = WsClient(port)
            try:
                msg = json.loads(ws.read_until_opcode(1))
                self.assertIn("early_warning", msg)
                self.assertEqual(msg["early_warning"]["escalated"], 1)
            finally:
                ws.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_scenario_news2(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2.json")
        )
        ward = build_ward(cfg, seed=42)
        for _ in range(40):
            snap = ward.tick()
        # Bed 0 stable: low, never escalates. Bed 1 hypoxic: medium
        # on a red SpO2 score, escalated from t=1. Bed 2
        # tachycardic: low. Bed 3 febrile: a medium episode on its
        # own at t=5-9, then a high one under the storm at t=10,
        # peaking at 11 and clearing at t=40.
        self.assertEqual(snap["beds"][0]["news2"]["risk"], "low")
        self.assertEqual(snap["beds"][1]["news2"]["risk"], "medium")
        self.assertTrue(snap["beds"][1]["news2"]["red_score"])
        self.assertEqual(snap["beds"][2]["news2"]["risk"], "low")
        summary = ward.early_warning_summary()
        self.assertEqual(summary["medium_risk"], 1)
        self.assertEqual(summary["low_risk"], 3)
        self.assertEqual(summary["escalated"], 1)
        self.assertEqual(summary["escalations_raised"], 3)
        bed3 = summary["beds"][3]
        self.assertEqual(bed3["escalations_raised"], 2)
        self.assertEqual(bed3["episodes"][0]["raised_t"], 5.0)
        self.assertEqual(bed3["episodes"][0]["peak_score"], 6)
        self.assertEqual(bed3["episodes"][1]["peak_score"], 11)
        self.assertEqual(bed3["episodes"][1]["peak_risk"], "high")
        self.assertEqual(bed3["episodes"][1]["cleared_t"], 40.0)

    def test_recorder_summary_carries_early_warning(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2.json")
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            summary = summarize(path)
            ew = summary["early_warning"]
            self.assertEqual(ew["medium_risk"], 1)
            self.assertEqual(ew["low_risk"], 3)
            self.assertEqual(ew["escalated"], 1)
            self.assertEqual(ew["escalations_raised"], 3)
        finally:
            os.unlink(path)


class TestOxygen(unittest.TestCase):
    def _signs(self, **over):
        signs = {
            "heart_rate": 70,
            "spo2": 98,
            "systolic": 120,
            "diastolic": 80,
            "temperature": 37.0,
            "resp_rate": 16,
        }
        signs.update(over)
        return signs

    def test_spo2_scale2_boundaries(self):
        # Scale 2 (on oxygen): <=83: 3, 84-85: 2, 86-87: 1, 88-92: 0,
        # 93-96: 2, >=97: 3. The target band is 88-92: above it is a
        # warning, not a reassurance.
        for spo2, want in (
            (83, 3), (84, 2), (85, 2), (86, 1), (87, 1),
            (88, 0), (92, 0), (93, 2), (96, 2), (97, 3),
        ):
            self.assertEqual(score_spo2(spo2, on_oxygen=True), want, spo2)
        # Scale 1 is unchanged: the same readings score differently.
        for spo2, want in ((91, 3), (94, 1), (96, 0), (97, 0)):
            self.assertEqual(score_spo2(spo2), want, spo2)
            self.assertEqual(score_spo2(spo2, on_oxygen=False), want, spo2)

    def test_score_signs_reports_spo2_scale(self):
        # SpO2 95 scores 1 on Scale 1 and 2 on Scale 2; 89 scores
        # 3 on Scale 1 and 0 on Scale 2 — the scale flips the
        # judgement, so the score says which table it used.
        on_air = score_signs(self._signs(spo2=95))
        self.assertEqual(on_air["components"]["spo2"], 1)
        self.assertEqual(on_air["spo2_scale"], 1)
        on_o2 = score_signs(self._signs(spo2=95), on_oxygen=True)
        self.assertEqual(on_o2["components"]["spo2"], 2)
        self.assertEqual(on_o2["spo2_scale"], 2)
        self.assertEqual(
            score_signs(self._signs(spo2=89))["components"]["spo2"], 3
        )
        self.assertEqual(
            score_signs(self._signs(spo2=89), on_oxygen=True)["components"]["spo2"],
            0,
        )
        # Everything else in the score is identical across scales.
        self.assertEqual(
            {k: v for k, v in on_air["components"].items() if k != "spo2"},
            {k: v for k, v in on_o2["components"].items() if k != "spo2"},
        )

    def test_reading_carries_on_oxygen_and_scale(self):
        ward = Ward(
            beds=2,
            profiles=["hypoxic", "hypoxic"],
            oxygen=[True, False],
            seed=42,
        )
        snap = ward.tick()
        on_o2, off_o2 = snap["beds"]
        self.assertTrue(on_o2["on_oxygen"])
        self.assertEqual(on_o2["news2"]["spo2_scale"], 2)
        self.assertFalse(off_o2["on_oxygen"])
        self.assertEqual(off_o2["news2"]["spo2_scale"], 1)
        detail = ward.monitors[0].news2_detail(ward.elapsed)
        self.assertTrue(detail["on_oxygen"])
        summary = ward.early_warning_summary()
        self.assertEqual(summary["on_oxygen"], 1)

    def test_oxygen_is_scoring_only(self):
        # Oxygen therapy changes the score, never the vitals or the
        # alarms: two wards with the same seed and profiles emit
        # identical signs and alarm lists, oxygen flag aside.
        plain = Ward(beds=2, profiles=["stable", "hypoxic"], seed=42)
        oxygenated = Ward(
            beds=2,
            profiles=["stable", "hypoxic"],
            oxygen=[True, True],
            seed=42,
        )
        for _ in range(40):
            a = plain.tick()
            b = oxygenated.tick()
            for ra, rb in zip(a["beds"], b["beds"]):
                for key in (
                    "heart_rate", "spo2", "systolic", "diastolic",
                    "temperature", "resp_rate",
                ):
                    self.assertEqual(ra[key], rb[key], key)
                self.assertEqual(ra["alarms"], rb["alarms"])

    def test_admit_resets_oxygen(self):
        ward = Ward(beds=1, profiles=["stable"], oxygen=[True], seed=42)
        ward.tick()
        self.assertTrue(ward.monitors[0].on_oxygen)
        ward.discharge(0)
        for _ in range(31):
            ward.tick()
        ward.admit(0)
        self.assertFalse(ward.monitors[0].on_oxygen)
        ward.discharge(0)
        for _ in range(31):
            ward.tick()
        ward.admit(0, on_oxygen=True)
        self.assertTrue(ward.monitors[0].on_oxygen)
        self.assertTrue(ward.tick()["beds"][0]["on_oxygen"])

    def test_oxygen_validation(self):
        with self.assertRaises(ValueError):
            Monitor(0, on_oxygen="yes")
        with self.assertRaises(ValueError):
            Ward(beds=1, oxygen=["yes"])
        with self.assertRaises(ValueError):
            Ward(beds=1, oxygen=[1])
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "oxygen.json")
        )
        cfg["profiles"] = ["stable"]
        cfg["oxygen"] = [1]
        with self.assertRaises(ValueError):
            build_ward(cfg, seed=42)

    def test_scenario_oxygen(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "oxygen.json")
        )
        ward = build_ward(cfg, seed=42)
        snap = ward.tick()
        # t=1: bed 0 stable on oxygen at SpO2 95.12 scores 2 on
        # Scale 2 (93-96 on oxygen) and stays low; bed 1 hypoxic
        # on oxygen at 88.08 is inside the 88-92 target band and
        # scores 0 on SpO2; bed 2 hypoxic off oxygen at 88.7
        # scores 3 on Scale 1 and is medium; bed 3 stable scores 0.
        self.assertEqual(snap["beds"][0]["news2"]["components"]["spo2"], 2)
        self.assertEqual(snap["beds"][0]["news2"]["risk"], "low")
        self.assertEqual(snap["beds"][1]["news2"]["components"]["spo2"], 0)
        self.assertEqual(snap["beds"][1]["news2"]["score"], 4)
        self.assertEqual(snap["beds"][2]["news2"]["components"]["spo2"], 3)
        self.assertEqual(snap["beds"][2]["news2"]["risk"], "medium")
        self.assertEqual(snap["beds"][3]["news2"]["score"], 0)
        for _ in range(39):
            snap = ward.tick()
        # Over 40 ticks the patient on oxygen drifts above the
        # target band and escalates twice, the hypoxic patient on
        # oxygen never escalates, and the hypoxic patient off
        # oxygen stays escalated from t=1.
        summary = ward.early_warning_summary()
        self.assertEqual(summary["on_oxygen"], 2)
        self.assertEqual(summary["medium_risk"], 2)
        self.assertEqual(summary["low_risk"], 2)
        self.assertEqual(summary["escalated"], 2)
        self.assertEqual(summary["escalations_raised"], 3)
        self.assertEqual(summary["beds"][0]["escalations_raised"], 2)
        self.assertEqual(summary["beds"][1]["escalations_raised"], 0)
        self.assertEqual(summary["beds"][2]["escalations_raised"], 1)

class TestACVPU(unittest.TestCase):
    def _signs(self, **over):
        signs = {
            "heart_rate": 70,
            "spo2": 98,
            "systolic": 120,
            "diastolic": 80,
            "temperature": 37.0,
            "resp_rate": 16,
        }
        signs.update(over)
        return signs

    def test_acvpu_scoring(self):
        # Alert scores 0; Confusion, Voice, Pain, Unresponsive
        # each score 3 — NEWS2's sixth parameter is all or nothing.
        self.assertEqual(score_consciousness("alert"), 0)
        for level in ("confusion", "voice", "pain", "unresponsive"):
            self.assertEqual(score_consciousness(level), 3, level)
        for bad in ("asleep", "ALERT", "Alert", "", None, 0, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_consciousness(bad)

    def test_score_signs_includes_consciousness(self):
        # A normal set of vitals with an Alert patient scores 0
        # and stays low; the same vitals with a patient responding
        # only to voice score 3 on consciousness alone, which is a
        # red score and makes the risk medium.
        normal = score_signs(self._signs())
        self.assertEqual(normal["components"]["consciousness"], 0)
        self.assertEqual(normal["score"], 0)
        self.assertEqual(normal["risk"], "low")
        voice = score_signs(self._signs(), consciousness="voice")
        self.assertEqual(voice["components"]["consciousness"], 3)
        self.assertEqual(voice["score"], 3)
        self.assertTrue(voice["red_score"])
        self.assertEqual(voice["risk"], "medium")
        self.assertIn("ward team", voice["response"])
        # The score now has six components, one per NEWS2 parameter.
        self.assertEqual(
            set(normal["components"]),
            {"resp_rate", "spo2", "systolic", "heart_rate",
             "temperature", "consciousness"},
        )
        # Consciousness stacks with the vitals: a hypoxic patient
        # who is also confused is high, not just medium.
        high = score_signs(
            self._signs(spo2=90), consciousness="confusion"
        )
        self.assertEqual(high["score"], 6)
        self.assertEqual(high["risk"], "medium")
        higher = score_signs(
            self._signs(spo2=90, resp_rate=22), consciousness="pain"
        )
        self.assertEqual(higher["score"], 8)
        self.assertEqual(higher["risk"], "high")

    def test_reading_carries_consciousness(self):
        ward = Ward(
            beds=2,
            profiles=["stable", "stable"],
            consciousness=["alert", "voice"],
            seed=42,
        )
        snap = ward.tick()
        alert, voice = snap["beds"]
        self.assertEqual(alert["consciousness"], "alert")
        self.assertEqual(alert["news2"]["components"]["consciousness"], 0)
        self.assertEqual(voice["consciousness"], "voice")
        self.assertEqual(voice["news2"]["components"]["consciousness"], 3)
        self.assertTrue(voice["news2"]["red_score"])
        # No consciousness named: Alert, the same as before.
        plain = Ward(beds=1, profiles=["stable"], seed=42)
        self.assertEqual(plain.tick()["beds"][0]["consciousness"], "alert")
        # The early-warning summary carries the level per bed and
        # counts the patients who are not Alert.
        summary = ward.early_warning_summary()
        self.assertEqual(summary["beds"][1]["consciousness"], "voice")
        self.assertEqual(summary["not_alert"], 1)

    def test_consciousness_is_scoring_only(self):
        # Consciousness changes the score, never the vitals or the
        # alarms: two wards with the same seed and profiles emit
        # identical signs and alarm lists, consciousness aside.
        plain = Ward(beds=2, profiles=["stable", "hypoxic"], seed=42)
        altered = Ward(
            beds=2,
            profiles=["stable", "hypoxic"],
            consciousness=["pain", "unresponsive"],
            seed=42,
        )
        for _ in range(40):
            a = plain.tick()
            b = altered.tick()
            for ra, rb in zip(a["beds"], b["beds"]):
                for key in (
                    "heart_rate", "spo2", "systolic", "diastolic",
                    "temperature", "resp_rate",
                ):
                    self.assertEqual(ra[key], rb[key], key)
                self.assertEqual(ra["alarms"], rb["alarms"])

    def test_admit_resets_consciousness(self):
        ward = Ward(
            beds=1, profiles=["stable"], consciousness=["pain"], seed=42
        )
        ward.tick()
        self.assertEqual(ward.monitors[0].consciousness, "pain")
        ward.discharge(0)
        for _ in range(31):
            ward.tick()
        ward.admit(0)
        self.assertEqual(ward.monitors[0].consciousness, "alert")
        self.assertEqual(ward.tick()["beds"][0]["consciousness"], "alert")
        ward.discharge(0)
        for _ in range(31):
            ward.tick()
        ward.admit(0, consciousness="voice")
        self.assertEqual(ward.monitors[0].consciousness, "voice")
        self.assertEqual(ward.tick()["beds"][0]["consciousness"], "voice")
        self.assertEqual(
            ward.tick()["beds"][0]["news2"]["components"]["consciousness"],
            3,
        )

    def test_consciousness_validation(self):
        with self.assertRaises(ValueError):
            Monitor(0, consciousness="asleep")
        with self.assertRaises(ValueError):
            Ward(beds=1, consciousness=["asleep"])
        with self.assertRaises(ValueError):
            Ward(beds=1, consciousness=[None])
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2-acvpu.json")
        )
        cfg["consciousness"] = ["asleep", "alert", "alert", "alert"]
        with self.assertRaises(ValueError):
            build_ward(cfg, seed=42)

    def test_scenario_news2_acvpu(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2-acvpu.json")
        )
        ward = build_ward(cfg, seed=42)
        snap = ward.tick()
        # t=1: bed 0 stable and Alert scores 0 and stays low.
        # Bed 1 is stable but responds only to voice: the
        # consciousness red score alone makes it medium. Bed 2 is
        # hypoxic and confused: SpO2 3 plus consciousness 3, high.
        # Bed 3 is febrile and unresponsive: high on consciousness
        # plus a febrile heart rate and temperature.
        self.assertEqual(snap["beds"][0]["news2"]["score"], 0)
        self.assertEqual(snap["beds"][0]["news2"]["risk"], "low")
        self.assertEqual(
            snap["beds"][1]["news2"]["components"]["consciousness"], 3
        )
        self.assertTrue(snap["beds"][1]["news2"]["red_score"])
        self.assertEqual(snap["beds"][1]["news2"]["risk"], "medium")
        self.assertEqual(snap["beds"][2]["news2"]["score"], 8)
        self.assertEqual(snap["beds"][2]["news2"]["risk"], "high")
        self.assertEqual(snap["beds"][3]["news2"]["score"], 7)
        self.assertEqual(snap["beds"][3]["news2"]["risk"], "high")
        for _ in range(39):
            snap = ward.tick()
        # Consciousness does not change over a run, so all three
        # non-Alert patients stay escalated from t=1; the two high
        # ones peak at 9.
        summary = ward.early_warning_summary()
        self.assertEqual(summary["not_alert"], 3)
        self.assertEqual(summary["high_risk"], 2)
        self.assertEqual(summary["medium_risk"], 1)
        self.assertEqual(summary["low_risk"], 1)
        self.assertEqual(summary["escalated"], 3)
        self.assertEqual(summary["escalations_raised"], 3)
        self.assertEqual(summary["beds"][0]["escalations_raised"], 0)
        self.assertEqual(summary["beds"][1]["escalations_raised"], 1)
        self.assertEqual(summary["beds"][1]["episodes"][0]["peak_score"], 4)
        self.assertEqual(summary["beds"][1]["episodes"][0]["peak_risk"], "medium")
        self.assertEqual(summary["beds"][2]["episodes"][0]["peak_score"], 9)
        self.assertEqual(summary["beds"][2]["episodes"][0]["peak_risk"], "high")
        self.assertEqual(summary["beds"][3]["episodes"][0]["peak_score"], 9)
        self.assertEqual(summary["beds"][3]["episodes"][0]["peak_risk"], "high")

    def test_recorder_summary_carries_not_alert(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2-acvpu.json")
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            summary = summarize(path)
            ew = summary["early_warning"]
            self.assertEqual(ew["not_alert"], 3)
            self.assertEqual(ew["high_risk"], 2)
            self.assertEqual(ew["medium_risk"], 1)
            self.assertEqual(ew["low_risk"], 1)
            self.assertEqual(ew["escalated"], 3)
            self.assertEqual(ew["escalations_raised"], 3)
        finally:
            os.unlink(path)

    def test_news2_endpoint_and_ws_frame_carry_consciousness(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2-acvpu.json")
        )
        ward = build_ward(cfg, seed=42)
        ward.tick()
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/news2")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["not_alert"], 3)
            self.assertEqual(data["beds"][1]["consciousness"], "voice")
            self.assertEqual(
                data["beds"][1]["news2"]["components"]["consciousness"], 3
            )
            self.assertEqual(data["beds"][3]["consciousness"], "unresponsive")
            ward.partitioned = True
            status, _ = get(port, "/news2")
            self.assertEqual(status, 503)
            ward.partitioned = False
        finally:
            server.shutdown()
            server.server_close()
        ward2 = Ward(
            beds=1,
            profiles=["stable"],
            consciousness=["pain"],
            rate_hz=20.0,
            seed=42,
        )
        server, port = serve_in_thread(ward2)
        try:
            ws = WsClient(port)
            try:
                msg = json.loads(ws.read_until_opcode(1))
                self.assertIn("early_warning", msg)
                self.assertEqual(msg["early_warning"]["not_alert"], 1)
                self.assertEqual(
                    msg["beds"][0]["news2"]["components"]["consciousness"],
                    3,
                )
            finally:
                ws.close()
        finally:
            server.shutdown()
            server.server_close()


class TestConsciousnessDynamics(unittest.TestCase):
    def test_dynamics_validation(self):
        # Off, defaults, and partial overrides.
        self.assertIsNone(validate_consciousness_dynamics(None))
        self.assertEqual(
            validate_consciousness_dynamics(True),
            {"deteriorate_after_s": 5.0, "recover_after_s": 10.0},
        )
        self.assertEqual(
            validate_consciousness_dynamics({"deteriorate_after_s": 3}),
            {"deteriorate_after_s": 3.0, "recover_after_s": 10.0},
        )
        self.assertEqual(
            validate_consciousness_dynamics(
                {"deteriorate_after_s": 2, "recover_after_s": 4}
            ),
            {"deteriorate_after_s": 2.0, "recover_after_s": 4.0},
        )
        for bad in (
            "fast",
            5,
            0,
            False,
            {"deteriorate_after_s": 0},
            {"deteriorate_after_s": -1},
            {"recover_after_s": "soon"},
            {"recover_after_s": True},
            {"recover_after_s": None},
            {"deteriorate_after_s": 3, "wake_up_after_s": 1},
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_consciousness_dynamics(bad)
        with self.assertRaises(ValueError):
            Monitor(0, consciousness_dynamics={"deteriorate_after_s": 0})
        with self.assertRaises(ValueError):
            Ward(
                beds=1,
                consciousness_dynamics=[{"recover_after_s": "soon"}],
            )
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2-acvpu-dynamics.json")
        )
        cfg["consciousness_dynamics"] = [
            {"deteriorate_after_s": -1},
            None,
            None,
            None,
        ]
        with self.assertRaises(ValueError):
            build_ward(cfg, seed=42)

    def test_recovery_steps_up_on_a_clock(self):
        # A stable patient's vitals-only risk is always low, so a
        # patient responding only to pain climbs back one level
        # every 2 s: pain at t=1, voice at t=2, confusion at t=4,
        # Alert at t=6 — and the episode closes when they get there.
        ward = Ward(
            beds=1,
            profiles=["stable"],
            consciousness=["pain"],
            consciousness_dynamics=[
                {"deteriorate_after_s": 2, "recover_after_s": 2}
            ],
            seed=42,
        )
        levels = [ward.tick()["beds"][0]["consciousness"] for _ in range(7)]
        self.assertEqual(
            levels,
            ["pain", "voice", "voice", "confusion", "confusion",
             "alert", "alert"],
        )
        detail = ward.early_warning_summary()["beds"][0]
        self.assertEqual(
            detail["consciousness_history"],
            [
                {"from": "pain", "to": "voice", "t": 2.0},
                {"from": "voice", "to": "confusion", "t": 4.0},
                {"from": "confusion", "to": "alert", "t": 6.0},
            ],
        )
        self.assertEqual(detail["consciousness_changes"], 3)
        self.assertFalse(detail["escalated"])
        self.assertEqual(detail["episodes"][0]["cleared_t"], 6.0)
        # The score follows the level: Alert again at t=6, so the
        # consciousness component is 0 and the risk is low.
        snap = ward.tick()
        self.assertEqual(
            snap["beds"][0]["news2"]["components"]["consciousness"], 0
        )
        self.assertEqual(snap["beds"][0]["news2"]["risk"], "low")

    def test_deterioration_steps_down_while_vitals_stay_bad(self):
        # A hypoxic patient's vitals-only risk sits at medium, so
        # they slide one ACVPU level worse every 2 s until there is
        # nowhere worse to go — and then stay Unresponsive.
        ward = Ward(
            beds=1,
            profiles=["hypoxic"],
            consciousness_dynamics=[
                {"deteriorate_after_s": 2, "recover_after_s": 2}
            ],
            seed=42,
        )
        for _ in range(20):
            snap = ward.tick()
        self.assertEqual(snap["beds"][0]["consciousness"], "unresponsive")
        self.assertEqual(
            snap["beds"][0]["news2"]["components"]["consciousness"], 3
        )
        self.assertEqual(snap["beds"][0]["news2"]["risk"], "high")
        detail = ward.early_warning_summary()["beds"][0]
        history = detail["consciousness_history"]
        self.assertEqual(len(history), 4)
        self.assertEqual(
            [(h["from"], h["to"]) for h in history],
            [
                ("alert", "confusion"),
                ("confusion", "voice"),
                ("voice", "pain"),
                ("pain", "unresponsive"),
            ],
        )
        self.assertEqual(
            ward.early_warning_summary()["consciousness_changes"], 4
        )
        self.assertEqual(ward.early_warning_summary()["not_alert"], 1)

    def test_dynamics_never_change_the_vitals(self):
        # Dynamics are deterministic and RNG-free: two wards with
        # the same seed emit identical vitals and alarms whether
        # dynamics are on or off — only the consciousness level and
        # the score differ.
        plain = Ward(beds=2, profiles=["stable", "hypoxic"], seed=42)
        dynamic = Ward(
            beds=2,
            profiles=["stable", "hypoxic"],
            consciousness_dynamics=[
                {"deteriorate_after_s": 2, "recover_after_s": 2},
                {"deteriorate_after_s": 2, "recover_after_s": 2},
            ],
            seed=42,
        )
        for _ in range(40):
            a = plain.tick()
            b = dynamic.tick()
            for ra, rb in zip(a["beds"], b["beds"]):
                for key in (
                    "heart_rate", "spo2", "systolic", "diastolic",
                    "temperature", "resp_rate",
                ):
                    self.assertEqual(ra[key], rb[key], key)
                self.assertEqual(ra["alarms"], rb["alarms"])
        # ...while the dynamics ward's hypoxic bed did deteriorate.
        self.assertEqual(
            b["beds"][1]["consciousness"], "unresponsive"
        )
        self.assertEqual(
            a["beds"][1]["consciousness"], "alert"
        )

    def test_dynamics_off_by_default(self):
        # No dynamics configured: a hypoxic patient stays Alert for
        # a whole run, exactly as before.
        ward = Ward(beds=1, profiles=["hypoxic"], seed=42)
        for _ in range(40):
            snap = ward.tick()
        self.assertEqual(snap["beds"][0]["consciousness"], "alert")
        self.assertEqual(
            ward.early_warning_summary()["consciousness_changes"], 0
        )
        # Dynamics configured with thresholds longer than the run
        # also never move the level.
        slow = Ward(
            beds=1,
            profiles=["hypoxic"],
            consciousness_dynamics=[
                {"deteriorate_after_s": 3600, "recover_after_s": 3600}
            ],
            seed=42,
        )
        for _ in range(40):
            snap = slow.tick()
        self.assertEqual(snap["beds"][0]["consciousness"], "alert")

    def test_admit_resets_consciousness_history(self):
        ward = Ward(
            beds=1,
            profiles=["hypoxic"],
            consciousness_dynamics=[
                {"deteriorate_after_s": 2, "recover_after_s": 2}
            ],
            seed=42,
        )
        for _ in range(20):
            ward.tick()
        self.assertEqual(
            ward.monitors[0].consciousness, "unresponsive"
        )
        self.assertEqual(len(ward.monitors[0].consciousness_history), 4)
        ward.discharge(0)
        for _ in range(31):
            ward.tick()
        ward.admit(0)
        self.assertEqual(ward.monitors[0].consciousness, "alert")
        self.assertEqual(ward.monitors[0].consciousness_history, [])
        # The dynamics config is the bed's scenario setting, so it
        # still applies to the new patient.
        for _ in range(20):
            ward.tick()
        self.assertEqual(
            ward.monitors[0].consciousness, "unresponsive"
        )

    def test_scenario_news2_acvpu_dynamics(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios", "news2-acvpu-dynamics.json"
            )
        )
        ward = build_ward(cfg, seed=42)
        snap = ward.tick()
        # t=1: bed 0 stable and Alert scores 0; bed 1 hypoxic and
        # Alert is already high on vitals alone; bed 2 febrile and
        # responding to voice is medium; bed 3 stable and confused
        # is medium on the consciousness red score alone.
        self.assertEqual(snap["beds"][0]["news2"]["score"], 0)
        self.assertEqual(snap["beds"][1]["news2"]["score"], 7)
        self.assertEqual(snap["beds"][1]["news2"]["risk"], "high")
        self.assertEqual(snap["beds"][2]["news2"]["score"], 6)
        self.assertEqual(snap["beds"][2]["news2"]["risk"], "medium")
        self.assertEqual(snap["beds"][3]["news2"]["score"], 3)
        self.assertEqual(snap["beds"][3]["news2"]["risk"], "medium")
        for _ in range(39):
            snap = ward.tick()
        summary = ward.early_warning_summary()
        # Bed 1 slid Alert -> confusion -> voice -> pain ->
        # unresponsive at 3 s intervals and stayed high; bed 2
        # recovered voice -> confusion -> Alert and its first
        # episode closed at 6 s; bed 3 recovered confusion -> Alert
        # at 3 s. Only bed 1 is not Alert at the end.
        self.assertEqual(summary["consciousness_changes"], 7)
        self.assertEqual(summary["not_alert"], 1)
        self.assertEqual(summary["high_risk"], 1)
        self.assertEqual(summary["medium_risk"], 1)
        self.assertEqual(summary["low_risk"], 2)
        self.assertEqual(summary["escalated"], 2)
        self.assertEqual(summary["escalations_raised"], 5)
        bed1 = summary["beds"][1]
        self.assertEqual(bed1["consciousness"], "unresponsive")
        self.assertEqual(
            [h["t"] for h in bed1["consciousness_history"]],
            [3.0, 6.0, 9.0, 12.0],
        )
        self.assertEqual(bed1["episodes"][0]["peak_score"], 9)
        self.assertIsNone(bed1["episodes"][0]["cleared_t"])
        bed2 = summary["beds"][2]
        self.assertEqual(bed2["consciousness"], "alert")
        self.assertEqual(
            [h["t"] for h in bed2["consciousness_history"]],
            [3.0, 6.0],
        )
        self.assertEqual(bed2["episodes"][0]["cleared_t"], 6.0)
        self.assertEqual(bed2["episodes"][0]["peak_score"], 6)
        bed3 = summary["beds"][3]
        self.assertEqual(bed3["consciousness"], "alert")
        self.assertEqual(
            bed3["consciousness_history"],
            [{"from": "confusion", "to": "alert", "t": 3.0}],
        )
        self.assertEqual(bed3["episodes"][0]["cleared_t"], 3.0)

    def test_recorder_summary_carries_consciousness_changes(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios", "news2-acvpu-dynamics.json"
            )
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            summary = summarize(path)
            ew = summary["early_warning"]
            self.assertEqual(ew["consciousness_changes"], 7)
            self.assertEqual(ew["not_alert"], 1)
            self.assertEqual(ew["escalations_raised"], 5)
        finally:
            os.unlink(path)

    def test_news2_endpoint_and_ws_frame_carry_changing_consciousness(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios", "news2-acvpu-dynamics.json"
            )
        )
        ward = build_ward(cfg, seed=42)
        for _ in range(40):
            ward.tick()
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/news2")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["consciousness_changes"], 7)
            self.assertEqual(data["not_alert"], 1)
            self.assertEqual(
                data["beds"][1]["consciousness"], "unresponsive"
            )
            self.assertEqual(
                data["beds"][1]["consciousness_changes"], 4
            )
            self.assertEqual(
                data["beds"][3]["consciousness_history"],
                [{"from": "confusion", "to": "alert", "t": 3.0}],
            )
            ward.partitioned = True
            status, _ = get(port, "/news2")
            self.assertEqual(status, 503)
            ward.partitioned = False
        finally:
            server.shutdown()
            server.server_close()
        ward2 = Ward(
            beds=1,
            profiles=["hypoxic"],
            consciousness_dynamics=[
                {"deteriorate_after_s": 0.2, "recover_after_s": 0.2}
            ],
            rate_hz=20.0,
            seed=42,
        )
        server, port = serve_in_thread(ward2)
        try:
            ws = WsClient(port)
            try:
                seen_unresponsive = False
                for _ in range(200):
                    msg = json.loads(ws.read_until_opcode(1))
                    if msg["beds"][0]["consciousness"] == "unresponsive":
                        seen_unresponsive = True
                        self.assertEqual(
                            msg["early_warning"]["consciousness_changes"],
                            4,
                        )
                        self.assertEqual(
                            msg["beds"][0]["news2"]["components"][
                                "consciousness"
                            ],
                            3,
                        )
                        break
                self.assertTrue(seen_unresponsive)
            finally:
                ws.close()
        finally:
            server.shutdown()
            server.server_close()


class TestNEWS2History(unittest.TestCase):
    def test_monitor_records_one_entry_per_fresh_tick(self):
        ward = Ward(beds=2, profiles=["stable", "hypoxic"], seed=42)
        snaps = [ward.tick() for _ in range(5)]
        for i, m in enumerate(ward.monitors):
            self.assertEqual(len(m.news2_history), 5)
            for entry, snap in zip(m.news2_history, snaps):
                self.assertEqual(entry["t"], snap["t"])
                self.assertEqual(
                    entry["score"], snap["beds"][i]["news2"]["score"]
                )
                self.assertEqual(
                    entry["risk"], snap["beds"][i]["news2"]["risk"]
                )
                self.assertEqual(
                    entry["components"],
                    snap["beds"][i]["news2"]["components"],
                )
                self.assertEqual(entry["consciousness"], "alert")
            detail = m.news2_history_detail(ward.elapsed)
            self.assertEqual(detail["points"], 5)
            self.assertEqual(detail["current"], dict(m.news2_history[-1]))
            self.assertEqual(
                detail["peak_score"],
                max(e["score"] for e in m.news2_history),
            )

    def test_stale_tick_does_not_duplicate(self):
        m = Monitor(0, "stable", rng=random.Random(1))
        m.tick(1.0)
        self.assertEqual(len(m.news2_history), 1)
        m.stale_until = 10.0
        stale = m.tick(2.0)
        self.assertTrue(stale["stale"])
        self.assertEqual(len(m.news2_history), 1)
        self.assertEqual(m.news2_history[0]["t"], 1.0)

    def test_history_capped_at_1000(self):
        m = Monitor(0, "stable", rng=random.Random(1))
        for i in range(1, 1006):
            m.tick(float(i))
        self.assertEqual(len(m.news2_history), 1000)
        self.assertEqual(m.news2_history[0]["t"], 6.0)
        self.assertEqual(m.news2_history[-1]["t"], 1005.0)

    def test_new_admission_clears_history(self):
        ward = Ward(beds=1, profiles=["stable"], cleaning_s=1.0, seed=1)
        for _ in range(3):
            ward.tick()
        self.assertEqual(len(ward.monitors[0].news2_history), 3)
        ward.discharge(0)
        ward.tick()  # cleaning finishes at t=4
        ward.admit(0, profile="stable")
        self.assertEqual(len(ward.monitors[0].news2_history), 0)
        ward.tick()
        self.assertEqual(len(ward.monitors[0].news2_history), 1)

    def test_ward_history_summary_single_and_all_beds(self):
        ward = Ward(beds=2, profiles=["stable", "hypoxic"], seed=42)
        for _ in range(3):
            ward.tick()
        all_beds = ward.news2_history_summary()
        self.assertEqual(all_beds["t"], 3.0)
        self.assertEqual(len(all_beds["beds"]), 2)
        self.assertEqual(all_beds["beds"][0]["points"], 3)
        one = ward.news2_history_summary(1)
        self.assertEqual(one["bed"], 1)
        self.assertEqual(one["points"], 3)
        self.assertEqual(one["peak_risk"], "medium")
        self.assertEqual(
            one["current"]["score"],
            ward.monitors[1].last_reading["news2"]["score"],
        )
        with self.assertRaises(ValueError):
            ward.news2_history_summary(99)

    def test_history_endpoint(self):
        ward = Ward(beds=2, profiles=["stable", "hypoxic"], seed=42)
        for _ in range(3):
            ward.tick()
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/news2/history")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(len(data["beds"]), 2)
            self.assertEqual(data["beds"][0]["points"], 3)
            status, body = get(port, "/news2/history?bed=1")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual(data["bed"], 1)
            self.assertEqual(data["points"], 3)
            self.assertEqual(data["peak_risk"], "medium")
            status, _ = get(port, "/news2/history?bed=99")
            self.assertEqual(status, 404)
            status, _ = get(port, "/news2/history?bed=x")
            self.assertEqual(status, 400)
            ward.partitioned = True
            status, _ = get(port, "/news2/history")
            self.assertEqual(status, 503)
            status, _ = get(port, "/news2/history?bed=0")
            self.assertEqual(status, 503)
            ward.partitioned = False
        finally:
            server.shutdown()
            server.server_close()

    def test_scenario_news2_history(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2-history.json")
        )
        ward = build_ward(cfg, seed=42)
        for _ in range(40):
            ward.tick()
        history = ward.news2_history_summary()
        bed0, bed1, bed2 = history["beds"]
        # Stable bed: low throughout, never above 1.
        self.assertEqual(bed0["points"], 40)
        self.assertEqual(bed0["peak_score"], 1)
        self.assertEqual(bed0["peak_risk"], "low")
        # Hypoxic bed: medium on a red SpO2 score, drifting high.
        self.assertEqual(bed1["points"], 40)
        self.assertEqual(bed1["peak_score"], 7)
        self.assertEqual(bed1["peak_risk"], "high")
        # Febrile bed: medium before the storm, 11 under it, low
        # again by the end — the history shows the whole arc.
        self.assertEqual(bed2["points"], 40)
        self.assertEqual(bed2["history"][0]["score"], 5)
        self.assertEqual(bed2["history"][0]["risk"], "medium")
        self.assertEqual(bed2["peak_score"], 11)
        self.assertEqual(bed2["peak_risk"], "high")
        self.assertEqual(bed2["history"][-1]["risk"], "low")

    def test_recorder_summary_carries_news2_history(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2-history.json")
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            summary = summarize(path)
            hist = summary["news2_history"]
            self.assertEqual(hist[0]["ticks"], 40)
            self.assertEqual(hist[0]["peak_score"], 1)
            self.assertEqual(hist[0]["ticks_low"], 40)
            self.assertEqual(hist[1]["peak_score"], 7)
            self.assertEqual(hist[1]["peak_risk"], "high")
            self.assertEqual(hist[1]["ticks_high"], 6)
            self.assertEqual(hist[2]["peak_score"], 11)
            self.assertEqual(hist[2]["peak_risk"], "high")
            self.assertEqual(hist[2]["ticks_high"], 10)
            self.assertEqual(hist[2]["ticks_low"], 17)
            self.assertEqual(hist[2]["first_t"], 1.0)
            self.assertEqual(hist[2]["last_t"], 40.0)
        finally:
            os.unlink(path)


class TestNEWS2Trend(unittest.TestCase):
    def _entries(self, scores, t0=1.0):
        return [
            {"t": t0 + i, "score": s} for i, s in enumerate(scores)
        ]

    def test_trend_for_jump_rule(self):
        # A rise of 2 inside the window is a jump, measured from
        # the earliest entry in the window.
        trend = trend_for(self._entries([3, 3, 5]), window_s=60.0, delta=2)
        self.assertTrue(trend["rising"])
        self.assertEqual(trend["rule"], "jump")
        self.assertEqual(trend["start_t"], 1.0)
        self.assertEqual(trend["start_score"], 3)
        self.assertEqual(trend["current_score"], 5)
        self.assertEqual(trend["delta"], 2)
        self.assertEqual(trend["window_s"], 60.0)
        # A rise of 1 is not a jump.
        trend = trend_for(self._entries([3, 3, 4]), window_s=60.0, delta=2)
        self.assertFalse(trend["rising"])
        self.assertIsNone(trend["rule"])
        self.assertEqual(trend["delta"], 1)

    def test_trend_for_creep_rule(self):
        # A slow climb — a point every couple of ticks, never
        # jumping — is a creep, when the window is too short for
        # the jump rule to see it.
        trend = trend_for(
            self._entries([3, 3, 4, 4, 5]),
            window_s=2.0,
            delta=5,
            trend_ticks=5,
            min_rise=2,
        )
        self.assertTrue(trend["rising"])
        self.assertEqual(trend["rule"], "creep")
        self.assertEqual(trend["start_t"], 1.0)
        self.assertEqual(trend["start_score"], 3)
        self.assertEqual(trend["delta"], 2)
        # A one-point wobble does not count as a trend.
        trend = trend_for(
            self._entries([0, 1, 0, 1, 0]),
            window_s=2.0,
            delta=5,
            trend_ticks=5,
            min_rise=2,
        )
        self.assertFalse(trend["rising"])
        # More falls than rises is not a creep either.
        trend = trend_for(
            self._entries([5, 4, 5, 4, 3]),
            window_s=2.0,
            delta=5,
            trend_ticks=5,
            min_rise=2,
        )
        self.assertFalse(trend["rising"])

    def test_trend_for_flat_falling_and_empty(self):
        flat = trend_for(self._entries([4, 4, 4, 4]))
        self.assertFalse(flat["rising"])
        self.assertEqual(flat["delta"], 0)
        falling = trend_for(self._entries([5, 4, 3]))
        self.assertFalse(falling["rising"])
        self.assertEqual(falling["delta"], -2)
        empty = trend_for([])
        self.assertFalse(empty["rising"])
        self.assertIsNone(empty["rule"])
        self.assertIsNone(empty["start_t"])
        self.assertIsNone(empty["start_score"])
        self.assertIsNone(empty["current_score"])

    def test_validate_news2_trend(self):
        defaults = validate_news2_trend(None)
        self.assertEqual(defaults["window_s"], 60.0)
        self.assertEqual(defaults["delta"], 2)
        self.assertEqual(defaults["trend_ticks"], 10)
        self.assertEqual(defaults["min_rise"], 2)
        custom = validate_news2_trend({"window_s": 10, "delta": 3})
        self.assertEqual(custom["window_s"], 10.0)
        self.assertEqual(custom["delta"], 3)
        self.assertEqual(custom["trend_ticks"], 10)
        with self.assertRaises(ValueError):
            validate_news2_trend({"window_s": 0})
        with self.assertRaises(ValueError):
            validate_news2_trend({"window_s": "fast"})
        with self.assertRaises(ValueError):
            validate_news2_trend({"delta": 0})
        with self.assertRaises(ValueError):
            validate_news2_trend({"delta": 1.5})
        with self.assertRaises(ValueError):
            validate_news2_trend({"trend_ticks": 1})
        with self.assertRaises(ValueError):
            validate_news2_trend({"min_rise": 0})
        with self.assertRaises(ValueError):
            validate_news2_trend({"window": 10})
        with self.assertRaises(ValueError):
            validate_news2_trend("fast")

    def test_monitor_opens_and_closes_trend_alert(self):
        # Febrile bed under an alarm storm: the score jumps 7 to
        # 11, so a trend alert opens on the jump rule, and it
        # closes when the storm ends and the score falls back.
        ward = Ward(
            beds=1,
            profiles=["febrile"],
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, bed=0)],
            seed=3,
        )
        ward.tick()
        m = ward.monitors[0]
        self.assertIsNone(m._trend_alert)
        ward.tick()
        self.assertIsNotNone(m._trend_alert)
        alert = m._trend_alert
        self.assertEqual(alert["raised_t"], 2.0)
        self.assertEqual(alert["start_score"], 7)
        self.assertEqual(alert["rule"], "jump")
        for _ in range(8):
            ward.tick()
        self.assertEqual(m._trend_alert["peak_score"], 11)
        self.assertEqual(m._trend_alert["peak_delta"], 4)
        self.assertEqual(m.trend_alerts_raised, 1)
        # The storm ends at t=11; the score falls and the alert
        # closes in the same tick.
        ward.tick()
        self.assertIsNone(m._trend_alert)
        self.assertEqual(len(m.trend_alert_history), 1)
        closed = m.trend_alert_history[0]
        self.assertEqual(closed["cleared_t"], 11.0)
        self.assertEqual(closed["peak_score"], 11)

    def test_discharge_closes_open_trend_alert(self):
        ward = Ward(
            beds=1,
            profiles=["hypoxic"],
            seed=3,
            news2_trend={"window_s": 10.0, "delta": 2},
        )
        for _ in range(40):
            ward.tick()
        m = ward.monitors[0]
        self.assertIsNotNone(m._trend_alert)
        raised = m.trend_alerts_raised
        ward.discharge(0)
        self.assertIsNone(m._trend_alert)
        self.assertEqual(m.trend_alerts_raised, raised)
        self.assertEqual(
            m.trend_alert_history[-1]["cleared_t"], 40.0
        )

    def test_trend_in_history_detail_and_early_warning(self):
        ward = Ward(
            beds=2,
            profiles=["stable", "febrile"],
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, bed=1)],
            seed=3,
            news2_trend={"window_s": 10.0, "delta": 2},
        )
        for _ in range(5):
            ward.tick()
        detail = ward.monitors[1].news2_history_detail(ward.elapsed)
        self.assertTrue(detail["trend"]["rising"])
        self.assertEqual(detail["trend"]["rule"], "jump")
        self.assertTrue(detail["trend_alert_active"])
        self.assertEqual(detail["trend_alerts_raised"], 1)
        self.assertEqual(len(detail["trend_alerts"]), 1)
        # The stable bed is not rising.
        stable = ward.monitors[0].news2_history_detail(ward.elapsed)
        self.assertFalse(stable["trend"]["rising"])
        self.assertEqual(stable["trend_alerts_raised"], 0)
        # The early-warning summary carries the ward totals, and
        # the snapshot carries them too.
        ew = ward.early_warning_summary()
        self.assertEqual(ew["trend_alerts"], 1)
        self.assertEqual(ew["trend_alerts_raised"], 1)
        self.assertEqual(
            ward.snapshot()["early_warning"]["trend_alerts"], 1
        )
        news2 = ward.monitors[1].news2_detail(ward.elapsed)
        self.assertTrue(news2["trend"]["rising"])
        self.assertTrue(news2["trend_alert_active"])

    def test_trend_does_not_touch_the_alarm_list(self):
        # Like the score itself, a trend alert is not a vital-sign
        # alarm: the alarm list is only what alarms_for raises.
        known = {
            "tachycardia", "bradycardia", "low_spo2", "hypertension",
            "hypotension", "fever", "hypothermia", "tachypnea",
            "missed_dose",
        }
        ward = Ward(
            beds=1,
            profiles=["febrile"],
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, bed=0)],
            seed=3,
        )
        for _ in range(40):
            snap = ward.tick()
            self.assertTrue(set(snap["beds"][0]["alarms"]) <= known)
        self.assertEqual(ward.monitors[0].trend_alerts_raised, 1)

    def test_history_endpoint_carries_trend(self):
        ward = Ward(
            beds=2,
            profiles=["stable", "febrile"],
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, bed=1)],
            seed=3,
            news2_trend={"window_s": 10.0, "delta": 2},
        )
        for _ in range(5):
            ward.tick()
        server, port = serve_in_thread(ward)
        try:
            status, body = get(port, "/news2/history?bed=1")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertTrue(data["trend"]["rising"])
            self.assertEqual(data["trend"]["rule"], "jump")
            self.assertEqual(data["trend_alerts_raised"], 1)
            self.assertEqual(data["trend_alerts"][0]["peak_score"], 11)
            # The whole-ward form carries the trend per bed too.
            status, body = get(port, "/news2/history")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertFalse(data["beds"][0]["trend"]["rising"])
            self.assertTrue(data["beds"][1]["trend"]["rising"])
        finally:
            server.shutdown()
            server.server_close()

    def test_scenario_news2_trend(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2-trend.json")
        )
        self.assertEqual(cfg["news2_trend"]["window_s"], 10.0)
        self.assertEqual(cfg["news2_trend"]["delta"], 2)
        ward = build_ward(cfg, seed=42)
        for _ in range(40):
            snap = ward.tick()
        history = ward.news2_history_summary()
        bed0, bed1, bed2 = history["beds"]
        # Stable bed: never rises two points, no alert.
        self.assertEqual(bed0["trend_alerts_raised"], 0)
        self.assertFalse(bed0["trend_alert_active"])
        # Hypoxic bed: drifts up twice — a creep alert at 12-18,
        # then a jump alert from 35 that is still open at the end.
        self.assertEqual(bed1["trend_alerts_raised"], 2)
        self.assertTrue(bed1["trend_alert_active"])
        self.assertEqual(bed1["trend"]["rule"], "jump")
        self.assertEqual(bed1["trend"]["delta"], 2)
        first = bed1["trend_alerts"][0]
        self.assertEqual(first["raised_t"], 12.0)
        self.assertEqual(first["cleared_t"], 18.0)
        self.assertEqual(first["rule"], "creep")
        # Febrile bed: one jump alert under the storm, 5 to 11,
        # clearing at 15 when the storm ends.
        self.assertEqual(bed2["trend_alerts_raised"], 1)
        self.assertFalse(bed2["trend_alert_active"])
        alert = bed2["trend_alerts"][0]
        self.assertEqual(alert["raised_t"], 5.0)
        self.assertEqual(alert["cleared_t"], 15.0)
        self.assertEqual(alert["start_t"], 1.0)
        self.assertEqual(alert["start_score"], 5)
        self.assertEqual(alert["peak_score"], 11)
        self.assertEqual(alert["peak_delta"], 6)
        self.assertEqual(alert["rule"], "jump")
        # Ward totals: one alert still open, three raised.
        self.assertEqual(snap["early_warning"]["trend_alerts"], 1)
        self.assertEqual(snap["early_warning"]["trend_alerts_raised"], 3)

    def test_escalation_response_recorded(self):
        ward = Ward(beds=1, profiles=["hypoxic"], seed=3)
        for _ in range(40):
            ward.tick()
        m = ward.monitors[0]
        self.assertIsNotNone(m._escalation)
        ep = m._escalation
        # Escalated at t=1 on a red SpO2 score; with an adequate
        # roster the ward team is due 30 s later and lands then.
        self.assertEqual(ep["raised_t"], 1.0)
        self.assertEqual(
            ep["response"],
            "urgent review by ward team (hourly monitoring)",
        )
        self.assertEqual(ep["response_due_t"], 31.0)
        self.assertEqual(ep["responded_t"], 31.0)
        self.assertEqual(ep["time_to_respond_s"], 30.0)

    def test_escalation_cleared_before_response_is_unresponded(self):
        # A storm pushes the febrile patient high, then the episode
        # clears at t=14 — long before the response due at t=31 —
        # so it keeps nulls: it cleared before anyone responded.
        ward = Ward(
            beds=1,
            profiles=["febrile"],
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, bed=0)],
            seed=3,
        )
        for _ in range(40):
            ward.tick()
        first = ward.monitors[0].escalation_history[0]
        self.assertEqual(first["raised_t"], 1.0)
        self.assertEqual(first["cleared_t"], 14.0)
        self.assertEqual(first["response_due_t"], 31.0)
        self.assertIsNone(first["responded_t"])
        self.assertIsNone(first["time_to_respond_s"])

    def test_thin_roster_slows_response(self):
        # No nurse on nights: the load factor is 4.0, so the
        # response is due 120 s after the escalation raises — still
        # pending at 40 ticks, landed at 121.
        ward = Ward(
            beds=1,
            profiles=["hypoxic"],
            seed=3,
            staffing={
                "start_hour": 20,
                "night": {"nurse": 0, "doctor": 1, "tech": 0},
            },
        )
        for _ in range(40):
            ward.tick()
        summary = ward.early_warning_summary()
        self.assertEqual(summary["responses"]["pending"], 1)
        self.assertEqual(summary["responses"]["responded"], 0)
        self.assertIsNone(
            summary["responses"]["mean_time_to_respond_s"]
        )
        self.assertIsNone(summary["responses"]["max_time_to_respond_s"])
        for _ in range(90):
            ward.tick()
        summary = ward.early_warning_summary()
        self.assertEqual(summary["responses"]["responded"], 1)
        self.assertEqual(summary["responses"]["pending"], 0)
        self.assertEqual(
            summary["responses"]["mean_time_to_respond_s"], 120.0
        )
        self.assertEqual(
            summary["responses"]["max_time_to_respond_s"], 120.0
        )
        ep = ward.monitors[0].news2_detail(ward.elapsed)["episodes"][0]
        self.assertEqual(ep["responded_t"], 121.0)
        self.assertEqual(ep["time_to_respond_s"], 120.0)

    def test_discharge_before_response_is_unresponded(self):
        ward = Ward(beds=1, profiles=["hypoxic"], seed=3)
        for _ in range(5):
            ward.tick()
        ward.discharge(0)
        ep = ward.monitors[0].escalation_history[0]
        self.assertEqual(ep["cleared_t"], 5.0)
        self.assertIsNone(ep["responded_t"])
        self.assertIsNone(ep["time_to_respond_s"])
        self.assertEqual(
            ward.early_warning_summary()["responses"]["unresponded"], 1
        )

    def test_scenario_news2_response(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2-response.json")
        )
        ward = build_ward(cfg, seed=42)
        for _ in range(40):
            snap = ward.tick()
        summary = ward.early_warning_summary()
        bed1, bed3 = summary["beds"][1], summary["beds"][3]
        # Hypoxic bed: escalated from t=1, ward team due at 31 —
        # it went high on its own, so the response recorded is the
        # critical care team called for at the risk it raised at.
        ep1 = bed1["episodes"][0]
        self.assertEqual(ep1["raised_t"], 1.0)
        self.assertEqual(ep1["response_due_t"], 31.0)
        self.assertEqual(ep1["responded_t"], 31.0)
        self.assertEqual(ep1["time_to_respond_s"], 30.0)
        # Febrile bed: a medium episode at 5-9 s clears before its
        # response is due at 35 s, so it stays unresponded; the high
        # episode under the storm is due at 40 s, the very tick the
        # risk returns to low — and the response still lands, the
        # team was already on its way.
        first, second = bed3["episodes"]
        self.assertEqual(first["raised_t"], 5.0)
        self.assertEqual(first["cleared_t"], 9.0)
        self.assertIsNone(first["responded_t"])
        self.assertIsNone(first["time_to_respond_s"])
        self.assertEqual(second["raised_t"], 10.0)
        self.assertEqual(second["responded_t"], 40.0)
        self.assertEqual(second["cleared_t"], 40.0)
        self.assertEqual(second["time_to_respond_s"], 30.0)
        # Ward totals, in the snapshot as well as the summary.
        self.assertEqual(summary["responses"]["responded"], 2)
        self.assertEqual(summary["responses"]["unresponded"], 1)
        self.assertEqual(summary["responses"]["pending"], 0)
        self.assertEqual(
            summary["responses"]["mean_time_to_respond_s"], 30.0
        )
        self.assertEqual(
            summary["responses"]["max_time_to_respond_s"], 30.0
        )
        self.assertEqual(
            snap["early_warning"]["responses"]["responded"], 2
        )

    def test_recorder_summary_carries_trend_alerts(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2-trend.json")
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            summary = summarize(path)
            ew = summary["early_warning"]
            self.assertEqual(ew["trend_alerts"], 1)
            self.assertEqual(ew["trend_alerts_raised"], 3)
            self.assertEqual(ew["escalations_raised"], 2)
        finally:
            os.unlink(path)

    def test_recorder_summary_carries_response_timing(self):
        cfg = load_scenario(
            os.path.join(HERE, "..", "scenarios", "news2-response.json")
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            summary = summarize(path)
            responses = summary["early_warning"]["responses"]
            self.assertEqual(responses["responded"], 2)
            self.assertEqual(responses["unresponded"], 1)
            self.assertEqual(responses["pending"], 0)
            self.assertEqual(
                responses["mean_time_to_respond_s"], 30.0
            )
            self.assertEqual(
                responses["max_time_to_respond_s"], 30.0
            )
        finally:
            os.unlink(path)


class TestTrendAlertHooks(unittest.TestCase):
    def _storm_ward(self, hooks=None):
        # Febrile bed under an alarm storm: the score jumps 7 to
        # 11, so a trend alert opens on the jump rule at t=2.
        return Ward(
            beds=1,
            profiles=["febrile"],
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, bed=0)],
            seed=3,
            trend_alert_hooks=hooks,
        )

    def test_validate_trend_alert_hooks(self):
        hooks = validate_trend_alert_hooks(None)
        self.assertIsInstance(hooks, TrendAlertHooks)
        self.assertFalse(hooks.configured)
        self.assertIsNone(hooks.webhook_url)
        self.assertIsNone(hooks.callback)
        self.assertEqual(hooks.timeout_s, 5.0)
        custom = validate_trend_alert_hooks(
            {"webhook_url": "https://example.invalid/hooks", "timeout_s": 2}
        )
        self.assertTrue(custom.configured)
        self.assertEqual(
            custom.webhook_url, "https://example.invalid/hooks"
        )
        self.assertEqual(custom.timeout_s, 2.0)
        seen = []
        append = seen.append
        cb = validate_trend_alert_hooks({"callback": append})
        self.assertTrue(cb.configured)
        self.assertIs(cb.callback, append)
        same = TrendAlertHooks(webhook_url="http://127.0.0.1:9/hooks")
        self.assertIs(validate_trend_alert_hooks(same), same)
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"webhook_url": "ftp://x/hooks"})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"webhook_url": 42})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"callback": "not callable"})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"timeout_s": 0})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"timeout_s": "fast"})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"url": "http://x/hooks"})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks("http://x/hooks")

    def test_hooks_off_by_default(self):
        # No hooks configured: the alert still fires, and it
        # records that no hook was invoked.
        ward = self._storm_ward()
        self.assertFalse(ward.trend_alert_hooks.configured)
        ward.tick()
        ward.tick()
        m = ward.monitors[0]
        self.assertIsNotNone(m._trend_alert)
        alert = m._trend_alert
        self.assertFalse(alert["hook_invoked"])
        self.assertEqual(alert["hook_errors"], [])
        summary = ward.early_warning_summary()["trend_alert_hooks"]
        self.assertEqual(
            summary, {"configured": False, "invoked": 0, "failed": 0, "retries": 0, "delivered": 0}
        )
        self.assertEqual(
            ward.snapshot()["early_warning"]["trend_alert_hooks"],
            summary,
        )

    def test_callback_hook_invoked_on_trend_alert(self):
        seen = []
        ward = self._storm_ward({"callback": seen.append})
        self.assertTrue(ward.trend_alert_hooks.configured)
        ward.tick()
        self.assertEqual(seen, [])
        ward.tick()
        self.assertEqual(len(seen), 1)
        payload = seen[0]
        self.assertEqual(payload["event"], "trend_alert")
        self.assertEqual(payload["bed"], 0)
        self.assertEqual(payload["rule"], "jump")
        self.assertEqual(payload["raised_t"], 2.0)
        self.assertEqual(payload["start_score"], 7)
        self.assertEqual(payload["peak_score"], 10)
        # The alert itself records the invocation.
        alert = ward.monitors[0]._trend_alert
        self.assertTrue(alert["hook_invoked"])
        self.assertEqual(alert["hook_errors"], [])
        # One invocation, no failure, in the ward totals.
        summary = ward.early_warning_summary()["trend_alert_hooks"]
        self.assertEqual(
            summary, {"configured": True, "invoked": 1, "failed": 0, "retries": 0, "delivered": 1}
        )
        # The alert fires once: later ticks while it stays open do
        # not invoke the hook again.
        for _ in range(5):
            ward.tick()
        self.assertEqual(len(seen), 1)
        self.assertEqual(
            ward.trend_alert_hooks.summary()["invoked"], 1
        )

    def test_callback_failure_is_not_fatal(self):
        def boom(payload):
            raise ValueError("pager is down")

        ward = self._storm_ward({"callback": boom})
        ward.tick()
        snap = ward.tick()
        # The tick still completes and the alert still fires.
        self.assertIsNotNone(ward.monitors[0]._trend_alert)
        self.assertTrue(snap["beds"][0]["online"])
        alert = ward.monitors[0]._trend_alert
        self.assertTrue(alert["hook_invoked"])
        self.assertEqual(alert["hook_errors"], ["pager is down"])
        summary = ward.early_warning_summary()["trend_alert_hooks"]
        self.assertEqual(summary["invoked"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_webhook_hook_posts_to_stub_server(self):
        server, port, received = capture_server()
        try:
            ward = self._storm_ward(
                {"webhook_url": "http://127.0.0.1:%d/hooks" % port}
            )
            ward.tick()
            ward.tick()
            self.assertTrue(ward.trend_alert_hooks.wait(timeout=5))
            self.assertEqual(len(received), 1)
            path, content_type, body = received[0]
            self.assertEqual(path, "/hooks")
            self.assertEqual(content_type, "application/json")
            payload = json.loads(body)
            self.assertEqual(payload["event"], "trend_alert")
            self.assertEqual(payload["bed"], 0)
            self.assertEqual(payload["rule"], "jump")
            self.assertEqual(payload["raised_t"], 2.0)
            self.assertEqual(payload["start_score"], 7)
            self.assertEqual(payload["peak_score"], 10)
            alert = ward.monitors[0]._trend_alert
            self.assertTrue(alert["hook_invoked"])
            self.assertEqual(alert["hook_errors"], [])
            summary = ward.early_warning_summary()["trend_alert_hooks"]
            self.assertEqual(
                summary, {"configured": True, "invoked": 1, "failed": 0, "retries": 0, "delivered": 1}
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_webhook_failure_is_not_fatal(self):
        # A port with nothing listening: the POST fails, the ward
        # keeps ticking, and the failure is counted, not raised.
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ward = self._storm_ward(
            {
                "webhook_url": "http://127.0.0.1:%d/hooks" % port,
                "timeout_s": 2,
            }
        )
        ward.tick()
        snap = ward.tick()
        self.assertIsNotNone(ward.monitors[0]._trend_alert)
        self.assertTrue(snap["beds"][0]["online"])
        # The hook was invoked (dispatched), even though the
        # delivery failed.
        alert = ward.monitors[0]._trend_alert
        self.assertTrue(alert["hook_invoked"])
        self.assertTrue(ward.trend_alert_hooks.wait(timeout=5))
        summary = ward.early_warning_summary()["trend_alert_hooks"]
        self.assertEqual(summary["invoked"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_webhook_does_not_block_the_tick(self):
        # A webhook endpoint that takes 10 s to answer: the tick
        # must still return at once, because the POST runs in a
        # daemon thread — the delivery is still in flight when
        # the tick returns.
        received = []

        class SlowHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                import time

                length = int(self.headers.get("Content-Length", 0))
                received.append(self.rfile.read(length))
                time.sleep(10)
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
        thread = threading.Thread(
            target=server.serve_forever, daemon=True
        )
        thread.start()
        try:
            hooks = TrendAlertHooks(
                webhook_url="http://127.0.0.1:%d/hooks"
                % server.server_address[1],
                timeout_s=30,
            )
            ward = self._storm_ward(hooks)
            import time

            start = time.monotonic()
            ward.tick()
            ward.tick()
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 5.0)
            # The POST was dispatched (it may take a moment to
            # reach the server under load); the server still takes
            # 10 s to answer, so the delivery is in flight.
            deadline = time.monotonic() + 5
            while len(received) < 1 and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(len(received), 1)
            self.assertFalse(hooks.wait(timeout=2))
        finally:
            server.shutdown()
            server.server_close()

    def test_scenario_news2_trend_hooks(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios", "news2-trend-hooks.json"
            )
        )
        self.assertTrue(cfg["trend_alert_hooks"].configured)
        server, port, received = capture_server()
        try:
            # The scenario's URL is a placeholder; point it at
            # the stub server for this run.
            cfg["trend_alert_hooks"] = validate_trend_alert_hooks(
                {"webhook_url": "http://127.0.0.1:%d/hooks" % port}
            )
            ward = build_ward(cfg, seed=42)
            for _ in range(40):
                snap = ward.tick()
            self.assertTrue(ward.trend_alert_hooks.wait(timeout=5))
            # Three trend alerts over the run, one POST each.
            self.assertEqual(len(received), 3)
            beds = sorted(json.loads(body)["bed"] for _, _, body in received)
            self.assertEqual(beds, [1, 1, 2])
            # Every alert records that its hook was invoked.
            history = ward.news2_history_summary()
            for bed in history["beds"]:
                for alert in bed["trend_alerts"]:
                    self.assertTrue(alert["hook_invoked"])
            # The totals are read from a fresh summary after the
            # wait: the snapshot from the last tick was taken
            # before the last delivery settled.
            self.assertEqual(
                ward.early_warning_summary()["trend_alert_hooks"],
                {"configured": True, "invoked": 3, "failed": 0, "retries": 0, "delivered": 3},
            )
            self.assertEqual(
                snap["early_warning"]["trend_alerts_raised"], 3
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_recorder_summary_carries_hook_totals(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios", "news2-trend-hooks.json"
            )
        )
        seen = []
        cfg["trend_alert_hooks"] = validate_trend_alert_hooks(
            {"callback": seen.append}
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            self.assertEqual(len(seen), 3)
            summary = summarize(path)
            hooks = summary["early_warning"]["trend_alert_hooks"]
            self.assertEqual(
                hooks, {"configured": True, "invoked": 3, "failed": 0, "retries": 0, "delivered": 3}
            )
        finally:
            os.unlink(path)


class TestEscalationHooks(unittest.TestCase):
    def _storm_ward(self, hooks=None):
        # Febrile bed under an alarm storm: the score is 7 at the
        # first tick, so a deterioration episode opens at t=1.
        return Ward(
            beds=1,
            profiles=["febrile"],
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, bed=0)],
            seed=3,
            escalation_hooks=hooks,
        )

    def test_validate_escalation_hooks(self):
        hooks = validate_escalation_hooks(None)
        self.assertIsInstance(hooks, TrendAlertHooks)
        self.assertFalse(hooks.configured)
        self.assertEqual(hooks.kind, "escalation")
        custom = validate_escalation_hooks(
            {"webhook_url": "https://example.invalid/hooks", "timeout_s": 2}
        )
        self.assertTrue(custom.configured)
        self.assertEqual(custom.kind, "escalation")
        self.assertEqual(custom.timeout_s, 2.0)
        seen = []
        append = seen.append
        cb = validate_escalation_hooks({"callback": append})
        self.assertTrue(cb.configured)
        self.assertIs(cb.callback, append)
        with self.assertRaises(ValueError):
            validate_escalation_hooks({"webhook_url": "ftp://x/hooks"})
        with self.assertRaises(ValueError):
            validate_escalation_hooks({"callback": "not callable"})
        with self.assertRaises(ValueError):
            validate_escalation_hooks({"timeout_s": 0})
        with self.assertRaises(ValueError):
            validate_escalation_hooks({"url": "http://x/hooks"})
        with self.assertRaises(ValueError):
            validate_escalation_hooks("http://x/hooks")

    def test_hooks_off_by_default(self):
        # No hooks configured: the episode still opens, and it
        # records that no hook was invoked.
        ward = self._storm_ward()
        self.assertFalse(ward.escalation_hooks.configured)
        ward.tick()
        m = ward.monitors[0]
        self.assertIsNotNone(m._escalation)
        episode = m._escalation
        self.assertFalse(episode["hook_invoked"])
        self.assertEqual(episode["hook_errors"], [])
        summary = ward.early_warning_summary()["escalation_hooks"]
        self.assertEqual(
            summary, {"configured": False, "invoked": 0, "failed": 0, "retries": 0, "delivered": 0}
        )
        self.assertEqual(
            ward.snapshot()["early_warning"]["escalation_hooks"],
            summary,
        )

    def test_callback_hook_invoked_on_escalation(self):
        seen = []
        ward = self._storm_ward({"callback": seen.append})
        self.assertTrue(ward.escalation_hooks.configured)
        ward.tick()
        self.assertEqual(len(seen), 1)
        payload = seen[0]
        self.assertEqual(payload["event"], "escalation")
        self.assertEqual(payload["bed"], 0)
        self.assertEqual(payload["raised_t"], 1.0)
        self.assertEqual(payload["peak_score"], 7)
        self.assertEqual(payload["peak_risk"], "high")
        self.assertEqual(
            payload["response"],
            "urgent review by critical care team (continuous monitoring)",
        )
        # The episode itself records the invocation.
        episode = ward.monitors[0]._escalation
        self.assertTrue(episode["hook_invoked"])
        self.assertEqual(episode["hook_errors"], [])
        # One invocation, no failure, in the ward totals.
        summary = ward.early_warning_summary()["escalation_hooks"]
        self.assertEqual(
            summary, {"configured": True, "invoked": 1, "failed": 0, "retries": 0, "delivered": 1}
        )
        # The hook fires once, when the episode opens: later ticks
        # while it stays open do not invoke it again.
        for _ in range(5):
            ward.tick()
        self.assertEqual(len(seen), 1)
        self.assertEqual(
            ward.escalation_hooks.summary()["invoked"], 1
        )

    def test_callback_failure_is_not_fatal(self):
        def boom(payload):
            raise ValueError("pager is down")

        ward = self._storm_ward({"callback": boom})
        snap = ward.tick()
        # The tick still completes and the episode still opens.
        self.assertIsNotNone(ward.monitors[0]._escalation)
        self.assertTrue(snap["beds"][0]["online"])
        episode = ward.monitors[0]._escalation
        self.assertTrue(episode["hook_invoked"])
        self.assertEqual(episode["hook_errors"], ["pager is down"])
        summary = ward.early_warning_summary()["escalation_hooks"]
        self.assertEqual(summary["invoked"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_webhook_hook_posts_to_stub_server(self):
        server, port, received = capture_server()
        try:
            ward = self._storm_ward(
                {"webhook_url": "http://127.0.0.1:%d/hooks" % port}
            )
            ward.tick()
            self.assertTrue(ward.escalation_hooks.wait(timeout=5))
            self.assertEqual(len(received), 1)
            path, content_type, body = received[0]
            self.assertEqual(path, "/hooks")
            self.assertEqual(content_type, "application/json")
            payload = json.loads(body)
            self.assertEqual(payload["event"], "escalation")
            self.assertEqual(payload["bed"], 0)
            self.assertEqual(payload["raised_t"], 1.0)
            self.assertEqual(payload["peak_score"], 7)
            self.assertEqual(payload["peak_risk"], "high")
            episode = ward.monitors[0]._escalation
            self.assertTrue(episode["hook_invoked"])
            self.assertEqual(episode["hook_errors"], [])
            summary = ward.early_warning_summary()["escalation_hooks"]
            self.assertEqual(
                summary, {"configured": True, "invoked": 1, "failed": 0, "retries": 0, "delivered": 1}
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_webhook_failure_is_not_fatal(self):
        # A port with nothing listening: the POST fails, the ward
        # keeps ticking, and the failure is counted, not raised.
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ward = self._storm_ward(
            {
                "webhook_url": "http://127.0.0.1:%d/hooks" % port,
                "timeout_s": 2,
            }
        )
        snap = ward.tick()
        self.assertIsNotNone(ward.monitors[0]._escalation)
        self.assertTrue(snap["beds"][0]["online"])
        # The hook was invoked (dispatched), even though the
        # delivery failed.
        episode = ward.monitors[0]._escalation
        self.assertTrue(episode["hook_invoked"])
        self.assertTrue(ward.escalation_hooks.wait(timeout=5))
        summary = ward.early_warning_summary()["escalation_hooks"]
        self.assertEqual(summary["invoked"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_scenario_news2_escalation_hooks(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios", "news2-escalation-hooks.json"
            )
        )
        self.assertTrue(cfg["escalation_hooks"].configured)
        server, port, received = capture_server()
        try:
            # The scenario's URL is a placeholder; point it at
            # the stub server for this run.
            cfg["escalation_hooks"] = validate_escalation_hooks(
                {"webhook_url": "http://127.0.0.1:%d/hooks" % port}
            )
            ward = build_ward(cfg, seed=42)
            for _ in range(40):
                snap = ward.tick()
            self.assertTrue(ward.escalation_hooks.wait(timeout=5))
            # Two escalations over the run, one POST each.
            self.assertEqual(len(received), 2)
            beds = sorted(json.loads(body)["bed"] for _, _, body in received)
            self.assertEqual(beds, [1, 2])
            # Every episode records that its hook was invoked.
            for bed in snap["early_warning"]["beds"]:
                for episode in bed["episodes"]:
                    self.assertTrue(episode["hook_invoked"])
            # The totals are read from a fresh summary after the
            # wait: the snapshot from the last tick was taken
            # before the last delivery settled.
            self.assertEqual(
                ward.early_warning_summary()["escalation_hooks"],
                {"configured": True, "invoked": 2, "failed": 0, "retries": 0, "delivered": 2},
            )
            self.assertEqual(
                snap["early_warning"]["escalations_raised"], 2
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_recorder_summary_carries_hook_totals(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios", "news2-escalation-hooks.json"
            )
        )
        seen = []
        cfg["escalation_hooks"] = validate_escalation_hooks(
            {"callback": seen.append}
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(40):
                    ward.tick()
            self.assertEqual(len(seen), 2)
            summary = summarize(path)
            hooks = summary["early_warning"]["escalation_hooks"]
            self.assertEqual(
                hooks, {"configured": True, "invoked": 2, "failed": 0, "retries": 0, "delivered": 2}
            )
        finally:
            os.unlink(path)


class TestHookRetries(unittest.TestCase):
    def _storm_ward(self, hooks=None):
        # Febrile bed under an alarm storm: the score jumps 7 to
        # 11, so a trend alert opens on the jump rule at t=2.
        return Ward(
            beds=1,
            profiles=["febrile"],
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, bed=0)],
            seed=3,
            trend_alert_hooks=hooks,
        )

    def _escalation_ward(self, hooks=None):
        # Febrile bed under an alarm storm: the score is 7 at the
        # first tick, so a deterioration episode opens at t=1.
        return Ward(
            beds=1,
            profiles=["febrile"],
            chaos=[ChaosEvent("alarm_storm", at=1, duration=10, bed=0)],
            seed=3,
            escalation_hooks=hooks,
        )

    def _flaky_server(self, fail_first=2, status=500):
        """A stub server that fails the first ``fail_first``
        attempts for each distinct payload with ``status``, then
        answers 200. Returns ``(server, port, received)``:
        ``received`` is a list of ``(path, content_type,
        body_bytes)`` in arrival order. The caller shuts the
        server down."""
        received = []
        attempts = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                received.append(
                    (self.path, self.headers.get("Content-Type"), body)
                )
                seen = attempts.get(body, 0)
                attempts[body] = seen + 1
                code = status if seen < fail_first else 200
                self.send_response(code)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=server.serve_forever, daemon=True
        )
        thread.start()
        return server, server.server_address[1], received

    def test_validate_retry_config(self):
        hooks = validate_trend_alert_hooks(None)
        self.assertEqual(hooks.max_retries, 2)
        self.assertEqual(hooks.retry_backoff_s, 1.0)
        custom = validate_trend_alert_hooks(
            {
                "webhook_url": "https://example.invalid/hooks",
                "max_retries": 0,
                "retry_backoff_s": 0.25,
            }
        )
        self.assertEqual(custom.max_retries, 0)
        self.assertEqual(custom.retry_backoff_s, 0.25)
        # max_retries 0 means one attempt, no retries — still valid.
        same = TrendAlertHooks(max_retries=5, retry_backoff_s=2)
        self.assertIs(validate_trend_alert_hooks(same), same)
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"max_retries": -1})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"max_retries": 1.5})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"max_retries": True})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"max_retries": "2"})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"retry_backoff_s": 0})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"retry_backoff_s": -1})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"retry_backoff_s": "fast"})
        with self.assertRaises(ValueError):
            validate_trend_alert_hooks({"retry_backoff_s": True})
        with self.assertRaises(ValueError):
            validate_escalation_hooks({"backoff": 1.0})

    def test_webhook_retries_then_delivers(self):
        # The endpoint 500s twice, then answers 200: the delivery
        # lands on the third attempt.
        server, port, received = self._flaky_server(fail_first=2)
        try:
            ward = self._storm_ward(
                {
                    "webhook_url": "http://127.0.0.1:%d/hooks" % port,
                    "max_retries": 2,
                    "retry_backoff_s": 0.05,
                }
            )
            ward.tick()
            ward.tick()
            alert = ward.monitors[0]._trend_alert
            self.assertIsNotNone(alert)
            self.assertTrue(alert["hook_invoked"])
            # The delivery is still in flight: the first attempt
            # failed and the retries have not landed yet.
            self.assertEqual(alert["hook_final_status"], "pending")
            self.assertTrue(ward.trend_alert_hooks.wait(timeout=5))
            self.assertEqual(len(received), 3)
            for _, _, body in received:
                self.assertEqual(
                    json.loads(body)["event"], "trend_alert"
                )
            # Two retries spent, and the alert now says delivered.
            self.assertEqual(alert["hook_retries"], 2)
            self.assertEqual(alert["hook_final_status"], "delivered")
            summary = ward.early_warning_summary()["trend_alert_hooks"]
            self.assertEqual(
                summary,
                {
                    "configured": True,
                    "invoked": 1,
                    "failed": 0,
                    "retries": 2,
                    "delivered": 1,
                },
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_webhook_gives_up_after_max_retries(self):
        # The endpoint 500s forever: three attempts, then it
        # gives up and the delivery counts as failed.
        server, port, received = self._flaky_server(fail_first=10)
        try:
            ward = self._storm_ward(
                {
                    "webhook_url": "http://127.0.0.1:%d/hooks" % port,
                    "max_retries": 2,
                    "retry_backoff_s": 0.05,
                }
            )
            ward.tick()
            ward.tick()
            alert = ward.monitors[0]._trend_alert
            self.assertTrue(ward.trend_alert_hooks.wait(timeout=5))
            self.assertEqual(len(received), 3)
            self.assertEqual(alert["hook_retries"], 2)
            self.assertEqual(alert["hook_final_status"], "failed")
            summary = ward.early_warning_summary()["trend_alert_hooks"]
            self.assertEqual(summary["invoked"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["retries"], 2)
            self.assertEqual(summary["delivered"], 0)
        finally:
            server.shutdown()
            server.server_close()

    def test_webhook_connection_error_is_retried(self):
        # A port with nothing listening: the POST fails, is
        # retried once, and then counts as failed.
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ward = self._storm_ward(
            {
                "webhook_url": "http://127.0.0.1:%d/hooks" % port,
                "max_retries": 1,
                "retry_backoff_s": 0.05,
                "timeout_s": 2,
            }
        )
        ward.tick()
        ward.tick()
        alert = ward.monitors[0]._trend_alert
        self.assertTrue(ward.trend_alert_hooks.wait(timeout=5))
        self.assertEqual(alert["hook_retries"], 1)
        self.assertEqual(alert["hook_final_status"], "failed")
        summary = ward.early_warning_summary()["trend_alert_hooks"]
        self.assertEqual(summary["invoked"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(summary["delivered"], 0)

    def test_retry_does_not_block_the_tick(self):
        # An endpoint that takes 0.3 s to answer 500, with a
        # 0.5 s backoff: the delivery takes seconds in its thread,
        # but the tick still returns at once.
        import time

        received = []

        class SlowHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                import time as _time

                length = int(self.headers.get("Content-Length", 0))
                received.append(
                    (
                        self.path,
                        self.headers.get("Content-Type"),
                        self.rfile.read(length),
                    )
                )
                _time.sleep(0.3)
                self.send_response(500)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
        thread = threading.Thread(
            target=server.serve_forever, daemon=True
        )
        thread.start()
        try:
            ward = self._storm_ward(
                {
                    "webhook_url": "http://127.0.0.1:%d/hooks"
                    % server.server_address[1],
                    "max_retries": 2,
                    "retry_backoff_s": 0.5,
                }
            )
            start = time.monotonic()
            ward.tick()
            ward.tick()
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 2.0)
            # The slow endpoint never answers 200, so this
            # delivery gives up — but the tick was not held up
            # by any of it.
            self.assertTrue(ward.trend_alert_hooks.wait(timeout=10))
            alert = ward.monitors[0]._trend_alert
            self.assertEqual(alert["hook_retries"], 2)
            self.assertEqual(alert["hook_final_status"], "failed")
            self.assertEqual(len(received), 3)
        finally:
            server.shutdown()
            server.server_close()

    def test_escalation_webhook_retries_then_delivers(self):
        # Retries are shared machinery: an escalation webhook
        # that 500s once lands on the second attempt.
        server, port, received = self._flaky_server(fail_first=1)
        try:
            ward = self._escalation_ward(
                {
                    "webhook_url": "http://127.0.0.1:%d/hooks" % port,
                    "max_retries": 2,
                    "retry_backoff_s": 0.05,
                }
            )
            ward.tick()
            episode = ward.monitors[0]._escalation
            self.assertIsNotNone(episode)
            self.assertTrue(ward.escalation_hooks.wait(timeout=5))
            self.assertEqual(len(received), 2)
            self.assertEqual(episode["hook_retries"], 1)
            self.assertEqual(episode["hook_final_status"], "delivered")
            summary = ward.early_warning_summary()["escalation_hooks"]
            self.assertEqual(summary["invoked"], 1)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["retries"], 1)
            self.assertEqual(summary["delivered"], 1)
        finally:
            server.shutdown()
            server.server_close()

    def test_callback_failure_is_not_retried(self):
        # A callback that raises is a bug, not a transient
        # network fault: it fails at once, with no retries.
        def boom(payload):
            raise ValueError("pager is down")

        ward = self._storm_ward({"callback": boom})
        ward.tick()
        ward.tick()
        alert = ward.monitors[0]._trend_alert
        self.assertEqual(alert["hook_errors"], ["pager is down"])
        self.assertEqual(alert["hook_retries"], 0)
        self.assertEqual(alert["hook_final_status"], "failed")
        summary = ward.early_warning_summary()["trend_alert_hooks"]
        self.assertEqual(summary["invoked"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["retries"], 0)
        self.assertEqual(summary["delivered"], 0)

    def test_no_hooks_records_not_invoked(self):
        ward = self._storm_ward()
        ward.tick()
        ward.tick()
        alert = ward.monitors[0]._trend_alert
        self.assertEqual(alert["hook_retries"], 0)
        self.assertEqual(alert["hook_final_status"], "not_invoked")
        summary = ward.early_warning_summary()["trend_alert_hooks"]
        self.assertEqual(
            summary,
            {
                "configured": False,
                "invoked": 0,
                "failed": 0,
                "retries": 0,
                "delivered": 0,
            },
        )

    def test_scenario_news2_trend_hooks_retry(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios", "news2-trend-hooks-retry.json"
            )
        )
        self.assertTrue(cfg["trend_alert_hooks"].configured)
        self.assertEqual(cfg["trend_alert_hooks"].max_retries, 2)
        server, port, received = self._flaky_server(fail_first=2)
        try:
            # The scenario's URL is a placeholder; point it at
            # the stub server for this run. The scenario's 0.5 s
            # backoff is shortened so the test stays fast; the
            # 2-retry bound from the file is what is exercised.
            cfg["trend_alert_hooks"].webhook_url = (
                "http://127.0.0.1:%d/hooks" % port
            )
            cfg["trend_alert_hooks"].retry_backoff_s = 0.05
            ward = build_ward(cfg, seed=42)
            for _ in range(40):
                snap = ward.tick()
            self.assertTrue(ward.trend_alert_hooks.wait(timeout=5))
            # Three trend alerts over the run, each delivered on
            # its third attempt: 9 POSTs, 6 retries. The totals
            # are read from a fresh summary after the wait: the
            # snapshot from the last tick was taken before the
            # last delivery settled.
            self.assertEqual(len(received), 9)
            history = ward.news2_history_summary()
            for bed in history["beds"]:
                for alert in bed["trend_alerts"]:
                    self.assertEqual(alert["hook_retries"], 2)
                    self.assertEqual(
                        alert["hook_final_status"], "delivered"
                    )
            self.assertEqual(
                ward.early_warning_summary()["trend_alert_hooks"],
                {
                    "configured": True,
                    "invoked": 3,
                    "failed": 0,
                    "retries": 6,
                    "delivered": 3,
                },
            )
            self.assertEqual(
                snap["early_warning"]["trend_alerts_raised"], 3
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_recorder_summary_carries_retry_totals(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios", "news2-trend-hooks-retry.json"
            )
        )
        server, port, received = self._flaky_server(fail_first=2)
        try:
            cfg["trend_alert_hooks"].webhook_url = (
                "http://127.0.0.1:%d/hooks" % port
            )
            cfg["trend_alert_hooks"].retry_backoff_s = 0.05
            with tempfile.NamedTemporaryFile(
                "w", suffix=".jsonl", delete=False
            ) as f:
                path = f.name
            try:
                with Recorder(
                    path, cfg["beds"], cfg["rate_hz"], seed=42
                ) as rec:
                    ward = build_ward(cfg, seed=42, recorder=rec)
                    for _ in range(40):
                        ward.tick()
                    # The last alert raises near the end of the
                    # run; let its retries land, then tick once
                    # more so the recording's last snapshot
                    # carries the settled totals.
                    self.assertTrue(
                        ward.trend_alert_hooks.wait(timeout=5)
                    )
                    ward.tick()
                summary = summarize(path)
                hooks = summary["early_warning"]["trend_alert_hooks"]
                self.assertEqual(
                    hooks,
                    {
                        "configured": True,
                        "invoked": 3,
                        "failed": 0,
                        "retries": 6,
                        "delivered": 3,
                    },
                )
            finally:
                os.unlink(path)
        finally:
            server.shutdown()
            server.server_close()


class TestBoardingFullHooks(unittest.TestCase):
    def _ward(self, beds=2, cleaning_s=10.0, **kw):
        ward = Ward(beds=beds, cleaning_s=cleaning_s, seed=42, **kw)
        ward.tick()
        return ward

    def _full_ward(self, hooks=None):
        # Queue of 1, boarding of 1, both occupied: one patient
        # waiting (t=1), one boarding (t=2).
        ward = self._ward(
            triage={"capacity": 1, "boarding_capacity": 1},
            boarding_full_hooks=hooks,
        )
        self.assertTrue(ward.triage_arrive(3, "stable"))  # t=1, queue
        ward.tick()  # t=2
        self.assertTrue(ward.triage_arrive(3, "febrile"))  # t=2, boards
        return ward

    def test_validate_boarding_full_hooks(self):
        hooks = validate_boarding_full_hooks(None)
        self.assertIsInstance(hooks, TrendAlertHooks)
        self.assertFalse(hooks.configured)
        self.assertEqual(hooks.kind, "boarding-full")
        custom = validate_boarding_full_hooks(
            {"webhook_url": "https://example.invalid/hooks", "timeout_s": 2}
        )
        self.assertTrue(custom.configured)
        self.assertEqual(custom.kind, "boarding-full")
        self.assertEqual(custom.timeout_s, 2.0)
        seen = []
        append = seen.append
        cb = validate_boarding_full_hooks({"callback": append})
        self.assertTrue(cb.configured)
        self.assertIs(cb.callback, append)
        with self.assertRaises(ValueError):
            validate_boarding_full_hooks({"webhook_url": "ftp://x/hooks"})
        with self.assertRaises(ValueError):
            validate_boarding_full_hooks({"callback": "not callable"})
        with self.assertRaises(ValueError):
            validate_boarding_full_hooks({"timeout_s": 0})
        with self.assertRaises(ValueError):
            validate_boarding_full_hooks({"url": "http://x/hooks"})
        with self.assertRaises(ValueError):
            validate_boarding_full_hooks("http://x/hooks")

    def test_hooks_off_by_default(self):
        # No hooks configured: the episode still raises, and it
        # records that no hook was invoked.
        ward = self._full_ward()
        self.assertFalse(ward.boarding_full_hooks.configured)
        self.assertFalse(ward.triage_arrive(1, "hypoxic"))  # t=2, diverted
        summary = ward.triage_summary()
        self.assertEqual(
            summary["boarding_full_hooks"],
            {"configured": False, "invoked": 0, "failed": 0, "retries": 0, "delivered": 0},
        )
        episode = summary["boarding_full_alarm_history"][0]
        self.assertFalse(episode["hook_invoked"])
        self.assertEqual(episode["hook_errors"], [])
        self.assertEqual(episode["hook_retries"], 0)
        self.assertEqual(episode["hook_final_status"], "not_invoked")
        self.assertEqual(
            ward.snapshot()["triage"]["boarding_full_hooks"],
            summary["boarding_full_hooks"],
        )

    def test_callback_hook_invoked_on_boarding_full_alarm(self):
        seen = []
        ward = self._full_ward({"callback": seen.append})
        self.assertTrue(ward.boarding_full_hooks.configured)
        self.assertFalse(ward.triage_arrive(1, "hypoxic"))  # t=2, diverted
        self.assertEqual(len(seen), 1)
        payload = seen[0]
        self.assertEqual(payload["event"], "boarding_full_alarm")
        self.assertEqual(payload["raised_t"], 2.0)
        self.assertEqual(payload["diverted"], 1)
        self.assertIsNone(payload["cleared_t"])
        # The episode itself records the invocation.
        episode = ward.triage_summary()["boarding_full_alarm_history"][0]
        self.assertTrue(episode["hook_invoked"])
        self.assertEqual(episode["hook_errors"], [])
        self.assertEqual(episode["hook_final_status"], "delivered")
        # One invocation, no failure, in the ward totals.
        summary = ward.triage_summary()["boarding_full_hooks"]
        self.assertEqual(
            summary, {"configured": True, "invoked": 1, "failed": 0, "retries": 0, "delivered": 1}
        )
        # Further diversions during the episode do not fire again.
        ward.triage_arrive(4)  # t=2, diverted: same episode
        self.assertEqual(len(seen), 1)
        # A second episode, after boarding refills, fires again.
        ward.discharge(0)  # t=2, bed 0 free at t=12 (cleaning is 10 s)
        for _ in range(10):
            ward.tick()
        # t=12: boarded patient admitted, first episode cleared.
        self.assertFalse(ward.triage_summary()["boarding_full_alarm_active"])
        self.assertTrue(ward.triage_arrive(3))  # boards
        self.assertFalse(ward.triage_arrive(4))  # diverted: second episode
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1]["raised_t"], 12.0)
        self.assertEqual(
            ward.triage.boarding_full_hooks.summary()["invoked"], 2
        )

    def test_callback_failure_is_not_fatal(self):
        def boom(payload):
            raise ValueError("pager is down")

        ward = self._full_ward({"callback": boom})
        self.assertFalse(ward.triage_arrive(1, "hypoxic"))  # t=2, diverted
        snap = ward.tick()  # t=3: the tick still completes
        self.assertTrue(snap["beds"][0]["online"])
        episode = ward.triage_summary()["boarding_full_alarm_history"][0]
        self.assertTrue(episode["hook_invoked"])
        self.assertEqual(episode["hook_errors"], ["pager is down"])
        self.assertEqual(episode["hook_final_status"], "failed")
        summary = ward.triage_summary()["boarding_full_hooks"]
        self.assertEqual(summary["invoked"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_webhook_hook_posts_to_stub_server(self):
        server, port, received = capture_server()
        try:
            ward = self._full_ward(
                {"webhook_url": "http://127.0.0.1:%d/hooks" % port}
            )
            self.assertFalse(ward.triage_arrive(1, "hypoxic"))
            self.assertTrue(ward.boarding_full_hooks.wait(timeout=5))
            self.assertEqual(len(received), 1)
            path, content_type, body = received[0]
            self.assertEqual(path, "/hooks")
            self.assertEqual(content_type, "application/json")
            payload = json.loads(body)
            self.assertEqual(payload["event"], "boarding_full_alarm")
            self.assertEqual(payload["raised_t"], 2.0)
            self.assertEqual(payload["diverted"], 1)
            episode = ward.triage_summary()["boarding_full_alarm_history"][0]
            self.assertTrue(episode["hook_invoked"])
            self.assertEqual(episode["hook_errors"], [])
            self.assertEqual(episode["hook_final_status"], "delivered")
            summary = ward.triage_summary()["boarding_full_hooks"]
            self.assertEqual(
                summary, {"configured": True, "invoked": 1, "failed": 0, "retries": 0, "delivered": 1}
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_webhook_failure_is_not_fatal(self):
        # A port with nothing listening: the POST fails, the ward
        # keeps ticking, and the failure is counted, not raised.
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ward = self._full_ward(
            {
                "webhook_url": "http://127.0.0.1:%d/hooks" % port,
                "timeout_s": 2,
            }
        )
        self.assertFalse(ward.triage_arrive(1, "hypoxic"))
        snap = ward.tick()  # t=3: the tick still completes
        self.assertTrue(snap["beds"][0]["online"])
        # The hook was invoked (dispatched), even though the
        # delivery failed.
        self.assertTrue(
            ward.triage_summary()["boarding_full_alarm_history"][0][
                "hook_invoked"
            ]
        )
        self.assertTrue(ward.boarding_full_hooks.wait(timeout=5))
        episode = ward.triage_summary()["boarding_full_alarm_history"][0]
        self.assertEqual(episode["hook_final_status"], "failed")
        summary = ward.triage_summary()["boarding_full_hooks"]
        self.assertEqual(summary["invoked"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_scenario_boarding_full_hooks(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios", "boarding-full-hooks.json"
            )
        )
        self.assertTrue(cfg["boarding_full_hooks"].configured)
        server, port, received = capture_server()
        try:
            # The scenario's URL is a placeholder; point it at
            # the stub server for this run.
            cfg["boarding_full_hooks"] = validate_boarding_full_hooks(
                {"webhook_url": "http://127.0.0.1:%d/hooks" % port}
            )
            ward = build_ward(cfg, seed=42)
            for _ in range(20):
                snap = ward.tick()
            self.assertTrue(ward.boarding_full_hooks.wait(timeout=5))
            # One boarding-full episode over the run, one POST.
            self.assertEqual(len(received), 1)
            payload = json.loads(received[0][2])
            self.assertEqual(payload["event"], "boarding_full_alarm")
            self.assertEqual(payload["raised_t"], 2.0)
            # Every episode records that its hook was invoked.
            for episode in snap["triage"]["boarding_full_alarm_history"]:
                self.assertTrue(episode["hook_invoked"])
            # The totals are read from a fresh summary after the
            # wait: the snapshot from the last tick was taken
            # before the last delivery settled.
            self.assertEqual(
                ward.triage_summary()["boarding_full_hooks"],
                {"configured": True, "invoked": 1, "failed": 0, "retries": 0, "delivered": 1},
            )
            self.assertEqual(
                snap["triage"]["boarding_full_alarms_raised"], 1
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_recorder_summary_carries_hook_totals(self):
        cfg = load_scenario(
            os.path.join(
                HERE, "..", "scenarios", "boarding-full-hooks.json"
            )
        )
        seen = []
        cfg["boarding_full_hooks"] = validate_boarding_full_hooks(
            {"callback": seen.append}
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            with Recorder(path, cfg["beds"], cfg["rate_hz"], seed=42) as rec:
                ward = build_ward(cfg, seed=42, recorder=rec)
                for _ in range(20):
                    ward.tick()
            self.assertEqual(len(seen), 1)
            summary = summarize(path)
            hooks = summary["triage"]["boarding_full_hooks"]
            self.assertEqual(
                hooks, {"configured": True, "invoked": 1, "failed": 0, "retries": 0, "delivered": 1}
            )
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
