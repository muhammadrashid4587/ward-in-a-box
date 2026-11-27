"""HTTP endpoint streaming ward vitals as JSON lines (stdlib only)."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .websocket import (
    OP_BINARY,
    OP_CLOSE,
    OP_CONTINUATION,
    OP_PING,
    OP_PONG,
    OP_TEXT,
    accept_key,
    encode_frame,
    read_frame,
)


def make_handler(ward):
    class WardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep stdout clean
            pass

        def _send_json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # The path is matched without its query string, so
            # /news2/history?bed=3 routes like /news2/history.
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "partitioned" if ward.partitioned else "ok",
                        "beds": ward.beds,
                    },
                )
            elif path == "/snapshot":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                snap = ward.snapshot()
                if snap is None:
                    snap = ward.tick()
                self._send_json(200, snap)
            elif path == "/alarms":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.alarm_summary())
            elif path == "/turnover":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.turnover_summary())
            elif path == "/staffing":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.staffing_summary())
            elif path == "/meds":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.meds_summary())
            elif path == "/triage":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.triage_summary())
            elif path == "/arrivals":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.arrivals_summary())
            elif path == "/discharge":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.discharge_summary())
            elif path == "/discharge-readiness":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.readiness_summary())
            elif path == "/discharge-priority":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.discharge_priority_summary())
            elif path == "/news2":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.early_warning_summary())
            elif path == "/news2/history":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                raw = query.get("bed", [None])[0]
                if raw is None:
                    self._send_json(200, ward.news2_history_summary())
                    return
                try:
                    bed_id = int(raw)
                except ValueError:
                    self._send_json(400, {"error": "bed must be an integer"})
                    return
                try:
                    self._send_json(
                        200, ward.news2_history_summary(bed_id)
                    )
                except ValueError:
                    self._send_json(404, {"error": "no bed %s" % raw})
            elif path == "/sensor-drift":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.sensor_drift_summary())
            elif path == "/bed-blocker":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.bed_blocker_summary())
            elif path == "/oxygen-outage":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.oxygen_outage_summary())
            elif path == "/power-outage":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.power_outage_summary())
            elif path == "/network-outage":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.network_outage_summary())
            elif path == "/medication-stockout":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.medication_stockout_summary())
            elif path == "/pump-failure":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.pump_failure_summary())
            elif path == "/lab-delay":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.lab_delay_summary())
            elif path == "/ventilator-failure":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.ventilator_failure_summary())
            elif path == "/dialysis-failure":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.dialysis_failure_summary())
            elif path == "/partition":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self._send_json(200, ward.partition_summary())
            elif path == "/stream":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                try:
                    while True:
                        snap = ward.tick()
                        self.wfile.write((json.dumps(snap) + "\n").encode())
                        self.wfile.flush()
                        time.sleep(1.0 / ward.rate_hz)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif path == "/ws":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                key = self.headers.get("Sec-WebSocket-Key")
                upgrade = (self.headers.get("Upgrade") or "").lower()
                connection = (self.headers.get("Connection") or "").lower()
                if not key or upgrade != "websocket" or "upgrade" not in connection:
                    self._send_json(400, {"error": "expected a WebSocket upgrade"})
                    return
                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept_key(key))
                self.end_headers()
                self._ws_loop(ward)
                self.close_connection = True
            else:
                self._send_json(404, {"error": "not found"})

        def _ws_loop(self, ward):
            """Stream one ward tick per text frame until the client closes.

            A reader thread handles client frames (ping/pong, close,
            fragmented messages); this thread sends the ticks. Writes share
            one lock so a pong and a tick frame can never interleave.
            """
            stop = threading.Event()
            write_lock = threading.Lock()

            def send(opcode, payload):
                with write_lock:
                    self.wfile.write(encode_frame(payload, opcode))
                    self.wfile.flush()

            def reader():
                try:
                    message = bytearray()
                    while not stop.is_set():
                        fin, opcode, payload = read_frame(self.rfile)
                        if opcode is None:  # clean EOF
                            break
                        if opcode == OP_CLOSE:
                            try:
                                send(OP_CLOSE, payload)
                            except OSError:
                                pass
                            break
                        if opcode == OP_PING:
                            send(OP_PONG, payload)
                            continue
                        if opcode == OP_PONG:
                            continue
                        if opcode in (OP_TEXT, OP_BINARY):
                            message = bytearray(payload)
                        elif opcode == OP_CONTINUATION:
                            message += payload
                        else:
                            continue
                        if fin:
                            # Client messages are accepted and ignored for
                            # now; the ward pushes ticks, it takes no input.
                            message = bytearray()
                except (OSError, ConnectionError):
                    pass
                finally:
                    stop.set()

            threading.Thread(target=reader, daemon=True).start()
            try:
                while not stop.is_set():
                    snap = ward.tick()
                    msg = {
                        "t": snap["t"],
                        "beds": snap["beds"],
                        "alarms": ward.alarm_summary(),
                        "staffing": ward.staffing_summary(),
                        "meds": ward.meds_summary(),
                        "triage": ward.triage_summary(),
                        "arrivals": ward.arrivals_summary(),
                        "discharge": ward.discharge_summary(),
                        "discharge_readiness": ward.readiness_summary(),
                        "discharge_priority": (
                            ward.discharge_priority_summary()
                        ),
                        "early_warning": ward.early_warning_summary(),
                        "sensor_drift": ward.sensor_drift_summary(),
                        "bed_blocker": ward.bed_blocker_summary(),
                        "oxygen_outage": ward.oxygen_outage_summary(),
                        "power_outage": ward.power_outage_summary(),
                        "network_outage": ward.network_outage_summary(),
                        "medication_stockout": (
                            ward.medication_stockout_summary()
                        ),
                        "pump_failure": ward.pump_failure_summary(),
                        "lab_delay": ward.lab_delay_summary(),
                        "ventilator_failure": (
                            ward.ventilator_failure_summary()
                        ),
                        "dialysis_failure": (
                            ward.dialysis_failure_summary()
                        ),
                        "partition": ward.partition_summary(),
                    }
                    send(OP_TEXT, json.dumps(msg))
                    stop.wait(1.0 / ward.rate_hz)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                stop.set()

    return WardHandler


def serve(ward, host="127.0.0.1", port=8080):
    """Serve the ward until Ctrl-C. The ward ticks inside /stream."""
    server = ThreadingHTTPServer((host, port), make_handler(ward))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
