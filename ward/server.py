"""HTTP endpoint streaming ward vitals as JSON lines (stdlib only)."""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "partitioned" if ward.partitioned else "ok",
                        "beds": ward.beds,
                    },
                )
            elif self.path == "/snapshot":
                if ward.partitioned:
                    self._send_json(503, {"error": "partitioned"})
                    return
                snap = ward.snapshot()
                if snap is None:
                    snap = ward.tick()
                self._send_json(200, snap)
            elif self.path == "/stream":
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
            else:
                self._send_json(404, {"error": "not found"})

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
