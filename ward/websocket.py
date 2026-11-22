"""Minimal RFC 6455 WebSocket framing (stdlib only).

Only what the ward endpoint needs: the opening handshake accept key,
server-to-client text frames, and reading client frames (masked, possibly
fragmented) with ping/pong and close handled by the caller.
"""

import base64
import hashlib
import struct

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


def accept_key(key):
    """The Sec-WebSocket-Accept value for a client's Sec-WebSocket-Key."""
    digest = hashlib.sha1((key + GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def encode_frame(payload, opcode=OP_TEXT):
    """One unmasked server-to-client frame (FIN set, no fragmentation)."""
    if isinstance(payload, str):
        payload = payload.encode()
    header = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header += bytes([n])
    elif n < 65536:
        header += bytes([126]) + struct.pack(">H", n)
    else:
        header += bytes([127]) + struct.pack(">Q", n)
    return header + payload


def read_frame(rfile):
    """Read one frame from a buffered reader.

    Returns ``(fin, opcode, payload)``; ``(False, None, b"")`` on a clean
    EOF before any frame byte. Raises ConnectionError on a truncated
    frame. Client payloads are unmasked when the mask bit is set.
    """
    head = rfile.read(2)
    if len(head) == 0:
        return False, None, b""
    if len(head) < 2:
        raise ConnectionError("truncated WebSocket frame header")
    fin = bool(head[0] & 0x80)
    opcode = head[0] & 0x0F
    masked = bool(head[1] & 0x80)
    length = head[1] & 0x7F
    if length == 126:
        length = struct.unpack(">H", rfile.read(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", rfile.read(8))[0]
    mask = rfile.read(4) if masked else None
    payload = rfile.read(length) if length else b""
    if len(payload) < length:
        raise ConnectionError("truncated WebSocket frame payload")
    if mask:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return fin, opcode, payload
