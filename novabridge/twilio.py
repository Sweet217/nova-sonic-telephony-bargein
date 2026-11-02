"""Twilio Media Streams (bidirectional) message helpers.

Shapes follow Twilio's documented websocket protocol:
  inbound  (Twilio -> app): connected, start, media, mark, stop
  outbound (app -> Twilio): media, mark, clear

Audio is base64 mu-law, 8 kHz mono. Twilio sends 20 ms (160-byte) frames.
"clear" tells Twilio to discard all outbound audio it has buffered but not yet
played -- the only way to make a bot stop talking quickly once audio has been
sent ahead of real time.
"""

from __future__ import annotations

import base64
import json

FRAME_MS = 20
SAMPLE_RATE = 8000
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 160


def media(stream_sid: str, ulaw: bytes) -> str:
    return json.dumps({
        "event": "media",
        "streamSid": stream_sid,
        "media": {"payload": base64.b64encode(ulaw).decode("ascii")},
    })


def clear(stream_sid: str) -> str:
    return json.dumps({"event": "clear", "streamSid": stream_sid})


def mark(stream_sid: str, name: str) -> str:
    return json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": name}})


# ---- messages Twilio would send (used by the fake phone in the testbed) ----

def connected() -> str:
    return json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"})


def start(stream_sid: str, call_sid: str = "CA-test", seq: int = 1) -> str:
    return json.dumps({
        "event": "start",
        "sequenceNumber": str(seq),
        "streamSid": stream_sid,
        "start": {
            "streamSid": stream_sid,
            "callSid": call_sid,
            "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": SAMPLE_RATE, "channels": 1},
            "customParameters": {},
        },
    })


def inbound_media(stream_sid: str, ulaw: bytes, seq: int, chunk: int, timestamp_ms: int) -> str:
    return json.dumps({
        "event": "media",
        "sequenceNumber": str(seq),
        "streamSid": stream_sid,
        "media": {
            "track": "inbound",
            "chunk": str(chunk),
            "timestamp": str(timestamp_ms),
            "payload": base64.b64encode(ulaw).decode("ascii"),
        },
    })


def stop(stream_sid: str, seq: int) -> str:
    return json.dumps({"event": "stop", "sequenceNumber": str(seq), "streamSid": stream_sid, "stop": {}})


def payload_bytes(msg: dict) -> bytes:
    return base64.b64decode(msg["media"]["payload"])
