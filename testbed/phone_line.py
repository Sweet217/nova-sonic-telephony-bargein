"""Simulated PSTN leg: Twilio playback buffer + caller handset/hybrid echo path.

What the caller hears is the 8 kHz mu-law audio Twilio plays out. What comes
back up the line is

    mic = caller_speech + echo(played) + line_noise      -> mu-law -> Twilio

with echo(played) = played * h, where h is a bulk delay (``echo_delay_ms``), a
telephone-band coloration (2nd-order 300-3400 Hz band-pass) with a few sparse
reflections, scaled so that its power gain equals -ERL dB.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter

from novabridge.g711 import float_to_ulaw, ulaw_to_float
from novabridge.twilio import FRAME_SAMPLES

FS = 8000


@dataclass
class LineConfig:
    echo_delay_ms: float = 150.0
    erl_db: float = 12.0
    noise_dbfs: float = -55.0
    coloration: bool = True


def build_echo_ir(cfg: LineConfig) -> np.ndarray:
    d = int(round(cfg.echo_delay_ms * FS / 1000))
    sparse = np.zeros(160)
    for k, a in ((0, 1.0), (29, 0.35), (71, -0.2), (113, 0.1)):
        sparse[k] = a
    if cfg.coloration:
        b, a = butter(2, [300 / (FS / 2), 3400 / (FS / 2)], btype="band")
        imp = np.zeros(400)
        imp[0] = 1
        sparse = np.convolve(sparse, lfilter(b, a, imp))[:400]
    h = np.concatenate([np.zeros(d), sparse])
    h *= np.sqrt(10 ** (-cfg.erl_db / 10) / np.sum(h * h))
    return h


class PhoneLine:
    """Caller side of the call: produces the inbound (caller -> Twilio) mu-law frames."""

    def __init__(self, cfg: LineConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.h = build_echo_ir(cfg)
        self._zi = np.zeros(len(self.h) - 1)
        # pink-ish line noise: white noise through a one-pole low-pass, normalised to the target level
        self._na = 0.9
        self._nzi = np.zeros(1)
        self._ngain = 10 ** (cfg.noise_dbfs / 20) * np.sqrt(1 - self._na ** 2)

    def mic_frame(self, played: np.ndarray, caller: np.ndarray) -> tuple[bytes, np.ndarray]:
        echo, self._zi = lfilter(self.h, [1.0], played.astype(np.float64), zi=self._zi)
        w = self.rng.standard_normal(len(played))
        noise, self._nzi = lfilter([self._ngain], [1.0, -self._na], w, zi=self._nzi)
        x = caller + echo + noise
        return float_to_ulaw(x), echo


class TwilioEmulator:
    """Twilio's side of a bidirectional Media Stream: buffers outbound media, honours ``clear``,
    plays one 20 ms frame per tick. One-way websocket delay ``link_ms`` in each direction."""

    def __init__(self, link_ms: float = 40.0):
        self.link_ms = link_ms
        self._in: deque[tuple[float, dict]] = deque()
        self.buffer: deque[bytes] = deque()
        self.cleared_at: list[float] = []
        self.max_buffer_frames = 0

    def send_to_twilio(self, msg: dict, now_ms: float) -> None:
        self._in.append((now_ms + self.link_ms, msg))

    def play(self, now_ms: float) -> np.ndarray | None:
        """Apply messages that have arrived, then return the frame played in [now, now+20) (None = silence)."""
        while self._in and self._in[0][0] <= now_ms:
            _, msg = self._in.popleft()
            if msg["event"] == "media":
                import base64
                self.buffer.append(base64.b64decode(msg["media"]["payload"]))
            elif msg["event"] == "clear":
                self.buffer.clear()
                self.cleared_at.append(now_ms)
        self.max_buffer_frames = max(self.max_buffer_frames, len(self.buffer))
        if not self.buffer:
            return None
        f = self.buffer.popleft()
        assert len(f) == FRAME_SAMPLES
        return ulaw_to_float(f)
