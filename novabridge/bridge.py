"""Transport-agnostic bridge core: Twilio (8 kHz mu-law) <-> Nova 2 Sonic.

``BridgeCore`` is a synchronous, clock-driven state machine so the exact same
code runs inside the asyncio websocket server (``novabridge.server``) and inside
the deterministic virtual-clock testbed (``testbed.sim``).

Inputs:  caller frames (Twilio ``media``), Nova output events, a 20 ms tick.
Outputs: Twilio messages (``media`` / ``clear``) and 16-bit PCM for Nova.

Modes (for the benchmark):
  naive          forward Nova audio to Twilio as soon as it arrives, ignore
                 interruptions (what you get if you only read the event docs
                 for the audio path).
  server_signal  forward immediately, but on Nova's INTERRUPTED signal flush,
                 send Twilio ``clear`` and drop late chunks (the documented
                 client duty; the aws-samples Twilio integration sends
                 ``clear`` on INTERRUPTED).
  full           paced playout (small lead), server signal OR local VAD on
                 echo-guarded audio triggers the flush immediately.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import twilio
from .bargein import BargeInController, Frame, PlayoutQueue
from .echo_guard import EchoGuard, EchoGuardConfig
from .g711 import float_to_pcm16, float_to_ulaw, ulaw_to_float
from .nova_events import parse_output
from .resample import StreamingResampler
from .vad import EnergyVAD

MODES = ("naive", "server_signal", "full")
FRAME = twilio.FRAME_SAMPLES


@dataclass
class BridgeConfig:
    mode: str = "full"
    echo_guard: bool = True
    nova_in_rate: int = 16000
    nova_out_rate: int = 24000
    lead_frames: int = 3                 # outbound frames kept ahead of real time (paced modes)
    pace: bool | None = None             # default: pace only in "full" mode
    local_vad_abs_dbfs: float = -45.0
    local_vad_min_frames: int = 3        # 60 ms of speech before a local barge-in
    guard: EchoGuardConfig = field(default_factory=EchoGuardConfig)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.pace is None:
            self.pace = self.mode == "full"


@dataclass
class SentFrame:
    t_ms: float
    content_id: str | None


class BridgeCore:
    def __init__(self, cfg: BridgeConfig | None = None, stream_sid: str = "MZ-test"):
        self.cfg = cfg or BridgeConfig()
        self.sid = stream_sid
        self.up = StreamingResampler(8000, self.cfg.nova_in_rate)
        self.down = StreamingResampler(self.cfg.nova_out_rate, 8000)
        self.queue = PlayoutQueue()
        self.ctl = BargeInController()
        self.guard = EchoGuard(self.cfg.guard) if self.cfg.echo_guard else None
        self.vad = EnergyVAD(abs_threshold_dbfs=self.cfg.local_vad_abs_dbfs,
                             min_on_frames=self.cfg.local_vad_min_frames)
        self.ref_fifo: deque[np.ndarray] = deque()   # mirror of Twilio's playback buffer
        self.current_ref = np.zeros(FRAME, dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)
        self._playing = False
        self.audio_content: str | None = None      # last AUDIO contentId from Nova
        self.audio_open = False
        self.outbox: list[str] = []
        self.sent: list[SentFrame] = []
        self.last_guard = None
        self.local_vad_onsets: list[float] = []

    # ------------------------------------------------------------- helpers
    def bot_active(self) -> bool:
        return (len(self.queue) > 0 or len(self.ref_fifo) > 0
                or (self.audio_open and not self.ctl.is_dead(self.audio_content)))

    def take_outbox(self) -> list[str]:
        out, self.outbox = self.outbox, []
        return out

    def barge_in(self, now_ms: float, source: str) -> bool:
        ev = self.ctl.interrupt(now_ms, source, self.audio_content, self.queue)
        if ev is None:
            return False
        self.ref_fifo.clear()
        self._playing = False
        self.down.reset()
        self._carry = np.zeros(0, dtype=np.float32)
        self.outbox.append(twilio.clear(self.sid))
        return True

    # -------------------------------------------------------------- inputs
    def on_caller_frame(self, ulaw: bytes, now_ms: float) -> bytes:
        """One inbound 20 ms frame. Returns PCM16 (nova_in_rate) to send upstream."""
        mic = ulaw_to_float(ulaw)
        ext_thr = 0.0
        vad_in = upstream = mic
        if self.guard is not None:
            g = self.guard.process(mic, self.current_ref)
            self.last_guard = g
            vad_in, upstream, ext_thr = g.residual, g.upstream, g.threshold
        onset = self.vad.update(vad_in, ext_thr)
        if onset:
            self.local_vad_onsets.append(now_ms)
            if self.cfg.mode == "full" and self.bot_active():
                self.barge_in(now_ms, "local_vad")
        return float_to_pcm16(self.up.process(upstream)).tobytes()

    def on_nova_event(self, event_json: dict | str, now_ms: float) -> None:
        ev = parse_output(event_json)
        if ev.kind == "content_start" and ev.type == "AUDIO":
            self.audio_content, self.audio_open = ev.content_id, True
            if ev.sample_rate and ev.sample_rate != self.cfg.nova_out_rate:
                raise ValueError(f"Nova output rate {ev.sample_rate} != configured {self.cfg.nova_out_rate}")
        elif ev.kind == "audio":
            if self.ctl.is_dead(ev.content_id):
                self.ctl.dropped_late_chunks += 1
                return
            x = np.concatenate([self._carry, self.down.process(ev.pcm.astype(np.float32) / 32768.0)])
            n = len(x) // FRAME
            for i in range(n):
                self.queue.push(Frame(float_to_ulaw(x[i * FRAME:(i + 1) * FRAME]), ev.content_id))
            self._carry = x[n * FRAME:]
        elif ev.kind == "content_end" and ev.type == "AUDIO" and ev.content_id == self.audio_content:
            self.audio_open = False
        elif ev.kind == "interrupted":
            if ev.type == "AUDIO" and ev.content_id == self.audio_content:
                self.audio_open = False
            if self.cfg.mode != "naive":
                self.barge_in(now_ms, "server")

    # ---------------------------------------------------------------- tick
    def tick(self, now_ms: float) -> list[str]:
        """Called every 20 ms. Returns Twilio messages to send now."""
        out = self.take_outbox()
        if not self.cfg.pace:
            n = len(self.queue)
        elif not self._playing:
            n = min(self.cfg.lead_frames, len(self.queue))
            self._playing = n > 0
        else:
            n = 1 if len(self.queue) else 0
            self._playing = n > 0
        for _ in range(n):
            f = self.queue.pop()
            out.append(twilio.media(self.sid, f.ulaw))
            self.sent.append(SentFrame(now_ms, f.content_id))
            self.ref_fifo.append(ulaw_to_float(f.ulaw))
        # the frame Twilio is (approximately) playing during the next 20 ms
        self.current_ref = self.ref_fifo.popleft() if self.ref_fifo else np.zeros(FRAME, dtype=np.float32)
        return out
