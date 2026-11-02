"""Deterministic virtual-clock telephony simulation of one bot turn.

Wires together, with one-way network delays:

  MockNovaSonic  <-- nova_net_ms -->  BridgeCore  <-- twilio_link_ms -->  TwilioEmulator  ->  PhoneLine (echo)
                                                                                                 ^ caller speech

Everything advances on a 20 ms clock (the Twilio frame), so all latencies are
quantised to 20 ms. The bridge code under test is the production
``novabridge.bridge.BridgeCore``; only the clock is virtual.
"""

from __future__ import annotations

import heapq
import itertools
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from novabridge import nova_events as ne
from novabridge.bridge import BridgeConfig, BridgeCore
from novabridge.twilio import FRAME_SAMPLES

from .mock_nova import MockNovaConfig, MockNovaSonic
from .phone_line import LineConfig, PhoneLine, TwilioEmulator

TICK = 20
DETECT_WINDOW_MS = 2000.0


@dataclass
class Scenario:
    sid: str
    kind: str                     # "barge" (caller interrupts at onset_ms) | "echo_only" (caller silent)
    seed: int
    bot_idx: int
    caller_idx: int
    onset_ms: float | None
    caller_level_dbfs: float
    line: LineConfig
    server_latency_ms: float
    nova_net_ms: float = 30.0
    twilio_link_ms: float = 40.0


@dataclass
class Result:
    sid: str
    kind: str
    mode: str
    echo_guard: bool
    echo_delay_ms: float
    erl_db: float
    noise_dbfs: float
    server_latency_ms: float
    onset_ms: float | None
    bot_audio_ms: float
    first_detect_ms: float | None = None
    first_detect_source: str | None = None     # local_vad | server
    false_bargein: bool = False
    false_source: str | None = None
    missed: bool = False
    detected: bool = False
    stop_latency_ear_ms: float | None = None   # onset -> end of last bot frame played to caller
    stop_latency_send_ms: float | None = None  # onset -> last bot frame sent by the bridge
    bot_played_ms: float = 0.0
    dropped_late_chunks: int = 0
    max_twilio_buffer_ms: float = 0.0
    delay_est_ms: float | None = None
    true_bulk_delay_ms: float | None = None
    erle_db: float | None = None
    guard_cpu_us_per_frame_p50: float | None = None
    guard_cpu_us_per_frame_p99: float | None = None
    extras: dict = field(default_factory=dict)


def run_scenario(sc: Scenario, bridge_cfg: BridgeConfig, corpus: dict,
                 mock_cfg: MockNovaConfig | None = None, keep_trace: bool = False) -> tuple[Result, dict | None]:
    rng = np.random.default_rng(sc.seed)
    mock_cfg = mock_cfg or MockNovaConfig()
    mock_cfg = MockNovaConfig(**{**asdict(mock_cfg), "server_latency_ms": sc.server_latency_ms})
    bot = corpus["bot"][sc.bot_idx]
    mock = MockNovaSonic(mock_cfg, [{"text": bot["text"], "pcm": bot["pcm"]}])
    bridge = BridgeCore(bridge_cfg)
    twl = TwilioEmulator(sc.twilio_link_ms)
    line = PhoneLine(sc.line, rng)

    # caller timeline at 8 kHz
    bot_ms = 1000 * len(bot["pcm"]) / bot["rate"]
    total_ms = bot_ms + 1500 + 1000
    n_total = int(total_ms * 8)
    caller = np.zeros(n_total)
    if sc.kind == "barge":
        c = corpus["caller"][sc.caller_idx]["pcm"].astype(np.float64) / 32768.0
        lvl = 10 * np.log10(np.mean(c[np.abs(c) > 1e-3] ** 2) + 1e-12)
        c *= 10 ** ((sc.caller_level_dbfs - lvl) / 20)
        s = int(sc.onset_ms * 8)
        e = min(n_total, s + len(c))
        caller[s:e] = c[: e - s]

    seq = itertools.count()
    nova_down: list = []   # (arrival, seq, event)   mock -> bridge
    nova_up: list = []     # (arrival, seq, event)   bridge -> mock
    inbound: list = []     # (arrival, seq, ulaw)    phone -> bridge (via Twilio)
    played_bot: list[tuple[float, bool]] = []
    cpu = []
    trace = {"t": [], "mic_db": [], "played": [], "echo_db": [],
             "tb": [], "in_db": [], "res_db": [], "thr_db": []} if keep_trace else None

    mock.start_response(0.0)
    n_ticks = int(total_ms // TICK)
    for k in range(n_ticks):
        t = k * TICK
        # 1. Nova -> bridge
        for te, ev in mock.poll(t):
            heapq.heappush(nova_down, (te + sc.nova_net_ms, next(seq), json.dumps(ev)))
        while nova_down and nova_down[0][0] <= t:
            _, _, ev = heapq.heappop(nova_down)
            bridge.on_nova_event(ev, t)
        # 2. caller audio -> bridge -> Nova
        while inbound and inbound[0][0] <= t:
            _, _, ulaw = heapq.heappop(inbound)
            t0 = time.perf_counter()
            pcm = bridge.on_caller_frame(ulaw, t)
            cpu.append(time.perf_counter() - t0)
            if keep_trace:
                from novabridge.g711 import ulaw_to_float as _u2f
                x_in = _u2f(ulaw).astype(np.float64)
                res = bridge.last_guard.residual if bridge.last_guard is not None else x_in
                trace["tb"].append(t)
                trace["in_db"].append(10 * np.log10(np.mean(x_in ** 2) + 1e-12))
                trace["res_db"].append(10 * np.log10(np.mean(res ** 2) + 1e-12))
                trace["thr_db"].append(10 * np.log10(bridge.vad.last_threshold + 1e-12))
                lg = bridge.last_guard
                trace.setdefault("guard", []).append(None if lg is None else
                                                     (lg.delay, round(lg.erle_db, 1), lg.double_talk, lg.converged,
                                                      round(lg.corr, 2),
                                                      round(10 * np.log10(np.mean(lg.upstream ** 2) + 1e-12), 1),
                                                      {k: round(float(v), 1) for k, v in bridge.guard.debug.items()}))
            ev = ne.audio_input("prompt-1", "audio-in", pcm)
            heapq.heappush(nova_up, (t + sc.nova_net_ms, next(seq), json.dumps(ev)))
        while nova_up and nova_up[0][0] <= t:
            ta, _, ev = heapq.heappop(nova_up)
            mock.on_input(json.loads(ev), ta)
        # 3. bridge -> Twilio
        for m in bridge.tick(t):
            twl.send_to_twilio(json.loads(m), t)
        # 4. Twilio plays [t, t+20)
        played = twl.play(t)
        played_bot.append((t, played is not None))
        if played is None:
            played = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        # 5. phone: echo + caller + noise for [t, t+20) -> arrives at the bridge after the link delay
        i0 = k * FRAME_SAMPLES
        ulaw, echo = line.mic_frame(played, caller[i0:i0 + FRAME_SAMPLES])
        heapq.heappush(inbound, (t + TICK + sc.twilio_link_ms, next(seq), ulaw))
        if keep_trace:
            from novabridge.g711 import ulaw_to_float
            mic = ulaw_to_float(ulaw)
            trace["t"].append(t)
            trace["mic_db"].append(10 * np.log10(np.mean(mic.astype(np.float64) ** 2) + 1e-12))
            trace["echo_db"].append(10 * np.log10(np.mean(echo ** 2) + 1e-12))
            trace["played"].append(float(np.sqrt(np.mean(played.astype(np.float64) ** 2))))

    # ------------------------------------------------------------ metrics
    r = Result(sid=sc.sid, kind=sc.kind, mode=bridge_cfg.mode, echo_guard=bridge_cfg.echo_guard,
               echo_delay_ms=sc.line.echo_delay_ms, erl_db=sc.line.erl_db, noise_dbfs=sc.line.noise_dbfs,
               server_latency_ms=sc.server_latency_ms, onset_ms=sc.onset_ms, bot_audio_ms=bot_ms)
    detections = []
    if bridge_cfg.mode == "full":
        detections += [(e.t_ms, "local_vad") for e in bridge.ctl.events if e.source == "local_vad"]
    # a server VAD onset while the response was playing makes the server truncate the
    # response, regardless of what the bridge does with the signal
    detections += [(i["t_detect"], "server") for i in mock.interrupts]
    detections.sort()
    played_times = [tt for tt, b in played_bot if b]
    r.bot_played_ms = TICK * len(played_times)
    r.dropped_late_chunks = bridge.ctl.dropped_late_chunks
    r.max_twilio_buffer_ms = TICK * twl.max_buffer_frames
    onset = sc.onset_ms
    if detections:
        r.first_detect_ms, r.first_detect_source = detections[0]
    false_d = [d for d in detections if onset is None or d[0] < onset]
    if false_d:
        r.false_bargein, r.false_source = True, false_d[0][1]
    if sc.kind == "barge" and not r.false_bargein:
        bot_playing_at_onset = any(onset - TICK <= tt <= onset + TICK for tt in played_times)
        true_d = [d for d in detections if onset <= d[0] <= onset + DETECT_WINDOW_MS]
        r.detected = bool(true_d)
        r.missed = bot_playing_at_onset and not true_d
        if bot_playing_at_onset:
            last_play = max(tt for tt in played_times if tt >= onset - TICK)
            r.stop_latency_ear_ms = last_play + TICK - onset
            sent = [s.t_ms for s in bridge.sent]
            r.stop_latency_send_ms = (max(sent) - onset) if sent else None
    if bridge.guard is not None:
        g = bridge.guard
        if g.est.delay is not None:
            r.delay_est_ms = g.est.delay / 8.0
        r.erle_db = bridge.last_guard.erle_db if bridge.last_guard else None
    # bulk delay the guard should find: Twilio link out + playout lead + echo + link in
    r.true_bulk_delay_ms = sc.twilio_link_ms * 2 + sc.line.echo_delay_ms
    if cpu:
        us = np.array(cpu) * 1e6
        r.guard_cpu_us_per_frame_p50 = float(np.percentile(us, 50))
        r.guard_cpu_us_per_frame_p99 = float(np.percentile(us, 99))
    r.extras = {"server_vad_onsets": mock.vad_onsets, "local_vad_onsets": bridge.local_vad_onsets,
                "bridge_events": [(e.t_ms, e.source, e.flushed_frames) for e in bridge.ctl.events],
                "twilio_clears": twl.cleared_at}
    if keep_trace:
        trace["detections"] = detections
        trace["onset_ms"] = onset
    return r, trace
