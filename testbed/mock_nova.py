"""Mock Amazon Nova 2 Sonic server (NOT the real model).

Speaks the documented bidirectional-stream JSON events (see
``novabridge.nova_events``) and reproduces the behaviour that matters for
barge-in:

* generates a spoken response faster than real time (``gen_speed``) in
  ``chunk_ms`` ``audioOutput`` chunks at 24 kHz, after ``first_audio_ms``;
* runs its own *naive energy VAD* on the received 16 kHz audio -- it has no
  echo reference, so echo on the caller leg can make it barge in on itself;
* on a detected user onset while its response is (estimated to be) playing,
  waits ``server_latency_ms`` (decision + model latency), stops generating,
  closes the AUDIO content (END_TURN, as Pipecat's Nova Sonic service observes
  on barge-in) and emits an assistant FINAL TEXT ``contentEnd`` with
  ``stopReason: "INTERRUPTED"``;
* optionally emits ``late_chunks_after_interrupt`` more audio chunks of the
  interrupted content to exercise the client's late-chunk dropping.

All timing numbers are assumptions, configurable, and are *not* measurements
of Nova 2 Sonic. Real-Nova numbers are pending (docs/RUN_WITH_REAL_NOVA.md).
"""

from __future__ import annotations

import base64
import heapq
import itertools
from dataclasses import dataclass

import numpy as np

from novabridge import nova_events as ne


@dataclass
class MockNovaConfig:
    out_rate: int = 24000
    in_rate: int = 16000
    first_audio_ms: float = 250.0
    chunk_ms: float = 80.0
    gen_speed: float = 2.0
    server_latency_ms: float = 300.0
    vad_threshold_dbfs: float = -40.0
    vad_min_ms: float = 100.0
    playout_slack_ms: float = 500.0
    late_chunks_after_interrupt: int = 0
    respond_after_user_turn: bool = False   # used by the real-time demo
    end_of_turn_silence_ms: float = 700.0


class MockNovaSonic:
    def __init__(self, cfg: MockNovaConfig, utterances: list[dict]):
        self.cfg = cfg
        self.utterances = utterances  # [{"text": str, "pcm": int16 @ out_rate}]
        self._heap: list = []
        self._seq = itertools.count()
        self.ids = {"sessionId": "mock-session", "promptName": "prompt-1", "completionId": "c0"}
        self._n_resp = 0
        self.resp: dict | None = None
        # VAD state
        self._run_ms = 0.0
        self._silence_ms = 0.0
        self._in_speech = False
        self.vad_onsets: list[float] = []
        self.interrupts: list[dict] = []
        self.audio_in_frames = 0

    # ------------------------------------------------------------ schedule
    def _push(self, t: float, ev: dict) -> None:
        heapq.heappush(self._heap, (t, next(self._seq), ev))

    def start_response(self, t0: float) -> None:
        u = self.utterances[self._n_resp % len(self.utterances)]
        self._n_resp += 1
        k = self._n_resp
        ids = dict(self.ids, completionId=f"completion-{k}")
        audio_id, spec_id, final_id = f"audio-{k}", f"text-spec-{k}", f"text-final-{k}"
        c = self.cfg
        self._push(t0, ne.out_completion_start(ids))
        self._push(t0, ne.out_text_content_start(ids, spec_id, "ASSISTANT", "SPECULATIVE"))
        self._push(t0, ne.out_text(ids, spec_id, u["text"]))
        self._push(t0, ne.out_content_end(ids, spec_id, "TEXT", "PARTIAL_TURN"))
        t_audio = t0 + c.first_audio_ms
        self._push(t_audio, ne.out_audio_content_start(ids, audio_id, c.out_rate))
        pcm = u["pcm"]
        n = int(c.out_rate * c.chunk_ms / 1000)
        chunks = [pcm[i:i + n] for i in range(0, len(pcm), n)]
        dt = c.chunk_ms / c.gen_speed
        for i, ch in enumerate(chunks):
            self._push(t_audio + i * dt, ne.out_audio(ids, audio_id, ch.astype("<i2").tobytes()))
        t_done = t_audio + len(chunks) * dt
        self._push(t_done, ne.out_content_end(ids, audio_id, "AUDIO", "END_TURN"))
        self._push(t_done, ne.out_text_content_start(ids, final_id, "ASSISTANT", "FINAL"))
        self._push(t_done, ne.out_text(ids, final_id, u["text"]))
        self._push(t_done, ne.out_content_end(ids, final_id, "TEXT", "END_TURN"))
        self._push(t_done, ne.out_completion_end(ids))
        self.resp = {"ids": ids, "audio_id": audio_id, "final_id": final_id, "t_audio": t_audio,
                     "t_done": t_done, "dur_ms": 1000 * len(pcm) / c.out_rate, "chunks": chunks,
                     "dt": dt, "text": u["text"], "interrupted": False}

    def _playing_estimate(self, t: float) -> bool:
        r = self.resp
        return (r is not None and not r["interrupted"]
                and r["t_audio"] <= t <= r["t_audio"] + r["dur_ms"] + self.cfg.playout_slack_ms)

    def _interrupt(self, t_detect: float) -> None:
        r = self.resp
        r["interrupted"] = True
        ti = t_detect + self.cfg.server_latency_ms
        ids = r["ids"]
        # cancel everything of this response not yet emitted at ti
        keep, late = [], []
        for item in self._heap:
            t, _, ev = item
            body = ev["event"]
            cid = next(iter(body.values())).get("contentId")
            if t > ti and (cid in (r["audio_id"], r["final_id"]) or "completionEnd" in body):
                if "audioOutput" in body:
                    late.append(item)
                continue
            keep.append(item)
        self._heap = keep
        heapq.heapify(self._heap)
        late.sort()
        for i, (_, _, ev) in enumerate(late[: self.cfg.late_chunks_after_interrupt]):
            self._push(ti + 1e-3 * (i + 1), ev)
        audio_open = ti < r["t_done"]
        t_end = ti + 0.01
        if audio_open:
            self._push(t_end, ne.out_content_end(ids, r["audio_id"], "AUDIO", "END_TURN"))
        spoken_frac = min(max((ti - r["t_audio"]) / max(r["dur_ms"], 1), 0), 1)
        words = r["text"].split()
        self._push(t_end, ne.out_text_content_start(ids, r["final_id"], "ASSISTANT", "FINAL"))
        self._push(t_end, ne.out_text(ids, r["final_id"], " ".join(words[: int(len(words) * spoken_frac)])))
        self._push(t_end, ne.out_content_end(ids, r["final_id"], "TEXT", "INTERRUPTED"))
        self._push(t_end, ne.out_completion_end(ids, "INTERRUPTED"))
        self.interrupts.append({"t_detect": t_detect, "t_signal": ti})

    # --------------------------------------------------------------- input
    def on_input(self, event: dict, t: float) -> None:
        body = event["event"]
        if "audioInput" not in body:
            return
        pcm = np.frombuffer(base64.b64decode(body["audioInput"]["content"]), dtype="<i2").astype(np.float64) / 32768.0
        self.audio_in_frames += 1
        frame_ms = 1000 * len(pcm) / self.cfg.in_rate
        e_db = 10 * np.log10(np.mean(pcm * pcm) + 1e-12)
        if e_db > self.cfg.vad_threshold_dbfs:
            self._run_ms += frame_ms
            self._silence_ms = 0.0
        else:
            self._run_ms = 0.0
            self._silence_ms += frame_ms
        if not self._in_speech and self._run_ms >= self.cfg.vad_min_ms:
            self._in_speech = True
            self.vad_onsets.append(t)
            if self._playing_estimate(t):
                self._interrupt(t)
        elif self._in_speech and self._silence_ms >= self.cfg.end_of_turn_silence_ms:
            self._in_speech = False
            if self.cfg.respond_after_user_turn:
                self.start_response(t + 200)

    # -------------------------------------------------------------- output
    def poll(self, t: float) -> list[tuple[float, dict]]:
        out = []
        while self._heap and self._heap[0][0] <= t:
            te, _, ev = heapq.heappop(self._heap)
            out.append((te, ev))
        return out
