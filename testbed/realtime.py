"""Real-time (wall-clock) pieces for the end-to-end websocket demo.

* ``MockNovaBackend`` -- the mock Nova server behind the same async interface as
  ``novabridge.nova_client.NovaSonicClient`` (JSON goes through
  ``json.dumps``/``loads`` like on the wire).
* ``fake_twilio_call`` -- a fake Twilio + caller: connects to the bridge's
  websocket, sends ``connected``/``start``/``media`` every 20 ms (caller speech
  + echo of what it played + noise, mu-law), buffers outbound ``media`` and
  honours ``clear`` exactly like Twilio's playback buffer.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque

import numpy as np

from novabridge import nova_events as ne
from novabridge import twilio
from novabridge.g711 import ulaw_to_float

from .mock_nova import MockNovaConfig, MockNovaSonic
from .phone_line import LineConfig, PhoneLine

_CORPUS = None


def _corpus():
    global _CORPUS
    if _CORPUS is None:
        from .audio import load_corpus
        _CORPUS = load_corpus()
    return _CORPUS


class MockNovaBackend:
    instances: list["MockNovaBackend"] = []

    def __init__(self, utterances: list[dict] | None = None, cfg: MockNovaConfig | None = None, net_ms: float = 30.0):
        if utterances is None:
            utterances = [{"text": b["text"], "pcm": b["pcm"]} for b in _corpus()["bot"]]
        self.cfg = cfg or MockNovaConfig(respond_after_user_turn=True)
        self.mock = MockNovaSonic(self.cfg, utterances)
        self.net_s = net_ms / 1000
        self._q: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._t0 = 0.0
        MockNovaBackend.instances.append(self)

    def _now(self) -> float:
        return (time.monotonic() - self._t0) * 1000

    async def start(self) -> None:
        self._t0 = time.monotonic()
        self.mock.start_response(0.0)
        self._task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        while True:
            for _, ev in self.mock.poll(self._now() - self.net_s * 1000):
                await self._q.put(json.dumps(ev))
            await asyncio.sleep(0.005)

    async def send_audio(self, pcm16: bytes) -> None:
        wire = json.dumps(ne.audio_input("prompt-1", "audio-in", pcm16))
        loop = asyncio.get_running_loop()
        loop.call_later(self.net_s, lambda: self.mock.on_input(json.loads(wire), self._now()))

    async def events(self):
        while True:
            yield json.loads(await self._q.get())

    async def close(self) -> None:
        if self._task:
            self._task.cancel()


async def fake_twilio_call(uri: str, line: LineConfig, caller_pcm8k: np.ndarray, onset_after_bot_ms: float,
                           seed: int = 0, max_s: float = 20.0, lead_in_pcm8k: np.ndarray | None = None) -> dict:
    """Place one call. The caller starts talking ``onset_after_bot_ms`` after first hearing the bot.

    ``lead_in_pcm8k``: optional caller utterance played from t=0 (e.g. a question), for backends
    such as real Nova Sonic that only speak after the user does."""
    import websockets

    rng = np.random.default_rng(seed)
    phone = PhoneLine(line, rng)
    sid = f"MZ{seed:08d}"
    buf: deque[bytes] = deque()
    clears: list[float] = []
    t0 = time.monotonic()
    now = lambda: (time.monotonic() - t0) * 1000  # noqa: E731
    played_log: list[tuple[float, bool]] = []
    caller = caller_pcm8k.astype(np.float64) / 32768.0
    lead = np.zeros(0) if lead_in_pcm8k is None else lead_in_pcm8k.astype(np.float64) / 32768.0

    async with websockets.connect(uri) as ws:
        async def rx():
            async for raw in ws:
                m = json.loads(raw)
                if m["event"] == "media":
                    buf.append(twilio.payload_bytes(m))
                elif m["event"] == "clear":
                    buf.clear()
                    clears.append(now())

        await ws.send(twilio.connected())
        await ws.send(twilio.start(sid))
        rx_task = asyncio.create_task(rx())
        loop = asyncio.get_running_loop()
        nxt = loop.time()
        first_heard = onset = None
        k = 0
        caller_pos = 0
        while now() < max_s * 1000:
            t = now()
            f = buf.popleft() if buf else None
            played_log.append((t, f is not None))
            if f is not None and first_heard is None:
                first_heard = t
                onset = t + onset_after_bot_ms
            played = ulaw_to_float(f) if f is not None else np.zeros(160, dtype=np.float32)
            frame = np.zeros(160)
            if first_heard is None and (k + 1) * 160 <= len(lead):
                frame = lead[k * 160:(k + 1) * 160].copy()
            if onset is not None and t >= onset and caller_pos < len(caller):
                seg = caller[caller_pos:caller_pos + 160]
                frame[: len(seg)] = seg
                caller_pos += 160
            ulaw, _ = phone.mic_frame(played, frame)
            k += 1
            await ws.send(twilio.inbound_media(sid, ulaw, k + 2, k, int(t)))
            if onset is not None and t > onset + 4000:
                break
            nxt += 0.020
            await asyncio.sleep(max(0.0, nxt - loop.time()))
        await ws.send(twilio.stop(sid, k + 3))
        rx_task.cancel()

    played_after = [tt for tt, b in played_log if b and onset is not None and tt >= onset - 20]
    lat = (max(played_after) + 20 - onset) if played_after else None
    still_talking = bool(played_log) and any(b for _, b in played_log[-5:])
    if still_talking:
        lat = None  # bot was still talking when the observation window (onset + 4 s) ended
    return {"onset_ms": onset, "first_heard_ms": first_heard, "stop_latency_ear_ms": lat, "clears": clears,
            "still_talking_at_onset_plus_4s": still_talking,
            "bot_frames_heard": sum(b for _, b in played_log)}
