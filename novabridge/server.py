"""Asyncio websocket server: Twilio Media Streams <-> Nova 2 Sonic.

    python -m novabridge.server --backend mock            # local demo, no AWS
    python -m novabridge.server --backend nova             # real Nova 2 Sonic (Python >= 3.12, AWS creds)

Point a Twilio ``<Connect><Stream url="wss://.../media"/></Connect>`` at it.
Per call: one ``BridgeCore`` + one backend (``NovaSonicClient`` or the mock).
Three tasks: Twilio receive loop, Nova event loop, and a 20 ms ticker that
paces outbound audio. ``clear`` messages are flushed as soon as a barge-in
fires, not on the next tick.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Callable

import websockets

from .bridge import BridgeConfig, BridgeCore

log = logging.getLogger("novabridge")


async def handle_call(ws, backend_factory: Callable, cfg: BridgeConfig, on_done: Callable | None = None) -> None:
    core: BridgeCore | None = None
    backend = None
    tasks: list[asyncio.Task] = []
    t0 = time.monotonic()
    now = lambda: (time.monotonic() - t0) * 1000.0  # noqa: E731
    send_lock = asyncio.Lock()

    async def flush(msgs):
        if msgs:
            async with send_lock:
                for m in msgs:
                    await ws.send(m)

    async def nova_loop():
        async for ev in backend.events():
            core.on_nova_event(ev, now())
            await flush(core.take_outbox())

    async def ticker():
        loop = asyncio.get_running_loop()
        nxt = loop.time()
        while True:
            await flush(core.tick(now()))
            nxt += 0.020
            await asyncio.sleep(max(0.0, nxt - loop.time()))

    try:
        async for raw in ws:
            msg = json.loads(raw)
            ev = msg.get("event")
            if ev == "start":
                core = BridgeCore(cfg, msg["start"]["streamSid"])
                backend = backend_factory()
                await backend.start()
                tasks = [asyncio.create_task(nova_loop()), asyncio.create_task(ticker())]
                log.info("call started %s mode=%s guard=%s", msg["start"]["streamSid"], cfg.mode, cfg.echo_guard)
            elif ev == "media" and core is not None:
                import base64
                pcm = core.on_caller_frame(base64.b64decode(msg["media"]["payload"]), now())
                await flush(core.take_outbox())
                await backend.send_audio(pcm)
            elif ev == "stop":
                break
    finally:
        for t in tasks:
            t.cancel()
        if backend is not None:
            await backend.close()
        if on_done is not None and core is not None:
            on_done(core)


def make_backend_factory(name: str):
    if name == "nova":
        from .nova_client import NovaSonicClient
        return NovaSonicClient
    if name == "mock":
        from testbed.realtime import MockNovaBackend
        return MockNovaBackend
    raise ValueError(name)


async def serve(host: str, port: int, backend: str, cfg: BridgeConfig) -> None:
    factory = make_backend_factory(backend)
    async with websockets.serve(lambda ws: handle_call(ws, factory, cfg), host, port):
        log.info("listening on ws://%s:%d (backend=%s)", host, port, backend)
        await asyncio.Future()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--backend", choices=["mock", "nova"], default="mock")
    ap.add_argument("--mode", choices=["naive", "server_signal", "full"], default="full")
    ap.add_argument("--no-echo-guard", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(a.host, a.port, a.backend, BridgeConfig(mode=a.mode, echo_guard=not a.no_echo_guard)))


if __name__ == "__main__":
    main()
