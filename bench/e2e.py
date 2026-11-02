"""End-to-end real-time run over real websockets (localhost, wall clock).

    python -m bench.e2e

Starts the production asyncio bridge (``novabridge.server.handle_call``) with
the mock Nova backend, then places calls from a fake Twilio/caller client that
streams 20 ms mu-law frames in real time, plays what the bridge sends, and
echoes it back through the simulated line. Measures the same stop latency as
the simulator, but in wall-clock time through the real async code path.
No network delay is added (localhost), so these numbers are lower than the
simulator's, which adds 40 ms Twilio and 30 ms Nova one-way delays.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
from pathlib import Path

import numpy as np
import websockets

from novabridge.bridge import BridgeConfig
from novabridge.server import handle_call
from testbed.audio import load_corpus
from testbed.mock_nova import MockNovaConfig
from testbed.phone_line import LineConfig
from testbed.realtime import MockNovaBackend, fake_twilio_call

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "naive (no client handler)": ("naive", False),
    "server signal only": ("server_signal", False),
    "server + local VAD + echo guard": ("full", True),
}
LINE = LineConfig(echo_delay_ms=100, erl_db=30, noise_dbfs=-60)  # benign line: this run is about latency


async def run(n_calls: int = 5, backend: str = "mock") -> dict:
    corpus = load_corpus()
    lead_in = None
    if backend == "nova":  # real Nova only talks after the caller does: ask a question first
        from testbed.audio import synth, trim
        lead_in = trim(synth("Hi, can you explain in detail how returns and refunds work at your store?",
                             "Daniel", 8000), 8000)
    results = {}
    port = 8765
    for name, (mode, guard) in CONFIGS.items():
        rows = []
        for i in range(n_calls):
            rng = np.random.default_rng(100 + i)
            bot = corpus["bot"][i % len(corpus["bot"])]
            caller = corpus["caller"][i % len(corpus["caller"])]
            lat_ms = float(rng.uniform(150, 450))
            if backend == "nova":
                from novabridge.nova_client import NovaSonicClient
                factory = NovaSonicClient
            else:
                factory = functools.partial(MockNovaBackend, [{"text": bot["text"], "pcm": bot["pcm"]}],
                                            MockNovaConfig(server_latency_ms=lat_ms))
            cores = []
            cfg = BridgeConfig(mode=mode, echo_guard=guard)
            server = await websockets.serve(lambda ws: handle_call(ws, factory, cfg, cores.append), "127.0.0.1", port)
            try:
                MockNovaBackend.instances.clear()
                # caller level about -22 dBFS
                pcm = caller["pcm"].astype(np.float64)
                pcm *= 10 ** ((-22 - 10 * np.log10(np.mean((pcm[np.abs(pcm) > 30] / 32768) ** 2))) / 20)
                onset_after = float(rng.uniform(2000, 4000))
                r = await fake_twilio_call(f"ws://127.0.0.1:{port}", LINE, pcm.astype(np.int16), onset_after, seed=i,
                                           max_s=40.0 if backend == "nova" else 20.0, lead_in_pcm8k=lead_in)
            finally:
                server.close()
                await server.wait_closed()
                port += 1
            await asyncio.sleep(0.2)
            mock = MockNovaBackend.instances[0].mock if MockNovaBackend.instances else None
            core = cores[0] if cores else None
            server_det = ([x["t_detect"] for x in mock.interrupts] if mock else
                          [e.t_ms for e in core.ctl.events if e.source == "server"] if core else [])
            local_det = [e.t_ms for e in core.ctl.events if e.source == "local_vad"] if core else []
            r.update({"call": i, "server_latency_ms": lat_ms if backend == "mock" else None, "server_detect_ms": server_det, "local_detect_ms": local_det})
            rows.append(r)
            r["bot_stopped_before_onset"] = r["stop_latency_ear_ms"] is None and not r["still_talking_at_onset_plus_4s"]
            print(f"{name:34s} call {i}: stop latency {r['stop_latency_ear_ms']} ms, clears={len(r['clears'])}")
        lat = [r["stop_latency_ear_ms"] for r in rows if r["stop_latency_ear_ms"] is not None]
        results[name] = {"calls": rows, "n": len(rows), "n_with_latency": len(lat),
                         "n_bot_stopped_before_caller_spoke": sum(r["bot_stopped_before_onset"] for r in rows),
                         "n_still_talking_4s_after_onset": sum(r["still_talking_at_onset_plus_4s"] for r in rows),
                         "stop_latency_ear_p50_ms": float(np.percentile(lat, 50)) if lat else None,
                         "stop_latency_ear_max_ms": float(np.max(lat)) if lat else None}
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "nova"], default="mock",
                    help="'nova' calls the real Bedrock API (costs money, needs Python >= 3.12 + AWS creds)")
    ap.add_argument("--calls", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    res = asyncio.run(run(a.calls, a.backend))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    if a.backend == "nova":
        payload = {"note": "REAL Nova 2 Sonic run via the bridge (fake caller over localhost websocket).",
                   "generated_unix": int(time.time()), "runtime_s": time.time() - t0, "results": res}
        (out / "e2e_real_nova.json").write_text(json.dumps(payload, indent=2, default=float))
        return
    payload = {"note": "Real-time localhost websocket run with the mock Nova backend (not real Nova). No network "
                       "delay added. Benign line (echo delay 100 ms, ERL 30 dB, noise -60 dBFS) so that the run isolates stop latency; false barge-ins are studied in the simulator.",
               "generated_unix": int(time.time()), "runtime_s": time.time() - t0, "results": res}
    (out / "e2e_realtime.json").write_text(json.dumps(payload, indent=2, default=float))
    from bench import report
    report.update_readme_e2e()
    for k, v in res.items():
        print(f"{k:34s} n={v['n']} p50={v['stop_latency_ear_p50_ms']} max={v['stop_latency_ear_max_ms']} "
              f"still_talking_after_4s={v['n_still_talking_4s_after_onset']}")


if __name__ == "__main__":
    main()
