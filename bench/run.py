"""Benchmark: barge-in stop latency and false/missed barge-ins on a simulated phone line.

    python -m bench.run            # full benchmark (eval seeds), writes results/
    python -m bench.run --quick    # small smoke run
    python -m bench.run --dev      # tuning seeds (0..9999), writes results_dev/

All numbers come from the mock Nova Sonic server + simulated PSTN line in
``testbed/``; they are NOT measurements of the real Nova 2 Sonic service.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from novabridge.bridge import BridgeConfig
from testbed.audio import load_corpus
from testbed.mock_nova import MockNovaConfig
from testbed.phone_line import LineConfig
from testbed.sim import Scenario, run_scenario

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    # name: (mode, echo_guard)
    "naive (no client handler)": ("naive", False),
    "server signal only": ("server_signal", False),
    "server signal + echo guard": ("server_signal", True),
    "server + local VAD, no guard": ("full", False),
    "server + local VAD + echo guard": ("full", True),
}
SWEEP_CONFIGS = ["server + local VAD, no guard", "server + local VAD + echo guard"]
NOMINAL = LineConfig(echo_delay_ms=150, erl_db=12, noise_dbfs=-55)
SWEEPS = {
    "echo_delay_ms": [50, 100, 150, 200, 300],
    "erl_db": [6, 9, 12, 18, 24, 30],
    "noise_dbfs": [-70, -60, -50, -45, -40],
}

_CORPUS = None


def _init():
    global _CORPUS
    _CORPUS = load_corpus()


def _bot_ms(corpus, i):
    b = corpus["bot"][i]
    return 1000 * len(b["pcm"]) / b["rate"]


def make_scenario(seed: int, kind: str, corpus: dict, line: LineConfig | None = None, tag: str = "") -> Scenario:
    rng = np.random.default_rng(seed)
    bot_idx = int(rng.integers(len(corpus["bot"])))
    caller_idx = int(rng.integers(len(corpus["caller"])))
    if line is None:
        line = LineConfig(echo_delay_ms=float(rng.choice([50, 100, 150, 200, 250, 300])),
                          erl_db=float(rng.uniform(6, 30)), noise_dbfs=float(rng.uniform(-65, -45)))
    onset = float(round(rng.uniform(1000, min(6000, _bot_ms(corpus, bot_idx) - 1500)) / 20) * 20)
    return Scenario(sid=f"{tag}{kind}-{seed}", kind=kind, seed=seed, bot_idx=bot_idx, caller_idx=caller_idx,
                    onset_ms=onset if kind == "barge" else None,
                    caller_level_dbfs=float(rng.uniform(-26, -18)), line=line,
                    server_latency_ms=float(rng.uniform(150, 450)))


def _job(args):
    sc, cfg_name, mode, guard = args
    r, _ = run_scenario(sc, BridgeConfig(mode=mode, echo_guard=guard), _CORPUS, MockNovaConfig())
    d = asdict(r)
    d["config"] = cfg_name
    d["extras"] = json.dumps(d["extras"])
    return d


def run_jobs(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        return list(ex.map(_job, jobs, chunksize=4))


def pct(x, q):
    x = [v for v in x if v is not None]
    return float(np.percentile(x, q)) if x else None


def summarise(rows, by):
    out = {}
    for key in dict.fromkeys(tuple(r[b] for b in by) for r in rows):  # first-appearance order
        sel = [r for r in rows if tuple(r[b] for b in by) == key]
        barge = [r for r in sel if r["kind"] == "barge"]
        echo = [r for r in sel if r["kind"] == "echo_only"]
        lat = [r["stop_latency_ear_ms"] for r in barge if r["stop_latency_ear_ms"] is not None and not r["missed"]]
        out[" | ".join(map(str, key))] = {
            **{b: k for b, k in zip(by, key)},
            "n_barge": len(barge), "n_echo_only": len(echo),
            "stop_latency_ear_p50_ms": pct(lat, 50), "stop_latency_ear_p95_ms": pct(lat, 95),
            "stop_latency_ear_n": len(lat),
            "false_bargein_echo_only_rate": (sum(r["false_bargein"] for r in echo) / len(echo)) if echo else None,
            "false_bargein_echo_only_local": sum(r["false_bargein"] and r["false_source"] == "local_vad" for r in echo),
            "false_bargein_echo_only_server": sum(r["false_bargein"] and r["false_source"] == "server" for r in echo),
            "false_bargein_before_onset_rate": (sum(r["false_bargein"] for r in barge) / len(barge)) if barge else None,
            "missed_rate": (sum(r["missed"] for r in barge) / len(barge)) if barge else None,
            "bot_played_fraction_echo_only": float(np.mean([r["bot_played_ms"] / r["bot_audio_ms"] for r in echo])) if echo else None,
        }
    return out


def cpu_probe(corpus, seed0, n=10):
    """Per-frame inbound-path cost, measured serially so parallel workers do not distort it."""
    import novabridge.bridge as nb
    times = []
    orig = nb.BridgeCore.on_caller_frame

    def timed(self, ulaw, now_ms):
        t = time.perf_counter()
        out = orig(self, ulaw, now_ms)
        times.append(time.perf_counter() - t)
        return out

    nb.BridgeCore.on_caller_frame = timed
    try:
        for i in range(n):
            sc = make_scenario(seed0 + 9000 + i, "barge" if i % 2 else "echo_only", corpus)
            run_scenario(sc, BridgeConfig(mode="full", echo_guard=True), corpus)
    finally:
        nb.BridgeCore.on_caller_frame = orig
    us = np.array(times) * 1e6
    return {"frames": int(us.size), "p50": float(np.percentile(us, 50)), "p99": float(np.percentile(us, 99)),
            "max": float(us.max())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dev", action="store_true", help="use tuning seeds and write to results_dev/")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    n_main = 12 if a.quick else 100
    n_sweep = 4 if a.quick else 20
    seed0 = 0 if a.dev else 10_000
    out = ROOT / ("results_dev" if a.dev else "results_quick" if a.quick else "results")
    out.mkdir(exist_ok=True)
    corpus = load_corpus()
    t0 = time.time()

    # ---- experiment A: all client configurations on the same randomised calls
    scen = [make_scenario(seed0 + i, kind, corpus) for kind in ("barge", "echo_only") for i in range(n_main)]
    jobs = [(sc, name, m, g) for name, (m, g) in CONFIGS.items() for sc in scen]
    rows_a = run_jobs(jobs, a.workers)
    print(f"experiment A: {len(rows_a)} runs in {time.time() - t0:.0f}s")

    # ---- experiment B: one-factor-at-a-time sweeps around the nominal line
    t1 = time.time()
    jobs, meta = [], []
    for param, values in SWEEPS.items():
        for v in values:
            line = LineConfig(**{**asdict(NOMINAL), param: float(v)})
            for kind in ("barge", "echo_only"):
                for i in range(n_sweep):
                    sc = make_scenario(seed0 + 5000 + i, kind, corpus, line=line, tag=f"{param}={v}-")
                    for name in SWEEP_CONFIGS:
                        m, g = CONFIGS[name]
                        jobs.append((sc, name, m, g))
                        meta.append((param, v))
    rows_b = run_jobs(jobs, a.workers)
    for r, (param, v) in zip(rows_b, meta):
        r["sweep_param"], r["sweep_value"] = param, v
    print(f"experiment B: {len(rows_b)} runs in {time.time() - t1:.0f}s")

    import csv
    for name, rows in (("runs_main.csv", rows_a), ("runs_sweep.csv", rows_b)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    guarded = [r for r in rows_a if r["echo_guard"] and r["guard_cpu_us_per_frame_p50"] is not None]
    cpu = cpu_probe(corpus, seed0)
    summary = {
        "note": "Mock Nova Sonic server + simulated PSTN line (testbed/). Not real Nova 2 Sonic measurements.",
        "generated_unix": int(time.time()),
        "platform": f"{platform.platform()} / Python {platform.python_version()}",
        "n_main_per_kind": n_main, "n_sweep_per_point_per_kind": n_sweep, "seed0": seed0,
        "assumptions": {
            "mock_nova": asdict(MockNovaConfig()),
            "server_latency_ms_range": [150, 450], "nova_net_one_way_ms": 30, "twilio_link_one_way_ms": 40,
            "main_line_sampling": {"echo_delay_ms": [50, 100, 150, 200, 250, 300], "erl_db": [6, 30],
                                   "noise_dbfs": [-65, -45]},
            "caller_level_dbfs": [-26, -18], "bot_level_dbfs": -20, "tick_ms": 20,
        },
        "main": summarise(rows_a, ["config"]),
        "sweep": summarise(rows_b, ["sweep_param", "sweep_value", "config"]),
        "guard_cpu_us_per_20ms_frame": {
            **cpu,
            "note": "wall-clock time of BridgeCore.on_caller_frame (mu-law decode + echo guard + VAD + 8k->16k "
                    "resample) per 20 ms frame; measured in a separate serial pass (no parallel workers) over "
                    "10 calls in 'server + local VAD + echo guard' mode",
        },
        "delay_estimate_abs_error_ms_p50": pct([abs(r["delay_est_ms"] - r["true_bulk_delay_ms"]) for r in guarded
                                                if r["delay_est_ms"] is not None], 50),
        "runtime_s": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    from bench import plots, report
    plots.make_all(rows_a, rows_b, summary, out, corpus)
    report.write_table(summary, out)
    if not a.dev and not a.quick:
        report.update_readme(summary)
    print(report.table_md(summary))


if __name__ == "__main__":
    main()
