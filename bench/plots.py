"""Matplotlib figures for the benchmark (static PNGs for the README)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
})


def latency_plot(rows_a, out: Path):
    names = list(dict.fromkeys(r["config"] for r in rows_a))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        lat = [r["stop_latency_ear_ms"] for r in rows_a if r["config"] == name and r["kind"] == "barge"
               and r["stop_latency_ear_ms"] is not None and not r["missed"]]
        if not lat:
            continue
        y = len(names) - 1 - i
        ax.scatter(lat, y + rng.uniform(-0.18, 0.18, len(lat)), s=10, color=SERIES[0], alpha=0.45, linewidths=0)
        p50, p95 = np.percentile(lat, [50, 95])
        ax.plot([p50, p50], [y - 0.32, y + 0.32], color=INK, lw=2)
        ax.plot([p95, p95], [y - 0.25, y + 0.25], color=INK2, lw=1.5, ls=":")
        ax.text(p95 * 1.08, y, f"p50 {p50:.0f} ms · p95 {p95:.0f} ms (n={len(lat)})", va="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yticks(range(len(names)), list(reversed(names)))
    ax.set_xlabel("caller speech onset → bot audio stops at the caller's ear (ms, log scale)")
    ax.set_title("Barge-in stop latency, true barge-ins only (mock Nova server, simulated PSTN)", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=ax.get_xlim()[1] * 6)
    fig.tight_layout()
    fig.savefig(out / "stop_latency.png", dpi=150)
    plt.close(fig)


def sweep_plot(rows_b, out: Path):
    params = [("echo_delay_ms", "echo delay (ms)"), ("erl_db", "echo return loss, ERL (dB)"),
              ("noise_dbfs", "line noise (dBFS)")]
    configs = list(dict.fromkeys(r["config"] for r in rows_b))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey="row")
    for j, (p, label) in enumerate(params):
        for k, cfg in enumerate(configs):
            sel = [r for r in rows_b if r["sweep_param"] == p and r["config"] == cfg]
            xs = sorted({r["sweep_value"] for r in sel})
            fb = [100 * np.mean([r["false_bargein"] for r in sel if r["sweep_value"] == x and r["kind"] == "echo_only"]) for x in xs]
            ms = [100 * np.mean([r["missed"] for r in sel if r["sweep_value"] == x and r["kind"] == "barge"]) for x in xs]
            color = SERIES[1] if "no guard" in cfg else SERIES[0]
            mk = "s" if "no guard" in cfg else "o"
            short = "no echo guard" if "no guard" in cfg else "echo guard"
            axes[0, j].plot(xs, fb, color=color, marker=mk, ms=7, label=short)
            axes[1, j].plot(xs, ms, color=color, marker=mk, ms=7, label=short)
        axes[1, j].set_xlabel(label)
        for i in range(2):
            axes[i, j].set_ylim(-4, 104)
    axes[0, 0].set_ylabel("false barge-in rate (%)\ncaller silent, echo present")
    axes[1, 0].set_ylabel("missed true barge-in (%)")
    axes[0, 0].legend(frameon=False, loc="center left")
    fig.suptitle("One-factor sweeps around nominal line (150 ms echo delay, 12 dB ERL, −55 dBFS noise); "
                 "server + local VAD mode, mock Nova server", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "sweeps.png", dpi=150)
    plt.close(fig)


def trace_plot(corpus, out: Path):
    from novabridge.bridge import BridgeConfig
    from testbed.phone_line import LineConfig
    from testbed.sim import Scenario, run_scenario

    sc = Scenario("trace", "barge", 7, 1, 1, 4000.0, -22.0, LineConfig(150, 9, -55), 300)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for ax, guard in zip(axes, (False, True)):
        r, tr = run_scenario(sc, BridgeConfig(mode="full", echo_guard=guard), corpus, keep_trace=True)
        tb = np.array(tr["tb"]) / 1000
        ax.plot(tb, tr["in_db"], color=INK2, lw=1, label="caller leg at bridge (echo + noise + caller)")
        if guard:
            ax.plot(tb, tr["res_db"], color=SERIES[0], lw=1.2, label="after echo guard (residual)")
        ax.plot(tb, tr["thr_db"], color=SERIES[1], lw=1.5, ls="--", label="local VAD threshold")
        ax.axvline(sc.onset_ms / 1000, color=INK, lw=1)
        ax.text(sc.onset_ms / 1000 - 0.05, -8, "caller starts talking", fontsize=8.5, ha="right")
        for t, src in tr["detections"][:1]:
            ax.axvline(t / 1000, color=SERIES[2], lw=2)
            ax.text(t / 1000 + 0.05, -8, f"barge-in fired ({src})", fontsize=8.5, color=INK)
        ax.set_ylim(-75, 0)
        ax.set_ylabel("frame energy (dBFS)")
        ax.set_title("with echo guard (threshold gaps = frame classified as echo by the correlation gate)" if guard
                     else "without echo guard", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3)
    axes[1].set_xlabel("time (s)")
    axes[1].set_xlim(0, 6)
    fig.suptitle("One call, 9 dB ERL, 150 ms echo: without a reference the bot's echo fires the VAD", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "example_trace.png", dpi=150)
    plt.close(fig)


def make_all(rows_a, rows_b, summary, out: Path, corpus):
    latency_plot(rows_a, out)
    sweep_plot(rows_b, out)
    trace_plot(corpus, out)
