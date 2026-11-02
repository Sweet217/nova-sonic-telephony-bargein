import numpy as np
import pytest

from novabridge.echo_guard import EchoGuard
from novabridge.g711 import float_to_ulaw, ulaw_to_float
from novabridge.vad import EnergyVAD
from testbed.phone_line import LineConfig, PhoneLine


def speechlike(n, rng, level=0.1):
    """Amplitude-modulated, coloured noise with pauses: a stand-in for speech in unit tests."""
    x = np.convolve(rng.standard_normal(n), [1, 0.8, 0.4, 0.2], mode="same")
    env = (np.sin(np.arange(n) * 2 * np.pi * 3 / 8000) > -0.3).astype(float)
    env = np.convolve(env, np.ones(160) / 160, mode="same")
    return level * x * env / np.std(x)


def run(erl_db, delay_ms, near=None, seconds=6.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(8000 * seconds)
    ref = ulaw_to_float(float_to_ulaw(speechlike(n, rng))).astype(np.float64)
    line = PhoneLine(LineConfig(echo_delay_ms=delay_ms, erl_db=erl_db, noise_dbfs=-60), rng)
    near = np.zeros(n) if near is None else near
    g, vad = EchoGuard(), EnergyVAD()
    onsets, frames = [], []
    for k in range(n // 160):
        s = slice(k * 160, (k + 1) * 160)
        ul, _ = line.mic_frame(ref[s], near[s])
        mic = ulaw_to_float(ul)
        f = g.process(mic, ref[s])
        frames.append((np.mean(mic.astype(float) ** 2), np.mean(f.residual ** 2)))
        if vad.update(f.residual, f.threshold):
            onsets.append(k * 20)
    return g, onsets, np.array(frames)


@pytest.mark.parametrize("delay_ms", [50, 200, 300])
def test_delay_estimate_locks(delay_ms):
    g, _, _ = run(12, delay_ms)
    assert g.est.delay is not None
    assert abs(g.est.delay / 8 - delay_ms) < 10


@pytest.mark.parametrize("erl_db", [6, 12, 20])
def test_cancels_echo_and_no_false_onset(erl_db):
    g, onsets, fr = run(erl_db, 150)
    tail = fr[len(fr) // 2:]
    erle = 10 * np.log10(tail[:, 0].sum() / tail[:, 1].sum())
    assert erle > 10, erle
    assert onsets == []


def test_without_guard_the_echo_triggers_a_plain_vad():
    rng = np.random.default_rng(0)
    ref = speechlike(8000 * 3, rng)
    line = PhoneLine(LineConfig(echo_delay_ms=150, erl_db=12, noise_dbfs=-60), rng)
    vad = EnergyVAD()
    fired = False
    for k in range(len(ref) // 160):
        ul, _ = line.mic_frame(ref[k * 160:(k + 1) * 160], np.zeros(160))
        fired |= vad.update(ulaw_to_float(ul))
    assert fired


def test_near_end_speech_still_detected_during_echo():
    rng = np.random.default_rng(5)
    n = 8000 * 6
    near = np.zeros(n)
    near[8000 * 4: 8000 * 5] = speechlike(8000, rng, level=0.08)
    _, onsets, _ = run(12, 150, near=near)
    assert onsets and 4000 <= onsets[0] <= 4200
