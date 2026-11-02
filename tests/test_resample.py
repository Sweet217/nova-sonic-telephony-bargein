import numpy as np
import pytest

from novabridge.resample import StreamingResampler


def tone(f, rate, seconds=1.0, amp=0.5):
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


@pytest.mark.parametrize("fin,fout,frame", [(8000, 16000, 160), (24000, 8000, 480), (16000, 8000, 320), (8000, 24000, 160)])
def test_streaming_equals_one_shot(fin, fout, frame):
    x = np.random.default_rng(0).standard_normal(fin).astype(np.float32) * 0.1
    one = StreamingResampler(fin, fout).process(x)
    r = StreamingResampler(fin, fout)
    chunks = np.concatenate([r.process(x[i:i + frame]) for i in range(0, len(x), frame)])
    np.testing.assert_allclose(chunks, one, atol=1e-6)
    assert len(one) == len(x) * fout // fin


def test_odd_chunk_sizes_keep_phase():
    x = np.random.default_rng(1).standard_normal(24000).astype(np.float32)
    one = StreamingResampler(24000, 8000).process(x)
    r = StreamingResampler(24000, 8000)
    out, i = [], 0
    for n in [7, 100, 333, 1920, 5, 4000]:
        out.append(r.process(x[i:i + n]))
        i += n
    out.append(r.process(x[i:]))
    np.testing.assert_allclose(np.concatenate(out), one, atol=1e-6)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def test_passband_tone_preserved_24k_to_8k():
    r = StreamingResampler(24000, 8000)
    y = r.process(tone(1000, 24000))
    assert abs(rms(y[400:]) / rms(tone(1000, 8000)) - 1) < 0.02


def test_alias_rejection_24k_to_8k():
    # 6 kHz cannot be represented at 8 kHz; without filtering it would alias to 2 kHz
    y = StreamingResampler(24000, 8000).process(tone(6000, 24000))
    assert 20 * np.log10(rms(y[400:]) / 0.354) < -50


def test_upsample_8k_to_16k_no_image():
    y = StreamingResampler(8000, 16000).process(tone(1000, 8000))
    spec = np.abs(np.fft.rfft(y[1000:9000] * np.hanning(8000)))
    f = np.fft.rfftfreq(8000, 1 / 16000)
    image = spec[(f > 6800) & (f < 7200)].max()
    main = spec[(f > 900) & (f < 1100)].max()
    assert 20 * np.log10(image / main) < -50
