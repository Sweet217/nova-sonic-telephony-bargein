import warnings

import numpy as np
import pytest

from novabridge.g711 import pcm16_to_ulaw, ulaw_to_pcm16

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import audioop  # stdlib up to Python 3.12, used only as a reference implementation
    except ImportError:  # pragma: no cover
        audioop = None


@pytest.mark.skipif(audioop is None, reason="audioop reference not available on this Python")
def test_decode_matches_stdlib_for_all_codes():
    codes = bytes(range(256))
    ref = np.frombuffer(audioop.ulaw2lin(codes, 2), dtype="<i2")
    np.testing.assert_array_equal(ulaw_to_pcm16(codes), ref)


@pytest.mark.skipif(audioop is None, reason="audioop reference not available on this Python")
def test_encode_matches_stdlib_over_int16_range():
    x = np.arange(-32768, 32768, dtype=np.int16)
    ref = audioop.lin2ulaw(x.astype("<i2").tobytes(), 2)
    assert pcm16_to_ulaw(x) == ref  # bit-exact


def test_known_codes_and_roundtrip_error():
    assert pcm16_to_ulaw(np.array([0], dtype=np.int16)) == b"\xff"
    assert ulaw_to_pcm16(b"\x00")[0] == -32124 and ulaw_to_pcm16(b"\x80")[0] == 32124
    x = (np.sin(np.arange(8000) * 2 * np.pi * 440 / 8000) * 20000).astype(np.int16)
    y = ulaw_to_pcm16(pcm16_to_ulaw(x)).astype(np.float64)
    snr = 10 * np.log10(np.sum(x.astype(float) ** 2) / np.sum((x - y) ** 2))
    assert snr > 35  # G.711 mu-law gives ~38 dB SQNR for a loud tone


def test_frame_size_is_160_bytes():
    assert len(pcm16_to_ulaw(np.zeros(160, dtype=np.int16))) == 160
