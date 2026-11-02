"""G.711 mu-law codec (ITU-T G.711), vectorised in numpy.

Twilio Media Streams carry 8 kHz, 8-bit mu-law ("audio/x-mulaw"), 160 bytes per
20 ms frame. This module converts between mu-law bytes and int16 / float32 PCM.

The encoder follows the classic Sun g711.c reference (bit-exact with CPython audioop).
"""

from __future__ import annotations

import numpy as np

BIAS = 0x84


def _build_decode_table() -> np.ndarray:
    u = np.arange(256, dtype=np.int32)
    u = ~u & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = (((mantissa << 3) + BIAS) << exponent) - BIAS
    sample = np.where(sign != 0, -sample, sample)
    return sample.astype(np.int16)


_DECODE = _build_decode_table()


def ulaw_to_pcm16(data: bytes | np.ndarray) -> np.ndarray:
    """Decode mu-law bytes to int16 PCM."""
    codes = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else np.asarray(data, dtype=np.uint8)
    return _DECODE[codes]


_SEG_UEND = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)


def pcm16_to_ulaw(pcm: np.ndarray) -> bytes:
    """Encode int16 PCM to mu-law bytes.

    Same decision levels as the classic Sun g711.c ``linear2ulaw`` on 14-bit
    input (as used by CPython's ``audioop.lin2ulaw``): the 16-bit sample is
    shifted to 14 bits, biased by 33, and the segment found by table search.
    """
    v = np.asarray(pcm, dtype=np.int32) >> 2
    mask = np.where(v < 0, 0x7F, 0xFF)
    mag = np.minimum(np.abs(v), 8159) + 33
    seg = np.searchsorted(_SEG_UEND, mag, side="left")
    uval = (np.minimum(seg, 7) << 4) | ((mag >> (np.minimum(seg, 7) + 1)) & 0x0F)
    uval = np.where(seg >= 8, 0x7F, uval)
    return (uval ^ mask).astype(np.uint8).tobytes()


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    """float32 in [-1, 1] -> int16 (clipped)."""
    return np.clip(np.round(np.asarray(x) * 32768.0), -32768, 32767).astype(np.int16)


def pcm16_to_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) / 32768.0


def ulaw_to_float(data: bytes) -> np.ndarray:
    return pcm16_to_float(ulaw_to_pcm16(data))


def float_to_ulaw(x: np.ndarray) -> bytes:
    return pcm16_to_ulaw(float_to_pcm16(x))
