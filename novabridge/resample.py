"""Streaming polyphase resampler (pure numpy).

Telephony is 8 kHz; Nova 2 Sonic is typically configured for 16 kHz input and
24 kHz output (the API also accepts 8000/16000/24000 on both sides). The bridge
therefore needs 8k -> 16k (upstream) and 24k -> 8k (downstream) on 20 ms
frames, with filter state carried across frames so there are no clicks at frame
boundaries.

Implementation: rational L/M resampling with a Kaiser-windowed-sinc low-pass,
zero-stuffing + FIR + decimation, keeping the FIR history and the decimation
phase between calls. For telephony ratios (L, M <= 3) this is cheap enough to
run per 20 ms frame in real time.
"""

from __future__ import annotations

from math import gcd

import numpy as np


def design_lowpass(up: int, down: int, half_taps: int = 24, beta: float = 7.0) -> np.ndarray:
    """Kaiser-windowed sinc for an L/M resampler, gain L, cutoff 0.9 * min Nyquist."""
    factor = max(up, down)
    n_taps = 2 * half_taps * factor + 1
    cutoff = 0.9 * 0.5 / factor  # cycles/sample at the up-sampled rate
    n = np.arange(n_taps) - (n_taps - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(n_taps, beta)
    h *= up / h.sum()
    return h.astype(np.float64)


class StreamingResampler:
    """Stateful L/M resampler for frame-by-frame audio."""

    def __init__(self, in_rate: int, out_rate: int, half_taps: int = 24, beta: float = 7.0):
        g = gcd(in_rate, out_rate)
        self.in_rate, self.out_rate = in_rate, out_rate
        self.up, self.down = out_rate // g, in_rate // g
        self.passthrough = self.up == 1 and self.down == 1
        self.h = design_lowpass(self.up, self.down, half_taps, beta)
        self._hist = np.zeros(len(self.h) - 1)
        self._pos = 0  # global index (up-sampled domain) of the next input sample

    @property
    def delay_out_samples(self) -> float:
        """Group delay of the filter, in output samples."""
        return (len(self.h) - 1) / 2 / self.down

    def reset(self) -> None:
        self._hist[:] = 0
        self._pos = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.passthrough:
            return x.astype(np.float32)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        xu = np.zeros(x.size * self.up)
        xu[:: self.up] = x
        buf = np.concatenate([self._hist, xu])
        y_full = np.convolve(buf, self.h, mode="valid")  # len == len(xu)
        first = (-self._pos) % self.down
        y = y_full[first :: self.down]
        self._pos += xu.size
        self._hist = buf[-(len(self.h) - 1):]
        return y.astype(np.float32)
