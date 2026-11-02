"""Frame-energy VAD with a tracked noise floor.

Deliberately simple: a frame is "speech" when its energy exceeds
max(absolute floor, noise floor + margin, optional external threshold), and a
speech *onset* is declared after ``min_on_frames`` consecutive speech frames.
The external threshold is how the echo guard makes the VAD playback-aware.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def energy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x * x)) if x.size else 0.0


def dbfs(e: float) -> float:
    """Energy (mean square, full scale = 1.0) -> dBFS."""
    return 10.0 * np.log10(e + EPS)


def from_db(db: float) -> float:
    return 10.0 ** (db / 10.0)


class EnergyVAD:
    def __init__(self, abs_threshold_dbfs: float = -45.0, noise_margin_db: float = 10.0,
                 min_on_frames: int = 3, floor_rise_db_per_s: float = 3.0, frame_ms: int = 20):
        self.abs_thr = from_db(abs_threshold_dbfs)
        self.noise_margin = from_db(noise_margin_db)
        self.min_on = min_on_frames
        self.rise = from_db(floor_rise_db_per_s * frame_ms / 1000.0)
        self.floor: float | None = None  # noise floor energy (minimum statistics)
        self.run = 0
        self.last_threshold = self.abs_thr

    def threshold(self, external: float = 0.0) -> float:
        floor_thr = (self.floor or 0.0) * self.noise_margin
        return max(self.abs_thr, floor_thr, external)

    def update(self, frame: np.ndarray, external_threshold: float = 0.0) -> bool:
        """Feed one frame. Returns True on the frame where an onset is declared."""
        e = energy(frame)
        thr = self.threshold(external_threshold)
        self.last_threshold = thr
        is_speech = e > thr
        # minimum-statistics style floor: fall fast, rise slowly. The first frame seeds it,
        # capped so that a call starting with speech does not hide that speech.
        if self.floor is None:
            self.floor = min(e, self.abs_thr / self.noise_margin * from_db(6.0))
        elif e < self.floor:
            self.floor = e
        else:
            self.floor = min(e, self.floor * self.rise)
        self.run = self.run + 1 if is_speech else 0
        return self.run == self.min_on

    @property
    def in_speech(self) -> bool:
        return self.run >= self.min_on
