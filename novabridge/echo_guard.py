"""Echo guard: use what the bridge played as an echo reference.

On a phone call the bot's own voice comes back on the caller leg (hybrid /
acoustic echo). Nova Sonic's server-side turn detection -- and any local VAD --
can hear that echo as "the caller is talking" and barge in on itself. The
bridge, unlike the server, knows *exactly* which samples it played (it encoded
them), so it can cancel or discount the echo. Five small, explainable pieces:

1. Delay estimator -- normalised cross-correlation (FFT, every 100 ms) of the
   played reference against the caller leg finds the bulk echo delay (Twilio
   buffer + network + echo path). Locks after two agreeing estimates and only
   re-locks on a large, persistent change.
2. NLMS canceller -- a short (256 taps = 32 ms) regularised normalised-LMS
   filter on the delay-aligned reference models the echo-path colouration and
   subtracts it. A frame the filter makes louder is passed through unchanged.
3. Double-talk detector -- a Geigel test scaled by the *measured* echo-path
   gain freezes adaptation while the caller talks over the bot.
4. Playback-aware VAD threshold -- raised by the expected echo on this frame:
   * an upper bound on the echo-path gain, max(mic)/max(already-echoed ref),
     tracked like a noise floor (starts at a worst-case 6 dB ERL);
   * once the canceller has converged (>= 10 dB ERLE), the expected *residual*
     echo, but falling back to the bound on any frame where the canceller is
     not actually removing echo (onsets after pauses, spectra it never saw).
5. Reference-correlation gate -- a frame whose waveform correlates strongly
   (> 0.7) with the canceller's echo estimate is echo, whatever its level.

The residual is also what goes upstream to Nova, through a simple non-linear
processor that attenuates echo-only frames by 30 dB, so the server's own turn
detection does not hear the bot either.

Pure numpy; per-frame cost is measured by ``bench.run`` (well under the 20 ms
frame budget).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vad import EPS, energy, from_db


@dataclass
class EchoGuardConfig:
    fs: int = 8000
    frame: int = 160
    max_delay_ms: int = 600       # bulk delay search range
    est_window_ms: int = 400      # correlation window
    est_every_frames: int = 5     # run the delay estimator every 100 ms
    min_ncc: float = 0.35         # minimum normalised correlation to accept a delay
    relock_samples: int = 48      # re-lock only for delay changes larger than this
    taps: int = 256               # NLMS length (32 ms at 8 kHz)
    pre_taps: int = 32            # taps placed before the estimated bulk delay
    mu: float = 0.5
    nlms_reg_dbfs: float = -50.0  # NLMS regularisation, as a per-sample reference power
    adapt_min_dbfs: float = -45.0 # adapt only while the aligned reference is at least this loud
    geigel_margin: float = 2.0    # double-talk if |mic| > margin * sqrt(echo gain) * max|ref|
    dt_hangover_frames: int = 4
    erl_assumed_db: float = 6.0   # worst-case ERL before anything is measured
    gain_rise_db_per_s: float = 1.0  # how fast the measured echo-gain bound may rise
    margin_db: float = 10.0       # VAD threshold margin above the expected residual echo (converged)
    bound_margin_db: float = 6.0  # margin above the measured echo bound (frame-alignment slack)
    min_erle_db: float = 10.0     # per-frame cancellation never assumed better than this
    min_frame_erle_db: float = 5.0  # below this cancellation on a frame, fall back to the bound
    corr_echo: float = 0.7        # mic/echo-estimate correlation above which a frame is echo
    corr_min_echo_frac: float = 0.05  # ...only tested when the echo estimate is >= 5% of mic energy
    converged_erle_db: float = 10.0
    ref_active_dbfs: float = -55.0
    nlp_atten_db: float = 30.0    # upstream attenuation for echo-only frames
    nlp_hold_frames: int = 10     # keep the gate open 200 ms after near-end speech
    nlp_open_frames: int = 2      # consecutive near-end-like frames needed to open the gate
    nlp_floor_dbfs: float = -45.0 # ...and they must also be louder than this


class DelayEstimator:
    def __init__(self, cfg: EchoGuardConfig):
        self.cfg = cfg
        self.max_lag = cfg.max_delay_ms * cfg.fs // 1000
        self.win = cfg.est_window_ms * cfg.fs // 1000
        self.ref = np.zeros(self.max_lag + self.win)
        self.mic = np.zeros(self.win)
        self.n = 0
        self.delay: int | None = None
        self._candidate: int | None = None
        self._agree = 0
        self.last_ncc = 0.0

    def push(self, ref: np.ndarray, mic: np.ndarray) -> int | None:
        k = len(ref)
        self.ref = np.concatenate([self.ref[k:], ref])
        self.mic = np.concatenate([self.mic[k:], mic])
        self.n += 1
        if self.n % self.cfg.est_every_frames == 0:
            self._estimate()
        return self.delay

    def _estimate(self) -> None:
        x, y = self.ref, self.mic
        ey = float(np.dot(y, y))
        if ey < EPS or energy(x) < from_db(self.cfg.ref_active_dbfs):
            return
        nfft = 1 << int(np.ceil(np.log2(len(x) + len(y))))
        c = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(y, nfft)), nfft)[: self.max_lag + 1]
        # c[k] = sum_n x[n + k] y[n]; lag (delay of mic behind ref) = max_lag - k
        cs = np.concatenate([[0.0], np.cumsum(x * x)])
        ex = cs[self.win: self.win + self.max_lag + 1] - cs[: self.max_lag + 1]
        ncc = c / np.sqrt(ey * ex + EPS)
        k = int(np.argmax(ncc))
        self.last_ncc = float(ncc[k])
        if ncc[k] < self.cfg.min_ncc:
            return
        lag = self.max_lag - k
        if self._candidate is not None and abs(lag - self._candidate) <= 4:
            self._agree += 1
        else:
            self._agree = 0
        self._candidate = lag
        # first lock after two agreeing estimates; afterwards only move for a
        # large, persistent change (small shifts are absorbed by the NLMS window)
        if self.delay is None and self._agree >= 1:
            self.delay = lag
        elif self.delay is not None and abs(lag - self.delay) > self.cfg.relock_samples and self._agree >= 2:
            self.delay = lag


class NLMS:
    def __init__(self, taps: int, mu: float, delta: float = 0.0):
        self.taps, self.mu = taps, mu
        self.delta = delta + 1e-9  # regularisation: keeps updates bounded when the reference is quiet
        self.w = np.zeros(taps)  # stored time-reversed so y = w . x[n-L+1 .. n]

    def reset(self) -> None:
        self.w[:] = 0

    def process(self, xh: np.ndarray, d: np.ndarray, adapt: bool) -> tuple[np.ndarray, np.ndarray]:
        """xh: reference with (taps-1) samples of history prepended; d: desired (mic)."""
        L, n_out = self.taps, len(d)
        e = np.empty(n_out)
        yhat = np.empty(n_out)
        w, mu = self.w, self.mu
        power = float(np.dot(xh[:L], xh[:L]))
        for n in range(n_out):
            seg = xh[n: n + L]
            if n:
                power += xh[n + L - 1] ** 2 - xh[n - 1] ** 2
            y = float(np.dot(w, seg))
            err = d[n] - y
            if adapt:
                w += (mu * err / (power + self.delta)) * seg
            e[n] = err
            yhat[n] = y
        return e, yhat


@dataclass
class GuardFrame:
    residual: np.ndarray          # echo-cancelled near-end frame
    upstream: np.ndarray          # what to send to Nova (residual after NLP)
    threshold: float              # playback-aware VAD threshold (energy; inf = frame is echo)
    playing: bool                 # reference active within the echo window
    double_talk: bool
    delay: int | None
    converged: bool
    erle_db: float
    corr: float = 0.0             # correlation of the mic frame with the echo estimate


class EchoGuard:
    def __init__(self, cfg: EchoGuardConfig | None = None):
        self.cfg = cfg = cfg or EchoGuardConfig()
        self.est = DelayEstimator(cfg)
        self.nlms = NLMS(cfg.taps, cfg.mu, delta=cfg.taps * from_db(cfg.nlms_reg_dbfs))
        hist = self.est.max_lag + cfg.taps + cfg.frame
        self.ref_hist = np.zeros(hist)
        self._delay_used: int | None = None
        self._dt_hold = 0
        self._ratio = 1.0          # residual/echo-estimate energy ratio (exp. weighted sums)
        self._num = 0.0
        self._den = 0.0
        self._ratio_frames = 0
        self._nlp_hold = 0
        self._nlp_run = 0
        self._prev_echo_e = 0.0
        self._prev2_echo_e = 0.0
        n_win = self.est.max_lag // cfg.frame + 2
        self._n_win = n_win
        self._ref_energy_hist = np.zeros(2 * n_win)   # per-frame energies, last 2 x max delay
        self._mic_energy_hist = np.zeros(2 * n_win)
        self._playing_run = 0
        # measured echo-path gain bound (energy), starts at the worst-case ERL assumption
        self._echo_gain = from_db(-cfg.erl_assumed_db)
        self._gain_rise = from_db(cfg.gain_rise_db_per_s * cfg.frame / cfg.fs)

    def _aligned_ref(self, delay: int) -> np.ndarray:
        """Reference aligned to the current mic frame, with taps-1 samples of history."""
        c = self.cfg
        start_delay = max(delay - c.pre_taps, 0)
        end = len(self.ref_hist) - start_delay
        return self.ref_hist[end - c.frame - (c.taps - 1): end]

    def process(self, mic: np.ndarray, ref: np.ndarray) -> GuardFrame:
        c = self.cfg
        mic = np.asarray(mic, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        self.ref_hist = np.concatenate([self.ref_hist[len(ref):], ref])
        self._ref_energy_hist = np.concatenate([self._ref_energy_hist[1:], [energy(ref)]])
        self._mic_energy_hist = np.concatenate([self._mic_energy_hist[1:], [energy(mic)]])
        max_ref_e = float(self._ref_energy_hist[-self._n_win:].max())   # can still echo now
        playing = max_ref_e > from_db(c.ref_active_dbfs)
        self._playing_run = self._playing_run + 1 if playing else 0
        # Measured upper bound on the echo-path gain: reference frames old enough to have
        # echoed back already, versus the loudest mic frame since. max(mic)/max(old ref)
        # can only over-estimate the gain; track it like a noise floor (fall fast, rise slowly).
        if self._delay_used is not None:   # bulk delay known: frames older than it have echoed
            d = int(np.ceil((self._delay_used + c.taps) / c.frame)) + 1
            old_ref = self._ref_energy_hist[-(d + self._n_win):-d]
            mic_win = self._mic_energy_hist[-(self._n_win + d):]  # covers every echo of old_ref
            ready = self._playing_run >= d + 5
        else:                              # unknown delay: assume the maximum
            old_ref = self._ref_energy_hist[: self._n_win]
            mic_win = self._mic_energy_hist
            ready = self._playing_run >= len(self._ref_energy_hist)
        old_ref_e = float(old_ref.max())
        if ready and old_ref_e > from_db(c.adapt_min_dbfs):
            g_obs = float(mic_win.max()) / old_ref_e
            self._echo_gain = min(g_obs, self._echo_gain * self._gain_rise, from_db(-c.erl_assumed_db))

        delay = self.est.push(ref, mic)
        if delay is not None and delay != self._delay_used:
            self.nlms.reset()
            self._ratio, self._ratio_frames, self._num, self._den = 1.0, 0, 0.0, 0.0
            self._delay_used = delay

        residual, echo_e, dt, aligned_ref_e, corr = mic, 0.0, False, max_ref_e, 0.0
        not_helping = False
        if self._delay_used is not None:
            xh = self._aligned_ref(self._delay_used)
            aligned_ref_e = max(energy(b) for b in np.array_split(xh, 3))
            ref_peak = float(np.max(np.abs(xh))) if xh.size else 0.0
            # Geigel test against the *measured* echo-path gain (6 dB margin), so a loud but
            # legitimate echo (low ERL) is not mistaken for the caller talking
            dt_now = float(np.max(np.abs(mic))) > c.geigel_margin * np.sqrt(self._echo_gain) * ref_peak + 1e-4
            self._dt_hold = c.dt_hangover_frames if dt_now else max(self._dt_hold - 1, 0)
            dt = self._dt_hold > 0
            ref_active = energy(xh) > from_db(c.adapt_min_dbfs)
            residual, yhat = self.nlms.process(xh, mic, adapt=ref_active and not dt)
            echo_e = energy(yhat)
            # reference-correlation test: a frame that looks like the predicted echo is echo,
            # even if the canceller left a lot of it behind (e.g. at onsets after a pause)
            if echo_e > c.corr_min_echo_frac * energy(mic):
                corr = float(np.dot(mic, yhat)) / (np.sqrt(np.dot(mic, mic) * np.dot(yhat, yhat)) + EPS)
            if energy(residual) > energy(mic):
                # the canceller is making this frame worse (mis-adapted): pass the mic through
                residual = mic
                not_helping = True
            if ref_active and not dt and echo_e > EPS:
                self._num = 0.95 * self._num + energy(residual)
                self._den = 0.95 * self._den + echo_e
                self._ratio = min(self._num / self._den, 1.0)
                self._ratio_frames += 1

        converged = self._ratio_frames >= 10 and self._ratio < from_db(-c.converged_erle_db)
        erle_db = -10 * np.log10(self._ratio + EPS)

        worst_case = max_ref_e * self._echo_gain  # echo can be no louder than this
        if not playing:
            echo_thr = 0.0
        elif converged:
            # residual echo is bursty at echo onsets: use the recent peak of the echo estimate
            bound = aligned_ref_e * self._echo_gain        # upper bound on the echo in this frame
            # if the filter predicts far less echo than the bound allows, or makes the frame
            # worse, it is not modelling this part of the signal (e.g. spectral content it was
            # never trained on): fall back to the bound; otherwise never assume more than
            # min_erle_db of cancellation
            frame_erle_ok = energy(residual) <= energy(mic) * from_db(-c.min_frame_erle_db)
            covered = echo_e >= bound * from_db(-c.min_erle_db) and not not_helping and frame_erle_ok
            echo_thr = max(self._ratio * max(echo_e, self._prev_echo_e, self._prev2_echo_e) * from_db(c.margin_db),
                           bound * (from_db(-c.min_erle_db) if covered else from_db(c.bound_margin_db)))
        else:
            echo_thr = worst_case * from_db(c.bound_margin_db)
        self._prev2_echo_e, self._prev_echo_e = self._prev_echo_e, echo_e

        self.debug = {"echo_gain_db": 10 * np.log10(self._echo_gain + EPS), "aligned_ref_db": 10 * np.log10(aligned_ref_e + EPS),
                      "echo_est_db": 10 * np.log10(echo_e + EPS), "max_ref_db": 10 * np.log10(max_ref_e + EPS)}
        echo_like = converged and corr > c.corr_echo
        if echo_like:
            echo_thr = float("inf")

        # NLP: attenuate upstream frames that look like echo only
        # (opens after nlp_open_frames consecutive near-end-like frames, then holds 200 ms)
        self._nlp_run = self._nlp_run + 1 if energy(residual) > max(echo_thr, from_db(c.nlp_floor_dbfs)) else 0
        if self._nlp_run >= c.nlp_open_frames:
            self._nlp_hold = c.nlp_hold_frames
        else:
            self._nlp_hold = max(self._nlp_hold - 1, 0)
        upstream = residual if (not playing or self._nlp_hold > 0) else residual * from_db(-c.nlp_atten_db / 2)

        return GuardFrame(residual=np.asarray(residual), upstream=np.asarray(upstream), threshold=echo_thr,
                          playing=playing, double_talk=dt, delay=self._delay_used, converged=converged,
                          erle_db=float(erle_db), corr=corr)
