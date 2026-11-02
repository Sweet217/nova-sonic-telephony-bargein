"""Synthetic speech corpus from macOS ``say`` + ffmpeg (generated at run time, cached).

No real call recordings are used. Bot lines are rendered at 24 kHz (Nova 2
Sonic's usual output rate); caller lines at 8 kHz (they only ever exist on the
phone side of the line).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "audio"

BOT_LINES = [
    "Thanks for calling. I can help you check an order, update your address, or answer billing questions. "
    "To get started, could you please tell me the phone number or the email address on your account?",
    "I found your account. Your last order shipped on Tuesday and is expected to arrive by Friday afternoon. "
    "If you would like, I can send the tracking link to the email address we have on file.",
    "Your current balance is forty two dollars and fifteen cents, and the next payment is due on the twelfth. "
    "You can pay now over the phone, or I can set up automatic payments so you never miss a due date.",
    "Before I make that change, I need to confirm a few details for your security. "
    "Please say the first line of your billing address, and then the zip code, one digit at a time.",
    "We are open Monday through Friday from eight in the morning until eight in the evening, and on Saturday until noon. "
    "You can also reach us any time through the mobile app or the help center on our website.",
]

CALLER_LINES = [
    "Wait, I need to talk to a real person.",
    "No, stop, that's not what I asked.",
    "Sorry, can you repeat the balance?",
    "Actually I want to cancel my order.",
    "Hold on, my address changed last month.",
]

BOT_VOICES = ["Samantha", "Flo (English (US))", "Shelley (English (US))"]
CALLER_VOICES = ["Daniel", "Eddy (English (US))", "Reed (English (US))", "Fred", "Ralph"]


def _need(tool: str) -> None:
    if shutil.which(tool) is None:
        raise RuntimeError(f"'{tool}' not found; the testbed needs macOS 'say' and ffmpeg to synthesise speech")


def synth(text: str, voice: str, rate: int) -> np.ndarray:
    """Render ``text`` with ``say`` and return int16 mono PCM at ``rate``."""
    key = hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()[:16]
    out = CACHE / f"{key}.s16"
    if not out.exists():
        _need("say")
        _need("ffmpeg")
        CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            aiff = Path(td) / "x.aiff"
            subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ac", "1", "-ar", str(rate),
                            "-f", "s16le", str(out)], check=True)
    return np.fromfile(out, dtype="<i2")


def trim(x: np.ndarray, rate: int, thr_dbfs: float = -45.0) -> np.ndarray:
    """Remove leading/trailing silence (10 ms frame energy below threshold)."""
    f = rate // 100
    xf = x.astype(np.float64) / 32768.0
    n = len(xf) // f
    e = (xf[: n * f].reshape(n, f) ** 2).mean(axis=1)
    on = np.nonzero(10 * np.log10(e + 1e-12) > thr_dbfs)[0]
    if on.size == 0:
        return x
    return x[on[0] * f: (on[-1] + 1) * f]


def active_level_dbfs(x: np.ndarray, rate: int) -> float:
    """RMS level over 20 ms frames within 30 dB of the loudest frame."""
    f = rate // 50
    xf = np.asarray(x, dtype=np.float64) / 32768.0
    n = len(xf) // f
    e = (xf[: n * f].reshape(n, f) ** 2).mean(axis=1)
    act = e[e > e.max() * 1e-3]
    return float(10 * np.log10(act.mean() + 1e-12))


def normalize(x: np.ndarray, rate: int, target_dbfs: float) -> np.ndarray:
    g = 10 ** ((target_dbfs - active_level_dbfs(x, rate)) / 20)
    return np.clip(np.round(x.astype(np.float64) * g), -32768, 32767).astype(np.int16)


def load_corpus(bot_dbfs: float = -20.0) -> dict:
    """Bot utterances at 24 kHz (normalised) and caller utterances at 8 kHz (trimmed, unnormalised)."""
    bot = []
    for i, text in enumerate(BOT_LINES):
        voice = BOT_VOICES[i % len(BOT_VOICES)]
        pcm = normalize(trim(synth(text, voice, 24000), 24000), 24000, bot_dbfs)
        bot.append({"text": text, "voice": voice, "pcm": pcm, "rate": 24000})
    caller = []
    for i, text in enumerate(CALLER_LINES):
        voice = CALLER_VOICES[i % len(CALLER_VOICES)]
        pcm = trim(synth(text, voice, 8000), 8000)
        caller.append({"text": text, "voice": voice, "pcm": pcm, "rate": 8000})
    return {"bot": bot, "caller": caller}


if __name__ == "__main__":
    c = load_corpus()
    for k in ("bot", "caller"):
        for u in c[k]:
            print(f"{k:6s} {u['voice']:24s} {len(u['pcm']) / u['rate']:5.2f}s  {active_level_dbfs(u['pcm'], u['rate']):6.1f} dBFS  {u['text'][:50]}")
