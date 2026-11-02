# Testbed: what is simulated, and what the metrics mean

## Signal chain (one bot turn per run)

```
MockNovaSonic ──30 ms──▶ BridgeCore ──40 ms──▶ TwilioEmulator ──▶ caller's ear
      ▲                     │  ▲                                      │
      └──────30 ms──────────┘  └────────40 ms──── μ-law ◀── caller speech + echo(played) + noise
```

* **Clock.** Virtual, 20 ms ticks (one Twilio frame). Every latency is quantised
  to 20 ms. The bridge code under test is the production `BridgeCore`; only the
  clock is simulated. `bench/e2e.py` runs the same code on the wall clock over
  real websockets.
* **Mock Nova server** (`testbed/mock_nova.py`). Uses the documented event JSON.
  First audio at 250 ms, then 80 ms chunks at 24 kHz, generated at 2× real time.
  It runs a naive energy VAD on the received 16 kHz audio: −40 dBFS for 100 ms.
  On an onset while it estimates its response is still playing, it waits
  `server_latency_ms` (drawn uniformly from 150–450 ms per call), stops
  generating, closes the AUDIO content with `END_TURN` and emits an assistant
  FINAL TEXT `contentEnd` with `stopReason: "INTERRUPTED"`. **These are
  assumptions, not Nova measurements.** Real Nova's turn detection is a model,
  not an energy threshold, and is probably more robust to echo than this
  stand-in.
* **Twilio emulator.** Buffers outbound `media`, plays one frame per tick, and
  empties the buffer on `clear`. One-way delay is 40 ms each way.
* **Phone line** (`testbed/phone_line.py`):
  - echo = played audio × impulse response. The response is a bulk delay (50–300 ms),
    a 2nd-order 300–3400 Hz band-pass with 3 sparse reflections, and is scaled so
    its white-noise power gain equals −ERL dB;
  - pink-ish line noise (−70…−40 dBFS);
  - caller speech at −26…−18 dBFS;
  - everything is μ-law encoded on the way back.
* **Audio.** macOS `say` (bot voices Samantha/Flo/Shelley at 24 kHz, normalised
  to −20 dBFS active level; caller voices Daniel/Eddy/Reed/Fred/Ralph at 8 kHz),
  synthesised at run time into `.cache/`. Five invented customer-service bot
  lines (9.8–12.4 s) and five invented caller interruptions (2.1–3.1 s).

## Client configurations

| Name | Outbound | On Nova `INTERRUPTED` | Local VAD | Echo guard |
|---|---|---|---|---|
| naive (no client handler) | forward immediately | ignored | – | – |
| server signal only | forward immediately | flush + `clear` + drop late chunks | – | – |
| server signal + echo guard | forward immediately | same | – | cleans/gates audio sent to Nova |
| server + local VAD, no guard | paced, 3-frame lead | same | on raw caller audio | – |
| server + local VAD + echo guard | paced, 3-frame lead | same | on echo-cancelled audio, playback-aware threshold | yes |

## Metrics

* **Stop latency (ear).** Time from caller speech onset (ground truth, first
  sample of the trimmed caller utterance) to the end of the last bot frame
  Twilio plays to the caller. Computed only on barge-in calls where the bot was
  still talking at onset and no false barge-in happened earlier. For "naive",
  it is simply the rest of the buffered response.
* **False barge-in (caller silent).** Share of caller-silent calls in which an
  interruption fired: the server VAD started an interruption (the server stops
  generating, whatever the client does), or the local VAD flushed playback.
* **False barge-in before caller spoke.** The same, on barge-in calls, for
  triggers before the onset.
* **Missed true barge-in.** The bot was talking at onset, there was no false
  trigger, and nothing fired within 2 s of onset.
* **Main table.** 100 barge-in + 100 caller-silent calls (seeds 10000+), each
  with random line conditions. Every configuration runs on the *same* calls.
* **Sweeps.** One factor at a time around 150 ms / 12 dB / −55 dBFS, 20 + 20
  calls per point (seeds 15000+), local-VAD mode with and without the guard.

## Tuning hygiene

Echo-guard parameters were tuned on development seeds (`python -m bench.run
--dev`, seeds below 10000, written to the git-ignored `results_dev/`), then frozen
before the evaluation run on seeds 10000+. The *speech corpus* is the same in
both (5 bot × 5 caller utterances), so the evaluation is not independent of the
material the guard was tuned on. Treat the guard's numbers as optimistic for
unseen voices and lines.

## Known limitations

* The echo path is linear and time-invariant. Real calls add codec transcoding
  (AMR/Opus legs), packet-loss concealment, AGC and noise suppression in the
  handset, which hurt cancellation more. The playback-aware threshold and
  correlation gate are meant to degrade gracefully, but that is not shown here.
* The bridge's reference timeline assumes Twilio plays exactly one frame per
  20 ms after the bridge's pacing. Real jitter would show up as small delay
  drift; the delay estimator re-locks only on changes > 6 ms.
* The server-side numbers depend entirely on the assumed decision latency and
  VAD. Only the *difference* between client configurations is meaningful.
