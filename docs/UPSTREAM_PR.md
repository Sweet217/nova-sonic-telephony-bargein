# Upstream contribution prep (draft, not submitted)

Nothing here has been opened as a PR, issue or comment. This is preparation only.
Facts were read from the repositories on 2026-11-01. Re-check them before acting.

## Candidates

### A. `pipecat-ai/pipecat` — `src/pipecat/services/aws/nova_sonic/llm.py` (Python, BSD-2-Clause)

What it already does:
- `AWSNovaSonicLLMService._handle_content_end_event` calls
  `broadcast_interruption()` on an assistant TEXT `contentEnd` with
  `stopReason == "INTERRUPTED"`. Its comment calls this "the canonical barge-in
  signal". It also handles `INTERRUPTED` on AUDIO `contentEnd`.
- The `InterruptionFrame` clears `BaseOutputTransport`'s audio buffer, and
  `serializers/twilio.py` turns it into `{"event": "clear", "streamSid": ...}`.
- Defaults: input 16 kHz, output 24 kHz, model `amazon.nova-2-5-sonic`,
  server-side VAD only. The docstring notes that it emits no
  `UserStartedSpeakingFrame` unless a local VAD analyzer is wired in.

So steps 1–3 of the docs' client duties (stop, flush, clear) are already there for Twilio.

What is missing for telephony:
- **No echo reference anywhere in the path.** Input audio filters
  (`BaseAudioFilter.filter(audio) -> audio`, called in
  `transports/base_input.py`) only see the input. They never see what
  `BaseOutputTransport.write_audio_frame` played. On an 8 kHz call the bot's own
  echo reaches Nova's server VAD, and any optional local VAD such as Silero,
  unchanged.
- If a user wires in a local VAD for faster barge-in, it has no playback
  awareness, so echo will trigger it. In this repo's mock testbed, a local VAD
  without an echo reference fired on 92% of caller-silent calls; see the README.

### B. `aws-samples/sample-amazon-nova-sonic-twilio-integration` — `src/server.ts` (TypeScript, MIT-0)

What it does:
- Forwards every Nova `audioOutput` to Twilio immediately as `media`. There is no
  pacing, so Twilio buffers whatever Nova has generated.
- On `contentEnd` with `stopReason == "INTERRUPTED"` it sends Twilio `clear`.
- It does not track which `contentId` was interrupted, so any `audioOutput` that
  arrives after the signal is forwarded and plays.
- It configures Nova for 8 kHz in and out (`consts.ts`) and uses the first-generation
  model id `amazon.nova-sonic-v1:0`.
- No echo handling.
- Activity: the only PRs in its history are Dependabot bumps. Last push 2026-02.

### C. Others looked at
- `aws-samples/sample-s2s-voip-gateway` (Java SIP gateway) and
  `aws-samples/sample-sonic-sip-server-js` (Node SIP server) have the same
  problem shape. They own the RTP playout buffer, so the patch would be in
  their media code. Not reviewed in depth.

## Which fits best

**Pipecat is the better long-term home. The aws-sample is the cheaper first PR.**

| | Pipecat | aws-samples Twilio |
|---|---|---|
| Reach | Large and very active; named in AWS's Nova Sonic telephony integration guide | Small (about 20 stars), reference sample |
| Language match with this repo | Python, code ports directly | TypeScript, needs a port |
| Flush + `clear` | Already done | Done |
| Late-chunk drop | Implicit (interruption resets the output task); worth a test | Missing |
| Echo reference / guard | Missing; needs a new hook (output audio to input filter) | Missing; easy because `server.ts` sees both directions |
| Review bar | High: changelog fragment, runnable repro required, explicit policy against unverified AI-generated contributions | Unknown; looks unmaintained |

Recommended sequence:
1. **First**, run `docs/RUN_WITH_REAL_NOVA.md` and get real Nova numbers.
   Without them, any upstream claim rests on a mock VAD. Pipecat's CONTRIBUTING
   asks for a runnable reproduction confirmed by the reporter.
2. Open a Pipecat **discussion/issue** (not a PR) with the real-Nova reproduction:
   "Realtime S2S services on telephony: bot echo causes false server-side
   barge-ins; transports have no playback reference for input filters." Ask
   whether maintainers prefer (a) an output→input reference tap in the transport
   or (b) a self-contained processor placed around the transport.
3. Then the PR below, shaped to their answer.
4. Optionally, in parallel, a small aws-samples PR (late-chunk drop + paced
   playout) that does not depend on any echo-guard design decision.

## What the Pipecat patch would change (proposal)

1. `src/pipecat/audio/echo/echo_guard.py` (new): port of `novabridge/echo_guard.py`
   (delay estimator, NLMS, double-talk, playback-aware threshold,
   reference-correlation gate, NLP), numpy only, about 250 lines.
2. A reference tap: in `BaseOutputTransport.write_audio_frame` (or the media
   sender), push each written chunk to an optional `EchoReference` object shared
   with the input transport. `BaseInputTransport` passes it to filters that
   declare they need a reference. An alternative with no transport change is a
   paired `EchoReferenceTap` processor placed right before the output transport,
   but its timing is less exact than the real write point.
3. `EchoGuardFilter(BaseAudioFilter)`: cancels and gates input audio before it
   reaches the realtime service, so Nova's server VAD does not hear the bot.
4. Optional `PlaybackAwareVADAnalyzer` wrapper so a local VAD's threshold follows
   the guard's expected-residual-echo estimate.
5. Tests: a unit test with synthetic echo (delay/ERL sweep, as in
   `tests/test_echo_guard.py`) and a `run_test()` pipeline test showing no
   `InterruptionFrame` on echo-only input.
6. `changelog/<PR>.added.md`.

## Draft PR description (Pipecat)

> **Add an optional echo guard for telephony with realtime speech-to-speech services (e.g. AWS Nova Sonic)**
>
> **Problem.** On PSTN calls (Twilio/Telnyx/Plivo serializers, 8 kHz μ-law) part
> of the bot's own audio comes back on the caller leg as line/handset echo.
> Realtime S2S services like `AWSNovaSonicLLMService` run turn detection
> server-side on that audio, and so does any local VAD a user adds. Echo can make
> the bot interrupt itself mid-sentence. Pipecat already flushes and `clear`s
> correctly once an interruption is signalled; the issue is false interruptions.
> No input filter can fix it today because input filters never see what the
> output transport played.
>
> **Change.**
> - `EchoReference`: optional ring buffer the output transport fills with the
>   audio it writes (8/16/24 kHz, resampled to the input rate).
> - `EchoGuardFilter(BaseAudioFilter)`: finds the bulk delay by reference
>   cross-correlation, cancels the echo with a 32 ms NLMS filter with Geigel
>   double-talk detection, and attenuates echo-only frames before they reach the
>   service. Pure numpy, about 0.2 ms per 20 ms frame in our measurements.
> - Opt-in: `TransportParams(audio_in_filter=EchoGuardFilter(), echo_reference=True)`.
>   Nothing changes for existing users.
>
> **Evidence.** *(to be filled from the real-Nova runs; mock-testbed numbers are
> in the linked repo and labelled as such)*. Reproduction: a fake Twilio caller
> with a simulated echo path (configurable delay/ERL/noise) against
> `AWSNovaSonicLLMService`; report false-interruption rate on caller-silent
> calls, and missed/late interruptions on calls where the caller barges in, with
> and without the filter.
>
> **Limitations.** Linear echo only (handset/hybrid). Non-linear codec paths
> (AMR, packet loss concealment) reduce cancellation; the guard then falls back
> to a playback-aware threshold, trading some sensitivity to very quiet callers
> during loud bot speech.
>
> **Tests.** Unit tests for delay lock (50–300 ms), ERLE > 10 dB on synthetic
> echo, no false onset on echo-only input, and near-end speech detected during
> double talk; a pipeline test with no `InterruptionFrame` on echo-only input.

## Draft PR description (aws-samples Twilio integration, smaller)

> **Drop late audio after an interruption, and pace playout to Twilio**
>
> - Track the `contentId` of the assistant audio being played. After
>   `contentEnd` with `INTERRUPTED`, ignore further `audioOutput` with that
>   `contentId` instead of forwarding it after the `clear`.
> - Pace outbound `media` to real time with a small lead (3 × 20 ms frames)
>   instead of forwarding everything immediately. On interruption the queue is
>   flushed locally, and at most about 60 ms is left in Twilio's buffer, even if
>   a `clear` is delayed. This also makes the played audio usable as an echo
>   reference.
> - (Optional, follow-up) Echo guard port to TypeScript.
>
> Tested with a fake Twilio client and a simulated echo line *(link)*.
