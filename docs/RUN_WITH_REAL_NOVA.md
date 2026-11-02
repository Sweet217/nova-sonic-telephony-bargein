# Running the bridge against real Amazon Nova 2 Sonic

**Status: not run yet.** Everything in `results/` comes from the mock server.
The real client (`novabridge/nova_client.py`) is written against the official
SDK and has been checked to import, resolve its config and build an
`AsyncBedrockRuntimeClient` (Python 3.13, `aws_sdk_bedrock_runtime==0.12.0`).
It has **not** opened a stream to Bedrock: no AWS session was available, and no
paid API was called while building this repo.

## 1. Prerequisites

| What | Value |
|---|---|
| Python | **3.12 or newer** for the SDK (`aws_sdk_bedrock_runtime` declares `Requires-Python >=3.12`). The rest of the repo runs on 3.11. |
| SDK | `pip install -r requirements-nova.txt` (pins `aws_sdk_bedrock_runtime==0.12.0`). In 0.12 the config must be built with `await AsyncBedrockRuntimeConfig.resolve(...)`; older examples, Pipecat included, use the pre-0.12 `Config(...)` constructor. |
| Model ID | `amazon.nova-2-sonic-v1:0` (default; override with `NOVA_MODEL_ID`). Pipecat's service now defaults to `amazon.nova-2-5-sonic`; check the Bedrock console for which IDs your account sees. |
| Region | Default `us-east-1` (override with `AWS_REGION`). Third-party docs list Nova 2 Sonic in us-east-1, us-west-2 and ap-northeast-1, and one also lists eu-north-1. Check the Bedrock model-availability page before you rely on any of them. |
| Model access | Enable Nova 2 Sonic in **Bedrock → Model access** for the account and region. |
| API | `InvokeModelWithBidirectionalStream`. A stream stays open for about 8 minutes (SDK docstring). |

### IAM

The minimum policy for the identity the bridge runs as:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["bedrock:InvokeModelWithBidirectionalStream"],
    "Resource": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-sonic-v1:0"
  }]
}
```

Add `bedrock:InvokeModel` if you also want the Bedrock console playground.

### Environment variables

```bash
export AWS_REGION=us-east-1
export NOVA_MODEL_ID=amazon.nova-2-sonic-v1:0
# credentials: when AWS_ACCESS_KEY_ID is set the client uses them explicitly
# (EnvironmentCredentialsResolver); otherwise it uses the SDK's default resolution.
eval "$(aws configure export-credentials --format env)"   # after `aws sso login` / `aws login`
```

## 2. Set up

```bash
python3.13 -m venv .venv-nova
.venv-nova/bin/pip install -r requirements-nova.txt
.venv-nova/bin/python -c "import asyncio; from novabridge.nova_client import NovaSonicClient as C; print(asyncio.run(C()._client()))"
```

The last line sends no request. It only checks that the SDK, the region and the credentials resolve.

## 3. Measure with the fake caller (no phone number needed)

`bench/e2e.py --backend nova` runs the production websocket bridge with the real
Nova client behind it. A fake Twilio caller on localhost first asks a question,
waits for Nova to answer, then barges in 2–4 s into the answer. The caller leg
carries the simulated echo line, so you measure real Nova's server-side
interruption and real echo-induced false barge-ins, with the same metric as the
simulator.

```bash
.venv-nova/bin/python -m bench.e2e --backend nova --calls 5   # ~15 calls x ~30 s of Nova audio. This is a PAID API.
```

Results go to `results/e2e_real_nova.json`. For each configuration this records:
- **stop latency at the caller's ear**. In `server signal only` mode this is the
  real Nova end-to-end interruption latency plus the client flush;
- **`server_detect_ms`**: when the bridge received Nova's `INTERRUPTED` signal;
- **`bot_stopped_before_onset`**: a false barge-in. On a benign line this
  should be 0; rerun with a harsher `LINE` (for example ERL 9 dB) to see whether
  real Nova barges in on its own echo.

Things to check on the first real run:
1. Which events carry the interruption. The parser accepts TEXT and AUDIO
   `contentEnd.stopReason == "INTERRUPTED"`. Log `ev.raw` once to see the actual order.
2. Whether Nova keeps emitting `audioOutput` for the interrupted `contentId`
   after the local barge-in. Those chunks are counted in `core.ctl.dropped_late_chunks`.
3. How far ahead of real time Nova streams. This decides how much audio Twilio
   would buffer without pacing.

## 4. A real phone call (Twilio)

```bash
.venv-nova/bin/python -m novabridge.server --backend nova --port 8080   # mode=full, echo guard on
ngrok http 8080
```

TwiML for the number:

```xml
<Response><Connect><Stream url="wss://YOUR-NGROK-HOST/media"/></Connect></Response>
```

Use `--mode server_signal --no-echo-guard` for the baseline. Turn on Twilio
dual-channel call recording to measure stop latency offline: the time from caller
onset to the end of bot audio, from the two channels.

## 5. Amazon Connect

Amazon Connect does not expose Twilio-style Media Streams. The same core
(`BridgeCore`) applies wherever the integration owns the playout buffer: a
Kinesis Video Streams consumer plus an outbound audio path, a SIP/Chime Voice
Connector media gateway (see `aws-samples/sample-s2s-voip-gateway`), or a
contact-center media fork. The parts that carry over unchanged are the flush,
the late-chunk drop, the echo reference (what the gateway played) and the echo
guard. Only `twilio.py` would be replaced.

## What stays pending in the README until this is done

- Real Nova 2 Sonic stop latency (server-signal path) and its distribution.
- Whether real Nova barges in on its own echo on an 8 kHz line, and at which ERL.
  The mock's energy VAD is a deliberately naive stand-in.
- The benefit of the echo guard against real Nova turn detection.
