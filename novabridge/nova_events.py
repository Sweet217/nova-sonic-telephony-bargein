"""Amazon Nova 2 Sonic bidirectional-stream event builders and parser.

Event names and shapes follow the Nova 2 Sonic user guide:
  https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-input-events.html
  https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html
  https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-barge-in.html

Interruption signal. The docs say Nova "sends a content notification indicating
the interruption has occurred"; the documented field is ``contentEnd.stopReason
== "INTERRUPTED"`` (documented on TEXT content). Pipecat's Nova Sonic service
treats the assistant FINAL-text ``contentEnd`` with ``INTERRUPTED`` as the
canonical barge-in signal and also accepts ``INTERRUPTED`` on AUDIO
``contentEnd``. This parser accepts both, plus (defensively, not documented
for Nova 2) a ``textOutput`` whose content is the JSON ``{"interrupted": true}``.

Both the real Bedrock client and the mock server in ``testbed/`` speak exactly
these JSON shapes, so the bridge code under test is the code that would run
against the real service.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

# ----------------------------------------------------------------- input side


def session_start(max_tokens: int = 1024, top_p: float = 0.9, temperature: float = 0.7,
                  endpointing: str | None = "MEDIUM") -> dict:
    ev: dict[str, Any] = {"sessionStart": {"inferenceConfiguration": {
        "maxTokens": max_tokens, "topP": top_p, "temperature": temperature}}}
    if endpointing:
        ev["sessionStart"]["turnDetectionConfiguration"] = {"endpointingSensitivity": endpointing}
    return {"event": ev}


def prompt_start(prompt_name: str, voice_id: str = "tiffany", out_rate: int = 24000) -> dict:
    return {"event": {"promptStart": {
        "promptName": prompt_name,
        "textOutputConfiguration": {"mediaType": "text/plain"},
        "audioOutputConfiguration": {
            "mediaType": "audio/lpcm", "sampleRateHertz": out_rate, "sampleSizeBits": 16,
            "channelCount": 1, "voiceId": voice_id, "encoding": "base64", "audioType": "SPEECH"},
    }}}


def text_content_start(prompt_name: str, content_name: str, role: str = "SYSTEM",
                       interactive: bool = False) -> dict:
    return {"event": {"contentStart": {
        "promptName": prompt_name, "contentName": content_name, "type": "TEXT",
        "interactive": interactive, "role": role,
        "textInputConfiguration": {"mediaType": "text/plain"}}}}


def text_input(prompt_name: str, content_name: str, text: str) -> dict:
    return {"event": {"textInput": {"promptName": prompt_name, "contentName": content_name, "content": text}}}


def audio_content_start(prompt_name: str, content_name: str, in_rate: int = 16000) -> dict:
    return {"event": {"contentStart": {
        "promptName": prompt_name, "contentName": content_name, "type": "AUDIO",
        "interactive": True, "role": "USER",
        "audioInputConfiguration": {
            "mediaType": "audio/lpcm", "sampleRateHertz": in_rate, "sampleSizeBits": 16,
            "channelCount": 1, "audioType": "SPEECH", "encoding": "base64"}}}}


def audio_input(prompt_name: str, content_name: str, pcm16: bytes) -> dict:
    return {"event": {"audioInput": {"promptName": prompt_name, "contentName": content_name,
                                     "content": base64.b64encode(pcm16).decode("ascii")}}}


def content_end(prompt_name: str, content_name: str) -> dict:
    return {"event": {"contentEnd": {"promptName": prompt_name, "contentName": content_name}}}


def prompt_end(prompt_name: str) -> dict:
    return {"event": {"promptEnd": {"promptName": prompt_name}}}


def session_end() -> dict:
    return {"event": {"sessionEnd": {}}}


# ---------------------------------------------------------------- output side


@dataclass
class NovaEvent:
    kind: str  # audio | interrupted | content_start | content_end | text | completion_start | completion_end | other
    name: str
    content_id: str | None = None
    pcm: np.ndarray | None = None  # int16, for kind == "audio"
    stop_reason: str | None = None
    role: str | None = None
    type: str | None = None
    generation_stage: str | None = None
    sample_rate: int | None = None
    text: str | None = None
    raw: dict | None = None


def _generation_stage(cs: dict) -> str | None:
    amf = cs.get("additionalModelFields")
    if not amf:
        return None
    try:
        return json.loads(amf).get("generationStage")
    except (ValueError, AttributeError):
        return None


def _is_legacy_interrupt_text(text: str) -> bool:
    try:
        obj = json.loads(text)
    except ValueError:
        return False
    return isinstance(obj, dict) and obj.get("interrupted") is True


def parse_output(event_json: dict | str | bytes) -> NovaEvent:
    """Normalise one Nova output event (``{"event": {...}}``)."""
    if isinstance(event_json, (str, bytes)):
        event_json = json.loads(event_json)
    ev = event_json.get("event", event_json)
    name = next(iter(ev)) if ev else ""
    body = ev.get(name, {}) if name else {}

    if name == "audioOutput":
        pcm = np.frombuffer(base64.b64decode(body["content"]), dtype="<i2")
        return NovaEvent("audio", name, body.get("contentId"), pcm=pcm, raw=event_json)
    if name == "contentStart":
        aoc = body.get("audioOutputConfiguration") or {}
        return NovaEvent("content_start", name, body.get("contentId"), role=body.get("role"),
                         type=body.get("type"), generation_stage=_generation_stage(body),
                         sample_rate=aoc.get("sampleRateHertz"), raw=event_json)
    if name == "contentEnd":
        sr = body.get("stopReason")
        kind = "interrupted" if sr == "INTERRUPTED" else "content_end"
        return NovaEvent(kind, name, body.get("contentId"), stop_reason=sr, type=body.get("type"), raw=event_json)
    if name == "textOutput":
        text = body.get("content", "")
        kind = "interrupted" if _is_legacy_interrupt_text(text) else "text"
        return NovaEvent(kind, name, body.get("contentId"), text=text, role=body.get("role"), raw=event_json)
    if name == "completionStart":
        return NovaEvent("completion_start", name, raw=event_json)
    if name == "completionEnd":
        return NovaEvent("completion_end", name, stop_reason=body.get("stopReason"), raw=event_json)
    return NovaEvent("other", name, raw=event_json)


# ------------------------------------------------- output builders (mock use)


def _ids(session_id: str, prompt_name: str, completion_id: str) -> dict:
    return {"sessionId": session_id, "promptName": prompt_name, "completionId": completion_id}


def out_completion_start(ids: dict) -> dict:
    return {"event": {"completionStart": dict(ids)}}


def out_audio_content_start(ids: dict, content_id: str, rate: int) -> dict:
    return {"event": {"contentStart": {**ids, "contentId": content_id, "type": "AUDIO", "role": "ASSISTANT",
                                       "audioOutputConfiguration": {"mediaType": "audio/lpcm", "sampleRateHertz": rate,
                                                                    "sampleSizeBits": 16, "encoding": "base64",
                                                                    "channelCount": 1}}}}


def out_audio(ids: dict, content_id: str, pcm16: bytes) -> dict:
    return {"event": {"audioOutput": {**ids, "contentId": content_id,
                                      "content": base64.b64encode(pcm16).decode("ascii")}}}


def out_text_content_start(ids: dict, content_id: str, role: str, stage: str) -> dict:
    return {"event": {"contentStart": {**ids, "contentId": content_id, "type": "TEXT", "role": role,
                                       "additionalModelFields": json.dumps({"generationStage": stage}),
                                       "textOutputConfiguration": {"mediaType": "text/plain"}}}}


def out_text(ids: dict, content_id: str, text: str) -> dict:
    return {"event": {"textOutput": {**ids, "contentId": content_id, "content": text}}}


def out_content_end(ids: dict, content_id: str, ctype: str, stop_reason: str) -> dict:
    return {"event": {"contentEnd": {**ids, "contentId": content_id, "type": ctype, "stopReason": stop_reason}}}


def out_completion_end(ids: dict, stop_reason: str = "END_TURN") -> dict:
    return {"event": {"completionEnd": {**ids, "stopReason": stop_reason}}}
