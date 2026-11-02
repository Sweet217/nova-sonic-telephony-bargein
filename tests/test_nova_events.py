import base64
import json

import numpy as np

from novabridge import nova_events as ne

IDS = {"sessionId": "s", "promptName": "p", "completionId": "c"}


def test_audio_output_parses_to_int16():
    pcm = np.arange(-5, 5, dtype="<i2")
    ev = ne.parse_output(json.dumps(ne.out_audio(IDS, "a1", pcm.tobytes())))
    assert ev.kind == "audio" and ev.content_id == "a1"
    np.testing.assert_array_equal(ev.pcm, pcm)


def test_interrupted_stop_reason_text_and_audio():
    for ctype in ("TEXT", "AUDIO"):
        ev = ne.parse_output(ne.out_content_end(IDS, "x", ctype, "INTERRUPTED"))
        assert ev.kind == "interrupted" and ev.type == ctype
    assert ne.parse_output(ne.out_content_end(IDS, "x", "AUDIO", "END_TURN")).kind == "content_end"


def test_legacy_interrupted_text_marker():
    assert ne.parse_output(ne.out_text(IDS, "t", '{ "interrupted" : true }')).kind == "interrupted"
    assert ne.parse_output(ne.out_text(IDS, "t", "hello")).kind == "text"


def test_content_start_fields():
    ev = ne.parse_output(ne.out_audio_content_start(IDS, "a1", 24000))
    assert (ev.kind, ev.type, ev.role, ev.sample_rate) == ("content_start", "AUDIO", "ASSISTANT", 24000)
    ev = ne.parse_output(ne.out_text_content_start(IDS, "t1", "ASSISTANT", "SPECULATIVE"))
    assert ev.generation_stage == "SPECULATIVE"


def test_input_events_match_documented_shapes():
    ev = ne.audio_content_start("p", "c", 16000)["event"]["contentStart"]
    assert ev["type"] == "AUDIO" and ev["role"] == "USER" and ev["interactive"] is True
    assert ev["audioInputConfiguration"]["sampleRateHertz"] == 16000
    ai = ne.audio_input("p", "c", b"\x01\x00")["event"]["audioInput"]
    assert base64.b64decode(ai["content"]) == b"\x01\x00"
    ss = ne.session_start()["event"]["sessionStart"]
    assert ss["turnDetectionConfiguration"]["endpointingSensitivity"] == "MEDIUM"
    assert ne.session_end() == {"event": {"sessionEnd": {}}}
