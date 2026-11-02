import json

import numpy as np

from novabridge import nova_events as ne
from novabridge.bridge import BridgeConfig, BridgeCore
from novabridge.g711 import float_to_ulaw

IDS = {"sessionId": "s", "promptName": "p", "completionId": "c"}


def tone_chunk(ms=80, rate=24000, amp=0.2):
    t = np.arange(int(rate * ms / 1000)) / rate
    return (amp * 32767 * np.sin(2 * np.pi * 440 * t)).astype("<i2").tobytes()


def feed_response(core, cid="audio-1", chunks=10, t=0.0):
    core.on_nova_event(ne.out_audio_content_start(IDS, cid, 24000), t)
    for _ in range(chunks):
        core.on_nova_event(ne.out_audio(IDS, cid, tone_chunk()), t)


def events(msgs):
    return [json.loads(m)["event"] for m in msgs]


def test_audio_is_resampled_and_framed():
    core = BridgeCore(BridgeConfig(mode="server_signal", echo_guard=False))
    feed_response(core, chunks=10)  # 10 x 80 ms = 800 ms = 40 frames
    assert len(core.queue) == 40
    out = core.tick(0)
    assert events(out) == ["media"] * 40  # unpaced: everything forwarded at once


def test_paced_mode_keeps_small_lead():
    core = BridgeCore(BridgeConfig(mode="full", echo_guard=False, lead_frames=3))
    feed_response(core, chunks=10)
    assert len(core.tick(0)) == 3
    assert len(core.tick(20)) == 1 and len(core.tick(40)) == 1


def test_server_interrupt_flushes_clears_and_drops_late_chunks():
    core = BridgeCore(BridgeConfig(mode="full", echo_guard=False))
    feed_response(core, chunks=10)
    core.tick(0)
    assert len(core.queue) > 0
    core.on_nova_event(ne.out_content_end(IDS, "text-final", "TEXT", "INTERRUPTED"), 100)
    assert len(core.queue) == 0
    out = core.take_outbox()
    assert events(out) == ["clear"]
    # late chunks of the interrupted response are dropped, not re-queued
    core.on_nova_event(ne.out_audio(IDS, "audio-1", tone_chunk()), 120)
    assert len(core.queue) == 0 and core.ctl.dropped_late_chunks == 1
    assert core.tick(140) == []
    # a second interrupt signal for the same response does not send another clear
    core.on_nova_event(ne.out_content_end(IDS, "audio-1", "AUDIO", "INTERRUPTED"), 160)
    assert core.take_outbox() == []
    # the next response plays normally
    feed_response(core, cid="audio-2", chunks=2, t=200)
    assert len(core.queue) == 8 and events(core.tick(220)) == ["media"] * 3


def test_naive_mode_ignores_interrupt():
    core = BridgeCore(BridgeConfig(mode="naive", echo_guard=False))
    feed_response(core, chunks=5)
    core.on_nova_event(ne.out_content_end(IDS, "t", "TEXT", "INTERRUPTED"), 10)
    assert core.take_outbox() == [] and len(core.queue) == 20


def test_local_vad_triggers_barge_in_while_bot_active():
    core = BridgeCore(BridgeConfig(mode="full", echo_guard=False, local_vad_min_frames=3))
    feed_response(core, chunks=10)
    core.tick(0)
    loud = float_to_ulaw(0.1 * np.sin(np.arange(160) * 0.3))
    for k in range(3):
        core.on_caller_frame(loud, 20 * (k + 1))
    assert [e.source for e in core.ctl.events] == ["local_vad"]
    assert events(core.take_outbox()) == ["clear"] and len(core.queue) == 0
    # Nova keeps streaming the same content until it notices: dropped
    core.on_nova_event(ne.out_audio(IDS, "audio-1", tone_chunk()), 80)
    assert len(core.queue) == 0


def test_local_vad_does_nothing_when_bot_silent():
    core = BridgeCore(BridgeConfig(mode="full", echo_guard=False))
    loud = float_to_ulaw(0.1 * np.sin(np.arange(160) * 0.3))
    for k in range(5):
        core.on_caller_frame(loud, 20 * k)
    assert core.ctl.events == [] and core.take_outbox() == []


def test_upstream_is_16k_pcm():
    core = BridgeCore(BridgeConfig(mode="full", echo_guard=True))
    pcm = core.on_caller_frame(float_to_ulaw(np.zeros(160)), 0)
    assert len(pcm) == 320 * 2
