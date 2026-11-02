"""Client-side barge-in handling: playout queue, flush, tombstones.

Nova 2 Sonic stops *generating* when it detects an interruption, but audio it
already produced may be sitting in three places: the bridge's playout queue,
in flight on the websocket, and in Twilio's own playback buffer. The handler:

1. stops sending outbound frames immediately,
2. flushes the bridge's playout queue,
3. sends Twilio ``clear`` so Twilio drops what it has buffered,
4. tombstones the interrupted ``contentId`` so late audio chunks from that
   response (already in flight from Nova, or still being generated because the
   local trigger fired before the server noticed) are dropped instead of
   resuming playback.

It can be triggered by the server signal (``contentEnd`` INTERRUPTED) or by a
local VAD running on the echo-guarded caller audio, whichever comes first.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Frame:
    ulaw: bytes
    content_id: str | None


@dataclass
class BargeInEvent:
    t_ms: float
    source: str            # "server" | "local_vad"
    content_id: str | None
    flushed_frames: int


class PlayoutQueue:
    def __init__(self) -> None:
        self._q: deque[Frame] = deque()

    def push(self, f: Frame) -> None:
        self._q.append(f)

    def pop(self) -> Frame | None:
        return self._q.popleft() if self._q else None

    def flush(self) -> int:
        n = len(self._q)
        self._q.clear()
        return n

    def __len__(self) -> int:
        return len(self._q)


@dataclass
class BargeInController:
    tombstones: set = field(default_factory=set)
    events: list = field(default_factory=list)
    dropped_late_chunks: int = 0

    def is_dead(self, content_id: str | None) -> bool:
        return content_id is not None and content_id in self.tombstones

    def interrupt(self, t_ms: float, source: str, content_id: str | None, queue: PlayoutQueue) -> BargeInEvent | None:
        """Returns the event if this is a new interruption of ``content_id``, else None."""
        if content_id is not None and content_id in self.tombstones:
            return None
        if content_id is not None:
            self.tombstones.add(content_id)
        ev = BargeInEvent(t_ms, source, content_id, queue.flush())
        self.events.append(ev)
        return ev
