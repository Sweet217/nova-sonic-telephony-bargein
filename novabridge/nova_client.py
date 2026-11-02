"""Real Amazon Nova 2 Sonic client over Bedrock ``InvokeModelWithBidirectionalStream``.

Uses the official experimental Python SDK (``aws_sdk_bedrock_runtime``, Smithy
based, async). That SDK requires **Python >= 3.12**; the rest of this repo runs
on 3.11, so the import is lazy and the module stays importable everywhere.

    pip install -r requirements-nova.txt     # in a Python 3.12+ venv

Not exercised against the live service yet (no AWS session was available when
this was written) -- see docs/RUN_WITH_REAL_NOVA.md. Event JSON comes from
``novabridge.nova_events`` and matches the Nova 2 Sonic user guide.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import AsyncIterator

from . import nova_events as ne

DEFAULT_MODEL_ID = "amazon.nova-2-sonic-v1:0"
DEFAULT_REGION = "us-east-1"
DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly phone assistant for a fictional retail store. Keep answers to one or two short "
    "sentences. If the caller interrupts, stop and address what they just said."
)


def sdk_available() -> bool:
    try:
        import aws_sdk_bedrock_runtime  # noqa: F401
    except ImportError:
        return False
    return True


class NovaSonicClient:
    """Backend with the same interface as ``testbed.realtime.MockNovaBackend``:
    ``start()``, ``send_audio(pcm16)``, ``events()``, ``close()``."""

    def __init__(self, model_id: str | None = None, region: str | None = None, voice_id: str = "tiffany",
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT, in_rate: int = 16000, out_rate: int = 24000,
                 endpointing: str | None = "MEDIUM"):
        self.model_id = model_id or os.getenv("NOVA_MODEL_ID", DEFAULT_MODEL_ID)
        self.region = region or os.getenv("AWS_REGION", DEFAULT_REGION)
        self.voice_id, self.system_prompt = voice_id, system_prompt
        self.in_rate, self.out_rate, self.endpointing = in_rate, out_rate, endpointing
        self.prompt_name = str(uuid.uuid4())
        self.audio_content = str(uuid.uuid4())
        self._stream = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._closed = False

    # ------------------------------------------------------------ plumbing
    async def _client(self):
        try:
            from aws_sdk_bedrock_runtime.client import AsyncBedrockRuntimeClient
            from aws_sdk_bedrock_runtime.config import AsyncBedrockRuntimeConfig
        except ImportError as e:  # pragma: no cover - depends on Python version
            raise RuntimeError("aws_sdk_bedrock_runtime is not installed (needs Python >= 3.12); "
                               "pip install -r requirements-nova.txt") from e
        kwargs = {"region": self.region}
        if os.getenv("AWS_ACCESS_KEY_ID"):
            from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver
            kwargs["aws_credentials_identity_resolver"] = EnvironmentCredentialsResolver()
        # SDK >= 0.12: configs are resolved asynchronously (env, config files, defaults + overrides)
        config = await AsyncBedrockRuntimeConfig.resolve(**kwargs)
        return AsyncBedrockRuntimeClient(config=config)

    async def _send(self, event: dict) -> None:
        from aws_sdk_bedrock_runtime.models import (
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk,
        )
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=json.dumps(event).encode("utf-8")))
        await self._stream.input_stream.send(chunk)

    async def _read_loop(self) -> None:
        try:
            _, output = await self._stream.await_output()
            async for item in output:
                value = getattr(item, "value", None)
                if value is None or getattr(value, "bytes_", None) is None:
                    # modelled exceptions arrive as union members without a payload
                    await self._queue.put({"event": {"error": {"type": type(item).__name__, "detail": str(item)}}})
                    continue
                await self._queue.put(json.loads(value.bytes_.decode("utf-8")))
        finally:
            await self._queue.put(None)

    # ----------------------------------------------------------- interface
    async def start(self) -> None:
        from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamOperationInput
        client = await self._client()
        self._stream = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id))
        sys_name = str(uuid.uuid4())
        for ev in (
            ne.session_start(endpointing=self.endpointing),
            ne.prompt_start(self.prompt_name, self.voice_id, self.out_rate),
            ne.text_content_start(self.prompt_name, sys_name, role="SYSTEM"),
            ne.text_input(self.prompt_name, sys_name, self.system_prompt),
            ne.content_end(self.prompt_name, sys_name),
            ne.audio_content_start(self.prompt_name, self.audio_content, self.in_rate),
        ):
            await self._send(ev)
        self._reader = asyncio.create_task(self._read_loop())

    async def send_audio(self, pcm16: bytes) -> None:
        if not self._closed:
            await self._send(ne.audio_input(self.prompt_name, self.audio_content, pcm16))

    async def events(self) -> AsyncIterator[dict]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def close(self) -> None:
        if self._closed or self._stream is None:
            return
        self._closed = True
        for ev in (ne.content_end(self.prompt_name, self.audio_content), ne.prompt_end(self.prompt_name),
                   ne.session_end()):
            try:
                await self._send(ev)
            except Exception:  # stream may already be closed by the service
                break
        await self._stream.input_stream.close()
        if self._reader:
            try:
                await asyncio.wait_for(self._reader, timeout=5)
            except (asyncio.TimeoutError, Exception):
                self._reader.cancel()
