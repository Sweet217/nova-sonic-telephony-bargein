import pytest

from novabridge import nova_client


def test_module_importable_without_sdk():
    c = nova_client.NovaSonicClient()
    assert c.model_id == nova_client.DEFAULT_MODEL_ID or c.model_id  # env may override
    assert c.in_rate == 16000 and c.out_rate == 24000


@pytest.mark.skipif(not nova_client.sdk_available(), reason="aws_sdk_bedrock_runtime needs Python >= 3.12")
def test_sdk_client_constructs_without_network():
    import asyncio
    c = nova_client.NovaSonicClient(region="us-east-1")
    assert asyncio.run(c._client()) is not None  # resolves config + builds the client; sends no request
