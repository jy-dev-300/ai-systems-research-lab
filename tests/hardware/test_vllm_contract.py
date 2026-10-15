import os

import pytest

from serving.vllm import VLLMBackend

pytestmark = pytest.mark.hardware


@pytest.mark.anyio
@pytest.mark.skipif(not os.getenv("VLLM_BASE_URL"), reason="set VLLM_BASE_URL for real-server test")
async def test_vllm_health_on_configured_server() -> None:
    backend = VLLMBackend(
        base_url=os.environ["VLLM_BASE_URL"],
        model=os.getenv("VLLM_MODEL", "configured-by-server"),
    )
    assert await backend.health()
