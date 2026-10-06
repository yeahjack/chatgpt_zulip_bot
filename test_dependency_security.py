"""Guard the dependency security floors across both supported install paths."""

from importlib.metadata import version

import anyio
import httpx
import pytest
from packaging.version import Version


@pytest.mark.parametrize(
    ("package", "minimum"),
    [
        ("anyio", "4.14.2"),
        ("idna", "3.15"),
        ("pygments", "2.20.0"),
        ("pytest", "9.0.3"),
        ("requests", "2.33.0"),
        ("urllib3", "2.7.0"),
    ],
)
def test_installed_dependency_security_floor(package, minimum):
    assert Version(version(package)) >= Version(minimum)


def test_httpx_async_transport_with_anyio():
    """Exercise the OpenAI SDK's HTTP stack without making network requests."""
    async def request():
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"ok": True}))
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.get("https://example.invalid/health")
        assert response.json() == {"ok": True}

    anyio.run(request)
