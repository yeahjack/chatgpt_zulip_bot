"""Guard the dependency security floors across both supported install paths."""

from importlib.metadata import version

import anyio
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
        ("urllib3", "2.8.0"),
    ],
)
def test_installed_dependency_security_floor(package, minimum):
    assert Version(version(package)) >= Version(minimum)


def test_anyio_task_group():
    """Exercise the upgraded async runtime without optional HTTP dependencies."""
    results = []

    async def record_result():
        await anyio.sleep(0)
        results.append("ok")

    async def run_tasks():
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(record_result)

    anyio.run(run_tasks)
    assert results == ["ok"]
