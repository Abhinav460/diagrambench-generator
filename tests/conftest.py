"""Test isolation for the family registry.

Registration mutates module-level state. Without restoring it between tests, a
fake family registered by one test stays visible to every later test, which makes
failures depend on execution order -- the kind of flakiness that wastes an
afternoon once the suite is large.
"""

from __future__ import annotations

import pytest

from generator import registry


@pytest.fixture(autouse=True)
def isolate_registry():
    """Snapshot the registry before each test and restore it afterwards."""
    saved = dict(registry._REGISTRY)
    saved_discovered = registry._DISCOVERED
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
    registry._DISCOVERED = saved_discovered
