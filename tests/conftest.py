"""Shared pytest configuration and fixtures for the test suite."""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: end-to-end tests requiring a running FalkorDB server"
    )


def _falkordb_available() -> bool:
    """Return True if a FalkorDB server is reachable on the configured host/port."""
    try:
        from falkordb import FalkorDB

        host = os.environ.get("FALKORDB_HOST", "localhost")
        port = int(os.environ.get("FALKORDB_PORT", "6379"))
        db = FalkorDB(host=host, port=port)
        g = db.select_graph("__ping__")
        g.query("RETURN 1")
        return True
    except Exception:
        return False


_falkordb_is_up = _falkordb_available()


def pytest_collection_modifyitems(config, items):
    """Auto-skip e2e tests when FalkorDB is not reachable."""
    if _falkordb_is_up:
        return
    skip_e2e = pytest.mark.skip(reason="FalkorDB server not available")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)
