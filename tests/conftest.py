"""Shared pytest configuration and fixtures for the test suite."""

import os
import socket
import warnings

import pytest


def _resolve_port() -> int:
    """Read ``FALKORDB_PORT``, falling back when it is unset or malformed.

    A bad value must not raise: this is read while pytest is still collecting,
    and an unusable port should degrade to a skipped e2e suite rather than
    taking down the whole run.
    """
    raw = os.environ.get("FALKORDB_PORT", "6379")
    try:
        return int(raw)
    except ValueError:
        warnings.warn(
            f"FALKORDB_PORT={raw!r} is not an integer; falling back to 6379.",
            stacklevel=2,
        )
        return 6379


# Resolved once. Test modules import these rather than re-reading the
# environment, so there is a single definition of the endpoint and a malformed
# port is reported once instead of on every probe.
falkordb_host = os.environ.get("FALKORDB_HOST", "localhost")
falkordb_port = _resolve_port()


def _falkordb_available() -> bool:
    """Return True if a FalkorDB server accepts a TCP connection.

    A plain socket probe with a short timeout is enough and — unlike a redis
    handshake — cannot hang collection for minutes against a host that drops
    packets instead of refusing them.
    """
    try:
        with socket.create_connection((falkordb_host, falkordb_port), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Skip e2e tests when FalkorDB is unreachable — loudly if REQUIRE_FALKORDB.

    ``trylast`` matters: pytest applies ``-m`` / ``-k`` deselection in its own
    implementation of this hook, so running first would see e2e items that are
    about to be deselected and probe (or hard-fail) for a server the run will
    never touch.

    The probe runs only when e2e tests were actually selected, and a skip is
    fail-open by default: set ``REQUIRE_FALKORDB=1`` (CI does) to turn a missing
    server into an error rather than a silently green run.
    """
    if not any("e2e" in item.keywords for item in items):
        return
    if _falkordb_available():
        return
    reason = f"FalkorDB server not available at {falkordb_host}:{falkordb_port}"
    if os.environ.get("REQUIRE_FALKORDB"):
        raise pytest.UsageError(
            f"{reason}, but REQUIRE_FALKORDB is set. Start one with: "
            f"docker run -p 6379:6379 -d falkordb/falkordb:edge"
        )
    skip_e2e = pytest.mark.skip(reason=reason)
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)
