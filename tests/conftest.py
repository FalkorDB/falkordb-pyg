"""Shared pytest configuration, fixtures and a faithful FalkorDB fake."""

from __future__ import annotations

import os
import re
import socket
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pytest

# ---------------------------------------------------------------------------
# A fake falkordb.Graph
# ---------------------------------------------------------------------------

# The exact query shapes produced by falkordb_pyg.utils. Matching on structure
# rather than on ad-hoc substrings means a change to a query builder shows up
# as a test failure instead of silently falling through to a default.
_IDENT = r"`((?:[^`]|``)*)`"
_NODE_IDS_RE = re.compile(rf"^MATCH \(n:{_IDENT}\) RETURN ID\(n\) ORDER BY ID\(n\)$")
_FEATURE_RE = re.compile(
    rf"^MATCH \(n:{_IDENT}\) RETURN n\.{_IDENT}, ID\(n\) ORDER BY ID\(n\)$"
)
_EDGE_RE = re.compile(
    rf"^MATCH \(s:{_IDENT}\)-\[r:{_IDENT}\]->\(d:{_IDENT}\) RETURN ID\(s\), ID\(d\)$"
)


def _unescape(ident: str) -> str:
    """Reverse the backtick doubling applied by ``utils.quote_identifier``."""
    return ident.replace("``", "`")


class FakeResult:
    """Stand-in for a falkordb QueryResult."""

    def __init__(self, rows: Iterable[Sequence[Any]]) -> None:
        self.result_set: List[Sequence[Any]] = [list(row) for row in rows]


class FakeFalkorGraph:
    """Minimal but faithful stand-in for ``falkordb.Graph``.

    Models the graph as data rather than as canned per-query responses, so the
    rows it returns have the same shape a real server produces — including
    ``None`` for a node that lacks the requested property, which is the norm in
    a schemaless graph database.

    Args:
        nodes: ``{label: {falkor_id: {property: value}}}``.
        edges: ``{(src_label, rel_type, dst_label): [(src_id, dst_id), ...]}``.
    """

    def __init__(
        self,
        nodes: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
        edges: Optional[Dict[Tuple[str, str, str], List[Tuple[int, int]]]] = None,
    ) -> None:
        self.nodes = nodes or {}
        self.edges = edges or {}
        self.calls: List[str] = []

    # -- falkordb.Graph surface the backend actually uses -------------------

    def query(self, q: str, params: Optional[dict] = None, **kwargs) -> FakeResult:
        self.calls.append(q)
        return self._dispatch(q)

    # -- internals ----------------------------------------------------------

    def _dispatch(self, q: str) -> FakeResult:
        match = _NODE_IDS_RE.match(q)
        if match:
            label = _unescape(match.group(1))
            return FakeResult([[nid] for nid in sorted(self.nodes.get(label, {}))])

        match = _FEATURE_RE.match(q)
        if match:
            label, prop = _unescape(match.group(1)), _unescape(match.group(2))
            props = self.nodes.get(label, {})
            return FakeResult([[props[nid].get(prop), nid] for nid in sorted(props)])

        match = _EDGE_RE.match(q)
        if match:
            key = (
                _unescape(match.group(1)),
                _unescape(match.group(2)),
                _unescape(match.group(3)),
            )
            return FakeResult([list(pair) for pair in self.edges.get(key, [])])

        raise AssertionError(f"FakeFalkorGraph received an unrecognised query: {q!r}")


# ---------------------------------------------------------------------------
# Shared graph fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def homo_graph() -> FakeFalkorGraph:
    """Three 'paper' nodes with non-contiguous IDs, and three 'cites' edges."""
    return FakeFalkorGraph(
        nodes={
            "paper": {
                10: {"x": [1.0, 2.0], "y": 0},
                11: {"x": [3.0, 4.0], "y": 1},
                12: {"x": [5.0, 6.0], "y": 2},
            }
        },
        edges={("paper", "cites", "paper"): [(10, 11), (11, 12), (11, 10)]},
    )


@pytest.fixture()
def hetero_graph() -> FakeFalkorGraph:
    """Two 'author' nodes and three 'paper' nodes, with 'writes' and 'cites'."""
    return FakeFalkorGraph(
        nodes={
            "author": {10: {"x": [0.5, 0.5]}, 11: {"x": [0.1, 0.9]}},
            "paper": {
                0: {"x": [1.0, 0.0], "y": 0},
                1: {"x": [0.0, 1.0], "y": 1},
                2: {"x": [0.5, 0.5], "y": 2},
            },
        },
        edges={
            ("author", "writes", "paper"): [(10, 0), (10, 1), (11, 2)],
            ("paper", "cites", "paper"): [(0, 1), (1, 2)],
        },
    )


# ---------------------------------------------------------------------------
# e2e gating
# ---------------------------------------------------------------------------


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
