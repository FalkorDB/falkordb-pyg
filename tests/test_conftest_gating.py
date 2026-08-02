"""Tests for the e2e gating hook in conftest.

These run pytest in a subprocess because the behaviour under test is collection
time hook ordering, which cannot be observed from inside a running session.
"""

import subprocess
import sys

import pytest

# A port nothing listens on, so the availability probe always fails.
UNREACHABLE = {"FALKORDB_HOST": "127.0.0.1", "FALKORDB_PORT": "65001"}


def _run(args, env_extra):
    import os

    env = {**os.environ, **env_extra}
    env.pop("REQUIRE_FALKORDB", None)
    env.update({k: v for k, v in env_extra.items()})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_require_falkordb_errors_when_e2e_is_selected_and_server_is_down():
    result = _run(
        ["tests/test_e2e.py", "--collect-only", "-q"],
        {**UNREACHABLE, "REQUIRE_FALKORDB": "1"},
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "REQUIRE_FALKORDB is set" in result.stdout + result.stderr


def test_require_falkordb_is_silent_when_e2e_is_deselected():
    """`-m "not e2e"` must not trip the probe for tests that will never run."""
    result = _run(
        ["tests/test_e2e.py", "-m", "not e2e", "--collect-only", "-q"],
        {**UNREACHABLE, "REQUIRE_FALKORDB": "1"},
    )
    combined = result.stdout + result.stderr
    assert "REQUIRE_FALKORDB is set" not in combined
    # Every test in the file is e2e, so deselecting them leaves nothing to run:
    # NO_TESTS_COLLECTED, not OK. The point is that it is not USAGE_ERROR.
    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED
    assert "deselected" in combined


def test_missing_server_only_skips_without_require_falkordb():
    result = _run(["tests/test_e2e.py", "-q"], UNREACHABLE)
    combined = result.stdout + result.stderr
    assert "REQUIRE_FALKORDB is set" not in combined
    assert result.returncode == pytest.ExitCode.OK
    assert "skipped" in combined
