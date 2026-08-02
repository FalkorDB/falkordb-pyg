# Project Guidelines

## Overview

falkordb-pyg is the PyTorch Geometric (PyG) remote backend for
[FalkorDB](https://github.com/FalkorDB/FalkorDB). It implements PyG's
`FeatureStore` and `GraphStore` interfaces so that GNN training and inference
can read graphs stored in FalkorDB through PyG's standard data loaders.

Note the current scope honestly: both stores fetch a whole feature column or a
whole edge list on first access and cache it in process. Peak memory is O(graph),
not O(batch). See the README's *Current limitations* section — do not add claims
to the docs that the code does not honour.

## Build & Install

```bash
uv sync                # install runtime dependencies
uv sync --extra test   # also install test dependencies (pytest, pytest-cov)
uv sync --group dev    # also install dev tools (ruff, mypy)
```

## Testing

Unit and integration tests are fully mocked and need no server. The `e2e`
subset requires a running FalkorDB instance:

```bash
docker run -p 6379:6379 -d falkordb/falkordb:edge
```

Run all tests (pytest lives in the `test` extra, so pass it):

```bash
uv run --extra test pytest
```

Run a single test file or test:

```bash
uv run --extra test pytest tests/test_graph_store.py
uv run --extra test pytest "tests/test_feature_store.py::TestPublicAPI::test_get_tensor"
```

Skip or require the e2e subset:

```bash
uv run --extra test pytest -m "not e2e"      # mocked tests only
REQUIRE_FALKORDB=1 uv run --extra test pytest  # fail instead of skipping e2e
```

Point the e2e tests at a non-default server with `FALKORDB_HOST` /
`FALKORDB_PORT`. When neither is reachable the e2e tests are skipped, unless
`REQUIRE_FALKORDB` is set — CI sets it so a missing service cannot pass silently.

With coverage (the config enforces `fail_under = 90` over `falkordb_pyg/`):

```bash
uv run --extra test pytest --cov --cov-report=xml
```

## Pre-commit Checks

Always run these checks before every commit:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy falkordb_pyg/
```

If formatting fails, fix with `uv run ruff format .` before committing.
If spellcheck fails, add missing words to `.github/wordlist.txt`.

## Code Style

- **Formatter/linter**: Ruff (line length 88, target Python 3.10)
- **Lint rules**: `F` (Pyflakes), `E`/`W` (pycodestyle), `I` (isort)
- **Type checking**: mypy with `ignore_missing_imports = true`
- **Python**: requires >= 3.10; CI tests 3.10 through 3.13

## Project Structure

```
falkordb_pyg/
  __init__.py         # Public API: get_remote_backend, stores, __version__
  feature_store.py    # FalkorDBFeatureStore — implements PyG FeatureStore ABC
  graph_store.py      # FalkorDBGraphStore — implements PyG GraphStore ABC
  utils.py            # NodeIDMapper, Cypher query builders, quote_identifier
  py.typed            # PEP 561 marker — keep it in the wheel
tests/
  __init__.py
  conftest.py             # FakeFalkorGraph, shared fixtures, e2e gating
  test_feature_store.py   # Unit tests for FalkorDBFeatureStore
  test_graph_store.py     # Unit tests for FalkorDBGraphStore
  test_integration.py     # Both stores together + NeighborSampler metadata
  test_e2e.py             # Marked `e2e`; runs against a live FalkorDB
examples/
  train_example.py        # Full-batch GNN training example
```

## Architecture Patterns

### PyG Remote Backend

The package implements PyG's remote backend protocol:

- `FalkorDBFeatureStore` subclasses `torch_geometric.data.FeatureStore` and
  registers `FalkorDBTensorAttr` as its `tensor_attr_cls` — without that, the
  entire public API (`get_tensor`, `store[...]`, `view`) raises.
- `FalkorDBGraphStore` subclasses `torch_geometric.data.GraphStore`
- `get_remote_backend()` returns `Tuple[FeatureStore, GraphStore]`
- The tuple plugs into `NeighborLoader(data=(feature_store, graph_store), ...)`,
  **but only after the node/edge types have been accessed once** — the stores
  register a type lazily, so a cold backend presents an empty graph. Schema
  auto-discovery is the planned fix.

### Testing conventions

- Test the **public** API. The `_`-prefixed methods are ABC hooks; a suite that
  only drives them can be green while the interface PyG calls is broken. That is
  exactly what happened before v0.2.2.
- Use `FakeFalkorGraph` from `tests/conftest.py` rather than ad-hoc `MagicMock`
  graphs. It models the graph as data and matches the generated Cypher
  structurally, so it returns realistic rows (including `None` for a node
  missing a property) and a query-builder change fails loudly.

### FalkorDB Integration

- Uses the `falkordb` Python client (`pip install FalkorDB`)
- Queries graph topology via Cypher: `MATCH (s)-[r]->(d) RETURN ID(s), ID(d)`
- Fetches features via Cypher: `MATCH (n:Label) RETURN n.property, ID(n) ORDER BY ID(n)`
- Reads go through `ro_query` (`GRAPH.RO_QUERY`) unless `read_only=False`
- Caches results locally to avoid repeated network round-trips
- **All labels, relationship types and property names must go through
  `utils.quote_identifier`** — it doubles embedded backticks. Interpolating them
  raw allows Cypher injection. Values belong in `params=`, never in an f-string.

### Node ID Mapping

- FalkorDB uses internal node IDs that may not be contiguous
- `NodeIDMapper` maps FalkorDB IDs to contiguous 0-based PyG indices
- Built lazily on first access per node type; reachable via
  `graph_store.id_mapper(node_type)`
- The feature store relies on both stores' `ORDER BY ID(n)` producing the same
  order. Do not change the ordering of one query builder without the other.

## CI/CD

- **`lint.yml`**: Runs ruff format, ruff check, and mypy on Python 3.13
- **`spellcheck.yml`**: Runs pyspelling on all `*.md` files; custom wordlist at `.github/wordlist.txt`
- **`test.yml`**: Runs pytest against a `falkordb/falkordb:edge` Docker service on Python 3.10–3.13; also builds the wheel, checks `py.typed` ships, and verifies the package import-errors helpfully without the `torch` extra; uploads coverage to Codecov
- **`publish.yml`**: Publishes to PyPI on version tags (`v*.*.*`) using `uv build` and `uv publish`

## Before Finishing a Task

After completing any task, review whether your changes require updates to:

- **`README.md`** — if public API, usage examples, limitations, or installation
  instructions changed
- **`AGENTS.md`** — if project structure, build commands, architecture patterns,
  or conventions changed
