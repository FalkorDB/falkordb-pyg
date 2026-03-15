# Project Guidelines

## Overview

falkordb-pyg is the PyTorch Geometric (PyG) remote backend for
[FalkorDB](https://github.com/FalkorDB/FalkorDB). It implements PyG's
`FeatureStore` and `GraphStore` interfaces, allowing GNN training and inference
directly on graphs stored in FalkorDB without loading the entire graph into
memory.

## Build & Install

```bash
uv sync                # install runtime dependencies
uv sync --extra test   # also install test dependencies (pytest, pytest-cov)
uv sync --group dev    # also install dev tools (ruff, mypy)
```

## Testing

Tests require a running FalkorDB instance on `localhost:6379`:

```bash
docker run -p 6379:6379 -d falkordb/falkordb:edge
```

Run all tests:

```bash
uv run pytest
```

Run a single test file or test:

```bash
uv run pytest tests/test_graph_store.py
uv run pytest tests/test_feature_store.py::TestFalkorDBFeatureStore::test_get_tensor_scalar_feature
```

With coverage:

```bash
uv run pytest --cov --cov-report=xml
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
  __init__.py         # Public API: get_remote_backend, FalkorDBFeatureStore, FalkorDBGraphStore
  feature_store.py    # FalkorDBFeatureStore — implements PyG FeatureStore ABC
  graph_store.py      # FalkorDBGraphStore — implements PyG GraphStore ABC
  utils.py            # NodeIDMapper, Cypher query builders, helpers
tests/
  test_feature_store.py   # Unit tests for FalkorDBFeatureStore (mocked FalkorDB)
  test_graph_store.py     # Unit tests for FalkorDBGraphStore (mocked FalkorDB)
  test_integration.py     # Integration tests with NeighborLoader (mocked)
examples/
  train_example.py        # Full GNN training example using FalkorDB backend
```

## Architecture Patterns

### PyG Remote Backend

The package implements PyG's remote backend protocol:

- `FalkorDBFeatureStore` subclasses `torch_geometric.data.FeatureStore`
- `FalkorDBGraphStore` subclasses `torch_geometric.data.GraphStore`
- `get_remote_backend()` returns `Tuple[FeatureStore, GraphStore]`
- The tuple plugs directly into `NeighborLoader(data=(feature_store, graph_store), ...)`

### FalkorDB Integration

- Uses the `falkordb` Python client (`pip install FalkorDB`)
- Queries graph topology via Cypher: `MATCH (s)-[r]->(d) RETURN ID(s), ID(d)`
- Fetches features via Cypher: `MATCH (n:Label) RETURN n.property ORDER BY ID(n)`
- Caches results locally to avoid repeated network round-trips

### Node ID Mapping

- FalkorDB uses internal node IDs that may not be contiguous
- `NodeIDMapper` maps FalkorDB IDs to contiguous 0-based PyG indices
- Built lazily on first access per node type

## CI/CD

- **`lint.yml`**: Runs ruff format, ruff check, and mypy on Python 3.13
- **`spellcheck.yml`**: Runs pyspelling on all `*.md` files; custom wordlist at `.github/wordlist.txt`
- **`test.yml`**: Runs pytest against a `falkordb/falkordb:edge` Docker service on Python 3.10–3.13; uploads coverage to Codecov
- **`publish.yaml`**: Publishes to PyPI on version tags (`v*.*.*`) using `uv build` and `uv publish`

## Before Finishing a Task

After completing any task, review whether your changes require updates to:

- **`README.md`** — if public API, usage examples, or installation instructions changed
- **`AGENTS.md`** — if project structure, build commands, architecture patterns, or conventions changed
