# falkordb-pyg

PyTorch Geometric (PyG) remote backend for [FalkorDB](https://github.com/FalkorDB/FalkorDB).

[![PyPI version](https://badge.fury.io/py/falkordb-pyg.svg)](https://badge.fury.io/py/falkordb-pyg)
[![Test](https://github.com/FalkorDB/falkordb-pyg/actions/workflows/test.yml/badge.svg)](https://github.com/FalkorDB/falkordb-pyg/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/FalkorDB/falkordb-pyg/branch/main/graph/badge.svg)](https://codecov.io/gh/FalkorDB/falkordb-pyg)

## Overview

`falkordb-pyg` implements PyG's `FeatureStore` and `GraphStore` interfaces,
allowing GNN training and inference directly on graphs stored in FalkorDB
without loading the entire graph into memory.

## Installation

```bash
pip install falkordb-pyg
```

## Quick Start

### 1. Start FalkorDB

```bash
docker run -p 6379:6379 -d falkordb/falkordb:edge
```

### 2. Load your graph data into FalkorDB

```python
from falkordb import FalkorDB

db = FalkorDB(host='localhost', port=6379)
graph = db.select_graph('my_graph')

# Create nodes with features
graph.query("CREATE (:Paper {x: [1.0, 0.0, 1.0], y: 0})")
graph.query("CREATE (:Paper {x: [0.0, 1.0, 0.0], y: 1})")

# Create edges
graph.query(
    "MATCH (a:Paper {y: 0}), (b:Paper {y: 1}) CREATE (a)-[:CITES]->(b)"
)
```

### 3. Use with PyG's NeighborLoader

```python
from falkordb_pyg import get_remote_backend
from torch_geometric.loader import NeighborLoader

feature_store, graph_store = get_remote_backend(
    host='localhost',
    port=6379,
    graph_name='my_graph',
)

loader = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors=[10, 5],
    input_nodes=('Paper', None),
    batch_size=32,
)

for batch in loader:
    # batch is a HeteroData object ready for GNN training
    pass
```

## API Reference

### `get_remote_backend(host, port, graph_name)`

Creates a `(FalkorDBFeatureStore, FalkorDBGraphStore)` tuple that can be
passed directly to any PyG loader.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `'localhost'` | FalkorDB host address |
| `port` | `int` | `6379` | FalkorDB port number |
| `graph_name` | `str` | `'default'` | Name of the graph in FalkorDB |

### `FalkorDBFeatureStore`

Implements `torch_geometric.data.FeatureStore`. Retrieves node and edge features
from FalkorDB via Cypher queries. Results are cached locally after the first
fetch.

### `FalkorDBGraphStore`

Implements `torch_geometric.data.GraphStore`. Retrieves edge connectivity from
FalkorDB via Cypher queries. Uses `NodeIDMapper` to remap FalkorDB's internal
node IDs to contiguous 0-based PyG indices.

## Architecture

```
FalkorDB Graph DB
       │
       │  Cypher queries (MATCH, RETURN)
       ▼
┌─────────────────────────────────────────────┐
│              falkordb-pyg                   │
│  ┌──────────────────┐  ┌──────────────────┐ │
│  │ FalkorDBFeature  │  │  FalkorDBGraph   │ │
│  │     Store        │  │     Store        │ │
│  └────────┬─────────┘  └────────┬─────────┘ │
└───────────┼─────────────────────┼───────────┘
            │                     │
            └──────────┬──────────┘
                       ▼
              PyG NeighborLoader /
              LinkNeighborLoader
                       │
                       ▼
                  GNN Model
```

## Contributing

See [AGENTS.md](AGENTS.md) for development guidelines, project structure, and
coding conventions.

## License

MIT — see [LICENSE](LICENSE).