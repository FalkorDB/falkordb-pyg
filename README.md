# falkordb-pyg

[![PyPI version](https://img.shields.io/pypi/v/falkordb-pyg.svg)](https://pypi.org/project/falkordb-pyg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**PyTorch Geometric Remote Backend for FalkorDB** — train GNNs directly on graphs stored in FalkorDB, without loading the entire graph into memory.

## What is it?

`falkordb-pyg` implements PyG's [Remote Backend interface](https://pytorch-geometric.readthedocs.io/en/latest/advanced/remote.html) (`FeatureStore` + `GraphStore`) for [FalkorDB](https://www.falkordb.com/), a high-performance graph database built on Redis. Once connected, you can plug the backend directly into `NeighborLoader`, `LinkNeighborLoader`, and other standard PyG data loaders — no changes to your model or training code required.

**Key features:**
- Zero-copy lazy loading — features and topology are fetched on demand and cached locally
- Heterogeneous graph support (multiple node and edge types)
- Automatic FalkorDB → PyG node ID remapping (non-contiguous IDs handled transparently)
- Drop-in replacement for any other PyG remote backend

## Installation

```bash
pip install falkordb-pyg
```

> **Requires:** Python ≥ 3.9, PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.4, FalkorDB Python client ≥ 1.0.

## Quick Start

### 1. Start FalkorDB

```bash
docker run -p 6379:6379 falkordb/falkordb:latest
```

### 2. Load data into FalkorDB

```python
from falkordb import FalkorDB

db = FalkorDB(host="localhost", port=6379)
graph = db.select_graph("papers")

# Create nodes with features and labels
graph.query("CREATE (:paper {x: [1.0, 0.0, 1.0], y: 0})")
graph.query("CREATE (:paper {x: [0.0, 1.0, 0.5], y: 1})")

# Create edges
graph.query(
    "MATCH (a:paper), (b:paper) "
    "WHERE ID(a) = 0 AND ID(b) = 1 "
    "CREATE (a)-[:cites]->(b)"
)
```

### 3. Create the remote backend

```python
from falkordb_pyg import get_remote_backend

feature_store, graph_store = get_remote_backend(
    host="localhost",
    port=6379,
    graph_name="papers",
)
```

### 4. Use with NeighborLoader

```python
from torch_geometric.loader import NeighborLoader
import torch

train_nodes = torch.tensor([0])

loader = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors={("paper", "cites", "paper"): [10, 10]},
    batch_size=32,
    input_nodes=("paper", train_nodes),
)

for batch in loader:
    paper_x = batch["paper"].x
    paper_y = batch["paper"].y
    edge_index = batch["paper", "cites", "paper"].edge_index
    # ... forward pass, loss, backward ...
```

## API Reference

### `get_remote_backend`

```python
from falkordb_pyg import get_remote_backend

feature_store, graph_store = get_remote_backend(
    host="localhost",           # FalkorDB / Redis hostname
    port=6379,                  # FalkorDB / Redis port
    graph_name="default",       # Graph name in FalkorDB
    node_type_to_label=None,    # Dict[str, str] — PyG type → FalkorDB label
    edge_type_to_rel=None,      # Dict[Tuple, str] — PyG edge triple → rel type
)
```

Returns a `(FalkorDBFeatureStore, FalkorDBGraphStore)` tuple.

---

### `FalkorDBFeatureStore`

Implements [`torch_geometric.data.FeatureStore`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.FeatureStore).

| Method | Description |
|---|---|
| `_get_tensor(attr)` | Fetch a node-feature tensor (lazy, cached) |
| `_put_tensor(tensor, attr)` | Store a tensor in the local cache |
| `_remove_tensor(attr)` | Remove a cached tensor |
| `_get_tensor_size(attr)` | Return the shape of a tensor |
| `get_all_tensor_attrs()` | List all registered `TensorAttr` objects |

**Constructor:**

```python
FalkorDBFeatureStore(
    graph,                     # falkordb.Graph instance
    node_type_to_label=None,   # Optional Dict[str, str]
)
```

---

### `FalkorDBGraphStore`

Implements [`torch_geometric.data.GraphStore`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.GraphStore).

| Method | Description |
|---|---|
| `_get_edge_index(attr)` | Fetch a COO edge index (lazy, cached) |
| `_put_edge_index(edge_index, attr)` | Store a COO edge index in the local cache |
| `_remove_edge_index(attr)` | Remove a cached edge index |
| `get_all_edge_attrs()` | List all registered `EdgeAttr` objects |

**Constructor:**

```python
FalkorDBGraphStore(
    graph,                     # falkordb.Graph instance
    node_type_to_label=None,   # Optional Dict[str, str]
    edge_type_to_rel=None,     # Optional Dict[Tuple[str,str,str], str]
)
```

---

### `FalkorDBTensorAttr`

A [`TensorAttr`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.TensorAttr) subclass where `index` defaults to `None` instead of `UNSET`.

```python
from falkordb_pyg.feature_store import FalkorDBTensorAttr

attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
attr_indexed = FalkorDBTensorAttr(group_name="paper", attr_name="x", index=torch.tensor([0, 1, 2]))
```

---

### `NodeIDMapper`

Bidirectional mapping between FalkorDB internal node IDs and contiguous 0-based PyG indices.

```python
from falkordb_pyg.utils import NodeIDMapper

mapper = NodeIDMapper(falkordb_ids=[100, 200, 300])
mapper.falkor_to_pyg(200)  # -> 1
mapper.pyg_to_falkor(1)    # -> 200
mapper.num_nodes            # -> 3
```

## Node ID Remapping

FalkorDB assigns internal integer IDs to nodes that may not be contiguous or start at zero. `falkordb-pyg` transparently builds a `NodeIDMapper` for each node type on first access, converting FalkorDB IDs to contiguous PyG indices. Edges referencing IDs not present in the mapper are silently dropped.

## Comparison with Kuzu PyG Integration

| Feature | Kuzu | FalkorDB |
|---|---|---|
| Database type | In-process embedded | Client-server (Redis-based) |
| Query language | Cypher | OpenCypher |
| PyG integration | Native `FeatureStore`/`GraphStore` | `falkordb-pyg` |
| Heterogeneous graphs | ✅ | ✅ |
| Lazy feature loading | ✅ | ✅ |
| Multi-host deployment | ❌ | ✅ |

## Examples

See [`examples/train_example.py`](examples/train_example.py) for a complete GraphSAGE training script.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/FalkorDB/falkordb-pyg).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests for your changes
4. Run the test suite: `pytest tests/`
5. Submit a pull request

## License

MIT — see [LICENSE](LICENSE).