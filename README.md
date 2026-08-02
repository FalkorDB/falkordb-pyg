# falkordb-pyg

[![PyPI version](https://img.shields.io/pypi/v/falkordb-pyg.svg)](https://pypi.org/project/falkordb-pyg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/falkordb-pyg.svg)](https://pypi.org/project/falkordb-pyg/)

**PyTorch Geometric Remote Backend for FalkorDB** — train GNNs on graphs stored in FalkorDB, using PyG's standard data loaders.

## What is it?

`falkordb-pyg` implements PyG's [Remote Backend interface](https://pytorch-geometric.readthedocs.io/en/latest/advanced/remote.html) (`FeatureStore` + `GraphStore`) for [FalkorDB](https://www.falkordb.com/), a high-performance graph database built on Redis. Once connected, the backend plugs into `NeighborLoader`, `LinkNeighborLoader`, and other standard PyG data loaders — no changes to your model or training code required.

**Key features:**

- Features and topology are read from FalkorDB with Cypher and cached in process, so repeated access costs no round-trips
- Heterogeneous graph support (multiple node and edge types)
- Automatic FalkorDB → PyG node ID remapping (non-contiguous IDs handled transparently)
- Reads use `GRAPH.RO_QUERY`, so they can be served by a replica

See [Current limitations](#current-limitations) before pointing this at a large graph.

## Installation

> **Prerequisite:** PyTorch and PyTorch Geometric must be installed first.
> Follow the [PyTorch](https://pytorch.org/get-started/locally/) and
> [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
> installation guides for your platform and CUDA version.

```bash
pip install falkordb-pyg
```

Or install with PyTorch and PyG included:

```bash
pip install 'falkordb-pyg[torch]'
```

> This resolves the default PyTorch wheel for your platform, which on Linux is a
> CUDA build. For CPU-only wheels, install torch first from
> `--index-url https://download.pytorch.org/whl/cpu`.

> **Neighbour sampling** additionally requires `pyg-lib` **or** `torch-sparse`,
> which are not installable from PyPI for every platform — see PyG's
> [wheel index](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
> Without one of them, `NeighborLoader` raises
> `ImportError: 'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'`.

> **Requires:** Python ≥ 3.10, PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.4, FalkorDB Python client ≥ 1.0.

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

# Create nodes and edges in one statement, binding them to local variables.
graph.query(
    """
    CREATE (p0:paper {x: [1.0, 0.0, 1.0], y: 0}),
           (p1:paper {x: [0.0, 1.0, 0.5], y: 1}),
           (p0)-[:cites]->(p1)
    """
)
```

> Never rely on `ID(n)` being 0-based, contiguous, or stable — FalkorDB reuses
> internal IDs after deletes. Match on your own key property (`MERGE (a:paper
> {paper_id: 1})`) when you need to address a specific node.

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

Both stores register a node type or edge type the first time it is accessed.
Until schema auto-discovery lands (see [Roadmap](#roadmap)), **prime the types
you intend to sample** before constructing a loader — otherwise the loader sees
an empty graph:

```python
from torch_geometric.loader import NeighborLoader
import torch

# Register the types with the stores. Note this is not free: the first access
# fetches and caches the whole feature column / whole edge list.
feature_store.get_tensor_size("paper", "x")
feature_store.get_tensor_size("paper", "y")
graph_store.get_edge_index(("paper", "cites", "paper"), layout="coo")

loader = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors={("paper", "cites", "paper"): [10, 10]},
    batch_size=32,
    input_nodes=("paper", torch.tensor([0])),
)

for batch in loader:
    paper_x = batch["paper"].x
    paper_y = batch["paper"].y
    edge_index = batch["paper", "cites", "paper"].edge_index
    # ... forward pass, loss, backward ...
```

> Batches are always `HeteroData`, even for a single-label graph. Call
> `batch.to_homogeneous()` if your model expects `batch.x` / `batch.edge_index`.

## Current limitations

Worth knowing before you scale up:

- **Peak memory is O(graph), not O(batch).** The first access to a node property
  fetches that entire column, and the first access to an edge type fetches the
  entire edge list. Both are then cached for the life of the process. PyG's
  `NeighborSampler` also builds a full CSC in client memory. The graph must fit
  in RAM.
- **No schema auto-discovery.** A freshly constructed backend reports no node
  types and no edge types; see [Quick Start step 4](#4-use-with-neighborloader).
- **No edge features or edge weights.** Requesting a tensor for an edge type
  raises `NotImplementedError`.
- **Caches are never invalidated automatically.** Neither store observes writes
  made after a value has been cached. After changing the graph, call
  `feature_store.clear_cache()` for features **and** `graph_store.clear_cache()`
  for topology — the latter also drops the node ID mappers, since adding or
  removing nodes changes the FalkorDB-ID to PyG-index assignment.
- **`num_workers > 0` is not supported.** Both stores hold a live connection and
  are not fork- or pickle-safe.
- **Feature properties must be uniform.** Every node carrying a label must define
  the property, with the same vector length and a numeric type. Violations raise
  an error naming the offending `ID(n)`.

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
    dtypes=None,                # Dict[Tuple[node_type, property], torch.dtype]
    read_only=True,             # Use GRAPH.RO_QUERY for all reads
    graph=None,                 # An existing falkordb.Graph to reuse
)
```

Returns a `(FalkorDBFeatureStore, FalkorDBGraphStore)` tuple.

Pass `graph=` to reuse a connection you already have — this is currently the way
to reach an authenticated or TLS-protected server, since the convenience
arguments do not yet cover credentials:

```python
from falkordb import FalkorDB
from falkordb_pyg import get_remote_backend

db = FalkorDB(host="…", port=6379, username="…", password="…", ssl=True)
feature_store, graph_store = get_remote_backend(graph=db.select_graph("papers"))
```

---

### `FalkorDBFeatureStore`

Implements [`torch_geometric.data.FeatureStore`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.FeatureStore).

| Method | Description |
|---|---|
| `get_tensor(group_name, attr_name, index=None)` | Fetch a node-feature tensor (lazy, cached) |
| `store[group_name, attr_name]` | Shorthand for `get_tensor` |
| `get_tensor_size(group_name, attr_name)` | Full shape of the stored feature matrix |
| `put_tensor(tensor, group_name, attr_name)` | Store a tensor in the local cache |
| `remove_tensor(group_name, attr_name)` | Evict a cached tensor |
| `multi_get_tensor(attrs)` | Fetch several tensors |
| `view(group_name)` | Attribute-style access, e.g. `store.view("paper").x` |
| `get_all_tensor_attrs()` | List all registered `TensorAttr` objects |
| `clear_cache(group_name=None, attr_name=None)` | Drop cached tensors so the next read hits FalkorDB |

The `_`-prefixed methods (`_get_tensor`, `_put_tensor`, …) are PyG ABC hooks, not
a consumer API — call the public methods above.

**Constructor:**

```python
FalkorDBFeatureStore(
    graph,                     # falkordb.Graph instance
    node_type_to_label=None,   # Optional Dict[str, str]
    dtypes=None,               # Optional Dict[Tuple[str, str], torch.dtype]
    read_only=True,            # Use GRAPH.RO_QUERY
)
```

**Dtypes.** Vector properties always become `torch.float`, since they are model
inputs. Scalar properties keep their FalkorDB type — integers become
`torch.long`, floats `torch.float`, booleans `torch.bool` — so labels work
directly with `F.cross_entropy`. Override per property with `dtypes`.

---

### `FalkorDBGraphStore`

Implements [`torch_geometric.data.GraphStore`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.GraphStore).

| Method | Description |
|---|---|
| `get_edge_index(edge_type, layout="coo")` | Fetch a COO edge index (lazy, cached) |
| `put_edge_index(edge_index, edge_type, layout, size)` | Store an edge index in the local cache |
| `remove_edge_index(edge_type, layout)` | Evict a cached edge index |
| `coo()` / `csr()` / `csc()` | Layout conversions over all registered edge types |
| `get_all_edge_attrs()` | List all registered `EdgeAttr` objects |
| `clear_cache(edge_type=None)` | Drop cached topology (and, with no argument, the ID mappers) |
| `id_mapper(node_type)` | The `NodeIDMapper` for a node type |
| `dropped_edges` | Per-edge-type count of edges dropped during remapping |

**Constructor:**

```python
FalkorDBGraphStore(
    graph,                     # falkordb.Graph instance
    node_type_to_label=None,   # Optional Dict[str, str]
    edge_type_to_rel=None,     # Optional Dict[Tuple[str,str,str], str]
    read_only=True,            # Use GRAPH.RO_QUERY
)
```

---

### `FalkorDBTensorAttr`

A [`TensorAttr`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.TensorAttr) subclass where `index` defaults to `None` instead of `UNSET`. The feature store installs it as its attribute class, so `store.get_tensor("paper", "x")` works without specifying an index.

```python
from falkordb_pyg import FalkorDBTensorAttr

attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
attr_indexed = FalkorDBTensorAttr(group_name="paper", attr_name="x", index=torch.tensor([0, 1, 2]))
```

---

### `NodeIDMapper`

Bidirectional mapping between FalkorDB internal node IDs and contiguous 0-based PyG indices.

```python
from falkordb_pyg import NodeIDMapper

mapper = NodeIDMapper(falkordb_ids=[100, 200, 300])
mapper.falkor_to_pyg(200)  # -> 1
mapper.pyg_to_falkor(1)    # -> 200
mapper.num_nodes           # -> 3
```

## Node ID Remapping

FalkorDB assigns internal integer IDs to nodes that may not be contiguous or start at zero. `falkordb-pyg` transparently builds a `NodeIDMapper` for each node type on first access, converting FalkorDB IDs to contiguous PyG indices.

To map a prediction back to the database node it came from:

```python
mapper = graph_store.id_mapper("paper")
falkor_id = mapper.pyg_to_falkor(pyg_index)
```

Edges referencing IDs not present in the mapper are dropped, with a warning and
a count in `graph_store.dropped_edges[edge_type]`. This normally means the
relationship also connects labels other than the ones in the edge type.

## Roadmap

| Capability | Status |
|---|---|
| Schema auto-discovery (`CALL db.labels()` etc.) so no priming is needed | Planned |
| Connection parity: username/password, TLS, `from_url` on the factory | Planned |
| Batched partial feature fetch (`WHERE ID(n) IN $ids`) instead of whole columns | Planned |
| Push-down neighbour sampling in Cypher | Planned |
| Writing embeddings back to FalkorDB + vector-index integration | Planned |
| Edge features and edge weights | Planned |

## Comparison with the Kùzu PyG integration

Both projects implement PyG's remote backend interface; they make different
deployment trade-offs. Roadmap items are marked *planned* rather than claimed.

| | Kùzu | falkordb-pyg |
|---|---|---|
| Deployment model | In-process embedded | Client–server (Redis protocol) |
| Multi-host / replicas | ❌ | ✅ (reads use `RO_QUERY`) |
| Auth / TLS from the factory function | n/a | ❌ (pass an existing handle) |
| Heterogeneous graphs | ✅ | ✅ |
| Schema auto-discovery | ✅ | ❌ (planned) |
| Feature fetch granularity | By node ID | Whole column (planned: by ID) |
| Neighbour sampling location | Client-side | Client-side (planned: pushed down) |
| Edge features | ✅ | ❌ (planned) |
| Native vector index | ❌ | ✅ in the DB, not yet wired into this package |
| Ships inside the PyG tree | ✅ | ❌ |

## Examples

See [`examples/train_example.py`](examples/train_example.py) for a complete GraphSAGE training script. Note that it does full-batch training on tensors fetched up front rather than mini-batching through a loader.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/FalkorDB/falkordb-pyg).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests for your changes
4. Run the checks CI runs:
   ```bash
   uv run ruff format --check . && uv run ruff check . && uv run mypy falkordb_pyg/
   uv run --extra test pytest
   ```
5. Submit a pull request

## License

MIT — see [LICENSE](LICENSE).
