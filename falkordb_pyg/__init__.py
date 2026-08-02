"""FalkorDB PyG remote backend package.

Exports the main classes and the convenience factory function
:func:`get_remote_backend`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from falkordb import FalkorDB

try:
    from .feature_store import FalkorDBFeatureStore, FalkorDBTensorAttr
    from .graph_store import FalkorDBGraphStore
except ImportError as exc:  # pragma: no cover - depends on the install extras
    # Only rewrite a genuinely missing optional dependency. Rewriting every
    # ImportError would bury real bugs inside the stores, or a version mismatch
    # inside torch_geometric, under a misleading "install torch" message.
    if (exc.name or "").split(".")[0] not in {"torch", "torch_geometric"}:
        raise
    raise ImportError(
        "falkordb-pyg requires PyTorch and PyTorch Geometric, which are not "
        "installed. Install them for your platform following "
        "https://pytorch.org/get-started/locally/ and "
        "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html, "
        "or install this package with its optional extra: "
        "pip install 'falkordb-pyg[torch]'"
    ) from exc

from .utils import NodeIDMapper

if TYPE_CHECKING:
    import torch

try:
    __version__ = _version("falkordb-pyg")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "FalkorDBFeatureStore",
    "FalkorDBGraphStore",
    "FalkorDBTensorAttr",
    "NodeIDMapper",
    "__version__",
    "get_remote_backend",
]


def get_remote_backend(
    host: str = "localhost",
    port: int = 6379,
    graph_name: str = "default",
    node_type_to_label: Optional[Dict[str, str]] = None,
    edge_type_to_rel: Optional[Dict[Tuple[str, str, str], str]] = None,
    dtypes: Optional[Dict[Tuple[str, str], "torch.dtype"]] = None,
    read_only: bool = True,
    graph=None,
) -> Tuple[FalkorDBFeatureStore, FalkorDBGraphStore]:
    """Create a FalkorDB-backed PyG Remote Backend.

    Connects to a running FalkorDB instance and returns a
    ``(FeatureStore, GraphStore)`` tuple that can be passed to
    :class:`~torch_geometric.loader.NeighborLoader` and other PyG loaders.

    .. note::

        Both stores currently register a node type / edge type only once it
        has been accessed.  Until schema auto-discovery lands, prime the types
        you intend to sample before constructing a loader — see the README's
        Quick Start.

    Args:
        host: Hostname of the FalkorDB / Redis server.
        port: Port of the FalkorDB / Redis server.
        graph_name: Name of the graph to use in FalkorDB.
        node_type_to_label: Optional mapping from PyG node type strings to
            FalkorDB node labels.  Pass this when your PyG node types differ
            from the labels stored in FalkorDB.
        edge_type_to_rel: Optional mapping from PyG edge type triples
            ``(src_type, rel_type, dst_type)`` to FalkorDB relationship type
            strings.  Defaults to using the middle element of the triple.
        dtypes: Optional mapping from ``(node_type, property)`` to an explicit
            :class:`torch.dtype`, overriding inference.  Keyed by the PyG node
            type, not by the FalkorDB label it maps to.
        read_only: Issue reads with ``GRAPH.RO_QUERY`` so they can be served
            by a replica.  Set to ``False`` for servers without RO_QUERY.
        graph: An existing ``falkordb.Graph`` handle to use instead of opening
            a new connection.  When given, ``host``/``port``/``graph_name`` are
            ignored.

    Returns:
        A ``(FalkorDBFeatureStore, FalkorDBGraphStore)`` tuple.

    Example::

        from falkordb_pyg import get_remote_backend
        from torch_geometric.loader import NeighborLoader

        feature_store, graph_store = get_remote_backend(
            host="localhost",
            port=6379,
            graph_name="papers",
        )

        # Prime the types the loader will sample:
        feature_store.get_tensor_size("paper", "x")
        graph_store.get_edge_index(("paper", "cites", "paper"), layout="coo")

        loader = NeighborLoader(
            data=(feature_store, graph_store),
            num_neighbors={("paper", "cites", "paper"): [10, 10]},
            batch_size=1024,
            input_nodes=("paper", train_nodes),
        )
    """
    if graph is None:
        db = FalkorDB(host=host, port=port)
        graph = db.select_graph(graph_name)

    feature_store = FalkorDBFeatureStore(
        graph=graph,
        node_type_to_label=node_type_to_label,
        dtypes=dtypes,
        read_only=read_only,
    )
    graph_store = FalkorDBGraphStore(
        graph=graph,
        node_type_to_label=node_type_to_label,
        edge_type_to_rel=edge_type_to_rel,
        read_only=read_only,
    )
    return feature_store, graph_store
