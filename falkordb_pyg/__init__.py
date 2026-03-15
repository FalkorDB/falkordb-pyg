"""falkordb-pyg: PyTorch Geometric remote backend for FalkorDB."""

from falkordb_pyg.feature_store import FalkorDBFeatureStore
from falkordb_pyg.graph_store import FalkorDBGraphStore

__all__ = [
    "FalkorDBFeatureStore",
    "FalkorDBGraphStore",
    "get_remote_backend",
]


def get_remote_backend(
    host: str = "localhost",
    port: int = 6379,
    graph_name: str = "default",
) -> tuple:
    """Create a FalkorDB remote backend for PyG.

    Args:
        host: FalkorDB host address.
        port: FalkorDB port number.
        graph_name: Name of the graph in FalkorDB.

    Returns:
        A tuple of (FalkorDBFeatureStore, FalkorDBGraphStore) that can be
        passed directly to PyG loaders such as NeighborLoader.

    """
    from falkordb import FalkorDB

    db = FalkorDB(host=host, port=port)
    graph = db.select_graph(graph_name)
    feature_store = FalkorDBFeatureStore(graph)
    graph_store = FalkorDBGraphStore(graph)
    return feature_store, graph_store
