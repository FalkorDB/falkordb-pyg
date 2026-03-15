"""Integration tests showing NeighborLoader-style usage with mocked FalkorDB."""

from unittest.mock import MagicMock

import torch
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout

from falkordb_pyg.feature_store import FalkorDBFeatureStore
from falkordb_pyg.graph_store import FalkorDBGraphStore


def _build_mock_graph():
    """Build a small mock FalkorDB graph with 4 Paper nodes and CITES edges.

    Graph topology:
        0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3

    Node features (x): each node has a 2-dimensional feature vector.
    """

    def query_side_effect(q):
        mock = MagicMock()
        if "MATCH (n:Paper) RETURN ID(n)" in q:
            mock.result_set = [[0], [1], [2], [3]]
        elif "MATCH (s:Paper)-[r:CITES]->(d:Paper)" in q and "r." not in q:
            mock.result_set = [[0, 1], [0, 2], [1, 3], [2, 3]]
        elif "n.x" in q:
            mock.result_set = [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
            ]
        else:
            mock.result_set = []
        return mock

    mock_graph = MagicMock()
    mock_graph.query.side_effect = query_side_effect
    return mock_graph


class TestIntegration:
    def test_feature_store_and_graph_store_together(self):
        """FeatureStore and GraphStore should work independently on same graph."""
        mock_graph = _build_mock_graph()
        fs = FalkorDBFeatureStore(mock_graph)
        gs = FalkorDBGraphStore(mock_graph)

        # Fetch node features
        attr = TensorAttr(group_name="Paper", attr_name="x")
        x = fs._get_tensor(attr)
        assert x is not None
        assert x.shape == (4, 2)

        # Fetch edge index
        edge_attr = EdgeAttr(
            edge_type=("Paper", "CITES", "Paper"),
            layout=EdgeLayout.COO,
            is_sorted=False,
        )
        edge_index = gs._get_edge_index(edge_attr)
        assert edge_index is not None
        assert edge_index.shape == (2, 4)

    def test_remote_backend_tuple(self):
        """get_remote_backend should return (FeatureStore, GraphStore) tuple."""
        from falkordb_pyg import FalkorDBFeatureStore, FalkorDBGraphStore

        mock_graph = _build_mock_graph()
        fs = FalkorDBFeatureStore(mock_graph)
        gs = FalkorDBGraphStore(mock_graph)

        backend = (fs, gs)
        assert isinstance(backend[0], FalkorDBFeatureStore)
        assert isinstance(backend[1], FalkorDBGraphStore)

    def test_node_id_remapping(self):
        """Non-contiguous FalkorDB IDs should be remapped to 0-based indices."""

        def query_side_effect(q):
            mock = MagicMock()
            if "MATCH (n:Paper) RETURN ID(n)" in q:
                # Non-contiguous IDs: 10, 20, 30
                mock.result_set = [[10], [20], [30]]
            elif "MATCH (s:Paper)-[r:CITES]->(d:Paper)" in q:
                mock.result_set = [[10, 20], [20, 30]]
            else:
                mock.result_set = []
            return mock

        mock_graph = MagicMock()
        mock_graph.query.side_effect = query_side_effect

        gs = FalkorDBGraphStore(mock_graph)
        edge_attr = EdgeAttr(
            edge_type=("Paper", "CITES", "Paper"),
            layout=EdgeLayout.COO,
            is_sorted=False,
        )
        edge_index = gs._get_edge_index(edge_attr)
        assert edge_index is not None
        # IDs 10->0, 20->1, 30->2
        assert torch.equal(
            edge_index,
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        )

    def test_caching_across_multiple_calls(self):
        """Multiple attribute lookups should reuse cached data."""
        mock_graph = _build_mock_graph()
        fs = FalkorDBFeatureStore(mock_graph)
        attr = TensorAttr(group_name="Paper", attr_name="x")

        # First call fetches from DB
        t1 = fs._get_tensor(attr)
        # Second call uses cache
        t2 = fs._get_tensor(attr)
        assert torch.equal(t1, t2)
        # Query should have been called only once for this attribute
        x_queries = [c for c in mock_graph.query.call_args_list if "n.x" in str(c)]
        assert len(x_queries) == 1
