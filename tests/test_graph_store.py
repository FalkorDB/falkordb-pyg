"""Unit tests for FalkorDBGraphStore."""

from unittest.mock import MagicMock

import torch
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout

from falkordb_pyg.graph_store import FalkorDBGraphStore


def _make_mock_graph(result_rows):
    """Create a mock FalkorDB graph object returning the given result rows."""
    mock_result = MagicMock()
    mock_result.result_set = result_rows
    mock_graph = MagicMock()
    mock_graph.query.return_value = mock_result
    return mock_graph


def _make_edge_attr(src, rel, dst, size=None):
    return EdgeAttr(
        edge_type=(src, rel, dst),
        layout=EdgeLayout.COO,
        is_sorted=False,
        size=size,
    )


class TestFalkorDBGraphStore:
    def test_get_edge_index_basic(self):
        """Should return correct COO edge index after remapping IDs."""

        # Node ID queries: Paper -> [10, 20], Paper -> [10, 20]
        # Edge query: (10->20), (20->10)
        def query_side_effect(q):
            mock = MagicMock()
            if "MATCH (n:Paper)" in q:
                mock.result_set = [[10], [20]]
            else:
                mock.result_set = [[10, 20], [20, 10]]
            return mock

        mock_graph = MagicMock()
        mock_graph.query.side_effect = query_side_effect

        store = FalkorDBGraphStore(mock_graph)
        attr = _make_edge_attr("Paper", "CITES", "Paper")
        edge_index = store._get_edge_index(attr)

        assert edge_index is not None
        assert edge_index.shape[0] == 2
        assert edge_index.dtype == torch.long
        # FalkorDB IDs 10->20 map to PyG 0->1
        assert edge_index[0, 0].item() == 0
        assert edge_index[1, 0].item() == 1

    def test_get_edge_index_cached(self):
        """Second call should return cached result without extra DB queries."""
        call_count = {"n": 0}

        def query_side_effect(q):
            call_count["n"] += 1
            mock = MagicMock()
            if "MATCH (n:" in q:
                mock.result_set = [[0], [1]]
            else:
                mock.result_set = [[0, 1]]
            return mock

        mock_graph = MagicMock()
        mock_graph.query.side_effect = query_side_effect

        store = FalkorDBGraphStore(mock_graph)
        attr = _make_edge_attr("A", "REL", "B")
        store._get_edge_index(attr)
        first_count = call_count["n"]
        store._get_edge_index(attr)
        # No additional queries on second call
        assert call_count["n"] == first_count

    def test_put_and_get_edge_index(self):
        """Manually put edge index should be retrievable."""
        mock_graph = MagicMock()
        store = FalkorDBGraphStore(mock_graph)
        attr = _make_edge_attr("A", "R", "B", size=(3, 3))
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        store._put_edge_index(edge_index, attr)
        retrieved = store._get_edge_index(attr)
        assert retrieved is not None
        assert torch.equal(retrieved, edge_index)

    def test_remove_edge_index(self):
        """Removing a cached edge index should return True and clear cache."""
        mock_graph = MagicMock()
        store = FalkorDBGraphStore(mock_graph)
        attr = _make_edge_attr("A", "R", "B")
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        store._put_edge_index(edge_index, attr)
        assert store._remove_edge_index(attr) is True
        assert store._remove_edge_index(attr) is False

    def test_get_all_edge_attrs(self):
        """get_all_edge_attrs should reflect cached edges."""
        mock_graph = MagicMock()
        store = FalkorDBGraphStore(mock_graph)
        attr = _make_edge_attr("A", "R", "B", size=(2, 2))
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        store._put_edge_index(edge_index, attr)
        all_attrs = store.get_all_edge_attrs()
        assert len(all_attrs) == 1
        assert all_attrs[0].edge_type == ("A", "R", "B")

    def test_get_edge_index_returns_none_on_error(self):
        """Should return None when FalkorDB query raises an exception."""
        mock_graph = MagicMock()
        mock_graph.query.side_effect = RuntimeError("connection error")
        store = FalkorDBGraphStore(mock_graph)
        attr = _make_edge_attr("A", "R", "B")
        result = store._get_edge_index(attr)
        assert result is None

    def test_empty_edge_result(self):
        """Empty graph (no edges) should yield an empty edge index."""

        def query_side_effect(q):
            mock = MagicMock()
            if "MATCH (n:" in q:
                mock.result_set = [[0], [1]]
            else:
                mock.result_set = []
            return mock

        mock_graph = MagicMock()
        mock_graph.query.side_effect = query_side_effect

        store = FalkorDBGraphStore(mock_graph)
        attr = _make_edge_attr("A", "REL", "B")
        edge_index = store._get_edge_index(attr)
        assert edge_index is not None
        assert edge_index.shape == (2, 0)
