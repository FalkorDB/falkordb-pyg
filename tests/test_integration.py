"""Integration tests that exercise the full Remote Backend stack with mocked
FalkorDB calls, verifying that the NeighborLoader-compatible interface works
end-to-end.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout

from falkordb_pyg import get_remote_backend
from falkordb_pyg.feature_store import FalkorDBFeatureStore, FalkorDBTensorAttr
from falkordb_pyg.graph_store import FalkorDBGraphStore


def _make_result(rows):
    result = MagicMock()
    result.result_set = rows
    return result


# ---------------------------------------------------------------------------
# Helper: build stores from a shared mock graph
# ---------------------------------------------------------------------------

def _build_stores(graph):
    feature_store = FalkorDBFeatureStore(graph)
    graph_store = FalkorDBGraphStore(graph)
    return feature_store, graph_store


# ---------------------------------------------------------------------------
# Tests – factory function
# ---------------------------------------------------------------------------

class TestGetRemoteBackend:
    def test_returns_tuple_of_correct_types(self):
        with patch("falkordb_pyg.FalkorDB") as mock_falkordb:
            mock_db = MagicMock()
            mock_graph = MagicMock()
            mock_falkordb.return_value = mock_db
            mock_db.select_graph.return_value = mock_graph

            feature_store, graph_store = get_remote_backend(
                host="localhost", port=6379, graph_name="test"
            )

        assert isinstance(feature_store, FalkorDBFeatureStore)
        assert isinstance(graph_store, FalkorDBGraphStore)

    def test_custom_mappings_forwarded(self):
        with patch("falkordb_pyg.FalkorDB") as mock_falkordb:
            mock_db = MagicMock()
            mock_graph = MagicMock()
            mock_falkordb.return_value = mock_db
            mock_db.select_graph.return_value = mock_graph

            feature_store, graph_store = get_remote_backend(
                host="host",
                port=1234,
                graph_name="g",
                node_type_to_label={"paper": "Paper"},
                edge_type_to_rel={("paper", "cites", "paper"): "CITES"},
            )

        assert graph_store._node_type_to_label == {"paper": "Paper"}
        assert graph_store._edge_type_to_rel == {
            ("paper", "cites", "paper"): "CITES"
        }
        assert feature_store._node_type_to_label == {"paper": "Paper"}


# ---------------------------------------------------------------------------
# Tests – homogeneous backend end-to-end
# ---------------------------------------------------------------------------

class TestHomogeneousBackend:
    @pytest.fixture()
    def stores(self):
        # Paper graph: 3 nodes with IDs 0,1,2; edges 0->1, 1->2
        node_result = _make_result([[0], [1], [2]])
        edge_result = _make_result([[0, 1], [1, 2]])
        x_result = _make_result([
            [[1.0, 0.0], 0],
            [[0.0, 1.0], 1],
            [[1.0, 1.0], 2],
        ])
        y_result = _make_result([[0, 0], [1, 1], [0, 2]])

        graph = MagicMock()
        def _query(q):
            if "RETURN ID(n)" in q:
                return node_result
            if "RETURN ID(s)" in q:
                return edge_result
            if "n.`x`" in q:
                return x_result
            if "n.`y`" in q:
                return y_result
            raise ValueError(f"Unexpected query: {q}")
        graph.query.side_effect = _query
        return _build_stores(graph)

    def test_feature_store_get_x(self, stores):
        feature_store, _ = stores
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        x = feature_store._get_tensor(attr)
        assert x.shape == (3, 2)

    def test_graph_store_get_edge_index(self, stores):
        _, graph_store = stores
        attr = EdgeAttr(
            edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO
        )
        ei = graph_store._get_edge_index(attr)
        assert ei is not None
        assert torch.equal(ei[0], torch.tensor([0, 1]))
        assert torch.equal(ei[1], torch.tensor([1, 2]))

    def test_edge_attr_has_correct_size(self, stores):
        _, graph_store = stores
        attr = EdgeAttr(
            edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO
        )
        graph_store._get_edge_index(attr)
        attrs = graph_store.get_all_edge_attrs()
        assert len(attrs) == 1
        assert attrs[0].size == (3, 3)


# ---------------------------------------------------------------------------
# Tests – heterogeneous backend end-to-end
# ---------------------------------------------------------------------------

class TestHeterogeneousBackend:
    @pytest.fixture()
    def stores(self):
        paper_ids = _make_result([[0], [1], [2]])
        author_ids = _make_result([[10], [11]])
        writes_edges = _make_result([[10, 0], [10, 1], [11, 2]])
        cites_edges = _make_result([[0, 1], [1, 2]])
        author_x = _make_result([[[0.5, 0.5], 10], [[0.1, 0.9], 11]])
        paper_x = _make_result([[[1.0, 0.0], 0], [[0.0, 1.0], 1], [[0.5, 0.5], 2]])

        graph = MagicMock()
        def _query(q):
            if "`paper`" in q and "RETURN ID(n)" in q:
                return paper_ids
            if "`author`" in q and "RETURN ID(n)" in q:
                return author_ids
            if "`writes`" in q:
                return writes_edges
            if "`cites`" in q:
                return cites_edges
            if "`author`" in q and "n.`x`" in q:
                return author_x
            if "`paper`" in q and "n.`x`" in q:
                return paper_x
            raise ValueError(f"Unexpected query: {q}")
        graph.query.side_effect = _query
        return _build_stores(graph)

    def test_paper_features(self, stores):
        feature_store, _ = stores
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        x = feature_store._get_tensor(attr)
        assert x.shape == (3, 2)

    def test_author_features(self, stores):
        feature_store, _ = stores
        attr = FalkorDBTensorAttr(group_name="author", attr_name="x")
        x = feature_store._get_tensor(attr)
        assert x.shape == (2, 2)

    def test_writes_edges(self, stores):
        _, graph_store = stores
        attr = EdgeAttr(
            edge_type=("author", "writes", "paper"), layout=EdgeLayout.COO
        )
        ei = graph_store._get_edge_index(attr)
        assert ei is not None
        assert ei[0].shape[0] == 3

    def test_cites_edges(self, stores):
        _, graph_store = stores
        attr = EdgeAttr(
            edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO
        )
        ei = graph_store._get_edge_index(attr)
        assert ei is not None
        assert torch.equal(ei[0], torch.tensor([0, 1]))
        assert torch.equal(ei[1], torch.tensor([1, 2]))

    def test_indexed_feature_access(self, stores):
        feature_store, _ = stores
        idx = torch.tensor([0, 2])
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x", index=idx)
        x = feature_store._get_tensor(attr)
        assert x.shape == (2, 2)


# ---------------------------------------------------------------------------
# Tests – NodeIDMapper integration
# ---------------------------------------------------------------------------

class TestNodeIDMapperIntegration:
    def test_non_contiguous_ids_remapped(self):
        """FalkorDB IDs 100, 200, 300 should map to PyG indices 0, 1, 2."""
        node_result = _make_result([[100], [200], [300]])
        edge_result = _make_result([[100, 200], [200, 300], [100, 300]])

        graph = MagicMock()
        def _query(q):
            if "RETURN ID(n)" in q:
                return node_result
            if "RETURN ID(s)" in q:
                return edge_result
            raise ValueError(q)
        graph.query.side_effect = _query

        _, graph_store = _build_stores(graph)
        attr = EdgeAttr(
            edge_type=("node", "rel", "node"), layout=EdgeLayout.COO
        )
        ei = graph_store._get_edge_index(attr)
        assert torch.equal(ei[0], torch.tensor([0, 1, 0]))
        assert torch.equal(ei[1], torch.tensor([1, 2, 2]))

    def test_unknown_ids_dropped(self):
        """Edges referencing IDs not in the node list are silently dropped."""
        node_result = _make_result([[1], [2]])
        # Edge 3->4 references IDs not in the node set
        edge_result = _make_result([[1, 2], [3, 4]])

        graph = MagicMock()
        def _query(q):
            if "RETURN ID(n)" in q:
                return node_result
            if "RETURN ID(s)" in q:
                return edge_result
            raise ValueError(q)
        graph.query.side_effect = _query

        _, graph_store = _build_stores(graph)
        attr = EdgeAttr(
            edge_type=("node", "rel", "node"), layout=EdgeLayout.COO
        )
        ei = graph_store._get_edge_index(attr)
        # Only the first edge (1->2) should survive
        assert ei[0].shape[0] == 1
        assert torch.equal(ei[0], torch.tensor([0]))
        assert torch.equal(ei[1], torch.tensor([1]))
