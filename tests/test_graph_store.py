"""Unit tests for FalkorDBGraphStore."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout

from falkordb_pyg.graph_store import FalkorDBGraphStore


def _make_result(rows):
    """Create a mock FalkorDB query result."""
    result = MagicMock()
    result.result_set = rows
    return result


def _make_graph(query_map):
    """Create a mock FalkorDB graph whose .query() returns results from *query_map*."""
    graph = MagicMock()

    def _query(q):
        for prefix, result in query_map.items():
            if q.startswith(prefix) or prefix in q:
                return result
        raise ValueError(f"Unexpected query: {q}")

    graph.query.side_effect = _query
    return graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def homo_graph():
    """A mock FalkorDB graph with one node type 'paper' and one edge type."""
    # Node IDs for 'paper' nodes (FalkorDB internal IDs 10, 11, 12)
    node_result = _make_result([[10], [11], [12]])
    # Edges: 10->11, 11->12
    edge_result = _make_result([[10, 11], [11, 12]])

    graph = MagicMock()
    def _query(q):
        if "RETURN ID(n)" in q:
            return node_result
        if "RETURN ID(s)" in q:
            return edge_result
        raise ValueError(f"Unexpected query: {q}")
    graph.query.side_effect = _query
    return graph


@pytest.fixture()
def hetero_graph():
    """A mock FalkorDB graph with two node types and two edge types."""
    paper_ids = _make_result([[0], [1], [2]])
    author_ids = _make_result([[10], [11]])
    writes_edges = _make_result([[10, 0], [10, 1], [11, 2]])
    cites_edges = _make_result([[0, 1], [1, 2]])

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
        raise ValueError(f"Unexpected query: {q}")
    graph.query.side_effect = _query
    return graph


# ---------------------------------------------------------------------------
# Tests – put / get / remove edge index
# ---------------------------------------------------------------------------

class TestPutGetRemoveEdgeIndex:
    def test_put_and_get_coo(self, homo_graph):
        store = FalkorDBGraphStore(homo_graph)
        edge_type = ("paper", "cites", "paper")
        src = torch.tensor([0, 1])
        dst = torch.tensor([1, 2])
        attr = EdgeAttr(edge_type=edge_type, layout=EdgeLayout.COO, size=(3, 3))

        assert store._put_edge_index((src, dst), attr) is True
        result = store._get_edge_index(attr)
        assert result is not None
        assert torch.equal(result[0], src)
        assert torch.equal(result[1], dst)

    def test_get_fetches_from_db_when_not_cached(self, homo_graph):
        store = FalkorDBGraphStore(homo_graph)
        edge_type = ("paper", "cites", "paper")
        attr = EdgeAttr(edge_type=edge_type, layout=EdgeLayout.COO)

        result = store._get_edge_index(attr)
        assert result is not None
        # FalkorDB IDs 10->11, 11->12 should remap to PyG indices 0->1, 1->2
        assert torch.equal(result[0], torch.tensor([0, 1]))
        assert torch.equal(result[1], torch.tensor([1, 2]))

    def test_get_returns_none_for_empty_graph(self):
        graph = MagicMock()
        graph.query.return_value = _make_result([])
        store = FalkorDBGraphStore(graph)
        attr = EdgeAttr(edge_type=("A", "rel", "B"), layout=EdgeLayout.COO)
        result = store._get_edge_index(attr)
        assert result is not None
        assert result[0].shape[0] == 0
        assert result[1].shape[0] == 0

    def test_remove_edge_index(self, homo_graph):
        store = FalkorDBGraphStore(homo_graph)
        edge_type = ("paper", "cites", "paper")
        src = torch.tensor([0, 1])
        dst = torch.tensor([1, 2])
        attr = EdgeAttr(edge_type=edge_type, layout=EdgeLayout.COO, size=(3, 3))

        store._put_edge_index((src, dst), attr)
        assert store._remove_edge_index(attr) is True
        # Removing again should return False
        assert store._remove_edge_index(attr) is False

    def test_remove_nonexistent_returns_false(self, homo_graph):
        store = FalkorDBGraphStore(homo_graph)
        attr = EdgeAttr(edge_type=("X", "y", "Z"), layout=EdgeLayout.COO)
        assert store._remove_edge_index(attr) is False

    def test_put_non_coo_raises(self, homo_graph):
        store = FalkorDBGraphStore(homo_graph)
        attr = EdgeAttr(edge_type=("A", "rel", "B"), layout=EdgeLayout.CSC)
        with pytest.raises(NotImplementedError):
            store._put_edge_index((torch.tensor([0]), torch.tensor([1])), attr)


# ---------------------------------------------------------------------------
# Tests – get_all_edge_attrs
# ---------------------------------------------------------------------------

class TestGetAllEdgeAttrs:
    def test_empty_initially(self):
        graph = MagicMock()
        store = FalkorDBGraphStore(graph)
        assert store.get_all_edge_attrs() == []

    def test_returns_registered_attrs(self, homo_graph):
        store = FalkorDBGraphStore(homo_graph)
        edge_type = ("paper", "cites", "paper")
        attr = EdgeAttr(edge_type=edge_type, layout=EdgeLayout.COO, size=(3, 3))
        store._put_edge_index((torch.tensor([0]), torch.tensor([1])), attr)
        attrs = store.get_all_edge_attrs()
        assert len(attrs) == 1
        assert attrs[0].edge_type == edge_type

    def test_auto_registers_on_get(self, homo_graph):
        store = FalkorDBGraphStore(homo_graph)
        edge_type = ("paper", "cites", "paper")
        attr = EdgeAttr(edge_type=edge_type, layout=EdgeLayout.COO)
        store._get_edge_index(attr)
        attrs = store.get_all_edge_attrs()
        assert len(attrs) == 1


# ---------------------------------------------------------------------------
# Tests – heterogeneous graphs
# ---------------------------------------------------------------------------

class TestHeterogeneousGraph:
    def test_hetero_edge_remapping(self, hetero_graph):
        store = FalkorDBGraphStore(hetero_graph)

        writes_attr = EdgeAttr(
            edge_type=("author", "writes", "paper"), layout=EdgeLayout.COO
        )
        result = store._get_edge_index(writes_attr)
        assert result is not None
        # author IDs 10->0, paper IDs 0,1,2->0,1,2
        # writes edges: (10,0),(10,1),(11,2) -> pyg (0,0),(0,1),(1,2)
        assert torch.equal(result[0], torch.tensor([0, 0, 1]))
        assert torch.equal(result[1], torch.tensor([0, 1, 2]))

    def test_multiple_edge_types_cached_separately(self, hetero_graph):
        store = FalkorDBGraphStore(hetero_graph)

        writes_attr = EdgeAttr(
            edge_type=("author", "writes", "paper"), layout=EdgeLayout.COO
        )
        cites_attr = EdgeAttr(
            edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO
        )
        writes = store._get_edge_index(writes_attr)
        cites = store._get_edge_index(cites_attr)
        assert writes[0].shape[0] == 3
        assert cites[0].shape[0] == 2


# ---------------------------------------------------------------------------
# Tests – node ID mapping via custom labels
# ---------------------------------------------------------------------------

class TestNodeTypeLabelMapping:
    def test_custom_label_used_in_query(self):
        graph = MagicMock()
        node_result = _make_result([[5], [6]])
        edge_result = _make_result([[5, 6]])

        def _query(q):
            if "RETURN ID(n)" in q:
                return node_result
            if "RETURN ID(s)" in q:
                return edge_result
            raise ValueError(q)
        graph.query.side_effect = _query

        store = FalkorDBGraphStore(
            graph,
            node_type_to_label={"paper": "Paper"},
            edge_type_to_rel={("paper", "cites", "paper"): "CITES"},
        )
        attr = EdgeAttr(edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO)
        result = store._get_edge_index(attr)
        assert result is not None
        # Verify that the queries used the mapped labels
        calls = [c.args[0] for c in graph.query.call_args_list]
        assert any("Paper" in c for c in calls)
        assert any("CITES" in c for c in calls)


# ---------------------------------------------------------------------------
# Tests – caching behaviour
# ---------------------------------------------------------------------------

class TestCaching:
    def test_db_not_queried_on_second_get(self, homo_graph):
        store = FalkorDBGraphStore(homo_graph)
        attr = EdgeAttr(edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO)

        store._get_edge_index(attr)
        call_count_after_first = homo_graph.query.call_count

        store._get_edge_index(attr)
        assert homo_graph.query.call_count == call_count_after_first

    def test_put_overwrites_cached_value(self, homo_graph):
        store = FalkorDBGraphStore(homo_graph)
        edge_type = ("paper", "cites", "paper")
        attr = EdgeAttr(edge_type=edge_type, layout=EdgeLayout.COO)

        store._get_edge_index(attr)  # populate cache from DB

        new_src = torch.tensor([2])
        new_dst = torch.tensor([0])
        store._put_edge_index((new_src, new_dst), attr)

        result = store._get_edge_index(attr)
        assert torch.equal(result[0], new_src)
        assert torch.equal(result[1], new_dst)
