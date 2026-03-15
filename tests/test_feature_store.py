"""Unit tests for FalkorDBFeatureStore."""

from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data.feature_store import TensorAttr, _FieldStatus

from falkordb_pyg.feature_store import FalkorDBFeatureStore, FalkorDBTensorAttr


def _make_result(rows):
    result = MagicMock()
    result.result_set = rows
    return result


def _make_graph(query_map):
    graph = MagicMock()

    def _query(q):
        for key, result in query_map.items():
            if key in q:
                return result
        raise ValueError(f"Unexpected query: {q}")

    graph.query.side_effect = _query
    return graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def paper_graph():
    """Mock graph with 'paper' nodes, each having a scalar 'y' label and a
    2-D 'x' feature vector."""
    # Rows: [property_value, node_id]
    x_result = _make_result([
        [[1.0, 2.0], 10],
        [[3.0, 4.0], 11],
        [[5.0, 6.0], 12],
    ])
    y_result = _make_result([
        [0, 10],
        [1, 11],
        [2, 12],
    ])

    graph = MagicMock()
    def _query(q):
        if "n.`x`" in q:
            return x_result
        if "n.`y`" in q:
            return y_result
        raise ValueError(f"Unexpected query: {q}")
    graph.query.side_effect = _query
    return graph


# ---------------------------------------------------------------------------
# Tests – put / get / remove tensor
# ---------------------------------------------------------------------------

class TestPutGetRemoveTensor:
    def test_put_and_get_full_tensor(self, paper_graph):
        store = FalkorDBFeatureStore(paper_graph)
        tensor = torch.arange(9, dtype=torch.float).reshape(3, 3)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")

        assert store._put_tensor(tensor, attr) is True
        result = store._get_tensor(attr)
        assert result is not None
        assert torch.equal(result, tensor)

    def test_get_fetches_from_db_when_not_cached(self, paper_graph):
        store = FalkorDBFeatureStore(paper_graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")

        result = store._get_tensor(attr)
        assert result is not None
        assert result.shape == (3, 2)
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))

    def test_get_scalar_property_as_column_tensor(self, paper_graph):
        store = FalkorDBFeatureStore(paper_graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="y")

        result = store._get_tensor(attr)
        assert result is not None
        assert result.shape == (3, 1)

    def test_get_with_index(self, paper_graph):
        store = FalkorDBFeatureStore(paper_graph)
        idx = torch.tensor([0, 2])
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x", index=idx)

        result = store._get_tensor(attr)
        assert result is not None
        assert result.shape == (2, 2)
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))
        assert torch.allclose(result[1], torch.tensor([5.0, 6.0]))

    def test_get_with_none_index_returns_full_tensor(self, paper_graph):
        store = FalkorDBFeatureStore(paper_graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x", index=None)

        result = store._get_tensor(attr)
        assert result is not None
        assert result.shape == (3, 2)

    def test_remove_tensor(self, paper_graph):
        store = FalkorDBFeatureStore(paper_graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")

        store._put_tensor(torch.zeros(3, 2), attr)
        assert store._remove_tensor(attr) is True
        assert store._remove_tensor(attr) is False

    def test_remove_nonexistent_returns_false(self):
        graph = MagicMock()
        store = FalkorDBFeatureStore(graph)
        attr = FalkorDBTensorAttr(group_name="nonexistent", attr_name="feat")
        assert store._remove_tensor(attr) is False

    def test_get_empty_graph(self):
        graph = MagicMock()
        graph.query.return_value = _make_result([])
        store = FalkorDBFeatureStore(graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        result = store._get_tensor(attr)
        assert result is not None
        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# Tests – get_tensor_size
# ---------------------------------------------------------------------------

class TestGetTensorSize:
    def test_returns_correct_shape(self, paper_graph):
        store = FalkorDBFeatureStore(paper_graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        size = store._get_tensor_size(attr)
        assert size == (3, 2)

    def test_put_then_get_size(self):
        graph = MagicMock()
        store = FalkorDBFeatureStore(graph)
        tensor = torch.zeros(5, 16)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        store._put_tensor(tensor, attr)
        assert store._get_tensor_size(attr) == (5, 16)


# ---------------------------------------------------------------------------
# Tests – get_all_tensor_attrs
# ---------------------------------------------------------------------------

class TestGetAllTensorAttrs:
    def test_empty_initially(self):
        graph = MagicMock()
        store = FalkorDBFeatureStore(graph)
        assert store.get_all_tensor_attrs() == []

    def test_returns_registered_attrs(self, paper_graph):
        store = FalkorDBFeatureStore(paper_graph)
        attr_x = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        attr_y = FalkorDBTensorAttr(group_name="paper", attr_name="y")
        store._put_tensor(torch.zeros(3, 2), attr_x)
        store._put_tensor(torch.zeros(3, 1), attr_y)

        attrs = store.get_all_tensor_attrs()
        names = {a.attr_name for a in attrs}
        assert names == {"x", "y"}

    def test_auto_registers_on_get(self, paper_graph):
        store = FalkorDBFeatureStore(paper_graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        store._get_tensor(attr)
        assert len(store.get_all_tensor_attrs()) == 1


# ---------------------------------------------------------------------------
# Tests – FalkorDBTensorAttr defaults
# ---------------------------------------------------------------------------

class TestFalkorDBTensorAttr:
    def test_index_defaults_to_none(self):
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        assert attr.index is None

    def test_group_name_and_attr_name_set(self):
        attr = FalkorDBTensorAttr(group_name="author", attr_name="feat")
        assert attr.group_name == "author"
        assert attr.attr_name == "feat"


# ---------------------------------------------------------------------------
# Tests – caching behaviour
# ---------------------------------------------------------------------------

class TestCaching:
    def test_db_not_queried_on_second_get(self, paper_graph):
        store = FalkorDBFeatureStore(paper_graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")

        store._get_tensor(attr)
        count_after_first = paper_graph.query.call_count

        store._get_tensor(attr)
        assert paper_graph.query.call_count == count_after_first

    def test_put_overwrites_cached_db_value(self, paper_graph):
        store = FalkorDBFeatureStore(paper_graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        store._get_tensor(attr)  # populate from DB

        new_tensor = torch.ones(3, 4)
        store._put_tensor(new_tensor, attr)
        result = store._get_tensor(attr)
        assert torch.equal(result, new_tensor)


# ---------------------------------------------------------------------------
# Tests – node type label mapping
# ---------------------------------------------------------------------------

class TestNodeTypeLabelMapping:
    def test_custom_label_used_in_query(self):
        x_result = _make_result([[[1.0, 2.0], 0]])
        graph = MagicMock()
        def _query(q):
            if "n.`x`" in q:
                return x_result
            raise ValueError(q)
        graph.query.side_effect = _query

        store = FalkorDBFeatureStore(
            graph, node_type_to_label={"paper": "Paper"}
        )
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        store._get_tensor(attr)

        calls = [c.args[0] for c in graph.query.call_args_list]
        assert any("Paper" in c for c in calls)
