"""Unit tests for FalkorDBFeatureStore."""

from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data.feature_store import TensorAttr

from falkordb_pyg.feature_store import FalkorDBFeatureStore


def _make_mock_graph(result_rows):
    """Create a mock FalkorDB graph returning given result rows."""
    mock_result = MagicMock()
    mock_result.result_set = result_rows
    mock_graph = MagicMock()
    mock_graph.query.return_value = mock_result
    return mock_graph


class TestFalkorDBFeatureStore:
    def test_get_tensor_scalar_feature(self):
        """Single scalar feature per node should produce shape (N, 1)."""
        mock_graph = _make_mock_graph([[1.0], [2.0], [3.0]])
        store = FalkorDBFeatureStore(mock_graph)
        attr = TensorAttr(group_name="Paper", attr_name="year")
        tensor = store._get_tensor(attr)
        assert tensor is not None
        assert tensor.shape == (3, 1)
        assert tensor[0, 0].item() == pytest.approx(1.0)

    def test_get_tensor_vector_feature(self):
        """Vector features per node should produce shape (N, D)."""
        mock_graph = _make_mock_graph([[[1.0, 2.0]], [[3.0, 4.0]]])
        store = FalkorDBFeatureStore(mock_graph)
        attr = TensorAttr(group_name="Paper", attr_name="x")
        tensor = store._get_tensor(attr)
        assert tensor is not None
        assert tensor.shape == (2, 2)

    def test_get_tensor_with_index(self):
        """Index attribute should slice the full tensor."""
        mock_graph = _make_mock_graph([[1.0], [2.0], [3.0], [4.0]])
        store = FalkorDBFeatureStore(mock_graph)
        index = torch.tensor([0, 2])
        attr = TensorAttr(group_name="Paper", attr_name="year", index=index)
        tensor = store._get_tensor(attr)
        assert tensor is not None
        assert tensor.shape[0] == 2
        assert tensor[0, 0].item() == pytest.approx(1.0)
        assert tensor[1, 0].item() == pytest.approx(3.0)

    def test_get_tensor_cached(self):
        """Second call for same attr should not issue another DB query."""
        mock_graph = _make_mock_graph([[1.0], [2.0]])
        store = FalkorDBFeatureStore(mock_graph)
        attr = TensorAttr(group_name="Paper", attr_name="year")
        store._get_tensor(attr)
        store._get_tensor(attr)
        assert mock_graph.query.call_count == 1

    def test_put_tensor(self):
        """Manually put tensor should be retrievable via _get_tensor."""
        mock_graph = MagicMock()
        store = FalkorDBFeatureStore(mock_graph)
        tensor = torch.tensor([[1.0], [2.0]])
        attr = TensorAttr(group_name="Paper", attr_name="x")
        store._put_tensor(tensor, attr)
        retrieved = store._get_tensor(attr)
        assert retrieved is not None
        assert torch.equal(retrieved, tensor)
        # No DB query should have been made
        mock_graph.query.assert_not_called()

    def test_remove_tensor(self):
        """Removing a tensor should clear the cache entry."""
        mock_graph = MagicMock()
        store = FalkorDBFeatureStore(mock_graph)
        tensor = torch.tensor([[1.0]])
        attr = TensorAttr(group_name="Paper", attr_name="x")
        store._put_tensor(tensor, attr)
        assert store._remove_tensor(attr) is True
        assert store._remove_tensor(attr) is False

    def test_get_tensor_size(self):
        """_get_tensor_size should return the shape tuple."""
        mock_graph = _make_mock_graph([[1.0], [2.0], [3.0]])
        store = FalkorDBFeatureStore(mock_graph)
        attr = TensorAttr(group_name="Paper", attr_name="year")
        size = store._get_tensor_size(attr)
        assert size == (3, 1)

    def test_get_tensor_empty_result(self):
        """Empty query result should return None."""
        mock_graph = _make_mock_graph([])
        store = FalkorDBFeatureStore(mock_graph)
        attr = TensorAttr(group_name="Paper", attr_name="missing")
        tensor = store._get_tensor(attr)
        assert tensor is None

    def test_get_all_tensor_attrs(self):
        """get_all_tensor_attrs should list all cached attributes."""
        mock_graph = MagicMock()
        store = FalkorDBFeatureStore(mock_graph)
        t1 = torch.tensor([[1.0]])
        t2 = torch.tensor([[2.0]])
        store._put_tensor(t1, TensorAttr(group_name="Paper", attr_name="x"))
        store._put_tensor(t2, TensorAttr(group_name="Author", attr_name="y"))
        attrs = store.get_all_tensor_attrs()
        assert len(attrs) == 2
        names = {(a.group_name, a.attr_name) for a in attrs}
        assert ("Paper", "x") in names
        assert ("Author", "y") in names

    def test_edge_feature_query(self):
        """Edge features (tuple group_name) should use relationship query."""
        mock_graph = _make_mock_graph([[0.5], [1.5]])
        store = FalkorDBFeatureStore(mock_graph)
        attr = TensorAttr(group_name=("Paper", "CITES", "Paper"), attr_name="weight")
        tensor = store._get_tensor(attr)
        assert tensor is not None
        assert tensor.shape == (2, 1)
        # Verify the query used relationship pattern
        call_args = mock_graph.query.call_args[0][0]
        assert "MATCH (s:Paper)-[r:CITES]->(d:Paper)" in call_args
