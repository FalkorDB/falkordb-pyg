"""Unit tests for FalkorDBFeatureStore.

These drive the PUBLIC PyG FeatureStore API wherever possible — the private
``_``-prefixed methods are ABC hooks, and testing only those is what allowed
the public surface to ship broken.
"""

import pytest
import torch

from falkordb_pyg.feature_store import FalkorDBFeatureStore, FalkorDBTensorAttr

from .conftest import FakeFalkorGraph


@pytest.fixture()
def store(homo_graph):
    return FalkorDBFeatureStore(homo_graph)


# ---------------------------------------------------------------------------
# Tests – public PyG API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """The API a PyG user actually calls. Every one of these failed in 0.2.1."""

    def test_tensor_attr_cls_is_registered(self, store):
        assert store._tensor_attr_cls is FalkorDBTensorAttr

    def test_get_tensor(self, store):
        assert store.get_tensor("paper", "x").shape == (3, 2)

    def test_getitem(self, store):
        assert store["paper", "x"].shape == (3, 2)

    def test_getitem_with_index(self, store):
        out = store["paper", "x", torch.tensor([0, 2])]
        assert out.shape == (2, 2)
        assert torch.allclose(out[1], torch.tensor([5.0, 6.0]))

    def test_get_tensor_size(self, store):
        assert store.get_tensor_size("paper", "x") == (3, 2)

    def test_put_tensor(self, store):
        tensor = torch.ones(3, 4)
        assert store.put_tensor(tensor, "paper", "z") is True
        assert torch.equal(store.get_tensor("paper", "z"), tensor)

    def test_setitem(self, store):
        store["paper", "z"] = torch.ones(3, 4)
        assert torch.equal(store["paper", "z"], torch.ones(3, 4))

    def test_view(self, store):
        assert store.view("paper").x.shape == (3, 2)

    def test_multi_get_tensor(self, store):
        x, y = store.multi_get_tensor(
            [
                FalkorDBTensorAttr("paper", "x"),
                FalkorDBTensorAttr("paper", "y"),
            ]
        )
        assert x.shape == (3, 2)
        assert y.shape == (3, 1)

    def test_remove_tensor(self, store):
        store.put_tensor(torch.zeros(3, 2), "paper", "z")
        assert store.remove_tensor("paper", "z") is True


# ---------------------------------------------------------------------------
# Tests – fetching and indexing
# ---------------------------------------------------------------------------


class TestFetch:
    def test_vector_feature(self, store):
        out = store.get_tensor("paper", "x")
        assert torch.allclose(out[0], torch.tensor([1.0, 2.0]))
        assert out.dtype == torch.float

    def test_scalar_feature_is_column(self, store):
        assert store.get_tensor("paper", "y").shape == (3, 1)

    def test_index_selects_rows(self, store):
        full = store.get_tensor("paper", "x")
        idx = torch.tensor([0, 2])
        assert torch.equal(store.get_tensor("paper", "x", index=idx), full[idx])

    def test_rows_are_ordered_by_falkordb_id(self, homo_graph):
        """Row i must correspond to the i-th node in ID order.

        This is the invariant that keeps features aligned with the topology
        the GraphStore builds; nothing else in the codebase enforces it.
        """
        store = FalkorDBFeatureStore(homo_graph)
        y = store.get_tensor("paper", "y").squeeze(1)
        ordered_ids = sorted(homo_graph.nodes["paper"])
        expected = [homo_graph.nodes["paper"][nid]["y"] for nid in ordered_ids]
        assert y.tolist() == expected

    def test_empty_label_returns_rank_two_tensor(self):
        store = FalkorDBFeatureStore(FakeFalkorGraph(nodes={"ghost": {}}))
        with pytest.warns(UserWarning, match="No nodes matched"):
            out = store.get_tensor("ghost", "x")
        assert out.shape == (0, 0)


# ---------------------------------------------------------------------------
# Tests – dtype inference
# ---------------------------------------------------------------------------


class TestDtype:
    def test_integer_scalars_stay_integers(self, store):
        """Labels must not be silently floated: cross_entropy needs int64."""
        assert store.get_tensor("paper", "y").dtype == torch.long

    def test_float_scalars_are_float(self):
        graph = FakeFalkorGraph(nodes={"paper": {0: {"t": 1.5}, 1: {"t": 2.5}}})
        assert FalkorDBFeatureStore(graph).get_tensor("paper", "t").dtype == torch.float

    def test_bool_scalars_are_bool(self):
        graph = FakeFalkorGraph(nodes={"paper": {0: {"m": True}, 1: {"m": False}}})
        assert FalkorDBFeatureStore(graph).get_tensor("paper", "m").dtype == torch.bool

    def test_integer_vectors_are_float(self):
        """Vectors are model inputs; conv layers want float."""
        graph = FakeFalkorGraph(nodes={"paper": {0: {"x": [1, 2]}, 1: {"x": [3, 4]}}})
        assert FalkorDBFeatureStore(graph).get_tensor("paper", "x").dtype == torch.float

    def test_explicit_override(self, homo_graph):
        store = FalkorDBFeatureStore(homo_graph, dtypes={("paper", "y"): torch.float})
        assert store.get_tensor("paper", "y").dtype == torch.float

    def test_override_is_keyed_by_pyg_group_name_not_falkordb_label(self):
        """dtypes must key off the name the caller uses, not the mapped label."""
        graph = FakeFalkorGraph(nodes={"Paper": {0: {"y": 1}, 1: {"y": 2}}})
        store = FalkorDBFeatureStore(
            graph,
            node_type_to_label={"paper": "Paper"},
            dtypes={("paper", "y"): torch.float},
        )
        assert store.get_tensor("paper", "y").dtype == torch.float

    def test_vector_override_is_keyed_by_group_name_too(self):
        graph = FakeFalkorGraph(nodes={"Paper": {0: {"x": [1.0, 2.0]}}})
        store = FalkorDBFeatureStore(
            graph,
            node_type_to_label={"paper": "Paper"},
            dtypes={("paper", "x"): torch.float64},
        )
        assert store.get_tensor("paper", "x").dtype == torch.float64


# ---------------------------------------------------------------------------
# Tests – malformed data produces actionable errors
# ---------------------------------------------------------------------------


class TestDataHazards:
    def test_missing_property_names_the_node(self):
        graph = FakeFalkorGraph(
            nodes={"paper": {7: {"x": [1.0, 2.0]}, 9: {}, 11: {"x": [5.0, 6.0]}}}
        )
        store = FalkorDBFeatureStore(graph)
        with pytest.raises(ValueError, match=r"ID\(n\)=9"):
            store.get_tensor("paper", "x")

    def test_ragged_vectors_name_the_node_and_lengths(self):
        graph = FakeFalkorGraph(
            nodes={"paper": {0: {"x": [1.0, 2.0]}, 3: {"x": [3.0]}}}
        )
        store = FalkorDBFeatureStore(graph)
        with pytest.raises(ValueError, match=r"inconsistent length.*ID\(n\)=3"):
            store.get_tensor("paper", "x")

    def test_string_property_is_rejected(self):
        graph = FakeFalkorGraph(nodes={"paper": {0: {"t": "hello"}, 1: {"t": "world"}}})
        store = FalkorDBFeatureStore(graph)
        with pytest.raises(ValueError, match="non-numeric type str"):
            store.get_tensor("paper", "t")

    def test_mixed_vector_then_scalar_is_rejected(self):
        graph = FakeFalkorGraph(nodes={"paper": {0: {"x": [1.0]}, 1: {"x": 2.0}}})
        store = FalkorDBFeatureStore(graph)
        with pytest.raises(ValueError, match="scalar on"):
            store.get_tensor("paper", "x")

    def test_mixed_scalar_then_vector_is_rejected(self):
        """The reverse order takes the scalar branch and must fail too."""
        graph = FakeFalkorGraph(nodes={"paper": {0: {"x": 1.0}, 1: {"x": [2.0]}}})
        store = FalkorDBFeatureStore(graph)
        with pytest.raises(ValueError, match=r"vector at ID\(n\)=1"):
            store.get_tensor("paper", "x")

    def test_non_numeric_element_inside_a_vector_is_rejected(self):
        graph = FakeFalkorGraph(
            nodes={"paper": {0: {"x": [1.0, "two"]}, 3: {"x": [3.0, 4.0]}}}
        )
        store = FalkorDBFeatureStore(graph)
        with pytest.raises(ValueError, match=r"non-numeric element at ID\(n\)=0"):
            store.get_tensor("paper", "x")

    def test_nested_vectors_are_rejected(self):
        graph = FakeFalkorGraph(
            nodes={"paper": {0: {"x": [[1.0], [2.0]]}, 1: {"x": [[3.0], [4.0]]}}}
        )
        store = FalkorDBFeatureStore(graph)
        with pytest.raises(ValueError, match="nested more than one level"):
            store.get_tensor("paper", "x")

    def test_edge_features_are_rejected_not_silently_wrong(self, store):
        """0.2.1 answered this with source-node features and no error."""
        with pytest.raises(NotImplementedError, match="does not support edge features"):
            store.get_tensor(("paper", "cites", "paper"), "x")


# ---------------------------------------------------------------------------
# Tests – caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_db_not_queried_on_second_get(self, store, homo_graph):
        store.get_tensor("paper", "x")
        count = len(homo_graph.calls)
        store.get_tensor("paper", "x")
        assert len(homo_graph.calls) == count

    def test_mutating_a_returned_tensor_does_not_poison_the_cache(self, store):
        first = store.get_tensor("paper", "x")
        first[0, 0] = 999.0
        assert store.get_tensor("paper", "x")[0, 0].item() == 1.0

    def test_mutating_a_put_tensor_does_not_poison_the_cache(self, store):
        tensor = torch.ones(3, 2)
        store.put_tensor(tensor, "paper", "z")
        tensor[0, 0] = 999.0
        assert store.get_tensor("paper", "z")[0, 0].item() == 1.0

    def test_clear_cache_forces_refetch(self, store, homo_graph):
        store.get_tensor("paper", "x")
        count = len(homo_graph.calls)
        store.clear_cache()
        store.get_tensor("paper", "x")
        assert len(homo_graph.calls) > count

    def test_clear_cache_by_group_name(self, store, homo_graph):
        store.get_tensor("paper", "x")
        calls = len(homo_graph.calls)
        store.clear_cache(group_name="other")  # no such group: nothing evicted
        store.get_tensor("paper", "x")
        assert len(homo_graph.calls) == calls
        store.clear_cache(group_name="paper")
        store.get_tensor("paper", "x")
        assert len(homo_graph.calls) > calls

    def test_clear_cache_is_selective(self, store, homo_graph):
        store.get_tensor("paper", "x")
        store.get_tensor("paper", "y")
        store.clear_cache(attr_name="x")
        count = len(homo_graph.calls)
        store.get_tensor("paper", "y")
        assert len(homo_graph.calls) == count  # 'y' still cached

    def test_put_overwrites_cached_db_value(self, store):
        store.get_tensor("paper", "x")
        store.put_tensor(torch.ones(3, 4), "paper", "x")
        assert torch.equal(store.get_tensor("paper", "x"), torch.ones(3, 4))


# ---------------------------------------------------------------------------
# Tests – attribute registry
# ---------------------------------------------------------------------------


class TestTensorAttrs:
    def test_empty_initially(self, store):
        assert store.get_all_tensor_attrs() == []

    def test_auto_registers_on_get(self, store):
        store.get_tensor("paper", "x")
        assert {a.attr_name for a in store.get_all_tensor_attrs()} == {"x"}

    def test_registry_is_not_leaked_to_callers(self, store):
        """PyG assembles batches by writing .index onto attrs it is handed."""
        store.get_tensor("paper", "x")
        assert store.get_tensor_size("paper", "x") == (3, 2)
        leaked = store.get_all_tensor_attrs()[0]
        leaked.index = torch.tensor([0])
        assert store.get_all_tensor_attrs()[0].index is None
        assert store.get_tensor_size("paper", "x") == (3, 2)

    def test_get_tensor_size_does_not_clone_the_matrix(self, store, monkeypatch):
        """Reading a shape must not copy a potentially huge feature matrix."""
        store.get_tensor_size("paper", "x")
        cached = store._tensor_cache[("paper", "x")]
        monkeypatch.setattr(
            type(cached), "clone", lambda self: pytest.fail("cloned to read a shape")
        )
        assert store.get_tensor_size("paper", "x") == (3, 2)

    def test_get_tensor_size_ignores_index(self, store):
        attr = FalkorDBTensorAttr("paper", "x", index=torch.tensor([0]))
        assert store._get_tensor_size(attr) == (3, 2)

    def test_index_defaults_to_none(self):
        assert FalkorDBTensorAttr(group_name="paper", attr_name="x").index is None


# ---------------------------------------------------------------------------
# Tests – label mapping
# ---------------------------------------------------------------------------


class TestNodeTypeLabelMapping:
    def test_custom_label_used_in_query(self):
        graph = FakeFalkorGraph(nodes={"Paper": {0: {"x": [1.0, 2.0]}}})
        store = FalkorDBFeatureStore(graph, node_type_to_label={"paper": "Paper"})
        assert store.get_tensor("paper", "x").shape == (1, 2)
        assert any("`Paper`" in call for call in graph.calls)
