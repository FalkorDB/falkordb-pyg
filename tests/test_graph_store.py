"""Unit tests for FalkorDBGraphStore."""

import pytest
import torch
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout

from falkordb_pyg.graph_store import FalkorDBGraphStore

from .conftest import FakeFalkorGraph

PAPER_CITES = ("paper", "cites", "paper")
AUTHOR_WRITES = ("author", "writes", "paper")


@pytest.fixture()
def store(homo_graph):
    return FalkorDBGraphStore(homo_graph)


def coo(edge_type, **kwargs):
    return EdgeAttr(edge_type=edge_type, layout=EdgeLayout.COO, **kwargs)


# ---------------------------------------------------------------------------
# Tests – public PyG API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_get_edge_index(self, store):
        row, col = store.get_edge_index(PAPER_CITES, layout="coo")
        assert row.tolist() == [0, 1, 1]
        assert col.tolist() == [1, 2, 0]

    def test_put_edge_index(self, store):
        src, dst = torch.tensor([0, 1]), torch.tensor([1, 2])
        store.put_edge_index((src, dst), PAPER_CITES, layout="coo", size=(3, 3))
        row, col = store.get_edge_index(PAPER_CITES, layout="coo")
        assert torch.equal(row, src) and torch.equal(col, dst)

    def test_remove_edge_index(self, store):
        store.get_edge_index(PAPER_CITES, layout="coo")
        assert store.remove_edge_index(PAPER_CITES, layout="coo") is True

    def test_coo_csr_csc(self, store):
        store.get_edge_index(PAPER_CITES, layout="coo")
        assert store.coo() is not None
        assert store.csr() is not None
        assert store.csc() is not None

    def test_csc_store_true_does_not_raise(self, store):
        """PyG memoises conversions via put_edge_index; 0.2.1 raised here."""
        store.get_edge_index(PAPER_CITES, layout="coo")
        store.csc(store=True)
        store.csr(store=True)


# ---------------------------------------------------------------------------
# Tests – ID remapping
# ---------------------------------------------------------------------------


class TestNodeIDRemapping:
    def test_non_contiguous_ids_are_remapped(self, store):
        """FalkorDB IDs 10, 11, 12 become PyG indices 0, 1, 2."""
        row, col = store.get_edge_index(PAPER_CITES, layout="coo")
        assert row.max() < 3 and col.max() < 3
        assert store.id_mapper("paper").num_nodes == 3
        assert store.id_mapper("paper").pyg_to_falkor(1) == 11
        assert store.id_mapper("paper").falkor_to_pyg(12) == 2

    def test_edges_to_unknown_nodes_are_dropped_with_a_warning(self):
        graph = FakeFalkorGraph(
            nodes={"paper": {1: {}, 2: {}}},
            # 3->4 references nodes outside the :paper id map
            edges={PAPER_CITES: [(1, 2), (3, 4)]},
        )
        store = FalkorDBGraphStore(graph)
        with pytest.warns(UserWarning, match="1 of 2"):
            row, col = store.get_edge_index(PAPER_CITES, layout="coo")
        assert row.tolist() == [0] and col.tolist() == [1]
        assert store.dropped_edges[PAPER_CITES] == 1

    def test_parallel_edges_are_preserved(self):
        """Two relationships between the same pair must yield two COO entries.

        The edge query projects ID(r) precisely to stop FalkorDB collapsing
        identical (ID(s), ID(d)) rows.
        """
        graph = FakeFalkorGraph(
            nodes={"paper": {0: {}, 1: {}}},
            edges={PAPER_CITES: [(0, 1), (0, 1), (1, 0)]},
        )
        row, col = FalkorDBGraphStore(graph).get_edge_index(PAPER_CITES, layout="coo")
        assert row.tolist() == [0, 0, 1]
        assert col.tolist() == [1, 1, 0]

    def test_edge_query_projects_relationship_id(self):
        from falkordb_pyg.utils import build_edge_query

        assert build_edge_query("paper", "cites", "paper").endswith(
            "RETURN ID(s), ID(d), ID(r)"
        )

    def test_mapper_is_shared_across_edge_types(self, hetero_graph):
        store = FalkorDBGraphStore(hetero_graph)
        store.get_edge_index(AUTHOR_WRITES, layout="coo")
        calls_before = len(hetero_graph.calls)
        store.get_edge_index(PAPER_CITES, layout="coo")
        # Only the edge query is new; the :paper id map is reused.
        assert len(hetero_graph.calls) == calls_before + 1


# ---------------------------------------------------------------------------
# Tests – edge attribute registry
# ---------------------------------------------------------------------------


class TestEdgeAttrs:
    def test_empty_initially(self, store):
        assert store.get_all_edge_attrs() == []

    def test_auto_registered_with_size(self, store):
        store.get_edge_index(PAPER_CITES, layout="coo")
        attrs = store.get_all_edge_attrs()
        assert len(attrs) == 1
        assert attrs[0].edge_type == PAPER_CITES
        assert attrs[0].size == (3, 3)

    def test_registry_is_not_leaked_to_callers(self, store):
        store.get_edge_index(PAPER_CITES, layout="coo")
        store.get_all_edge_attrs()[0].size = (99, 99)
        assert store.get_all_edge_attrs()[0].size == (3, 3)

    def test_put_without_size_backfills_it(self, store):
        """A missing size truncates the CSC colptr and drops trailing nodes."""
        store.put_edge_index(
            (torch.tensor([0]), torch.tensor([1])), PAPER_CITES, layout="coo"
        )
        assert store.get_all_edge_attrs()[0].size == (3, 3)

    def test_hetero_sizes_are_per_endpoint(self, hetero_graph):
        store = FalkorDBGraphStore(hetero_graph)
        store.get_edge_index(AUTHOR_WRITES, layout="coo")
        assert store.get_all_edge_attrs()[0].size == (2, 3)


# ---------------------------------------------------------------------------
# Tests – layout handling
# ---------------------------------------------------------------------------


class TestLayout:
    def test_non_coo_get_returns_none_rather_than_coo_tensors(self, store):
        """0.2.1 answered a CSR request with COO tensors and no error."""
        assert (
            store._get_edge_index(EdgeAttr(PAPER_CITES, layout=EdgeLayout.CSR)) is None
        )

    def test_put_accepts_converted_layouts(self, store):
        attr = EdgeAttr(PAPER_CITES, layout=EdgeLayout.CSC, size=(3, 3))
        assert store._put_edge_index(
            (torch.tensor([0, 1]), torch.tensor([0, 1, 2, 2])), attr
        )
        assert store._get_edge_index(attr) is not None

    def test_coo_fetch_after_a_converted_put_keeps_one_registration(self, store):
        """A CSC put registers the type; the later COO fetch must not re-register."""
        csc = EdgeAttr(PAPER_CITES, layout=EdgeLayout.CSC, size=(3, 3))
        store._put_edge_index((torch.tensor([0, 1]), torch.tensor([0, 1, 2, 2])), csc)
        assert len(store.get_all_edge_attrs()) == 1

        row, col = store.get_edge_index(PAPER_CITES, layout="coo")
        assert row.tolist() == [0, 1, 1]  # fetched from the graph, not the CSC cache
        attrs = store.get_all_edge_attrs()
        assert len(attrs) == 1
        assert attrs[0].size == (3, 3)

    def test_converted_layout_does_not_replace_the_coo_registration(self, store):
        store.get_edge_index(PAPER_CITES, layout="coo")
        store.csc(store=True)
        attrs = store.get_all_edge_attrs()
        assert len(attrs) == 1
        assert attrs[0].layout == EdgeLayout.COO


# ---------------------------------------------------------------------------
# Tests – caching and diagnostics
# ---------------------------------------------------------------------------


class TestCachingAndDiagnostics:
    def test_db_not_queried_on_second_get(self, store, homo_graph):
        store.get_edge_index(PAPER_CITES, layout="coo")
        count = len(homo_graph.calls)
        store.get_edge_index(PAPER_CITES, layout="coo")
        assert len(homo_graph.calls) == count

    def test_empty_edge_type_warns(self):
        graph = FakeFalkorGraph(nodes={"paper": {0: {}}}, edges={})
        store = FalkorDBGraphStore(graph)
        with pytest.warns(UserWarning, match="No edges matched"):
            row, _ = store.get_edge_index(("paper", "TYPO", "paper"), layout="coo")
        assert row.numel() == 0

    def test_clear_cache_drops_topology_and_mappers(self, store, homo_graph):
        store.get_edge_index(PAPER_CITES, layout="coo")
        calls = len(homo_graph.calls)
        store.clear_cache()
        assert store._id_mappers == {}
        store.get_edge_index(PAPER_CITES, layout="coo")
        assert len(homo_graph.calls) > calls

    def test_clear_cache_is_selective(self, hetero_graph):
        store = FalkorDBGraphStore(hetero_graph)
        store.get_edge_index(PAPER_CITES, layout="coo")
        store.get_edge_index(AUTHOR_WRITES, layout="coo")
        store.clear_cache(PAPER_CITES)
        calls = len(hetero_graph.calls)
        store.get_edge_index(AUTHOR_WRITES, layout="coo")
        assert len(hetero_graph.calls) == calls  # still cached

    def test_remove_then_get_refetches(self, store, homo_graph):
        store.get_edge_index(PAPER_CITES, layout="coo")
        store.remove_edge_index(PAPER_CITES, layout="coo")
        count = len(homo_graph.calls)
        store.get_edge_index(PAPER_CITES, layout="coo")
        assert len(homo_graph.calls) > count


# ---------------------------------------------------------------------------
# Tests – type mapping
# ---------------------------------------------------------------------------


class TestTypeMapping:
    def test_custom_label_and_rel_used_in_query(self):
        graph = FakeFalkorGraph(
            nodes={"Paper": {0: {}, 1: {}}},
            edges={("Paper", "CITES", "Paper"): [(0, 1)]},
        )
        store = FalkorDBGraphStore(
            graph,
            node_type_to_label={"paper": "Paper"},
            edge_type_to_rel={PAPER_CITES: "CITES"},
        )
        row, _ = store.get_edge_index(PAPER_CITES, layout="coo")
        assert row.tolist() == [0]


# ---------------------------------------------------------------------------
# Tests – read_only
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_reads_use_ro_query_by_default(self, homo_graph, monkeypatch):
        seen = []
        monkeypatch.setattr(
            homo_graph,
            "ro_query",
            lambda q, params=None, **kw: seen.append(q) or homo_graph.query(q),
        )
        FalkorDBGraphStore(homo_graph).get_edge_index(PAPER_CITES, layout="coo")
        assert seen

    def test_read_only_false_uses_query(self, homo_graph, monkeypatch):
        monkeypatch.setattr(
            homo_graph,
            "ro_query",
            lambda *a, **k: pytest.fail("ro_query used with read_only=False"),
        )
        store = FalkorDBGraphStore(homo_graph, read_only=False)
        assert store.get_edge_index(PAPER_CITES, layout="coo") is not None

    def test_handle_without_ro_query_falls_back_with_a_warning(self, legacy_graph):
        """An older client must degrade, not AttributeError mid-epoch."""
        store = FalkorDBGraphStore(legacy_graph)
        with pytest.warns(UserWarning, match="no ro_query"):
            assert store.get_edge_index(PAPER_CITES, layout="coo") is not None
        # Warned once, then stops trying.
        assert store._read_only is False
        assert store.id_mapper("paper").num_nodes == 3
