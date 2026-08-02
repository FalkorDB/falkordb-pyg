"""Integration tests for the full Remote Backend stack against a fake FalkorDB.

These exercise what PyG itself calls: the metadata that :class:`NeighborSampler`
reads off the two stores, and the factory that wires them together.

Note that iterating a :class:`~torch_geometric.loader.NeighborLoader` also needs
``pyg-lib`` or ``torch-sparse``, which are not declared dependencies; these tests
therefore assert on the sampler's view of the graph rather than on batches.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout
from torch_geometric.sampler import NeighborSampler

from falkordb_pyg import get_remote_backend
from falkordb_pyg.feature_store import FalkorDBFeatureStore
from falkordb_pyg.graph_store import FalkorDBGraphStore

from .conftest import FakeFalkorGraph

PAPER_CITES = ("paper", "cites", "paper")
AUTHOR_WRITES = ("author", "writes", "paper")


def _stores(graph):
    return FalkorDBFeatureStore(graph), FalkorDBGraphStore(graph)


# ---------------------------------------------------------------------------
# Tests – factory function
# ---------------------------------------------------------------------------


class TestGetRemoteBackend:
    def test_returns_tuple_of_correct_types(self):
        with patch("falkordb_pyg.FalkorDB") as mock_falkordb:
            mock_falkordb.return_value.select_graph.return_value = MagicMock()
            feature_store, graph_store = get_remote_backend(graph_name="test")
        assert isinstance(feature_store, FalkorDBFeatureStore)
        assert isinstance(graph_store, FalkorDBGraphStore)

    def test_custom_mappings_forwarded(self):
        with patch("falkordb_pyg.FalkorDB") as mock_falkordb:
            mock_falkordb.return_value.select_graph.return_value = MagicMock()
            feature_store, graph_store = get_remote_backend(
                graph_name="g",
                node_type_to_label={"paper": "Paper"},
                edge_type_to_rel={PAPER_CITES: "CITES"},
            )
        assert graph_store._node_type_to_label == {"paper": "Paper"}
        assert graph_store._edge_type_to_rel == {PAPER_CITES: "CITES"}
        assert feature_store._node_type_to_label == {"paper": "Paper"}

    def test_version_is_exported(self):
        import falkordb_pyg

        assert isinstance(falkordb_pyg.__version__, str)


# ---------------------------------------------------------------------------
# Tests – what PyG's sampler sees
# ---------------------------------------------------------------------------


class TestSamplerMetadata:
    def test_primed_backend_is_visible_to_neighbor_sampler(self, homo_graph):
        feature_store, graph_store = _stores(homo_graph)

        # Prime the types the loader will sample (see README Quick Start).
        feature_store.get_tensor_size("paper", "x")
        graph_store.get_edge_index(PAPER_CITES, layout="coo")

        sampler = NeighborSampler((feature_store, graph_store), num_neighbors=[2, 2])
        assert sampler.node_types == ["paper"]
        assert sampler.edge_types == [PAPER_CITES]
        assert sampler.num_nodes == {"paper": 3}

    def test_hetero_backend_is_visible_to_neighbor_sampler(self, hetero_graph):
        feature_store, graph_store = _stores(hetero_graph)
        for node_type in ("paper", "author"):
            feature_store.get_tensor_size(node_type, "x")
        for edge_type in (PAPER_CITES, AUTHOR_WRITES):
            graph_store.get_edge_index(edge_type, layout="coo")

        sampler = NeighborSampler((feature_store, graph_store), num_neighbors=[2])
        assert set(sampler.node_types) == {"paper", "author"}
        assert set(sampler.edge_types) == {PAPER_CITES, AUTHOR_WRITES}
        assert sampler.num_nodes == {"paper": 3, "author": 2}

    def test_sampler_metadata_survives_repeated_reads(self, homo_graph):
        """Guards the attr-registry leak: a second sampler must see the same graph."""
        feature_store, graph_store = _stores(homo_graph)
        feature_store.get_tensor_size("paper", "x")
        graph_store.get_edge_index(PAPER_CITES, layout="coo")

        first = NeighborSampler((feature_store, graph_store), num_neighbors=[2])
        second = NeighborSampler((feature_store, graph_store), num_neighbors=[2])
        assert first.num_nodes == second.num_nodes == {"paper": 3}

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Schema auto-discovery is not implemented (planned for v0.3.0): a "
            "cold backend advertises no node or edge types, so NeighborLoader "
            "silently samples an empty graph. See README 'Current limitations'."
        ),
    )
    def test_cold_backend_is_visible_to_neighbor_sampler(self, homo_graph):
        feature_store, graph_store = _stores(homo_graph)
        sampler = NeighborSampler((feature_store, graph_store), num_neighbors=[2, 2])
        assert sampler.edge_types == [PAPER_CITES]

    def test_cold_backend_reports_nothing_and_queries_nothing(self, homo_graph):
        """Pins the current behaviour so the v0.3.0 fix is a visible change."""
        feature_store, graph_store = _stores(homo_graph)
        assert feature_store.get_all_tensor_attrs() == []
        assert graph_store.get_all_edge_attrs() == []
        assert homo_graph.calls == []


# ---------------------------------------------------------------------------
# Tests – heterogeneous data flow
# ---------------------------------------------------------------------------


class TestHeterogeneousBackend:
    def test_features_per_node_type(self, hetero_graph):
        feature_store, _ = _stores(hetero_graph)
        assert feature_store.get_tensor("paper", "x").shape == (3, 2)
        assert feature_store.get_tensor("author", "x").shape == (2, 2)

    def test_edges_per_edge_type_are_independent(self, hetero_graph):
        _, graph_store = _stores(hetero_graph)
        writes, _ = graph_store.get_edge_index(AUTHOR_WRITES, layout="coo")
        cites, _ = graph_store.get_edge_index(PAPER_CITES, layout="coo")
        assert writes.shape[0] == 3
        assert cites.shape[0] == 2

    def test_cross_type_edges_use_the_right_mappers(self, hetero_graph):
        """author IDs 10,11 -> 0,1 and paper IDs 0,1,2 -> 0,1,2."""
        _, graph_store = _stores(hetero_graph)
        row, col = graph_store.get_edge_index(AUTHOR_WRITES, layout="coo")
        assert row.tolist() == [0, 0, 1]
        assert col.tolist() == [0, 1, 2]

    def test_features_align_with_topology_indices(self, hetero_graph):
        """Feature row i and edge index i must refer to the same node."""
        feature_store, graph_store = _stores(hetero_graph)
        x = feature_store.get_tensor("author", "x")
        mapper = graph_store.id_mapper("author")
        for pyg_idx in range(mapper.num_nodes):
            falkor_id = mapper.pyg_to_falkor(pyg_idx)
            expected = hetero_graph.nodes["author"][falkor_id]["x"]
            assert x[pyg_idx].tolist() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Tests – Cypher generation safety
# ---------------------------------------------------------------------------


class TestQuerySafety:
    @pytest.mark.parametrize(
        "hostile",
        [
            "paper`) DETACH DELETE n //",
            "pa`per",
            "with space",
            "with\nnewline",
            "}brace{",
        ],
    )
    def test_identifiers_round_trip_instead_of_injecting(self, hostile):
        """A quoted identifier must parse back to exactly the input string."""
        from falkordb_pyg.utils import quote_identifier

        quoted = quote_identifier(hostile)
        assert quoted.startswith("`") and quoted.endswith("`")
        # Undo the doubling the way Cypher's lexer would.
        assert quoted[1:-1].replace("``", "`") == hostile

    def test_hostile_label_does_not_add_a_second_statement(self):
        from falkordb_pyg.utils import build_node_ids_query

        query = build_node_ids_query("paper`) DETACH DELETE n //")
        assert query.count("MATCH") == 1
        assert query.endswith("RETURN ID(n) ORDER BY ID(n)")

    def test_hostile_label_round_trips_through_the_fake(self):
        hostile = "pa`per"
        graph = FakeFalkorGraph(nodes={hostile: {0: {"x": [1.0]}}})
        store = FalkorDBFeatureStore(graph)
        assert store.get_tensor(hostile, "x").shape == (1, 1)


# ---------------------------------------------------------------------------
# Tests – layout plumbing through the public API
# ---------------------------------------------------------------------------


class TestLayoutPlumbing:
    def test_csc_conversion_matches_manual_sort(self, homo_graph):
        _, graph_store = _stores(homo_graph)
        graph_store.get_edge_index(PAPER_CITES, layout="coo")
        row, colptr, perm = graph_store.csc()
        assert colptr[PAPER_CITES].tolist() == [0, 1, 2, 3]
        assert row[PAPER_CITES].numel() == 3
        assert perm[PAPER_CITES].numel() == 3

    def test_put_then_get_roundtrip(self, homo_graph):
        _, graph_store = _stores(homo_graph)
        src, dst = torch.tensor([0, 1]), torch.tensor([1, 2])
        graph_store.put_edge_index((src, dst), PAPER_CITES, layout="coo", size=(3, 3))
        attr = EdgeAttr(PAPER_CITES, layout=EdgeLayout.COO)
        got_src, got_dst = graph_store._get_edge_index(attr)
        assert torch.equal(got_src, src) and torch.equal(got_dst, dst)
