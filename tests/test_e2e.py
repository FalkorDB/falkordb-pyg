"""End-to-end tests that run against a live FalkorDB server.

These tests are skipped when no FalkorDB instance is reachable (see
``conftest.py``).  Set ``FALKORDB_HOST`` / ``FALKORDB_PORT`` environment
variables to point at a non-default server.

Start a local FalkorDB instance with::

    docker run -p 6379:6379 falkordb/falkordb:latest
"""

from __future__ import annotations

import os
import uuid

import pytest
import torch
from falkordb import FalkorDB
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout

from falkordb_pyg import get_remote_backend
from falkordb_pyg.feature_store import FalkorDBFeatureStore, FalkorDBTensorAttr
from falkordb_pyg.graph_store import FalkorDBGraphStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOST = os.environ.get("FALKORDB_HOST", "localhost")
_PORT = int(os.environ.get("FALKORDB_PORT", "6379"))


def _unique_graph_name() -> str:
    """Return a unique graph name to avoid collisions between tests."""
    return f"test_{uuid.uuid4().hex[:12]}"


def _drop_graph(graph_name: str) -> None:
    """Delete the graph from FalkorDB."""
    db = FalkorDB(host=_HOST, port=_PORT)
    g = db.select_graph(graph_name)
    try:
        g.query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def homo_graph_name():
    """Create a homogeneous 'paper' graph with features and edges.

    Graph layout::

        paper0 --cites--> paper1 --cites--> paper2
                                  paper1 --cites--> paper0

    Each paper node has:
        x: 2-D feature vector
        y: scalar label
    """
    name = _unique_graph_name()
    db = FalkorDB(host=_HOST, port=_PORT)
    g = db.select_graph(name)

    g.query(
        """
        CREATE (p0:paper {x: [1.0, 2.0], y: 0}),
               (p1:paper {x: [3.0, 4.0], y: 1}),
               (p2:paper {x: [5.0, 6.0], y: 2}),
               (p0)-[:cites]->(p1),
               (p1)-[:cites]->(p2),
               (p1)-[:cites]->(p0)
        """
    )

    yield name

    _drop_graph(name)


@pytest.fixture()
def hetero_graph_name():
    """Create a heterogeneous graph with 'paper' and 'author' nodes.

    Graph layout::

        author0 --writes--> paper0
        author0 --writes--> paper1
        author1 --writes--> paper2
        paper0 --cites--> paper1
        paper1 --cites--> paper2

    Each author node has:
        x: 2-D feature vector
    Each paper node has:
        x: 2-D feature vector
        y: scalar label
    """
    name = _unique_graph_name()
    db = FalkorDB(host=_HOST, port=_PORT)
    g = db.select_graph(name)

    g.query(
        """
        CREATE (a0:author {x: [0.5, 0.5]}),
               (a1:author {x: [0.1, 0.9]}),
               (p0:paper {x: [1.0, 0.0], y: 0}),
               (p1:paper {x: [0.0, 1.0], y: 1}),
               (p2:paper {x: [0.5, 0.5], y: 2}),
               (a0)-[:writes]->(p0),
               (a0)-[:writes]->(p1),
               (a1)-[:writes]->(p2),
               (p0)-[:cites]->(p1),
               (p1)-[:cites]->(p2)
        """
    )

    yield name

    _drop_graph(name)


def _connect(graph_name: str):
    """Return a FalkorDB graph handle."""
    db = FalkorDB(host=_HOST, port=_PORT)
    return db.select_graph(graph_name)


# ---------------------------------------------------------------------------
# Tests – Feature Store
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestFeatureStoreE2E:
    def test_get_vector_features(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBFeatureStore(graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")

        x = store._get_tensor(attr)
        assert x.shape == (3, 2)
        # Values should be the ones we inserted (sorted by ID)
        assert x.dtype == torch.float

    def test_get_scalar_property_as_column(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBFeatureStore(graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="y")

        y = store._get_tensor(attr)
        assert y.shape == (3, 1)
        # Labels 0, 1, 2 in order
        assert set(y.squeeze().tolist()) == {0.0, 1.0, 2.0}

    def test_indexed_feature_access(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBFeatureStore(graph)

        # First load full tensor
        full_attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        full = store._get_tensor(full_attr)

        # Then access with index
        idx = torch.tensor([0, 2])
        indexed_attr = FalkorDBTensorAttr(group_name="paper", attr_name="x", index=idx)
        indexed = store._get_tensor(indexed_attr)
        assert indexed.shape == (2, 2)
        assert torch.equal(indexed, full[idx])

    def test_caching_avoids_second_query(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBFeatureStore(graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")

        first = store._get_tensor(attr)
        second = store._get_tensor(attr)
        # Same object reference means it came from cache
        assert first is second

    def test_get_tensor_size(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBFeatureStore(graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        assert store._get_tensor_size(attr) == (3, 2)

    def test_put_and_remove_tensor(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBFeatureStore(graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="z")

        tensor = torch.randn(3, 8)
        assert store._put_tensor(tensor, attr) is True
        assert torch.equal(store._get_tensor(attr), tensor)
        assert store._remove_tensor(attr) is True
        assert store._remove_tensor(attr) is False

    def test_get_all_tensor_attrs(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBFeatureStore(graph)

        store._get_tensor(FalkorDBTensorAttr(group_name="paper", attr_name="x"))
        store._get_tensor(FalkorDBTensorAttr(group_name="paper", attr_name="y"))

        attrs = store.get_all_tensor_attrs()
        names = {a.attr_name for a in attrs}
        assert names == {"x", "y"}


# ---------------------------------------------------------------------------
# Tests – Graph Store
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestGraphStoreE2E:
    def test_get_edge_index_coo(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBGraphStore(graph)
        attr = EdgeAttr(edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO)

        ei = store._get_edge_index(attr)
        assert ei is not None
        src, dst = ei
        # 3 edges created
        assert src.shape[0] == 3
        assert dst.shape[0] == 3
        # Indices should be in range [0, 3)
        assert src.min() >= 0 and src.max() < 3
        assert dst.min() >= 0 and dst.max() < 3

    def test_node_id_remapping(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBGraphStore(graph)
        attr = EdgeAttr(edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO)

        store._get_edge_index(attr)
        # Verify mapper was built with 3 nodes
        mapper = store._id_mappers["paper"]
        assert mapper.num_nodes == 3

    def test_edge_attr_auto_registered(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBGraphStore(graph)
        attr = EdgeAttr(edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO)

        store._get_edge_index(attr)
        all_attrs = store.get_all_edge_attrs()
        assert len(all_attrs) == 1
        assert all_attrs[0].edge_type == ("paper", "cites", "paper")
        assert all_attrs[0].size == (3, 3)

    def test_caching_avoids_second_query(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBGraphStore(graph)
        attr = EdgeAttr(edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO)

        first = store._get_edge_index(attr)
        second = store._get_edge_index(attr)
        # Same tuple object from cache
        assert first[0] is second[0]
        assert first[1] is second[1]

    def test_put_and_remove_edge_index(self, homo_graph_name):
        graph = _connect(homo_graph_name)
        store = FalkorDBGraphStore(graph)
        edge_type = ("paper", "cites", "paper")
        attr = EdgeAttr(edge_type=edge_type, layout=EdgeLayout.COO, size=(3, 3))

        src = torch.tensor([0, 1])
        dst = torch.tensor([1, 2])
        assert store._put_edge_index((src, dst), attr) is True
        result = store._get_edge_index(attr)
        assert torch.equal(result[0], src)
        assert torch.equal(result[1], dst)

        assert store._remove_edge_index(attr) is True
        assert store._remove_edge_index(attr) is False


# ---------------------------------------------------------------------------
# Tests – Heterogeneous Graph
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestHeterogeneousE2E:
    def test_paper_features(self, hetero_graph_name):
        graph = _connect(hetero_graph_name)
        store = FalkorDBFeatureStore(graph)
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")

        x = store._get_tensor(attr)
        assert x.shape == (3, 2)

    def test_author_features(self, hetero_graph_name):
        graph = _connect(hetero_graph_name)
        store = FalkorDBFeatureStore(graph)
        attr = FalkorDBTensorAttr(group_name="author", attr_name="x")

        x = store._get_tensor(attr)
        assert x.shape == (2, 2)

    def test_writes_edges(self, hetero_graph_name):
        graph = _connect(hetero_graph_name)
        store = FalkorDBGraphStore(graph)
        attr = EdgeAttr(edge_type=("author", "writes", "paper"), layout=EdgeLayout.COO)

        ei = store._get_edge_index(attr)
        assert ei is not None
        assert ei[0].shape[0] == 3  # 3 writes edges

    def test_cites_edges(self, hetero_graph_name):
        graph = _connect(hetero_graph_name)
        store = FalkorDBGraphStore(graph)
        attr = EdgeAttr(edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO)

        ei = store._get_edge_index(attr)
        assert ei is not None
        assert ei[0].shape[0] == 2  # 2 cites edges

    def test_multiple_edge_types_independent(self, hetero_graph_name):
        graph = _connect(hetero_graph_name)
        store = FalkorDBGraphStore(graph)

        writes_attr = EdgeAttr(
            edge_type=("author", "writes", "paper"), layout=EdgeLayout.COO
        )
        cites_attr = EdgeAttr(
            edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO
        )

        writes_ei = store._get_edge_index(writes_attr)
        cites_ei = store._get_edge_index(cites_attr)

        # Both should be present
        attrs = store.get_all_edge_attrs()
        edge_types = {a.edge_type for a in attrs}
        assert ("author", "writes", "paper") in edge_types
        assert ("paper", "cites", "paper") in edge_types

        # Edge counts should be independent
        assert writes_ei[0].shape[0] == 3
        assert cites_ei[0].shape[0] == 2


# ---------------------------------------------------------------------------
# Tests – get_remote_backend factory
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestGetRemoteBackendE2E:
    def test_returns_correct_types(self, homo_graph_name):
        feature_store, graph_store = get_remote_backend(
            host=_HOST, port=_PORT, graph_name=homo_graph_name
        )
        assert isinstance(feature_store, FalkorDBFeatureStore)
        assert isinstance(graph_store, FalkorDBGraphStore)

    def test_feature_access_through_backend(self, homo_graph_name):
        feature_store, _ = get_remote_backend(
            host=_HOST, port=_PORT, graph_name=homo_graph_name
        )
        attr = FalkorDBTensorAttr(group_name="paper", attr_name="x")
        x = feature_store._get_tensor(attr)
        assert x.shape == (3, 2)

    def test_edge_access_through_backend(self, homo_graph_name):
        _, graph_store = get_remote_backend(
            host=_HOST, port=_PORT, graph_name=homo_graph_name
        )
        attr = EdgeAttr(edge_type=("paper", "cites", "paper"), layout=EdgeLayout.COO)
        ei = graph_store._get_edge_index(attr)
        assert ei is not None
        assert ei[0].shape[0] == 3

    def test_custom_mappings(self, hetero_graph_name):
        """Custom node_type_to_label and edge_type_to_rel should work.

        Here PyG type names ("Paper", "Author") differ from the FalkorDB
        labels ("paper", "author") stored in the graph.  The mappings
        translate the capitalized PyG types to the lowercase DB labels.
        """
        feature_store, graph_store = get_remote_backend(
            host=_HOST,
            port=_PORT,
            graph_name=hetero_graph_name,
            node_type_to_label={"Paper": "paper", "Author": "author"},
            edge_type_to_rel={
                ("Author", "writes", "Paper"): "writes",
                ("Paper", "cites", "Paper"): "cites",
            },
        )

        attr = FalkorDBTensorAttr(group_name="Paper", attr_name="x")
        x = feature_store._get_tensor(attr)
        assert x.shape == (3, 2)

        attr = EdgeAttr(edge_type=("Author", "writes", "Paper"), layout=EdgeLayout.COO)
        ei = graph_store._get_edge_index(attr)
        assert ei is not None
        assert ei[0].shape[0] == 3
