"""Unit tests for the Cypher query builders and NodeIDMapper."""

import pytest

from falkordb_pyg.utils import (
    NodeIDMapper,
    build_edge_query,
    build_feature_query,
    build_node_ids_query,
    quote_identifier,
)

HOSTILE = [
    "paper`) DETACH DELETE n //",
    "pa`per",
    "``",
    "with space",
    "with\nnewline",
    "}brace{",
    "-hyphen-",
]


class TestQuoteIdentifier:
    @pytest.mark.parametrize("name", HOSTILE + ["paper", "x", "CITES"])
    def test_round_trips(self, name):
        """A quoted identifier must parse back to exactly the input string.

        Cypher escapes a literal backtick by doubling it, so undoing the
        doubling inside the outer quotes has to recover the original.
        """
        quoted = quote_identifier(name)
        assert quoted.startswith("`") and quoted.endswith("`")
        assert quoted[1:-1].replace("``", "`") == name

    def test_backtick_is_doubled(self):
        assert quote_identifier("a`b") == "`a``b`"

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            quote_identifier("")

    def test_non_string_is_rejected(self):
        with pytest.raises(TypeError, match="must be a string"):
            quote_identifier(None)


class TestQueryBuilders:
    def test_node_ids_query(self):
        assert (
            build_node_ids_query("paper")
            == "MATCH (n:`paper`) RETURN ID(n) ORDER BY ID(n)"
        )

    def test_feature_query(self):
        assert build_feature_query("paper", "x") == (
            "MATCH (n:`paper`) RETURN n.`x`, ID(n) ORDER BY ID(n)"
        )

    def test_edge_query(self):
        assert build_edge_query("author", "writes", "paper") == (
            "MATCH (s:`author`)-[r:`writes`]->(d:`paper`) RETURN ID(s), ID(d)"
        )

    @pytest.mark.parametrize("hostile", HOSTILE)
    def test_hostile_label_cannot_add_a_clause(self, hostile):
        """The injected text must stay inside the quoted identifier."""
        query = build_node_ids_query(hostile)
        assert query.count("MATCH") == 1
        assert query.endswith(") RETURN ID(n) ORDER BY ID(n)")

    @pytest.mark.parametrize("hostile", HOSTILE)
    def test_hostile_property_cannot_add_a_clause(self, hostile):
        query = build_feature_query("paper", hostile)
        assert query.count("MATCH") == 1
        assert query.endswith(", ID(n) ORDER BY ID(n)")

    @pytest.mark.parametrize("hostile", HOSTILE)
    def test_hostile_relationship_cannot_add_a_clause(self, hostile):
        query = build_edge_query("paper", hostile, "paper")
        assert query.count("MATCH") == 1
        assert query.endswith(" RETURN ID(s), ID(d)")


class TestNodeIDMapper:
    def test_maps_both_directions(self):
        mapper = NodeIDMapper([100, 200, 300])
        assert mapper.num_nodes == 3
        assert mapper.falkor_to_pyg(200) == 1
        assert mapper.pyg_to_falkor(1) == 200

    def test_unknown_id_returns_none(self):
        assert NodeIDMapper([1, 2]).falkor_to_pyg(99) is None

    def test_empty(self):
        assert NodeIDMapper([]).num_nodes == 0


class TestRemapEdges:
    def test_homogeneous_uses_one_mapping(self):
        mapper = NodeIDMapper([10, 11, 12])
        src, dst = mapper.remap_edges([10, 11], [11, 12])
        assert src == [0, 1] and dst == [1, 2]

    def test_heterogeneous_needs_the_destination_mapper(self):
        """The old signature applied one mapper to both endpoints."""
        authors = NodeIDMapper([10, 11])
        papers = NodeIDMapper([0, 1, 2])
        src, dst = authors.remap_edges([10, 10, 11], [0, 1, 2], dst_mapper=papers)
        assert src == [0, 0, 1]
        assert dst == [0, 1, 2]

    def test_without_the_destination_mapper_hetero_ids_are_dropped(self):
        authors = NodeIDMapper([10, 11])
        src, dst = authors.remap_edges([10], [0])
        assert src == [] and dst == []

    def test_unknown_endpoints_are_dropped(self):
        mapper = NodeIDMapper([1, 2])
        src, dst = mapper.remap_edges([1, 3], [2, 4])
        assert src == [0] and dst == [1]
