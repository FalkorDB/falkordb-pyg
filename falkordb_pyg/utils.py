"""Utility helpers for falkordb-pyg."""

from typing import Dict, List, Optional


class NodeIDMapper:
    """Maps FalkorDB internal node IDs to contiguous 0-based PyG indices.

    FalkorDB uses internal node IDs that may not be contiguous or start at 0.
    This class builds a bidirectional mapping between FalkorDB IDs and the
    contiguous 0-based indices that PyG expects.
    """

    def __init__(self) -> None:
        self._falkordb_to_pyg: Dict[int, int] = {}
        self._pyg_to_falkordb: List[int] = []

    def build(self, falkordb_ids: List[int]) -> None:
        """Build the mapping from a list of FalkorDB node IDs.

        Args:
            falkordb_ids: List of FalkorDB internal node IDs (may be unsorted).

        """
        sorted_ids = sorted(set(falkordb_ids))
        self._pyg_to_falkordb = sorted_ids
        self._falkordb_to_pyg = {fid: idx for idx, fid in enumerate(sorted_ids)}

    def to_pyg(self, falkordb_id: int) -> int:
        """Convert a FalkorDB node ID to a PyG index.

        Args:
            falkordb_id: FalkorDB internal node ID.

        Returns:
            Corresponding 0-based PyG index.

        """
        return self._falkordb_to_pyg[falkordb_id]

    def to_falkordb(self, pyg_idx: int) -> int:
        """Convert a PyG index to a FalkorDB node ID.

        Args:
            pyg_idx: 0-based PyG node index.

        Returns:
            Corresponding FalkorDB internal node ID.

        """
        return self._pyg_to_falkordb[pyg_idx]

    def __len__(self) -> int:
        return len(self._pyg_to_falkordb)

    @property
    def is_built(self) -> bool:
        """Return True if the mapper has been built."""
        return len(self._pyg_to_falkordb) > 0


def build_node_id_query(label: str) -> str:
    """Build a Cypher query to fetch all node IDs for a label.

    Args:
        label: Node label in FalkorDB.

    Returns:
        Cypher query string.

    """
    return f"MATCH (n:{label}) RETURN ID(n)"


def build_feature_query(label: str, property_name: str) -> str:
    """Build a Cypher query to fetch a node feature by property name.

    Args:
        label: Node label in FalkorDB.
        property_name: Property name to fetch.

    Returns:
        Cypher query string returning (ID(n), n.property) pairs.

    """
    return f"MATCH (n:{label}) RETURN ID(n), n.{property_name}"


def build_edge_query(
    src_label: str,
    rel_type: str,
    dst_label: str,
    src_mapper: Optional[NodeIDMapper] = None,
    dst_mapper: Optional[NodeIDMapper] = None,
) -> str:
    """Build a Cypher query to fetch edge indices.

    Args:
        src_label: Source node label.
        rel_type: Relationship type.
        dst_label: Destination node label.
        src_mapper: Optional mapper (unused, kept for API consistency).
        dst_mapper: Optional mapper (unused, kept for API consistency).

    Returns:
        Cypher query string returning (ID(s), ID(d)) pairs.

    """
    return f"MATCH (s:{src_label})-[r:{rel_type}]->(d:{dst_label}) RETURN ID(s), ID(d)"
