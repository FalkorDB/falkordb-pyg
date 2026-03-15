"""Utility helpers for the FalkorDB PyG backend.

Provides node ID remapping (FalkorDB internal IDs → contiguous 0-based PyG
indices) and small Cypher query builders used by both stores.
"""

from typing import Dict, List, Optional, Tuple


class NodeIDMapper:
    """Bidirectional mapping between FalkorDB internal node IDs and
    contiguous 0-based PyG node indices.

    FalkorDB assigns internal integer IDs to nodes that may not be contiguous
    or start at zero.  PyG's samplers require contiguous indices starting from
    0, so we maintain an explicit mapping.

    Args:
        falkordb_ids: Ordered list of FalkorDB node IDs.  Position ``i`` in
            the list becomes PyG index ``i``.
    """

    def __init__(self, falkordb_ids: List[int]) -> None:
        self._pyg_to_falkor: List[int] = falkordb_ids
        self._falkor_to_pyg: Dict[int, int] = {
            fid: pyg_idx for pyg_idx, fid in enumerate(falkordb_ids)
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Total number of nodes tracked by this mapper."""
        return len(self._pyg_to_falkor)

    def falkor_to_pyg(self, falkor_id: int) -> Optional[int]:
        """Return the PyG index for a given FalkorDB node ID, or ``None``."""
        return self._falkor_to_pyg.get(falkor_id)

    def pyg_to_falkor(self, pyg_idx: int) -> int:
        """Return the FalkorDB node ID for a given PyG index."""
        return self._pyg_to_falkor[pyg_idx]

    def remap_edges(
        self, src_ids: List[int], dst_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Remap lists of FalkorDB src/dst IDs to PyG indices.

        Pairs where either endpoint is missing from the mapping are silently
        dropped.
        """
        new_src, new_dst = [], []
        for s, d in zip(src_ids, dst_ids):
            ps = self._falkor_to_pyg.get(s)
            pd = self._falkor_to_pyg.get(d)
            if ps is not None and pd is not None:
                new_src.append(ps)
                new_dst.append(pd)
        return new_src, new_dst


# ---------------------------------------------------------------------------
# Cypher query builders
# ---------------------------------------------------------------------------


def build_node_ids_query(label: str) -> str:
    """Return a Cypher query that fetches all internal node IDs for *label*."""
    return f"MATCH (n:`{label}`) RETURN ID(n) ORDER BY ID(n)"


def build_feature_query(label: str, prop: str) -> str:
    """Return a Cypher query that fetches a node property ordered by ID."""
    return f"MATCH (n:`{label}`) RETURN n.`{prop}`, ID(n) ORDER BY ID(n)"


def build_edge_query(src_label: str, rel_type: str, dst_label: str) -> str:
    """Return a Cypher query that fetches (src_id, dst_id) for an edge type."""
    return (
        f"MATCH (s:`{src_label}`)-[r:`{rel_type}`]->(d:`{dst_label}`) "
        f"RETURN ID(s), ID(d)"
    )
