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
        self,
        src_ids: List[int],
        dst_ids: List[int],
        dst_mapper: Optional["NodeIDMapper"] = None,
    ) -> Tuple[List[int], List[int]]:
        """Remap lists of FalkorDB src/dst IDs to PyG indices.

        Pairs where either endpoint is missing from the relevant mapping are
        silently dropped.

        Args:
            src_ids: FalkorDB IDs of the source endpoints.
            dst_ids: FalkorDB IDs of the destination endpoints.
            dst_mapper: Mapper for the destination node type.  Required for a
                heterogeneous edge, where the endpoints belong to different
                node types and therefore to different index spaces.  Defaults
                to ``self`` for a homogeneous edge.
        """
        dst = dst_mapper if dst_mapper is not None else self
        new_src, new_dst = [], []
        for s, d in zip(src_ids, dst_ids):
            ps = self._falkor_to_pyg.get(s)
            pd = dst._falkor_to_pyg.get(d)
            if ps is not None and pd is not None:
                new_src.append(ps)
                new_dst.append(pd)
        return new_src, new_dst


# ---------------------------------------------------------------------------
# Cypher query builders
# ---------------------------------------------------------------------------


def quote_identifier(name: str) -> str:
    """Return *name* as a backtick-quoted Cypher identifier.

    Cypher escapes a literal backtick inside a quoted identifier by doubling
    it.  Without this, a label or property containing a backtick terminates
    the quoted section early and the remainder is parsed as Cypher — turning
    a stray identifier into arbitrary query execution.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"Cypher identifier must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("Cypher identifier must not be empty")
    escaped = name.replace("`", "``")
    return f"`{escaped}`"


def build_node_ids_query(label: str) -> str:
    """Return a Cypher query that fetches all internal node IDs for *label*."""
    return f"MATCH (n:{quote_identifier(label)}) RETURN ID(n) ORDER BY ID(n)"


def build_feature_query(label: str, prop: str) -> str:
    """Return a Cypher query that fetches a node property ordered by ID."""
    return (
        f"MATCH (n:{quote_identifier(label)}) "
        f"RETURN n.{quote_identifier(prop)}, ID(n) ORDER BY ID(n)"
    )


def build_edge_query(src_label: str, rel_type: str, dst_label: str) -> str:
    """Return a Cypher query that fetches (src_id, dst_id, rel_id) for an edge type.

    ``ID(r)`` is projected even though callers only need the endpoints: without
    it FalkorDB collapses identical ``(ID(s), ID(d))`` rows, silently dropping
    parallel edges between the same pair of nodes.
    """
    return (
        f"MATCH (s:{quote_identifier(src_label)})"
        f"-[r:{quote_identifier(rel_type)}]->"
        f"(d:{quote_identifier(dst_label)}) "
        f"RETURN ID(s), ID(d), ID(r)"
    )
