"""FalkorDB GraphStore implementation for PyG remote backend."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data.graph_store import EdgeAttr, GraphStore

from falkordb_pyg.utils import NodeIDMapper, build_edge_query, build_node_id_query


class FalkorDBGraphStore(GraphStore):
    """PyG GraphStore backed by FalkorDB.

    Implements the GraphStore interface using Cypher queries to retrieve
    edge connectivity from a FalkorDB graph. Results are cached locally
    after the first fetch to avoid repeated network round-trips.

    Args:
        graph: A FalkorDB graph object (e.g. ``db.select_graph('name')``).

    """

    def __init__(self, graph: Any) -> None:
        super().__init__()
        self._graph = graph
        self._edge_index_cache: Dict[Tuple, torch.Tensor] = {}
        self._node_id_mappers: Dict[str, NodeIDMapper] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_build_mapper(self, label: str) -> NodeIDMapper:
        """Return (building if needed) the NodeIDMapper for *label*."""
        if label not in self._node_id_mappers:
            mapper = NodeIDMapper()
            query = build_node_id_query(label)
            result = self._graph.query(query)
            ids = [int(row[0]) for row in result.result_set]
            mapper.build(ids)
            self._node_id_mappers[label] = mapper
        return self._node_id_mappers[label]

    def _fetch_edge_index(self, edge_attr: EdgeAttr) -> torch.Tensor:
        """Fetch edge index from FalkorDB and remap to PyG indices."""
        edge_type: Tuple[str, str, str] = edge_attr.edge_type
        src_label, rel_type, dst_label = edge_type

        src_mapper = self._get_or_build_mapper(src_label)
        dst_mapper = self._get_or_build_mapper(dst_label)

        query = build_edge_query(src_label, rel_type, dst_label)
        result = self._graph.query(query)

        src_indices: List[int] = []
        dst_indices: List[int] = []
        for row in result.result_set:
            src_fid, dst_fid = int(row[0]), int(row[1])
            src_indices.append(src_mapper.to_pyg(src_fid))
            dst_indices.append(dst_mapper.to_pyg(dst_fid))

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        return edge_index

    # ------------------------------------------------------------------
    # GraphStore abstract method implementations
    # ------------------------------------------------------------------

    def _put_edge_index(self, edge_index: Any, edge_attr: EdgeAttr) -> bool:
        """Store an edge index in the cache (does not write to FalkorDB)."""
        key = self._to_cache_key(edge_attr)
        self._edge_index_cache[key] = edge_index
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[torch.Tensor]:
        """Retrieve an edge index, fetching from FalkorDB if not cached."""
        key = self._to_cache_key(edge_attr)
        if key not in self._edge_index_cache:
            try:
                edge_index = self._fetch_edge_index(edge_attr)
                self._edge_index_cache[key] = edge_index
            except Exception:
                return None
        return self._edge_index_cache[key]

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        """Remove an edge index from the local cache."""
        key = self._to_cache_key(edge_attr)
        if key in self._edge_index_cache:
            del self._edge_index_cache[key]
            return True
        return False

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        """Return all edge attributes currently held in the local cache."""
        attrs = []
        for key in self._edge_index_cache:
            edge_type, layout, is_sorted, size = key
            attr = EdgeAttr(
                edge_type=edge_type,
                layout=layout,
                is_sorted=is_sorted,
                size=size,
            )
            attrs.append(attr)
        return attrs

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _to_cache_key(edge_attr: EdgeAttr) -> Tuple:
        """Convert an EdgeAttr to a hashable cache key."""
        return (
            tuple(edge_attr.edge_type) if edge_attr.edge_type is not None else None,
            edge_attr.layout,
            edge_attr.is_sorted,
            tuple(edge_attr.size) if edge_attr.size is not None else None,
        )
