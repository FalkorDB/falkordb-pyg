"""FalkorDB implementation of PyG's GraphStore abstract class."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout, GraphStore

from .utils import NodeIDMapper, build_edge_query, build_node_ids_query


class FalkorDBGraphStore(GraphStore):
    """A PyG :class:`~torch_geometric.data.GraphStore` backed by FalkorDB.

    Edges are fetched on first access via Cypher queries and then cached
    locally so that subsequent calls do not round-trip to the database.

    Args:
        graph: A ``falkordb.Graph`` instance (the result of
            ``FalkorDB(...).select_graph(name)``).
        node_type_to_label: Optional mapping from PyG node type strings to
            FalkorDB node labels.  Defaults to the identity mapping.
        edge_type_to_rel: Optional mapping from PyG edge type triples
            ``(src_type, rel_type, dst_type)`` to FalkorDB relationship type
            strings.  Defaults to using the middle element of the triple.
    """

    def __init__(
        self,
        graph,
        node_type_to_label: Optional[Dict[str, str]] = None,
        edge_type_to_rel: Optional[Dict[Tuple[str, str, str], str]] = None,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._node_type_to_label: Dict[str, str] = node_type_to_label or {}
        self._edge_type_to_rel: Dict[Tuple[str, str, str], str] = edge_type_to_rel or {}

        # Cache: edge_type -> (src_tensor, dst_tensor) in COO format
        self._edge_index_cache: Dict[
            Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        # Cache: node_type -> NodeIDMapper
        self._id_mappers: Dict[str, NodeIDMapper] = {}
        # Registered edge attrs (populated by put_edge_index or discovered lazily)
        self._edge_attrs: Dict[Tuple[str, str, str], EdgeAttr] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label(self, node_type: str) -> str:
        """Resolve a PyG node type to a FalkorDB label."""
        return self._node_type_to_label.get(node_type, node_type)

    def _rel_type(self, edge_type: Tuple[str, str, str]) -> str:
        """Resolve a PyG edge type triple to a FalkorDB relationship type."""
        return self._edge_type_to_rel.get(edge_type, edge_type[1])

    def _get_or_build_mapper(self, node_type: str) -> NodeIDMapper:
        """Return (and cache) the NodeIDMapper for *node_type*."""
        if node_type not in self._id_mappers:
            label = self._label(node_type)
            query = build_node_ids_query(label)
            result = self._graph.query(query)
            ids = [int(row[0]) for row in result.result_set]
            self._id_mappers[node_type] = NodeIDMapper(ids)
        return self._id_mappers[node_type]

    def _fetch_edge_index(
        self, edge_type: Tuple[str, str, str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query FalkorDB and return a COO edge index remapped to PyG indices."""
        src_type, _, dst_type = edge_type
        rel = self._rel_type(edge_type)
        src_label = self._label(src_type)
        dst_label = self._label(dst_type)

        query = build_edge_query(src_label, rel, dst_label)
        result = self._graph.query(query)

        raw_src = [int(row[0]) for row in result.result_set]
        raw_dst = [int(row[1]) for row in result.result_set]

        src_mapper = self._get_or_build_mapper(src_type)
        dst_mapper = self._get_or_build_mapper(dst_type)

        new_src, new_dst = [], []
        for s, d in zip(raw_src, raw_dst):
            ps = src_mapper.falkor_to_pyg(s)
            pd = dst_mapper.falkor_to_pyg(d)
            if ps is not None and pd is not None:
                new_src.append(ps)
                new_dst.append(pd)

        src_t = torch.tensor(new_src, dtype=torch.long)
        dst_t = torch.tensor(new_dst, dtype=torch.long)
        return src_t, dst_t

    # ------------------------------------------------------------------
    # GraphStore abstract method implementations
    # ------------------------------------------------------------------

    def _put_edge_index(
        self, edge_index: Tuple[torch.Tensor, torch.Tensor], edge_attr: EdgeAttr
    ) -> bool:
        """Store an edge index in the local cache (does not write to DB)."""
        et = edge_attr.edge_type
        if edge_attr.layout != EdgeLayout.COO:
            raise NotImplementedError(
                "FalkorDBGraphStore only supports COO layout for put_edge_index."
            )
        self._edge_index_cache[et] = (edge_index[0], edge_index[1])
        self._edge_attrs[et] = edge_attr
        return True

    def _get_edge_index(
        self, edge_attr: EdgeAttr
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return a COO edge index, fetching from FalkorDB if not cached."""
        et = edge_attr.edge_type
        if et not in self._edge_index_cache:
            src_t, dst_t = self._fetch_edge_index(et)
            self._edge_index_cache[et] = (src_t, dst_t)
            # Auto-register the edge attr if not already present
            if et not in self._edge_attrs:
                src_type, _, dst_type = et
                src_mapper = self._get_or_build_mapper(src_type)
                dst_mapper = self._get_or_build_mapper(dst_type)
                self._edge_attrs[et] = EdgeAttr(
                    edge_type=et,
                    layout=EdgeLayout.COO,
                    is_sorted=False,
                    size=(src_mapper.num_nodes, dst_mapper.num_nodes),
                )
        return self._edge_index_cache[et]

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        """Remove a cached edge index."""
        et = edge_attr.edge_type
        existed = et in self._edge_index_cache
        self._edge_index_cache.pop(et, None)
        self._edge_attrs.pop(et, None)
        return existed

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        """Return all registered :class:`~torch_geometric.data.EdgeAttr` objects."""
        return list(self._edge_attrs.values())
