"""FalkorDB implementation of PyG's GraphStore abstract class."""

from __future__ import annotations

import copy
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout, GraphStore

from .utils import NodeIDMapper, build_edge_query, build_node_ids_query

EdgeType = Tuple[str, str, str]
EdgeTensors = Tuple[torch.Tensor, torch.Tensor]


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
        edge_type_to_rel: Optional[Dict[EdgeType, str]] = None,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._node_type_to_label: Dict[str, str] = node_type_to_label or {}
        self._edge_type_to_rel: Dict[EdgeType, str] = edge_type_to_rel or {}

        # Cache: (edge_type, layout) -> (row_tensor, col_tensor)
        self._edge_index_cache: Dict[Tuple[EdgeType, EdgeLayout], EdgeTensors] = {}
        # Cache: node_type -> NodeIDMapper
        self._id_mappers: Dict[str, NodeIDMapper] = {}
        # Registered edge attrs (populated by put_edge_index or discovered lazily)
        self._edge_attrs: Dict[EdgeType, EdgeAttr] = {}
        # Per-edge-type count of edges dropped during ID remapping
        self.dropped_edges: Dict[EdgeType, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label(self, node_type: str) -> str:
        """Resolve a PyG node type to a FalkorDB label."""
        return self._node_type_to_label.get(node_type, node_type)

    def _rel_type(self, edge_type: EdgeType) -> str:
        """Resolve a PyG edge type triple to a FalkorDB relationship type."""
        return self._edge_type_to_rel.get(edge_type, edge_type[1])

    def _get_or_build_mapper(self, node_type: str) -> NodeIDMapper:
        """Return (and cache) the NodeIDMapper for *node_type*."""
        if node_type not in self._id_mappers:
            label = self._label(node_type)
            result = self._graph.query(build_node_ids_query(label))
            ids = [int(row[0]) for row in result.result_set]
            self._id_mappers[node_type] = NodeIDMapper(ids)
        return self._id_mappers[node_type]

    def id_mapper(self, node_type: str) -> NodeIDMapper:
        """Return the :class:`NodeIDMapper` for *node_type*, building it lazily.

        Use this to translate between FalkorDB internal node IDs and the
        contiguous PyG indices the model sees.
        """
        return self._get_or_build_mapper(node_type)

    def _fetch_edge_index(self, edge_type: EdgeType) -> EdgeTensors:
        """Query FalkorDB and return a COO edge index remapped to PyG indices."""
        src_type, _, dst_type = edge_type
        rel = self._rel_type(edge_type)
        src_label = self._label(src_type)
        dst_label = self._label(dst_type)

        result = self._graph.query(build_edge_query(src_label, rel, dst_label))

        src_mapper = self._get_or_build_mapper(src_type)
        dst_mapper = self._get_or_build_mapper(dst_type)

        new_src, new_dst = [], []
        total = 0
        for row in result.result_set:
            total += 1
            ps = src_mapper.falkor_to_pyg(int(row[0]))
            pd = dst_mapper.falkor_to_pyg(int(row[1]))
            if ps is not None and pd is not None:
                new_src.append(ps)
                new_dst.append(pd)

        dropped = total - len(new_src)
        self.dropped_edges[edge_type] = dropped
        if dropped:
            warnings.warn(
                f"{dropped} of {total} {edge_type} edges referenced nodes "
                f"outside the :{src_label}/:{dst_label} ID maps and were "
                f"dropped. This usually means the relationship also connects "
                f"other labels than the ones in the edge type.",
                stacklevel=2,
            )
        elif total == 0:
            warnings.warn(
                f"No edges matched {edge_type} "
                f"(:{src_label})-[:{rel}]->(:{dst_label}). Registering an empty "
                f"edge type — check the relationship type and label spellings.",
                stacklevel=2,
            )

        src_t = torch.tensor(new_src, dtype=torch.long)
        dst_t = torch.tensor(new_dst, dtype=torch.long)
        return src_t, dst_t

    def _size_for(self, edge_type: EdgeType) -> Tuple[int, int]:
        """Return ``(num_src_nodes, num_dst_nodes)`` for *edge_type*."""
        src_type, _, dst_type = edge_type
        return (
            self._get_or_build_mapper(src_type).num_nodes,
            self._get_or_build_mapper(dst_type).num_nodes,
        )

    # ------------------------------------------------------------------
    # Public cache management
    # ------------------------------------------------------------------

    def clear_cache(self, edge_type: Optional[EdgeType] = None) -> None:
        """Drop cached topology so the next access re-reads from FalkorDB.

        With no argument this also drops the node ID mappers, since a write
        that adds or removes nodes invalidates the FalkorDB-ID to PyG-index
        assignment for every type.  The store does not otherwise observe
        writes made after a value has been cached.
        """
        if edge_type is None:
            self._edge_index_cache.clear()
            self._id_mappers.clear()
            self.dropped_edges.clear()
            return
        for key in [k for k in self._edge_index_cache if k[0] == edge_type]:
            del self._edge_index_cache[key]
        self.dropped_edges.pop(edge_type, None)

    # ------------------------------------------------------------------
    # GraphStore abstract method implementations
    # ------------------------------------------------------------------

    def _put_edge_index(self, edge_index: EdgeTensors, edge_attr: EdgeAttr) -> bool:
        """Store an edge index in the local cache (does not write to DB).

        Converted layouts (CSR/CSC) are accepted so that PyG's own
        ``csc(store=True)`` / ``csr(store=True)`` conversion caching works.
        """
        et = edge_attr.edge_type
        self._edge_index_cache[(et, edge_attr.layout)] = (edge_index[0], edge_index[1])

        attr = copy.copy(edge_attr)
        if attr.size is None:
            # A missing size truncates the CSC colptr, silently dropping
            # trailing nodes that have no outgoing edges.
            attr.size = self._size_for(et)
        # Keep the canonical COO attr in the registry; conversions are
        # bookkeeping, not new edge types.
        if attr.layout == EdgeLayout.COO or et not in self._edge_attrs:
            self._edge_attrs[et] = attr
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensors]:
        """Return a COO edge index, fetching from FalkorDB if not cached."""
        et = edge_attr.edge_type
        layout = edge_attr.layout

        cached = self._edge_index_cache.get((et, layout))
        if cached is not None:
            return cached

        if layout != EdgeLayout.COO:
            # Never answer a CSR/CSC request with COO tensors — that is
            # silently misinterpreted topology.  Returning None lets PyG's
            # base class perform (or report the absence of) a conversion.
            return None

        src_t, dst_t = self._fetch_edge_index(et)
        self._edge_index_cache[(et, EdgeLayout.COO)] = (src_t, dst_t)
        if et not in self._edge_attrs:
            self._edge_attrs[et] = EdgeAttr(
                edge_type=et,
                layout=EdgeLayout.COO,
                is_sorted=False,
                size=self._size_for(et),
            )
        return self._edge_index_cache[(et, EdgeLayout.COO)]

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        """Evict a cached edge index.

        This is cache eviction, not a database delete: a subsequent get will
        re-read the relationship from FalkorDB.
        """
        et = edge_attr.edge_type
        existed = any(key[0] == et for key in self._edge_index_cache)
        for key in [k for k in self._edge_index_cache if k[0] == et]:
            del self._edge_index_cache[key]
        self._edge_attrs.pop(et, None)
        self.dropped_edges.pop(et, None)
        return existed

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        """Return all registered :class:`~torch_geometric.data.EdgeAttr` objects.

        Copies are returned so that callers mutating them cannot corrupt the
        registry.
        """
        return [copy.copy(attr) for attr in self._edge_attrs.values()]
