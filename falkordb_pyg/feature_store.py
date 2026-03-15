"""FalkorDB implementation of PyG's FeatureStore abstract class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch_geometric.data.feature_store import (
    FeatureStore,
    TensorAttr,
    _FieldStatus,
)

from .utils import build_feature_query, build_node_ids_query


@dataclass
class FalkorDBTensorAttr(TensorAttr):
    """A :class:`TensorAttr` that defaults ``index`` to ``None`` instead of
    :data:`UNSET`, matching the convention used by most remote backends."""

    def __init__(
        self,
        group_name: Optional[Any] = _FieldStatus.UNSET,
        attr_name: Optional[str] = _FieldStatus.UNSET,
        index: Optional[Any] = None,
    ) -> None:
        super().__init__(
            group_name=group_name,
            attr_name=attr_name,
            index=index,
        )


class FalkorDBFeatureStore(FeatureStore):
    """A PyG :class:`~torch_geometric.data.FeatureStore` backed by FalkorDB.

    Node features are fetched on first access via Cypher queries and then
    cached locally so that subsequent calls do not round-trip to the database.

    Args:
        graph: A ``falkordb.Graph`` instance (the result of
            ``FalkorDB(...).select_graph(name)``).
        node_type_to_label: Optional mapping from PyG node type strings to
            FalkorDB node labels.  Defaults to the identity mapping.
    """

    def __init__(
        self,
        graph,
        node_type_to_label: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._node_type_to_label: Dict[str, str] = node_type_to_label or {}

        # Cache: (group_name, attr_name) -> full tensor
        self._tensor_cache: Dict[Tuple, torch.Tensor] = {}
        # Registered tensor attrs
        self._tensor_attrs: Dict[Tuple, FalkorDBTensorAttr] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label(self, group_name: Union[str, Tuple]) -> str:
        """Resolve a PyG group name (node type) to a FalkorDB label.

        For heterogeneous node types the group name is a plain string.
        For edge features the group name is a tuple — we use its first element.
        """
        if isinstance(group_name, tuple):
            key = group_name[0]
        else:
            key = group_name
        return self._node_type_to_label.get(key, key)

    def _cache_key(self, attr: TensorAttr) -> Tuple:
        return (attr.group_name, attr.attr_name)

    def _fetch_tensor(self, attr: TensorAttr) -> torch.Tensor:
        """Query FalkorDB and return the full feature tensor for *attr*."""
        label = self._label(attr.group_name)
        prop = attr.attr_name
        query = build_feature_query(label, prop)
        result = self._graph.query(query)

        rows = result.result_set
        if not rows:
            return torch.zeros(0)

        # Each row is [property_value, node_id].  Property values may be
        # scalars or lists (multi-dimensional features).
        values = [row[0] for row in rows]

        first = values[0]
        if isinstance(first, (list, tuple)):
            tensor = torch.tensor(values, dtype=torch.float)
        else:
            tensor = torch.tensor(values, dtype=torch.float).unsqueeze(1)

        return tensor

    # ------------------------------------------------------------------
    # FeatureStore abstract method implementations
    # ------------------------------------------------------------------

    def _put_tensor(self, tensor: torch.Tensor, attr: TensorAttr) -> bool:
        """Store a tensor in the local cache (does not write to DB)."""
        key = self._cache_key(attr)
        self._tensor_cache[key] = tensor
        self._tensor_attrs[key] = FalkorDBTensorAttr(
            group_name=attr.group_name,
            attr_name=attr.attr_name,
            index=None,
        )
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[torch.Tensor]:
        """Return the tensor for *attr*, fetching from FalkorDB if not cached."""
        key = self._cache_key(attr)
        if key not in self._tensor_cache:
            tensor = self._fetch_tensor(attr)
            self._tensor_cache[key] = tensor
            # Auto-register the attr
            if key not in self._tensor_attrs:
                self._tensor_attrs[key] = FalkorDBTensorAttr(
                    group_name=attr.group_name,
                    attr_name=attr.attr_name,
                    index=None,
                )

        full_tensor = self._tensor_cache[key]

        index = attr.index
        if index is None or (
            isinstance(index, _FieldStatus) and index == _FieldStatus.UNSET
        ):
            return full_tensor
        return full_tensor[index]

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        """Remove a cached tensor."""
        key = self._cache_key(attr)
        existed = key in self._tensor_cache
        self._tensor_cache.pop(key, None)
        self._tensor_attrs.pop(key, None)
        return existed

    def _get_tensor_size(self, attr: TensorAttr) -> Optional[Tuple[int, ...]]:
        """Return the size of the tensor for *attr*."""
        tensor = self._get_tensor(attr)
        if tensor is None:
            return None
        return tuple(tensor.shape)

    def get_all_tensor_attrs(self) -> List[FalkorDBTensorAttr]:
        """Return all registered :class:`FalkorDBTensorAttr` objects."""
        return list(self._tensor_attrs.values())
