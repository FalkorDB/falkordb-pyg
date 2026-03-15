"""FalkorDB FeatureStore implementation for PyG remote backend."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data.feature_store import FeatureStore, TensorAttr, _FieldStatus


class FalkorDBFeatureStore(FeatureStore):
    """PyG FeatureStore backed by FalkorDB.

    Implements the FeatureStore interface using Cypher queries to retrieve
    node features from a FalkorDB graph. Results are cached locally after
    the first fetch to avoid repeated network round-trips.

    Args:
        graph: A FalkorDB graph object (e.g. ``db.select_graph('name')``).

    """

    def __init__(self, graph: Any) -> None:
        super().__init__()
        self._graph = graph
        self._tensor_cache: Dict[Tuple, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_tensor(self, attr: TensorAttr) -> Optional[torch.Tensor]:
        """Fetch a feature tensor from FalkorDB via Cypher."""
        group_name = attr.group_name
        attr_name = attr.attr_name

        if isinstance(group_name, tuple):
            # Edge feature: (src_label, rel_type, dst_label)
            src_label, rel_type, dst_label = group_name
            query = (
                f"MATCH (s:{src_label})-[r:{rel_type}]->(d:{dst_label}) "
                f"RETURN r.{attr_name} ORDER BY ID(r)"
            )
        else:
            # Node feature
            query = f"MATCH (n:{group_name}) RETURN n.{attr_name} ORDER BY ID(n)"

        result = self._graph.query(query)
        rows = result.result_set
        if not rows:
            return None

        # Build tensor from query results
        values = [row[0] for row in rows]
        if isinstance(values[0], (list, tuple)):
            tensor = torch.tensor(values, dtype=torch.float)
        else:
            tensor = torch.tensor(values, dtype=torch.float).unsqueeze(-1)
        return tensor

    @staticmethod
    def _to_cache_key(attr: TensorAttr) -> Tuple:
        """Convert a TensorAttr to a hashable cache key (without index)."""
        group = (
            tuple(attr.group_name)
            if isinstance(attr.group_name, (list, tuple))
            else attr.group_name
        )
        return (group, attr.attr_name)

    # ------------------------------------------------------------------
    # FeatureStore abstract method implementations
    # ------------------------------------------------------------------

    def _put_tensor(self, tensor: torch.Tensor, attr: TensorAttr) -> bool:
        """Store a tensor in the local cache."""
        key = self._to_cache_key(attr)
        self._tensor_cache[key] = tensor
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[torch.Tensor]:
        """Retrieve a feature tensor, fetching from FalkorDB if not cached."""
        key = self._to_cache_key(attr)
        if key not in self._tensor_cache:
            tensor = self._fetch_tensor(attr)
            if tensor is None:
                return None
            self._tensor_cache[key] = tensor

        full_tensor = self._tensor_cache[key]

        # Apply index if specified (skip if UNSET or None)
        index = attr.index
        if index is not None and index is not _FieldStatus.UNSET:
            return full_tensor[index]
        return full_tensor

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        """Remove a tensor from the local cache."""
        key = self._to_cache_key(attr)
        if key in self._tensor_cache:
            del self._tensor_cache[key]
            return True
        return False

    def _get_tensor_size(self, attr: TensorAttr) -> Optional[Tuple[int, ...]]:
        """Return the size of a feature tensor."""
        tensor = self._get_tensor(attr)
        if tensor is None:
            return None
        return tuple(tensor.shape)

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        """Return all tensor attributes currently held in the local cache."""
        attrs = []
        for key in self._tensor_cache:
            group_name, attr_name = key
            attr = TensorAttr(group_name=group_name, attr_name=attr_name)
            attrs.append(attr)
        return attrs
