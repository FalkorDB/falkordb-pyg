"""FalkorDB implementation of PyG's FeatureStore abstract class."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch_geometric.data.feature_store import (
    FeatureStore,
    TensorAttr,
    _FieldStatus,
)

from .utils import build_feature_query


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


def _is_unset(value: Any) -> bool:
    """Return True if *value* is PyG's ``UNSET`` sentinel or ``None``."""
    return value is None or (
        isinstance(value, _FieldStatus) and value == _FieldStatus.UNSET
    )


class FalkorDBFeatureStore(FeatureStore):
    """A PyG :class:`~torch_geometric.data.FeatureStore` backed by FalkorDB.

    Node features are fetched on first access via Cypher queries and then
    cached locally so that subsequent calls do not round-trip to the database.
    The whole column for a ``(label, property)`` pair is materialised on first
    access; see the README's *Current limitations* section.

    Args:
        graph: A ``falkordb.Graph`` instance (the result of
            ``FalkorDB(...).select_graph(name)``).
        node_type_to_label: Optional mapping from PyG node type strings to
            FalkorDB node labels.  Defaults to the identity mapping.
        dtypes: Optional mapping from ``(group_name, attr_name)`` to an
            explicit :class:`torch.dtype`, overriding inference.
    """

    def __init__(
        self,
        graph,
        node_type_to_label: Optional[Dict[str, str]] = None,
        dtypes: Optional[Dict[Tuple[str, str], torch.dtype]] = None,
    ) -> None:
        super().__init__(tensor_attr_cls=FalkorDBTensorAttr)
        self._graph = graph
        self._node_type_to_label: Dict[str, str] = node_type_to_label or {}
        self._dtypes: Dict[Tuple[str, str], torch.dtype] = dtypes or {}

        # Cache: (group_name, attr_name) -> full tensor
        self._tensor_cache: Dict[Tuple, torch.Tensor] = {}
        # Registered tensor attrs
        self._tensor_attrs: Dict[Tuple, FalkorDBTensorAttr] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label(self, group_name: Union[str, Tuple]) -> str:
        """Resolve a PyG group name (node type) to a FalkorDB label.

        Only node types are supported.  Edge features would arrive here as a
        ``(src, rel, dst)`` triple; those are rejected rather than silently
        answered with source-node data.
        """
        if isinstance(group_name, tuple):
            raise NotImplementedError(
                "FalkorDBFeatureStore does not support edge features. "
                f"Expected a node type string, got the edge type {group_name!r}."
            )
        return self._node_type_to_label.get(group_name, group_name)

    def _cache_key(self, attr: TensorAttr) -> Tuple:
        return (attr.group_name, attr.attr_name)

    def _rows_to_tensor(
        self,
        rows: Sequence[Sequence[Any]],
        label: str,
        prop: str,
        group_name: Optional[str] = None,
    ) -> torch.Tensor:
        """Convert FalkorDB result rows into a 2-D feature tensor.

        Each row is ``[property_value, node_id]``.  Property values may be
        scalars or lists (multi-dimensional features).  ``node_id`` is used
        only to point at the offending node when validation fails.

        ``label`` names the FalkorDB label for error messages; ``group_name``
        is the PyG node type and is what ``dtypes`` is keyed by.
        """
        where = f"property '{prop}' on :{label}"
        dtype_key = (group_name if group_name is not None else label, prop)

        values = [row[0] for row in rows]
        node_ids = [row[1] if len(row) > 1 else None for row in rows]

        missing = [nid for val, nid in zip(values, node_ids) if val is None]
        if missing:
            raise ValueError(
                f"{where} is missing (NULL) on {len(missing)} of {len(values)} "
                f"nodes, first at ID(n)={missing[0]}. Every node carrying the "
                f"label must define the property; filter the label, backfill "
                f"the property, or use COALESCE in a custom query."
            )

        first = values[0]

        if isinstance(first, (list, tuple)):
            width = len(first)
            for val, nid in zip(values, node_ids):
                if not isinstance(val, (list, tuple)):
                    raise ValueError(
                        f"{where} is a vector on some nodes and a scalar on "
                        f"others (scalar at ID(n)={nid})."
                    )
                if len(val) != width:
                    raise ValueError(
                        f"{where} has inconsistent length: {width} at "
                        f"ID(n)={node_ids[0]} but {len(val)} at ID(n)={nid}. "
                        f"Feature vectors must all be the same length."
                    )
                if any(isinstance(v, (list, tuple)) for v in val):
                    raise ValueError(
                        f"{where} is nested more than one level deep at "
                        f"ID(n)={nid}; only flat numeric vectors are supported."
                    )
                if any(
                    not isinstance(v, (int, float)) or isinstance(v, bool) for v in val
                ):
                    raise ValueError(
                        f"{where} contains a non-numeric element at "
                        f"ID(n)={nid}; only numeric feature vectors are supported."
                    )
            # Vectors are model inputs: always float, regardless of whether
            # FalkorDB returned them as integers.
            dtype = self._dtypes.get(dtype_key, torch.float)
            return torch.tensor(values, dtype=dtype)

        for val, nid in zip(values, node_ids):
            if isinstance(val, (list, tuple)):
                raise ValueError(
                    f"{where} is a scalar on some nodes and a vector on "
                    f"others (vector at ID(n)={nid})."
                )
            if not isinstance(val, (int, float, bool)):
                raise ValueError(
                    f"{where} has non-numeric type {type(val).__name__} at "
                    f"ID(n)={nid} (value {val!r}). Only numeric properties can "
                    f"be converted to tensors; encode text features before "
                    f"storing them."
                )

        # Scalars carry labels, timestamps and masks, where the Python type is
        # meaningful: preserve it instead of flattening everything to float32.
        override = self._dtypes.get(dtype_key)
        if override is not None:
            scalar_dtype = override
        elif all(isinstance(v, bool) for v in values):
            scalar_dtype = torch.bool
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            scalar_dtype = torch.long
        else:
            scalar_dtype = torch.float
        return torch.tensor(values, dtype=scalar_dtype).unsqueeze(1)

    def _fetch_tensor(self, attr: TensorAttr) -> torch.Tensor:
        """Query FalkorDB and return the full feature tensor for *attr*."""
        label = self._label(attr.group_name)
        prop = attr.attr_name
        result = self._graph.query(build_feature_query(label, prop))

        rows = result.result_set
        if not rows:
            warnings.warn(
                f"No nodes matched :{label} when fetching property '{prop}'. "
                f"Returning an empty tensor — check the label spelling and the "
                f"node_type_to_label mapping.",
                stacklevel=2,
            )
            return torch.zeros((0, 0))

        return self._rows_to_tensor(rows, label, prop, attr.group_name)

    def _register(self, attr: TensorAttr) -> None:
        """Record *attr* in the registry if it is not already known."""
        key = self._cache_key(attr)
        if key not in self._tensor_attrs:
            self._tensor_attrs[key] = FalkorDBTensorAttr(
                group_name=attr.group_name,
                attr_name=attr.attr_name,
                index=None,
            )

    # ------------------------------------------------------------------
    # Public cache management
    # ------------------------------------------------------------------

    def clear_cache(
        self,
        group_name: Optional[str] = None,
        attr_name: Optional[str] = None,
    ) -> None:
        """Drop cached tensors so the next access re-reads from FalkorDB.

        With no arguments the whole cache is dropped.  The store does not
        observe writes made to the graph after a tensor has been cached, so
        call this when the underlying data has changed.
        """
        if group_name is None and attr_name is None:
            self._tensor_cache.clear()
            return
        for key in list(self._tensor_cache):
            if group_name is not None and key[0] != group_name:
                continue
            if attr_name is not None and key[1] != attr_name:
                continue
            del self._tensor_cache[key]

    # ------------------------------------------------------------------
    # FeatureStore abstract method implementations
    # ------------------------------------------------------------------

    def _put_tensor(self, tensor: torch.Tensor, attr: TensorAttr) -> bool:
        """Store a tensor in the local cache (does not write to DB)."""
        key = self._cache_key(attr)
        self._tensor_cache[key] = tensor.clone()
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
            self._tensor_cache[key] = self._fetch_tensor(attr)
            self._register(attr)

        full_tensor = self._tensor_cache[key]

        # Always hand out a copy: callers (and PyG transforms) mutate tensors
        # in place, which would otherwise silently corrupt the cache.
        if _is_unset(attr.index):
            return full_tensor.clone()
        return full_tensor[attr.index].clone()

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        """Evict a cached tensor.

        This is cache eviction, not a database delete: a subsequent get will
        re-read the property from FalkorDB.
        """
        key = self._cache_key(attr)
        existed = key in self._tensor_cache
        self._tensor_cache.pop(key, None)
        self._tensor_attrs.pop(key, None)
        return existed

    def _get_tensor_size(self, attr: TensorAttr) -> Optional[Tuple[int, ...]]:
        """Return the full size of the tensor for *attr*.

        ``attr.index`` is deliberately ignored: this reports the size of the
        stored feature matrix, not of a particular slice of it.
        """
        key = self._cache_key(attr)
        if key not in self._tensor_cache:
            full_attr = FalkorDBTensorAttr(
                group_name=attr.group_name,
                attr_name=attr.attr_name,
                index=None,
            )
            self._tensor_cache[key] = self._fetch_tensor(full_attr)
            self._register(full_attr)
        # Read the shape off the cached tensor directly: going through
        # _get_tensor would clone the whole matrix just to report its size.
        return tuple(self._tensor_cache[key].shape)

    def get_all_tensor_attrs(self) -> List[FalkorDBTensorAttr]:
        """Return all registered :class:`FalkorDBTensorAttr` objects.

        Copies are returned so that callers mutating ``.index`` — which PyG
        does while assembling a batch — cannot corrupt the registry.
        """
        return [copy.copy(attr) for attr in self._tensor_attrs.values()]
