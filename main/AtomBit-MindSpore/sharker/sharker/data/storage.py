import copy
import warnings
import weakref
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from typing_extensions import Self

import numpy as np
import mindspore as ms
from mindspore import ops, mint

from .view import ItemsView, KeysView, ValuesView
from ..utils.coalesce import coalesce, coalesce_np
from ..utils.undirected import is_undirected
from ..utils.select import select
from ..utils.sort_edge_index import sort_edge_index, sort_edge_index_np
from ..utils.isolated import contains_isolated_nodes


N_KEYS = {"x", "feat", "pos", "batch", "node_type", "n_id", "tf"}
E_KEYS = {"edge_index", "edge_weight", "edge_attr", "edge_type", "e_id"}


class AttrType(Enum):
    NODE = "NODE"
    EDGE = "EDGE"
    OTHER = "OTHER"


class BaseStorage(MutableMapping):
    # This class wraps a Python dictionary and extends it as follows:
    # 1. It allows attribute assignments, e.g.:
    #    `storage.x = ...` in addition to `storage['x'] = ...`
    # 2. It allows private attributes that are not exposed to the user, e.g.:
    #    `storage._{key} = ...` and accessible via `storage._{key}`
    # 3. It holds an (optional) weak reference to its parent object, e.g.:
    #    `storage._parent = weakref.ref(parent)`
    # 4. It allows iterating over only a subset of keys, e.g.:
    #    `storage.values('x', 'y')` or `storage.items('x', 'y')
    # 5. It adds additional Mindspore Tensor functionality, e.g.:
    #    `storage.numpy()`, `storage.tensor()` or `storage.share_memory_()`.
    def __init__(
        self,
        _mapping: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._mapping: Dict[str, Any] = {}
        for key, value in (_mapping or {}).items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def _key(self) -> Any:
        return None

    def _pop_cache(self, key: str) -> None:
        for cache in getattr(self, "_cached_attr", {}).values():
            cache.discard(key)

    def __len__(self) -> int:
        return len(self._mapping)

    def __getattr__(self, key: str) -> Any:
        if key == "_mapping":
            self._mapping = {}
            return self._mapping
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, "fset", None) is not None:
            propobj.fset(self, value)
        elif key == "_parent":
            self.__dict__[key] = weakref.ref(value)
        elif key[:1] == "_":
            self.__dict__[key] = value
        else:
            self[key] = value

    def __delattr__(self, key: str) -> None:
        if key[:1] == "_":
            del self.__dict__[key]
        else:
            del self[key]

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._pop_cache(key)
        if value is None and key in self._mapping:
            del self._mapping[key]
        elif value is not None:
            self._mapping[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._mapping:
            self._pop_cache(key)
            del self._mapping[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._mapping)

    def __copy__(self) -> Self:
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if key != "_cached_attr":
                out.__dict__[key] = value
        out._mapping = copy.copy(out._mapping)
        return out

    def __deepcopy__(self, memo: Optional[Dict[int, Any]]) -> Self:
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.deepcopy(out._mapping, memo)
        return out

    def __getstate__(self) -> Dict[str, Any]:
        out = self.__dict__.copy()

        _parent = out.get("_parent", None)
        if _parent is not None:
            out["_parent"] = _parent()

        return out

    def __setstate__(self, mapping: Dict[str, Any]) -> None:
        for key, value in mapping.items():
            self.__dict__[key] = value

        _parent = self.__dict__.get("_parent", None)
        if _parent is not None:
            self.__dict__["_parent"] = weakref.ref(_parent)

    def __repr__(self) -> str:
        return repr(self._mapping)

    # Allow iterating over subsets ############################################

    # In contrast to standard `keys()`, `values()` and `items()` functions of
    # Python dictionaries, we allow to only iterate over a subset of items
    # denoted by a list of keys `args`.
    # This is especially useful for adding MindSpore Tensor functionality to the
    # storage object, e.g., in case we only want to transfer a subset of keys
    # to the GPU (i.e. the ones that are relevant to the deep learning model).

    def keys(self, *args: str) -> KeysView:  # type: ignore
        return KeysView(self._mapping, *args)

    def values(self, *args: str) -> ValuesView:  # type: ignore
        return ValuesView(self._mapping, *args)

    def items(self, *args: str) -> ItemsView:  # type: ignore
        return ItemsView(self._mapping, *args)

    def apply(self, func: Callable, *args: str) -> Self:
        r"""Applies the function :obj:`func`, either to all attributes or only
        the ones given in :obj:`*args`.
        """
        for key, value in self.items(*args):
            self[key] = recursive_apply(value, func)
        return self

    # Additional functionality ################################################

    def get(self, key: str, value: Optional[Any] = None) -> Any:
        return self._mapping.get(key, value)

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        out_dict = copy.copy(self._mapping)
        # Needed to preserve individual `num_nodes` attributes when calling
        # `BaseData.collate`.
        # TODO (matthias) Try to make this more generic.
        if "_num_nodes" in self.__dict__:
            out_dict["_num_nodes"] = self.__dict__["_num_nodes"]
        return out_dict

    def copy(self, *args: str) -> Self:
        r"""Performs a deep-copy of the object."""
        return copy.deepcopy(self)

    def numpy(self, *args: str) -> Self:
        r"""Copies attributes to CPU memory, either for all attributes or only
        the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.asnumpy(), *args)

    def tensor(self, *args: str) -> Self:
        r"""Copies attributes to CPU memory, either for all attributes or only
        the ones given in :obj:`*args`.
        """
        return self.apply(
            lambda x: Tensor.from_numpy(x) if isinstance(x, np.ndarray) else x, *args
        )

    # Time Handling ###########################################################

    def _cat_dims(self, keys: Iterable[str]) -> Dict[str, int]:
        return {key: self._parent().__cat_dim__(key, self[key], self) for key in keys}

    def _select(
        self,
        keys: Iterable[str],
        index_or_mask: ms.Tensor,
    ) -> Self:

        for key, dim in self._cat_dims(keys).items():
            self[key] = select(self[key], index_or_mask, dim)

        return self

    def concat(self, other: Self) -> Self:
        if not (set(self.keys()) == set(other.keys())):
            raise AttributeError("Given storage is not compatible")

        for key, dim in self._cat_dims(self.keys()).items():
            value1 = self[key]
            value2 = other[key]

            if key in {"num_nodes", "num_edges"}:
                self[key] = value1 + value2

            elif isinstance(value1, list):
                self[key] = value1 + value2

            elif isinstance(value1, np.ndarray):
                self[key] = np.concatenate([value1, value2], axis=dim)

            elif isinstance(value1, ms.Tensor):
                self[key] = mint.cat([value1, value2], dim=dim)

            else:
                raise NotImplementedError(
                    f"'{self.__class__.__name__}.concat' not yet implemented "
                    f"for '{type(value1)}'"
                )

        return self

    def is_sorted_by_time(self) -> bool:
        if "time" in self:
            return bool(np.all(self.time[:-1] <= self.time[1:]))
        return True

    def sort_by_time(self) -> Self:
        if self.is_sorted_by_time():
            return self

        if "time" in self:
            perm = np.argsort(self.time)

            if self.is_node_attr("time"):
                keys = self.node_attrs()
            elif self.is_edge_attr("time"):
                keys = self.edge_attrs()

            self._select(keys, perm)

        return self

    def snapshot(
        self,
        start_time: Union[float, int],
        end_time: Union[float, int],
    ) -> Self:
        if "time" in self:
            mask = np.logical_and((self.time >= start_time), (self.time <= end_time))

            if self.is_node_attr("time"):
                keys = self.node_attrs()
            elif self.is_edge_attr("time"):
                keys = self.edge_attrs()

            self._select(keys, mask)

            if self.is_node_attr("time") and "num_nodes" in self:
                self.num_nodes: Optional[int] = int(mask.sum())

        return self

    def up_to(self, time: Union[float, int]) -> Self:
        if "time" in self:
            return self.snapshot(self.time.min().item(), time)
        return self


class NodeStorage(BaseStorage):
    r"""A storage for node-level information."""

    @property
    def _key(self) -> str:
        key = self.__dict__.get("_key", None)
        if key is None or not isinstance(key, str):
            raise ValueError("'_key' does not denote a valid node type")
        return key

    @property
    def can_infer_num_nodes(self) -> bool:
        keys = set(self.keys())
        num_node_keys = {
            "num_nodes",
            "x",
            "crd",
            "batch",
            "adj",
            "adj_t",
            "edge_index",
            "face",
        }
        if len(keys & num_node_keys) > 0:
            return True
        elif len([key for key in keys if "node" in key]) > 0:
            return True
        else:
            return False

    @property
    def num_nodes(self) -> Optional[int]:
        # We sequentially access attributes that reveal the number of nodes.
        if "num_nodes" in self:
            return self["num_nodes"]
        for key, value in self.items():
            if isinstance(value, ms.Tensor) and key in N_KEYS:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
            if isinstance(value, np.ndarray) and key in N_KEYS:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
        for key, value in self.items():
            if isinstance(value, ms.Tensor) and "node" in key:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
            if isinstance(value, np.ndarray) and "node" in key:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
        warnings.warn(
            f"Unable to accurately infer 'num_nodes' from the attribute set "
            f"'{set(self.keys())}'. Please explicitly set 'num_nodes' as an "
            f"attribute of "
            + ("'data'" if self._key is None else f"'data[{self._key}]'")
            + " to suppress this warning"
        )
        if "edge_index" in self and isinstance(self.edge_index, ms.Tensor):
            if self.edge_index.numel() > 0:
                return int(mint.max(self.edge_index)) + 1
            return 0
        if "face" in self and isinstance(self.face, ms.Tensor):
            if self.face.numel() > 0:
                return int(mint.max(self.face)) + 1
            return 0
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes: Optional[int]) -> None:
        self["num_nodes"] = num_nodes

    @property
    def num_node_features(self) -> int:
        x: Optional[Any] = self.get("x")
        if isinstance(x, ms.Tensor):
            return 1 if x.dim() == 1 else x.shape[-1]
        if isinstance(x, np.ndarray):
            return 1 if x.ndim == 1 else x.shape[-1]
        return 0

    @property
    def num_features(self) -> int:
        return self.num_node_features

    def is_node_attr(self, key: str) -> bool:
        if "_cached_attr" not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.NODE]:
            return True
        if key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self[key]

        if not isinstance(value, (ms.Tensor, np.ndarray)):
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        if value.ndim == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        cat_dim = self._parent().__cat_dim__(key, value, self)
        if value.shape[cat_dim] != self.num_nodes:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        self._cached_attr[AttrType.NODE].add(key)
        return True

    def is_edge_attr(self, key: str) -> bool:
        return False

    def node_attrs(self) -> List[str]:
        return [key for key in self.keys() if self.is_node_attr(key)]


class EdgeStorage(BaseStorage):
    r"""A storage for edge-level information.

    We support multiple ways to store edge connectivity in a
    :class:`EdgeStorage` object:

    * :obj:`edge_index`: A :class:`ms.Tensor` holding edge indices in
      COO format with shape :obj:`[2, num_edges]` (the default format)

    * :obj:`adj`: A :class:`mindspore.SparseTensor` holding edge indices in
      a sparse format, supporting both COO and CSR format.

    * :obj:`adj_t`: A **transposed** :class:`mindspore.SparseTensor` holding
      edge indices in a sparse format, supporting both COO and CSR format.
      This is the most efficient one for graph-based deep learning models as
      indices are sorted based on target nodes.
    """

    @property
    def _key(self) -> Tuple[str, str, str]:
        key = self.__dict__.get("_key", None)
        if key is None or not isinstance(key, tuple) or not len(key) == 3:
            raise ValueError("'_key' does not denote a valid edge type")
        return key

    @property
    def edge_index(self) -> ms.Tensor:
        if "edge_index" in self:
            return self["edge_index"]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute "
            f"'edge_index', 'adj' or 'adj_t'"
        )

    @edge_index.setter
    def edge_index(self, edge_index: Optional[ms.Tensor]) -> None:
        self["edge_index"] = edge_index

    @property
    def num_edges(self) -> int:
        # We sequentially access attributes that reveal the number of edges.
        if "num_edges" in self:
            return self["num_edges"]
        for key, value in self.items():
            if isinstance(value, ms.Tensor) and key in E_KEYS:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
            if isinstance(value, np.ndarray) and key in E_KEYS:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
        for key, value in self.items():
            if isinstance(value, ms.Tensor) and "edge" in key:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
            if isinstance(value, np.ndarray) and "edge" in key:
                cat_dim = self._parent().__cat_dim__(key, value, self)
                return value.shape[cat_dim]
        return 0

    @property
    def num_edge_features(self) -> int:
        edge_attr: Optional[Any] = self.get("edge_attr")
        if isinstance(edge_attr, ms.Tensor):
            return 1 if edge_attr.dim() == 1 else edge_attr.shape[-1]
        if isinstance(edge_attr, np.ndarray):
            return 1 if edge_attr.ndim == 1 else edge_attr.shape[-1]
        return 0

    @property
    def num_features(self) -> int:
        return self.num_edge_features

    @property
    def shape(self) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:

        if self._key is None:
            raise NameError(
                "Unable to infer 'size' without explicit " "'_key' assignment"
            )

        size = (
            self._parent()[self._key[0]].num_nodes,
            self._parent()[self._key[-1]].num_nodes,
        )

        return size

    def is_node_attr(self, key: str) -> bool:
        return False

    def is_edge_attr(self, key: str) -> bool:
        if "_cached_attr" not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.EDGE]:
            return True
        if key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self[key]

        if not isinstance(value, (ms.Tensor, np.ndarray)):
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        if value.ndim == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        cat_dim = self._parent().__cat_dim__(key, value, self)
        if value.shape[cat_dim] != self.num_edges:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        self._cached_attr[AttrType.EDGE].add(key)
        return True

    def edge_attrs(self) -> List[str]:
        return [key for key in self.keys() if self.is_edge_attr(key)]

    def is_sorted(self, sort_by_row: bool = True) -> bool:
        if "edge_index" in self:
            index = self.edge_index[0] if sort_by_row else self.edge_index[1]
            return bool(np.all(index[:-1] <= index[1:]))
        return True 

    def sort(self, sort_by_row: bool = True) -> Self:
        if "edge_index" in self:
            edge_attrs = self.edge_attrs()
            edge_attrs.remove("edge_index")
            edge_feats = [self[edge_attr] for edge_attr in edge_attrs]
            self.edge_index, edge_feats = sort_edge_index_np(
                self.edge_index, edge_feats, sort_by_row=sort_by_row
            )
            for key, edge_feat in zip(edge_attrs, edge_feats):
                self[key] = edge_feat
        return self

    def is_coalesced(self) -> bool:
        for value in self.values("adj", "adj_t"):
            return value.is_coalesced()

        if "edge_index" in self:
            size = [s for s in self.shape if s is not None]
            num_nodes = max(size) if len(size) > 0 else None
            new_edge_index = coalesce_np(self.edge_index, num_nodes=num_nodes)
            return self.edge_index.size == new_edge_index.size and np.all(
                self.edge_index == new_edge_index
            )

        return True

    def coalesce(self, reduce: str = "sum") -> Self:
        for key, value in self.items("adj", "adj_t"):
            self[key] = value.coalesce(reduce)

        if "edge_index" in self:

            size = [s for s in self.shape if s is not None]
            num_nodes = max(size) if len(size) > 0 else None

            self.edge_index, self.edge_attr = coalesce_np(
                self.edge_index,
                edge_attr=self.get("edge_attr"),
                num_nodes=num_nodes,
            )

        return self

    def has_isolated_nodes(self) -> bool:
        edge_index, num_nodes = self.edge_index, self.shape[1]
        if num_nodes is None:
            raise NameError("Unable to infer 'num_nodes'")
        if self.is_bipartite():
            return np.unique(edge_index[1]).size < num_nodes
        else:
            return contains_isolated_nodes(edge_index, num_nodes)

    def has_self_loops(self) -> bool:
        if self.is_bipartite():
            return False
        edge_index = self.edge_index
        return int((edge_index[0] == edge_index[1]).sum()) > 0

    def is_undirected(self) -> bool:
        if self.is_bipartite():
            return False

        for value in self.values("adj", "adj_t"):
            return value.is_symmetric()

        edge_index = self.edge_index
        edge_attr = self.edge_attr if "edge_attr" in self else None
        return is_undirected(edge_index, edge_attr, num_nodes=self.shape[0])

    def is_directed(self) -> bool:
        return not self.is_undirected()

    def is_bipartite(self) -> bool:
        return self._key is not None and self._key[0] != self._key[-1]


class GlobalStorage(NodeStorage, EdgeStorage):
    r"""A storage for both node-level and edge-level information."""

    @property
    def _key(self) -> Any:
        return None

    @property
    def num_features(self) -> int:
        return self.num_node_features

    @property
    def shape(self) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        size = (self.num_nodes, self.num_nodes)
        return size

    def is_node_attr(self, key: str) -> bool:
        if "_cached_attr" not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.NODE]:
            return True
        if key in self._cached_attr[AttrType.EDGE]:
            return False
        if key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self[key]

        if (isinstance(value, (list, tuple))
                and len(value) == self.num_nodes):
            self._cached_attr[AttrType.NODE].add(key)
            return True

        if not isinstance(value, (ms.Tensor, np.ndarray)):
            return False

        if value.ndim == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        cat_dim = self._parent().__cat_dim__(key, value, self)
        num_nodes, num_edges = self.num_nodes, self.num_edges

        if value.shape[cat_dim] != num_nodes:
            if value.shape[cat_dim] == num_edges:
                self._cached_attr[AttrType.EDGE].add(key)
            else:
                self._cached_attr[AttrType.OTHER].add(key)
            return False

        if num_nodes != num_edges:
            self._cached_attr[AttrType.NODE].add(key)
            return True

        if "edge" not in key:
            self._cached_attr[AttrType.NODE].add(key)
            return True
        else:
            self._cached_attr[AttrType.EDGE].add(key)
            return False

    def is_edge_attr(self, key: str) -> bool:
        if "_cached_attr" not in self.__dict__:
            self._cached_attr = defaultdict(set)

        if key in self._cached_attr[AttrType.EDGE]:
            return True
        if key in self._cached_attr[AttrType.NODE]:
            return False
        if key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self[key]

        if not isinstance(value, (ms.Tensor, np.ndarray)):
            return False

        if value.ndim == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        cat_dim = self._parent().__cat_dim__(key, value, self)
        num_nodes, num_edges = self.num_nodes, self.num_edges

        if value.shape[cat_dim] != num_edges:
            if value.shape[cat_dim] == num_nodes:
                self._cached_attr[AttrType.NODE].add(key)
            else:
                self._cached_attr[AttrType.OTHER].add(key)
            return False

        if num_edges != num_nodes:
            self._cached_attr[AttrType.EDGE].add(key)
            return True

        if "edge" in key:
            self._cached_attr[AttrType.EDGE].add(key)
            return True
        else:
            self._cached_attr[AttrType.NODE].add(key)
            return False


def recursive_apply(data: Any, func: Callable) -> Any:
    if isinstance(data, ms.Tensor):
        return func(data)
    elif isinstance(data, tuple) and hasattr(data, "_fields"):
        return type(data)(*(recursive_apply(d, func) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [recursive_apply(d, func) for d in data]
    elif isinstance(data, Mapping):
        return {key: recursive_apply(data[key], func) for key in data}
    else:
        try:
            return func(data)
        except Exception:
            return data


