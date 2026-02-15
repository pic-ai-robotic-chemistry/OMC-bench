import copy
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import mindspore as ms
from typing_extensions import Self
from mindspore import Tensor, ops, mint

from .storage import BaseStorage, EdgeStorage, GlobalStorage, NodeStorage


class Data:
    def __getattr__(self, key: str) -> Any:
        raise NotImplementedError

    def __setattr__(self, key: str, value: Any):
        raise NotImplementedError

    def __delattr__(self, key: str):
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError

    def __delitem__(self, key: str):
        raise NotImplementedError

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memo):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def stores_as(self, data: Self):
        raise NotImplementedError

    @property
    def stores(self) -> List[BaseStorage]:
        raise NotImplementedError

    @property
    def node_stores(self) -> List[NodeStorage]:
        raise NotImplementedError

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        raise NotImplementedError

    def concat(self, data: Self, return_tensor: bool = True) -> Self:
        r"""Concatenates :obj:`self` with another :obj:`data` object.
        All values needs to have matching shapes at non-concat dimensions.
        """
        out = copy.copy(self)
        for store, other_store in zip(out.stores, data.stores):
            store.concat(other_store)
        if return_tensor == True:
            out.tensor()
        else:
            out.numpy()
        return out

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the dimension for which the value :obj:`value` of the
        attribute :obj:`key` will get concatenated when creating mini-batches
        using :class:`sharker.loader.DataLoader`.

        .. note::

            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the incremental count to cumulatively increase the value
        :obj:`value` of the attribute :obj:`key` when creating mini-batches
        using :class:`sharker.loader.DataLoader`.

        .. note::

            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError

    ###########################################################################

    def keys(self) -> List[str]:
        r"""Returns a list of all graph attribute names."""
        out = []
        for store in self.stores:
            out += list(store.keys())
        return list(set(out))

    def __len__(self) -> int:
        r"""Returns the number of graph attributes."""
        return len(self.keys())

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the
        data.
        """
        return key in self.keys()

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.__dict__[key] = value

    @property
    def num_nodes(self) -> Optional[int]:
        r"""Returns the number of nodes in the graph.

        .. note::
            The number of nodes in the data object is automatically inferred
            in case node-level attributes are present, *e.g.*, :obj:`data.x`.
            In some cases, however, a graph may only be given without any
            node-level attributes.
            :mindgeometric:`MindGeometric` then *guesses* the number of nodes according to
            :obj:`edge_index.max().item() + 1`.
            However, in case there exists isolated nodes, this number does not
            have to be correct which can result in unexpected behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        try:
            size = sum([v.num_nodes for v in self.node_stores])
            if isinstance(size, Tensor):
                size = size.item()
            return size
        except TypeError:
            return None

    @property
    def num_edges(self) -> int:
        r"""Returns the number of edges in the graph.
        For undirected graphs, this will return the number of bi-directional
        edges, which is double the amount of unique edges.
        """
        size = sum([v.num_edges for v in self.edge_stores])
        if isinstance(size, Tensor):
            size = size.item()
        return size

    def node_attrs(self) -> List[str]:
        r"""Returns all node-level tensor attribute names."""
        return list(set(chain(*[s.node_attrs() for s in self.node_stores])))

    def edge_attrs(self) -> List[str]:
        r"""Returns all edge-level tensor attribute names."""
        return list(set(chain(*[s.edge_attrs() for s in self.edge_stores])))

    def apply(self, func: Callable, *args: str):
        r"""Applies the function :obj:`func`, either to all attributes or only
        the ones given in :obj:`*args`.
        """
        for store in self.stores:
            store.apply(func, *args)
        return self

    def numpy(self, *args: str):
        r"""Copies attributes to CPU memory, either for all attributes or only
        the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.asnumpy() if isinstance(x, Tensor) else x, *args)

    def tensor(self, *args: str):
        r"""Copies attributes to CPU memory, either for all attributes or only
        the ones given in :obj:`*args`.
        """
        return self.apply(
            lambda x: Tensor.from_numpy(x) if isinstance(x, np.ndarray) else x, *args
        )

    def copy(self, *args: str):
        r"""Performs cloning of tensors, either for all attributes or only the
        ones given in :obj:`*args`.
        """
        return copy.copy(self).apply(lambda x: x.copy(), *args)
###############################################################################

@ms.jit_class
class Graph(Data):
    r"""A graph object describing a homogeneous graph.
    The data object can hold node-level, link-level and graph-level attributes.
    In general, :class:`~sharker.data.Graph` tries to mimic the
    behavior of a regular :python:`Python` dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic MindSpore tensor functionalities.
    See `here <https://sharker.readthedocs.io/en/latest/get_started/
    introduction.html#data-handling-of-graphs>`__ for the accompanying
    tutorial.

    .. code-block:: python

        from sharker.data import Graph

        data = Graph(x=x, edge_index=edge_index, ...)

        # Add additional arguments to `data`:
        data.train_idx = Tensor([...], dtype=ms.int64)
        data.test_mask = Tensor([...], dtype=ms.bool_)

        # Analyzing the graph structure:
        data.num_nodes
        >>> 23

        data.is_directed()
        >>> False

        # MindSpore tensor functionality:
        data = data.pin_memory()

    Args:
        x (Tensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph-level or node-level ground-truth
            labels with arbitrary shape. (default: :obj:`None`)
        crd (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        time (Tensor, optional): The timestamps for each event with shape
            :obj:`[num_edges]` or :obj:`[num_nodes]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """

    def __init__(
        self,
        x: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        y: Optional[Union[Tensor, int, float]] = None,
        crd: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        **kwargs,
    ):
        self.__dict__["_store"] = GlobalStorage(_parent=self)

        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y
        if crd is not None:
            self.crd = crd
        if time is not None:
            self.time = time

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        if "_store" not in self.__dict__:
            raise RuntimeError(
                "The 'data' object was created by an older version of MindeGometric. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again."
            )
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, "fset", None) is not None:
            propobj.fset(self, value)
        else:
            setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any):
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__["_store"] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        has_dict = any([isinstance(v, Mapping) for v in self._store.values()])

        if not has_dict:
            info = [size_repr(k, v) for k, v in self._store.items()]
            info = ", ".join(info)
            return f"{cls}({info})"
        else:
            info = [size_repr(k, v, indent=2) for k, v in self._store.items()]
            info = ",\n".join(info)
            return f"{cls}(\n{info}\n)"

    @property
    def num_nodes(self) -> Optional[int]:
        return super().num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: Optional[int]):
        self._store.num_nodes = num_nodes

    def stores_as(self, data: Self):
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        return self._store.to_dict()

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "adj" in key:
            return (0, 1)
        elif "index" in key or key == "face":
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "batch" in key and isinstance(value, (Tensor, np.ndarray)):
            return int(value.max()) + 1
        elif "index" in key or key == "face":
            return self.num_nodes
        else:
            return 0

    def is_node_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes a
        node-level tensor attribute.
        """
        return self._store.is_node_attr(key)

    def is_edge_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes an
        edge-level tensor attribute.
        """
        return self._store.is_edge_attr(key)

    def to_hetero(
        self,
        node_type: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
        node_type_names: Optional[List[str]] = None,
        edge_type_names: Optional[List[Tuple[str, str, str]]] = None,
        return_tensor: bool = True
    ):
        r"""Converts to HeteroGraph (removed in minimal build)."""
        raise NotImplementedError("HeteroGraph support was removed in minimal build.")

    ###########################################################################

    def __iter__(self) -> Iterable:
        r"""Iterates over all attributes in the data, yielding their attribute
        names and values.
        """
        for key, value in self._store.items():
            yield key, value

    @property
    def x(self) -> Optional[Tensor]:
        return self["x"] if "x" in self._store else None

    @x.setter
    def x(self, x: Optional[Tensor]):
        self._store.x = x

    @property
    def edge_index(self) -> Optional[Tensor]:
        return self["edge_index"] if "edge_index" in self._store else None

    @edge_index.setter
    def edge_index(self, edge_index: Optional[Tensor]):
        self._store.edge_index = edge_index

    @property
    def edge_weight(self) -> Optional[Tensor]:
        return self["edge_weight"] if "edge_weight" in self._store else None

    @edge_weight.setter
    def edge_weight(self, edge_weight: Optional[Tensor]):
        self._store.edge_weight = edge_weight

    @property
    def edge_attr(self) -> Optional[Tensor]:
        return self["edge_attr"] if "edge_attr" in self._store else None

    @edge_attr.setter
    def edge_attr(self, edge_attr: Optional[Tensor]):
        self._store.edge_attr = edge_attr

    @property
    def y(self) -> Optional[Union[Tensor, int, float]]:
        return self["y"] if "y" in self._store else None

    @y.setter
    def y(self, y: Optional[Tensor]):
        self._store.y = y

    @property
    def crd(self) -> Optional[Tensor]:
        return self["crd"] if "crd" in self._store else None

    @crd.setter
    def crd(self, crd: Optional[Tensor]):
        self._store.crd = crd

    @property
    def batch(self) -> Optional[Tensor]:
        return self["batch"] if "batch" in self._store else None

    @batch.setter
    def batch(self, batch: Optional[Tensor]):
        self._store.batch = batch

    @property
    def time(self) -> Optional[Tensor]:
        return self["time"] if "time" in self._store else None

    @time.setter
    def time(self, time: Optional[Tensor]):
        self._store.time = time

    @property
    def face(self) -> Optional[Tensor]:
        return self["face"] if "face" in self._store else None

    @face.setter
    def face(self, face: Optional[Tensor]):
        self._store.face = face


###############################################################################


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, Tensor) and getattr(value, "is_nested", False):
        out = str(list(value.to_padded_tensor(padding=0.0).shape))
    elif isinstance(value, Tensor):
        out = str(list(value.shape))
    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = "{}"
    elif (
        isinstance(value, Mapping)
        and len(value) == 1
        and not isinstance(list(value.values())[0], Mapping)
    ):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = "{ " + ", ".join(lines) + " }"
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = "{\n" + ",\n".join(lines) + ",\n" + pad + "}"
    else:
        out = str(value)

    key = str(key).replace("'", "")
    return f"{pad}{key}={out}"


