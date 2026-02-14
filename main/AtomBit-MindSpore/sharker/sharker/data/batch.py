import inspect
from collections.abc import Sequence, Mapping
from typing import Any, List, Optional, Type, Union
from typing_extensions import Self

import mindspore as ms
from mindspore import mint
import numpy as np
from .graph import Graph
from .collate import collate
from .separate import separate
from .dataset import IndexType


class DynamicInheritance(type):
    # A meta class that sets the base class of a `Batch` object, e.g.:
    # * `Batch(Graph)` in case `Graph` objects are batched together
    # * `Batch(HeteroGraph)` in case `HeteroGraph` objects are batched together
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        base_cls = kwargs.pop("_base_cls", Graph)

        if issubclass(base_cls, Batch):
            new_cls = base_cls
        else:
            name = f"{base_cls.__name__}{cls.__name__}"

            # NOTE `MetaResolver` is necessary to resolve metaclass conflict
            # problems between `DynamicInheritance` and the metaclass of
            # `base_cls`. In particular, it creates a new common metaclass
            # from the defined metaclasses.
            class MetaResolver(type(cls), type(base_cls)):  # type: ignore
                pass

            if name not in globals():
                globals()[name] = MetaResolver(name, (cls, base_cls), {})
            new_cls = globals()[name]

        params = list(inspect.signature(base_cls.__init__).parameters.items())
        for i, (k, v) in enumerate(params[1:]):
            if k == "args" or k == "kwargs":
                continue
            if i < len(args) or k in kwargs:
                continue
            if v.default is not inspect.Parameter.empty:
                continue
            kwargs[k] = None

        return super(DynamicInheritance, new_cls).__call__(*args, **kwargs)


class DynamicInheritanceGetter:
    def __call__(self, cls: Type, base_cls: Type) -> Self:
        return cls(_base_cls=base_cls)


@ms.jit_class
class Batch(metaclass=DynamicInheritance):
    r"""A data object describing a batch of graphs as one big (disconnected)
    graph.
    Inherits from :class:`sharker.data.Graph` or
    :class:`sharker.data.HeteroGraph`.
    In addition, single graphs can be identified via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.

    :mindgeometric:`MindGeometric` allows modification to the underlying batching procedure by
    overwriting the :meth:`~Data.__inc__` and :meth:`~Data.__cat_dim__`
    functionalities.
    The :meth:`~Data.__inc__` method defines the incremental count between two
    consecutive graph attributes.
    By default, :mindgeometric:`MindGeometric` increments attributes by the number of nodes
    whenever their attribute names contain the substring :obj:`index`
    (for historical reasons), which comes in handy for attributes such as
    :obj:`edge_index` or :obj:`node_index`.
    However, note that this may lead to unexpected behavior for attributes
    whose names contain the substring :obj:`index` but should not be
    incremented.
    To make sure, it is best practice to always double-check the output of
    batching.
    Furthermore, :meth:`~Data.__cat_dim__` defines in which dimension graph
    tensors of the same attribute should be concatenated together.
    """

    @classmethod
    def from_data_list(
        cls,
        data_list: List[Graph],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        return_tensor: bool = True
    ) -> Self:
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            return_tensor = False,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        if return_tensor == True:
            batch.tensor()
        return batch

    def get_example(self, idx: int) -> Graph:
        r"""Gets the :class:`~sharker.data.Graph` or
        :class:`~sharker.data.HeteroGraph` object at index :obj:`idx`.
        The :class:`~sharker.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial object.
        """
        if not hasattr(self, "_slice_dict"):
            raise RuntimeError(
                (
                    "Cannot reconstruct 'Data' object from 'Batch' because "
                    "'Batch' was not created via 'Batch.from_data_list()'"
                )
            )

        data = separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=getattr(self, "_slice_dict"),
            inc_dict=getattr(self, "_inc_dict"),
            decrement=True,
        )

        return data

    def __getitem__(self, idx: Union[int, np.integer, str, IndexType]) -> Any:
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, ms.Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            return self.get_example(idx)
        elif isinstance(idx, str) or (
            isinstance(idx, tuple) and isinstance(idx[0], str)
        ):
            # Accessing attributes or node/edge types:
            return super().__getitem__(idx)
        else:
            if isinstance(idx, slice):
                index = list(range(self.num_graphs)[idx])
            elif isinstance(idx, ms.Tensor):
                index = mint.nonzero(idx).flatten().asnumpy().tolist() if idx.dtype == ms.bool_ else idx.flatten().tolist()
            elif isinstance(idx, np.ndarray):
                index = idx.flatten().nonzero()[0].tolist() if idx.dtype == bool else idx.flatten().tolist()
            else:
                index = list(idx)
            return [self.get_example(i) for i in index]

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        if hasattr(self, "_num_graphs"):
            return self._num_graphs
        elif hasattr(self, "ptr"):
            return self.ptr.numel() - 1
        elif hasattr(self, "batch"):
            return int(self.batch.max()) + 1
        else:
            raise ValueError("Can not infer the number of graphs")

    def __len__(self) -> int:
        return self.num_graphs

    def __reduce__(self) -> Any:
        state = self.__dict__.copy()
        return DynamicInheritanceGetter(), self.__class__.__bases__, state
