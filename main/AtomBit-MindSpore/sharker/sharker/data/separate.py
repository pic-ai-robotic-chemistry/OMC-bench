from collections.abc import Mapping, Sequence
from typing import Any, Type, TypeVar

import numpy as np
import mindspore as ms

from .graph import Graph
from .storage import BaseStorage
from ..utils import narrow


T = TypeVar("T")


def separate(
    cls: Type[T],
    batch: Any,
    idx: int,
    slice_dict: Any,
    inc_dict: Any = None,
    decrement: bool = True,
) -> T:
    # Separates the individual element from a `batch` at index `idx`.
    # `separate` can handle both homogeneous and heterogeneous data objects by
    # individually separating all their stores.
    # In addition, `separate` can handle nested data structures such as
    # dictionaries and lists.

    data = cls().stores_as(batch)

    # Iterate over each storage object and recursively separate its attributes:
    for batch_store, data_store in zip(batch.stores, data.stores):
        key = batch_store._key
        if key is not None:
            attrs = slice_dict[key].keys()
        else:
            attrs = set(batch_store.keys())
            attrs = [attr for attr in slice_dict.keys() if attr in attrs]

        for attr in attrs:
            if key is not None:
                slices = slice_dict[key][attr]
                incs = inc_dict[key][attr] if decrement else None
            else:
                slices = slice_dict[attr]
                incs = inc_dict[attr] if decrement else None

            data_store[attr] = _separate(
                attr,
                batch_store[attr],
                idx,
                slices,
                incs,
                batch,
                batch_store,
                decrement,
            )

        # The `num_nodes` attribute needs special treatment, as we cannot infer
        # the real number of nodes from the total number of nodes alone:
        if hasattr(batch_store, "_num_nodes"):
            data_store.num_nodes = batch_store._num_nodes[idx]

    return data


def _separate(
    key: str,
    values: Any,
    idx: int,
    slices: Any,
    incs: Any,
    batch: Graph,
    store: BaseStorage,
    decrement: bool,
) -> Any:

    if isinstance(values, (ms.Tensor, np.ndarray)):
        # Narrow a `Tensor` based on `slices`.
        # NOTE: We need to take care of decrementing elements appropriately.
        key = str(key)
        is_tensor = False
        if isinstance(values, ms.Tensor):
            is_tensor = True
        slices_np = slices.asnumpy().astype(np.int64) if isinstance(slices, ms.Tensor) else slices
        incs_np = incs.asnumpy().astype(np.int64) if isinstance(incs, ms.Tensor) else incs
        values_np = values.asnumpy() if isinstance(values, ms.Tensor) else values

        cat_dim = batch.__cat_dim__(key, values, store)
        start, end = int(slices_np[idx]), int(slices_np[idx + 1])
        value = narrow(values_np, cat_dim or 0, start, end - start)
        value = np.squeeze(value, axis=0) if cat_dim is None else value

        if decrement and incs is not None and (incs.ndim > 1 or incs_np[idx] != 0):
            value = value - incs_np[idx]

        return ms.Tensor(value) if is_tensor else value

    elif isinstance(values, Mapping):
        # Recursively separate elements of dictionaries.
        return {
            key: _separate(
                key,
                value,
                idx,
                slices=slices[key],
                incs=incs[key] if decrement else None,
                batch=batch,
                store=store,
                decrement=decrement,
            )
            for key, value in values.items()
        }

    elif (
        isinstance(values, Sequence)
        and isinstance(values[0], Sequence)
        and not isinstance(values[0], str)
        and len(values[0]) > 0
        and isinstance(values[0][0], ms.Tensor)
        and isinstance(slices, Sequence)
    ):
        # Recursively separate elements of lists of lists.
        return [value[idx] for value in values]

    elif (
        isinstance(values, Sequence)
        and not isinstance(values, str)
        and isinstance(values[0], ms.Tensor)
        and isinstance(slices, Sequence)
    ):
        # Recursively separate elements of lists of Tensors/SparseTensors.
        return [
            _separate(
                key,
                value,
                idx,
                slices=slices[i],
                incs=incs[i] if decrement else None,
                batch=batch,
                store=store,
                decrement=decrement,
            )
            for i, value in enumerate(values)
        ]

    else:
        return values[idx]

		