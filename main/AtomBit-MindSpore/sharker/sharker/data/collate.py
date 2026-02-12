from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import numpy as np
import mindspore as ms
from mindspore import ops, mint
from .graph import Graph
from .storage import BaseStorage, NodeStorage
from ..utils.functions import cumsum, cumsum_np

T = TypeVar("T")
SliceDictType = Dict[str, Union[ms.Tensor, Dict[str, ms.Tensor]]]
IncDictType = Dict[str, Union[ms.Tensor, Dict[str, ms.Tensor]]]

def collate(
    cls: Type[T],
    data_list: List[Graph],
    increment: bool = True,
    add_batch: bool = True,
    return_tensor: bool = True,
    follow_batch: Optional[Iterable[str]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
) -> Tuple[T, SliceDictType, IncDictType]:
    # Collates a list of `data` objects into a single object of type `cls`.
    # `collate` can handle both homogeneous and heterogeneous data objects by
    # individually collating all their stores.
    # In addition, `collate` can handle nested data structures such as
    # dictionaries and lists.

    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:  # Dynamic inheritance.
        out = cls(_base_cls=data_list[0].__class__)
    else:
        out = cls()

    # Create empty stores:
    out.stores_as(data_list[0])

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_stores = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    slice_dict: SliceDictType = {}
    inc_dict: IncDictType = {}
    for out_store in out.stores:
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():

            if attr in exclude_keys:  # Do not include top-level attribute.
                continue

            values = [store[attr] for store in stores]

            # The `num_nodes` attribute needs special treatment, as we need to
            # sum their values up instead of merging them to a list:
            if attr == "num_nodes":
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            # Skip batching of `ptr` vectors for now:
            if attr == "ptr":
                continue

            # Collate attributes into a unified representation:
            value, slices, incs = _collate(attr, values, data_list, stores, increment)

            out_store[attr] = value

            if key is not None:  # Heterogeneous:
                store_slice_dict = slice_dict.get(key, {})
                assert isinstance(store_slice_dict, dict)
                store_slice_dict[attr] = slices
                slice_dict[key] = store_slice_dict

                store_inc_dict = inc_dict.get(key, {})
                assert isinstance(store_inc_dict, dict)
                store_inc_dict[attr] = incs
                inc_dict[key] = store_inc_dict
            else:  # Homogeneous:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute:
            if attr in follow_batch:
                batch, ptr = _batch_and_ptr(slices)
                out_store[f"{attr}_batch"] = batch
                out_store[f"{attr}_ptr"] = ptr

        # In case of node-level storages, we add a top-level batch vector it:
        if (
            add_batch
            and isinstance(stores[0], NodeStorage)
            and stores[0].can_infer_num_nodes
        ):
            repeats = [int(store.num_nodes) or 0 for store in stores]
            out_store.batch = repeat(repeats)
            out_store.ptr = cumsum_np(np.array(repeats))

    if return_tensor == True:
        out = out.tensor()
    return out, slice_dict, inc_dict


def _collate(
    key: str,
    values: List[Any],
    data_list: List[Graph],
    stores: List[BaseStorage],
    increment: bool,
) -> Tuple[Any, Any, Any]:

    elem = values[0]

    if isinstance(elem, ms.Tensor):
        # Concatenate a list of `Tensor` along the `cat_dim`.
        # NOTE: We need to take care of incrementing elements appropriately.
        values = [value.asnumpy() for value in values]
        value, slices, incs = _collate(key, values, data_list, stores, increment)
        return value, slices, incs

    elif isinstance(elem, np.ndarray):
        key = str(key)
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        if elem.ndim == 0:
            values = [value.reshape(1) for value in values]
        if cat_dim is None:
            values = [value[None,:] for value in values]
        sizes = np.array([value.shape[cat_dim or 0] for value in values])
        slices = cumsum_np(sizes)
        if increment:
            incs = get_incs_np(key, values, data_list, stores)
            if incs.ndim > 1 or incs[-1] != 0:
                values = [value + inc for value, inc in zip(values, incs)]
        else:
            incs = None

        value = np.concatenate(values, axis=cat_dim or 0)
        return value, slices, incs

    elif isinstance(elem, (int, float)):
        # Convert a list of numerical values to a `Tensor`.
        value = np.array(values)
        if increment:
            incs = get_incs_np(key, values, data_list, stores)
            if (incs[-1]) != 0:
                value += incs
        else:
            incs = None
        slices = np.arange(len(values) + 1)
        return value, slices, incs

    elif isinstance(elem, Mapping):
        # Recursively collate elements of dictionaries.
        value_dict, slice_dict, inc_dict = {}, {}, {}
        for key in elem.keys():
            value_dict[key], slice_dict[key], inc_dict[key] = _collate(
                key, [v[key] for v in values], data_list, stores, increment
            )
        return value_dict, slice_dict, inc_dict

    elif (
        isinstance(elem, Sequence)
        and not isinstance(elem, str)
        and len(elem) > 0
        and isinstance(elem[0], ms.Tensor)
    ):
        # Recursively collate elements of lists.
        value_list, slice_list, inc_list = [], [], []
        for i in range(len(elem)):
            value, slices, incs = _collate(
                key, [v[i] for v in values], data_list, stores, increment
            )
            value_list.append(value)
            slice_list.append(slices)
            inc_list.append(incs)
        return value_list, slice_list, inc_list

    else:
        # Other-wise, just return the list of values as it is.
        slices = np.arange(len(values) + 1)
        return values, slices, None


def _batch_and_ptr(
    slices: Any,
) -> Tuple[Any, Any]:
    if isinstance(slices, ms.Tensor) and slices.dim() == 1:
        # Default case, turn slices tensor into batch.
        slices_np = slices.asnumpy()
        batch, ptr = _batch_and_ptr(slices_np)
        return batch, ptr

    if isinstance(slices, np.ndarray) and slices.ndim == 1:
        # Default case, turn slices tensor into batch.
        repeats = slices[1:] - slices[:-1]
        batch = repeat(repeats)
        ptr = slices
        return batch, ptr

    elif isinstance(slices, Mapping):
        # Recursively batch elements of dictionaries.
        batch, ptr = {}, {}
        for k, v in slices.items():
            batch[k], ptr[k] = _batch_and_ptr(v)
        return batch, ptr

    elif (
        isinstance(slices, Sequence)
        and not isinstance(slices, str)
        and isinstance(slices[0], (ms.Tensor,np.ndarray))
    ):
        # Recursively batch elements of lists.
        batch, ptr = [], []
        for s in slices:
            sub_batch, sub_ptr = _batch_and_ptr(s)
            batch.append(sub_batch)
            ptr.append(sub_ptr)
        return batch, ptr

    else:
        # Failure of batching, usually due to slices.dim() != 1
        return None, None


###############################################################################


def repeat(
    repeats) -> np.ndarray:
    if isinstance(repeats, List):
        repeats = np.array(repeats)
    if isinstance(repeats, ms.Tensor):
        repeaats = repeats.asnumpy()
    outs = np.repeat(np.arange(repeats.shape[0]), repeats, 0)
    return outs


def get_incs(
    key, values: List[Any], data_list: List[Graph], stores: List[BaseStorage]
) -> ms.Tensor:
    repeats = [
        data.__inc__(key, value, store)
        for value, data, store in zip(values, data_list, stores)
    ]
    if isinstance(repeats[0], ms.Tensor):
        repeats = mint.stack(repeats, dim=0)
    else:
        repeats = ms.Tensor(repeats)
    return cumsum(mint.narrow(repeats, 0, 0, repeats.shape[0] - 1))

def get_incs_np(
    key, values: List[Any], data_list: List[Graph], stores: List[BaseStorage]
) -> np.ndarray:
    repeats = [data.__inc__(key, value, store)
               for value, data, store in zip(values, data_list, stores)]
    repeats = np.stack(repeats, axis=0)
    incs = cumsum_np(repeats[:-1], axis=0)
    return incs
