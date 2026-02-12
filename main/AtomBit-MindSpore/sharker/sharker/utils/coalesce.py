from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, nn, mint
from .num_nodes import maybe_num_nodes
from . import scatter

MISSING = "???"


def coalesce(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    reduce: str = "sum",
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]], Tuple[Tensor, List[Tensor]]]:
    """Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are merged by scattering them
    together according to the given :obj:`reduce` option.

    Args:
        edge_index (Tensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights
            or multi-dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`, :obj:`"any"`). (default: :obj:`"sum"`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).

    Example:
        >>> edge_index = Tensor([[1, 1, 2, 3],
        ...                            [3, 3, 1, 2]])
        >>> edge_attr = Tensor([1., 1., 1., 1.])
        >>> coalesce(edge_index)
        tensor([[1, 2, 3],
                [3, 1, 2]])

        >>> # Sort `edge_index` column-wise
        >>> coalesce(edge_index, sort_by_row=False)
        tensor([[2, 3, 1],
                [1, 2, 3]])

        >>> coalesce(edge_index, edge_attr)
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([2., 1., 1.]))

        >>> # Use 'mean' operation to merge edge features
        >>> coalesce(edge_index, edge_attr, reduce='mean')
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([1., 1., 1.]))
    """
    num_edges = edge_index[0].shape[0]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    idx = mint.neg(mint.ones(num_edges + 1, dtype=ms.int64))
    idx_from_1 = edge_index[1 - int(sort_by_row)]
    idx_from_1 = mint.add(edge_index[int(sort_by_row)], idx_from_1, alpha=num_nodes)

    if not is_sorted:
        idx_from_1, perm = mint.sort(idx_from_1)
        if isinstance(edge_index, Tensor):
            edge_index = mint.index_select(edge_index, 1, perm)
        elif isinstance(edge_index, tuple):
            edge_index = (edge_index[0][perm], edge_index[1][perm])
        else:
            raise NotImplementedError
        if isinstance(edge_attr, Tensor):
            edge_attr = mint.index_select(edge_attr, 0, perm)
        elif isinstance(edge_attr, (list, tuple)):
            edge_attr = [e[perm] for e in edge_attr]

    idx[1:] = idx_from_1
    mask = mint.greater(idx_from_1, idx[:-1])

    # Only perform expensive merging in case there exists duplicates::
    if mint.all(mask):
        if edge_attr is None or isinstance(edge_attr, Tensor):
            return edge_index, edge_attr
        if isinstance(edge_attr, (list, tuple)):
            return edge_index, edge_attr
        return edge_index
    if isinstance(edge_index, Tensor):
        # edge_index = edge_index[:, mask]
        edge_index = edge_index[:, mask]
    elif isinstance(edge_index, tuple):
        edge_index = (edge_index[0][mask], edge_index[1][mask])
    else:
        raise NotImplementedError

    dim_size = None
    if isinstance(edge_attr, (Tensor, list, tuple)) and len(edge_attr) > 0:
        dim_size = edge_index.shape[1]
        idx = mint.arange(0, num_edges)
        idx -= mint.cumsum(~mask, dim=0)

    if edge_attr is None:
        return edge_index, None
    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, dim_size, reduce)
        return edge_index, edge_attr
    if isinstance(edge_attr, (list, tuple)):
        if len(edge_attr) == 0:
            return edge_index, edge_attr
        edge_attr = [scatter(e, idx, 0, dim_size, reduce) for e in edge_attr]
        return edge_index, edge_attr

    return edge_index

def coalesce_np(  # noqa: F811
    edge_index: np.ndarray,
    edge_attr: Union[Optional[np.ndarray], List[np.ndarray], str] = MISSING,
    num_nodes: Optional[int] = None,
    reduce: str = "sum",
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, List[np.ndarray]]]:

    num_edges = edge_index[0].shape[0]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = -np.ones(num_edges + 1, dtype=np.int32)
    idx_from_1 = edge_index[1 - int(sort_by_row)]
    idx_from_1 = idx_from_1 * num_nodes + edge_index[int(sort_by_row)]

    if not is_sorted:
        perm = np.argsort(idx_from_1)
        idx_from_1 = np.sort(idx_from_1)
        if isinstance(edge_index, np.ndarray):
            edge_index = edge_index[:, perm]
        elif isinstance(edge_index, tuple):
            edge_index = (edge_index[0][perm], edge_index[1][perm])
        else:
            raise NotImplementedError
        if isinstance(edge_attr, np.ndarray):
            edge_attr = edge_attr[perm]
        elif isinstance(edge_attr, (list, tuple)):
            edge_attr = [e[perm] for e in edge_attr]

    idx[1:] = idx_from_1

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        if edge_attr is None or isinstance(edge_attr, np.ndarray):
            return edge_index, edge_attr
        if isinstance(edge_attr, (list, tuple)):
            return edge_index, edge_attr
        return edge_index
    if isinstance(edge_index, np.ndarray):
        edge_index = edge_index[:, mask]
    elif isinstance(edge_index, tuple):
        edge_index = (edge_index[0][mask], edge_index[1][mask])
    else:
        raise NotImplementedError

    dim_size = None
    if isinstance(edge_attr, (np.ndarray, list, tuple)) and len(edge_attr) > 0:
        dim_size = edge_index.shape[1]
        idx = np.arange(0, num_edges)
        idx -= np.cumsum(~mask, axis=0)

    if edge_attr is None:
        return edge_index, None
    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, dim_size, reduce)
        return edge_index, edge_attr
    if isinstance(edge_attr, (list, tuple)):
        if len(edge_attr) == 0:
            return edge_index, edge_attr
        edge_attr = [scatter(e, idx, 0, dim_size, reduce) for e in edge_attr]
        return edge_index, edge_attr

    return edge_index
