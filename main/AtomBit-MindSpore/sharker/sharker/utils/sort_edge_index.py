from typing import List, Optional, Tuple, Union
import numpy as np
from mindspore import Tensor, ops, mint

from .num_nodes import maybe_num_nodes

MISSING = "???"


def sort_edge_index(  
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]], Tuple[Tensor, List[Tensor]]]:
    """Row-wise sorts :obj:`edge_index`.

    Args:
        edge_index (Tensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights
            or multi-dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        sort_by_src (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise/by destination node.
            (default: :obj:`True`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).

    Examples:
        >>> edge_index = Tensor([[2, 1, 1, 0],
                                [1, 2, 0, 1]])
        >>> edge_attr = Tensor [2], [3], [4]])
        >>> sort_edge_index(edge_index)
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]])

        >>> sort_edge_index(edge_index, edge_attr)
        (tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]),
        tensor([[4],
                [3],
                [2],
                [1]]))
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index[1 - int(sort_by_row)] * num_nodes
    idx += edge_index[int(sort_by_row)]
    _, perm = mint.sort(idx)

    if isinstance(edge_index, Tensor):
        edge_index = mint.index_select(edge_index, 1, perm)
    elif isinstance(edge_index, tuple):
        edge_index = (edge_index[0][perm], edge_index[1][perm])
    else:
        raise NotImplementedError

    if edge_attr is None:
        return edge_index, None
    if isinstance(edge_attr, Tensor):
        return edge_index, mint.index_select(edge_attr, 0, perm)
    if isinstance(edge_attr, (list, tuple)):
        return edge_index, [mint.index_select(e, 0, perm) for e in edge_attr]

    return edge_index

def sort_edge_index_np(  
    edge_index: np.ndarray,
    edge_attr: Union[Optional[np.ndarray], List[np.ndarray], str] = MISSING,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, List[np.ndarray]]]:

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index[1 - int(sort_by_row)] * num_nodes
    idx += edge_index[int(sort_by_row)]
    perm = np.argsort(idx)

    if isinstance(edge_index, np.ndarray):
        edge_index = edge_index[:, perm]
    elif isinstance(edge_index, tuple):
        edge_index = (edge_index[0][perm], edge_index[1][perm])
    else:
        raise NotImplementedError

    if edge_attr is None:
        return edge_index, None
    if isinstance(edge_attr, np.ndarray):
        return edge_index, edge_attr[perm]
    if isinstance(edge_attr, (list, tuple)):
        return edge_index, [e[perm] for e in edge_attr]

    return edge_index
