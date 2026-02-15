from typing import Optional, Tuple, Union

from mindspore import Tensor, COOTensor, CSRTensor, ops, mint

from . import scatter
from .num_nodes import maybe_num_nodes


def contains_self_loops(edge_index: Tensor) -> bool:
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    self-loops.

    Args:
        edge_index (LongTensor): The edge indices.

    :rtype: bool

    Examples:
        >>> edge_index = Tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> contains_self_loops(edge_index)
        True

        >>> edge_index = Tensor([[0, 1, 1],
        ...                            [1, 0, 2]])
        >>> contains_self_loops(edge_index)
        False
    """
    mask = edge_index[0] == edge_index[1]
    return mask.sum().item() > 0


def remove_self_loops(  
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Example:
        >>> edge_index = Tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_attr = [[1, 2], [3, 4], [5, 6]]
        >>> edge_attr = Tensor(edge_attr)
        >>> remove_self_loops(edge_index, edge_attr)
        (tensor([[0, 1],
                [1, 0]]),
        tensor([[1, 2],
                [3, 4]]))
    """
    value = None
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    if value is not None:
        value = value[mask]

    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def segregate_self_loops(  
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    r"""Segregates self-loops from the graph.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`LongTensor`,
        :class:`Tensor`)

    Example:
        >>> edge_index = Tensor([[0, 0, 1],
        ...                            [0, 1, 0]])
        >>> (edge_index, edge_attr,
        ...  loop_edge_index,
        ...  loop_edge_attr) = segregate_self_loops(edge_index)
        >>>  loop_edge_index
        tensor([[0],
                [0]])
    """
    mask = mint.ne(edge_index[0],edge_index[1])
    inv_mask = mint.logical_not(mask)
    loop_edge_index = mint.index_select(edge_index, 1, mint.nonzero(inv_mask).view(-1))
    loop_edge_attr = None if edge_attr is None else edge_attr[inv_mask]
    edge_index = mint.index_select(edge_index, 1, mint.nonzero(mask).view(-1))
    edge_attr = None if edge_attr is None else edge_attr[mask]

    return edge_index, edge_attr, loop_edge_index, loop_edge_attr


def add_self_loops(  
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    fill_value: Optional[Union[float, Tensor, str]] = None,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of self-loops will be added
    according to :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> edge_index = Tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = Tensor0.5])
        >>> add_self_loops(edge_index)
        (tensor([[0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1]]),
        None)

        >>> add_self_loops(edge_index, edge_weight)
        (tensor([[0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1]]),
        tensor([0.5000, 0.5000, 0.5000, 1.0000, 1.0000]))

        >>> # edge features of self-loops are filled by constant `2.0`
        >>> add_self_loops(edge_index, edge_weight,
        ...                fill_value=2.)
        (tensor([[0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1]]),
        tensor([0.5000, 0.5000, 0.5000, 2.0000, 2.0000]))

        >>> # Use 'add' operation to merge edge features for self-loops
        >>> add_self_loops(edge_index, edge_weight,
        ...                fill_value='add')
        (tensor([[0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1]]),
        tensor([0.5000, 0.5000, 0.5000, 1.0000, 0.5000]))
    """
    value = None

    if isinstance(num_nodes, (tuple, list)):
        size = (num_nodes[0], num_nodes[1])
        N = min(size)
    else:
        N = maybe_num_nodes(edge_index, num_nodes)
        size = (N, N)

    loop_index = mint.arange(N).view(1, -1).tile((2, 1))

    full_edge_index = mint.cat([edge_index, loop_index], dim=1)

    if edge_attr is not None:
        loop_attr = compute_loop_attr(edge_index, edge_attr, N, fill_value)  #
        edge_attr = mint.cat([edge_attr, loop_attr], dim=0)

    return full_edge_index, edge_attr


def compute_loop_attr(  
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
    fill_value: Optional[Union[float, Tensor, str]] = None,
) -> Tensor:

    if fill_value is None:
        size = (num_nodes,) + edge_attr.shape[1:]
        return mint.ones(size, dtype=edge_attr.dtype)

    elif isinstance(fill_value, (int, float)):
        size = (num_nodes,) + edge_attr.shape[1:]
        return ops.full(size, fill_value, dtype=edge_attr.dtype)

    elif isinstance(fill_value, Tensor):
        size = (num_nodes,) + edge_attr.shape[1:]
        loop_attr = fill_value.astype(edge_attr.dtype)
        if edge_attr.dim() != loop_attr.dim():
            loop_attr = loop_attr.unsqueeze(0)
        return loop_attr.broadcast_to(size)

    elif isinstance(fill_value, str):
        col = edge_index[1]
        return scatter(edge_attr, col, 0, num_nodes, fill_value)
    else:
        raise AttributeError("No valid 'fill_value' provided")


def add_remaining_self_loops(  
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    fill_value: Optional[Union[float, Tensor, str]] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of non-existing self-loops will
    be added according to :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Example:
        >>> edge_index = Tensor([[0, 1],
        ...                            [1, 0]])
        >>> edge_weight = Tensor
        >>> add_remaining_self_loops(edge_index, edge_weight)
        (tensor([[0, 1, 0, 1],
                [1, 0, 0, 1]]),
        tensor([0.5000, 0.5000, 1.0000, 1.0000]))
    """
    N = maybe_num_nodes(edge_index, num_nodes)
    mask = edge_index[0] != edge_index[1]

    loop_index = mint.arange(N).view(1, -1).tile((2, 1)).astype(edge_index.dtype)

    if edge_attr is not None:

        loop_attr = compute_loop_attr(edge_index, edge_attr, N, fill_value)  #

        inv_mask = ~mask
        if inv_mask.any():
            loop_attr[edge_index[0][inv_mask]] = edge_attr[inv_mask]

        edge_attr = mint.cat([edge_attr[mask], loop_attr], dim=0)

    edge_index = edge_index[:, mask]

    edge_index = mint.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_attr


def get_self_loop_attr(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:
        >>> edge_index = Tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = Tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = mint.ones(loop_index.numel())

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = mint.zeros(
        (num_nodes,) + loop_attr.shape[1:], dtype=loop_attr.dtype
    )
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr
