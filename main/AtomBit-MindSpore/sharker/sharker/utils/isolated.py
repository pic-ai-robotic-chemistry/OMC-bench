from typing import Optional, Tuple

import mindspore as ms
from mindspore import Tensor, ops, nn, mint
from .loop import remove_self_loops, segregate_self_loops
from .num_nodes import maybe_num_nodes


def contains_isolated_nodes(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
) -> bool:
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    isolated nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: bool

    Examples:
        >>> edge_index = Tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> contains_isolated_nodes(edge_index)
        False

        >>> contains_isolated_nodes(edge_index, num_nodes=3)
        True
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    return mint.unique(edge_index.view(-1)).numel() < num_nodes


def remove_isolated_nodes(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    r"""Removes the isolated nodes from the graph given by :attr:`edge_index`
    with optional edge attributes :attr:`edge_attr`.
    In addition, returns a mask of shape :obj:`[num_nodes]` to manually filter
    out isolated node features later on.
    Self-loops are preserved for non-isolated nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (LongTensor, Tensor, BoolTensor)

    Examples:
        >>> edge_index = Tensor
        ...                            [1, 0, 0]])
        >>> edge_index, edge_attr, mask = remove_isolated_nodes(edge_index)
        >>> mask # node mask (2 nodes)
        tensor([True, True])

        >>> edge_index, edge_attr, mask = remove_isolated_nodes(edge_index,
        ...                                                     num_nodes=3)
        >>> mask # node mask (3 nodes)
        tensor([True, True, False])
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    out = segregate_self_loops(edge_index, edge_attr)
    edge_index, edge_attr, loop_edge_index, loop_edge_attr = out

    mask = mint.zeros(num_nodes, dtype=ms.bool_)
    mask[edge_index.view(-1)] = 1

    assoc = ops.full((num_nodes,), -1, dtype=ms.int64)
    assoc[mask] = mint.arange(mask.sum().item())  # type: ignore
    edge_index = assoc[edge_index]

    loop_mask = mint.zeros_like(mask)
    loop_mask[loop_edge_index[0]] = 1
    loop_mask = mint.logical_and(loop_mask, mask)
    loop_assoc = ops.full_like(assoc, -1)
    loop_assoc[loop_edge_index[0]] = mint.arange(loop_edge_index.shape[1])
    loop_idx = loop_assoc[loop_mask]
    loop_edge_index = assoc[loop_edge_index[:, loop_idx]]

    edge_index = mint.cat([edge_index, loop_edge_index], dim=1)

    if edge_attr is not None:
        assert loop_edge_attr is not None
        loop_edge_attr = loop_edge_attr[loop_idx]
        edge_attr = mint.cat([edge_attr, loop_edge_attr], dim=0)

    return edge_index, edge_attr, mask
