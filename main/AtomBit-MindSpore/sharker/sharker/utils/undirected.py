from typing import List, Optional, Tuple, Union
from mindspore import Tensor, ops, mint
from .coalesce import coalesce
from .sort_edge_index import sort_edge_index
from .num_nodes import maybe_num_nodes

MISSING = "???"


def is_undirected(  
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor]] = None,
    num_nodes: Optional[int] = None,
) -> bool:
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` is
    undirected.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will check for equivalence in all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)

    :rtype: bool

    Examples:
        >>> edge_index = Tensor([[0, 1, 0],
        ...                         [1, 0, 0]])
        >>> weight = Tensor([0, 0, 1])
        >>> is_undirected(edge_index, weight)
        True

        >>> weight = Tensor([0, 1, 1])
        >>> is_undirected(edge_index, weight)
        False

    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_attrs: List[Tensor] = []
    if isinstance(edge_attr, Tensor):
        edge_attrs.append(edge_attr)
    elif isinstance(edge_attr, (list, tuple)):
        edge_attrs = edge_attr

    edge_index1, edge_attrs1 = sort_edge_index(
        edge_index,
        edge_attrs,
        num_nodes=num_nodes,
        sort_by_row=True,
    )
    edge_index2, edge_attrs2 = sort_edge_index(
        edge_index,
        edge_attrs,
        num_nodes=num_nodes,
        sort_by_row=False,
    )

    if not ops.equal(edge_index1[0], edge_index2[1]).all():
        return False

    if not ops.equal(edge_index1[1], edge_index2[0]).all():
        return False

    assert isinstance(edge_attrs1, list) and isinstance(edge_attrs2, list)
    for edge_attr1, edge_attr2 in zip(edge_attrs1, edge_attrs2):
        if not ops.equal(edge_attr1, edge_attr2).all():
            return False

    return True


def to_undirected(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]], Tuple[Tensor, List[Tensor]]]:
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will remove duplicates for all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).

    Examples:
        >>> edge_index = Tensor([[0, 1, 1],
        ...                            [1, 0, 2]])
        >>> to_undirected(edge_index)
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]])

        >>> edge_index = Tensor1, 1],
        ...                            [1, 0, 2]])
        >>> edge_weight = Tensor([1., 1., 1.])
        >>> to_undirected(edge_index, edge_weight)
        (tensor([[0, 1, 1, 2],
                 [1, 0, 2, 1]]),
        tensor([2., 2., 1., 1.]))

        >>> # Use 'mean' operation to merge edge features
        >>>  to_undirected(edge_index, edge_weight, reduce='mean')
        (tensor([[0, 1, 1, 2],
                 [1, 0, 2, 1]]),
        tensor([1., 1., 1., 1.]))
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        num_nodes = edge_attr
        edge_attr = MISSING

    src, dst = edge_index[0], edge_index[1]
    src, dst = mint.cat([src, dst], dim=0), mint.cat([dst, src], dim=0)
    edge_index = mint.stack([src, dst], dim=0)

    if isinstance(edge_attr, Tensor):
        edge_attr = mint.cat([edge_attr, edge_attr], dim=0)
    elif isinstance(edge_attr, (list, tuple)):
        edge_attr = [mint.cat([e, e], dim=0) for e in edge_attr]

    return coalesce(edge_index, edge_attr, num_nodes, reduce)
