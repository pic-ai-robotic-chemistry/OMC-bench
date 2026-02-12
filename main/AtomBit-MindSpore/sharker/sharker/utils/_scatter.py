import mindspore as ms
from typing import List, Optional, Tuple, Union
from mindspore import Tensor, ops, mint
from .functions import cumsum, index_select, index_fill, broadcast_to


_scatter_max = ops.MultitypeFuncGraph('_scatter_max')
_scatter_amax = ops.MultitypeFuncGraph('_scatter_amax')
_scatter_min = ops.MultitypeFuncGraph('_scatter_min')
_scatter_amin = ops.MultitypeFuncGraph('_scatter_amin')

_scatter_sum = ops.MultitypeFuncGraph('_scatter_sum')
_scatter_mean = ops.MultitypeFuncGraph('_scatter_mean')
_scatter_mul = ops.MultitypeFuncGraph('_scatter_mul')


@_scatter_max.register('Number', 'Tensor', 'Tensor', 'Number')
def _scatter_max(dim: int,  index: Tensor, src: Tensor, scope: int):
    mask = index == scope
    out = index_select(src, mask, dim)
    if out.shape[dim] in [0, 1]:
        return out
    return out.max(dim, True)


@_scatter_amax.register('Number', 'Tensor', 'Tensor', 'Number')
def _scatter_amax(dim: int,  index: Tensor, src: Tensor, scope: int):
    mask = index != scope
    out = index_fill(src.copy(), mask, src.min()-1, dim)
    shape = list(src.shape)
    if shape[dim] == 0:
        shape[dim] = 1
        return ops.zeros(shape).long()
    return out.argmax(dim, keepdims=True)


@_scatter_min.register('Number', 'Tensor', 'Tensor', 'Number')
def _scatter_min(dim: int,  index: Tensor, src: Tensor, scope: int):
    mask = index == scope
    out = index_select(src, mask, dim)
    if out.shape[dim] in [0, 1]:
        return out
    return out.min(dim, keepdims=True)


@_scatter_amin.register('Number', 'Tensor', 'Tensor', 'Number')
def _scatter_amin(dim: int,  index: Tensor, src: Tensor, scope: int):
    mask = index != scope
    out = index_fill(src.copy(), mask, src.max()+1, dim)
    shape = list(src.shape)
    if shape[dim] == 0:
        shape[dim] = 1
        return mint.zeros(shape).int()
    return out.argmin(dim, keepdims=True)


@_scatter_sum.register('Number', 'Tensor', 'Tensor', 'Number')
def _scatter_sum(dim: int,  index: Tensor, src: Tensor, scope: int):
    mask = index == scope
    out = index_select(src, mask, dim)
    if out.shape[dim] in [0, 1]:
        return out
    return out.sum(dim, keepdims=True)


@_scatter_mean.register('Number', 'Tensor', 'Tensor', 'Number')
def _scatter_mean(dim: int,  index: Tensor, src: Tensor, scope: int):
    mask = index == scope
    out = index_select(src, mask, dim)
    if out.shape[dim] in [0, 1]:
        return out
    return out.mean(dim, keep_dims=True)


@_scatter_mul.register('Number', 'Tensor', 'Tensor', 'Number')
def _scatter_mul(dim: int,  index: Tensor, src: Tensor, scope: int):
    mask = index == scope
    out = index_select(src, mask, dim)
    if out.shape[dim] in [0, 1]:
        return out
    return out.prod(dim, keep_dims=True)


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    if dim > src.ndim-1:
        raise ValueError(f"`dim` must lay between 0 and {src.ndim-1}")
    if dim_size is None:
        dim_size = int(ops.amax(index)) + 1
    shape = list(src.shape)
    shape[dim] = dim_size
    out = mint.zeros(shape, dtype=src.dtype)
    scope = ops.unique(index)[0].astype(ms.int64)
    common_map = ops.Map()
    if reduce == "any":
        index = broadcast_to(index, src, dim=dim, is_dense=(reduce == 'any'))
        out = ops.scatter(out, dim, index, src)
        return out
    if reduce in ['sum', 'add']:
        vals = common_map(ops.partial(_scatter_sum, dim, index, src), scope)
    elif reduce == 'mean':
        vals = common_map(ops.partial(_scatter_mean, dim, index, src), scope)
    elif reduce == 'mul':
        vals = common_map(ops.partial(_scatter_mul, dim, index, src), scope)
    elif reduce == 'max':
        vals = common_map(ops.partial(_scatter_max, dim, index, src), scope)
    elif reduce == 'amax':
        out = out.long().fill(src.shape[dim])
        vals = common_map(ops.partial(_scatter_amax, dim, index, src), scope)
    elif reduce == 'min':
        vals = common_map(ops.partial(_scatter_min, dim, index, src), scope)
    elif reduce == 'amin':
        out = out.int().fill(src.shape[dim])
        vals = common_map(ops.partial(_scatter_amin, dim, index, src), scope)
    else:
        raise ValueError(f"invalid `reduce` argument '{reduce}'")
    out = index_fill(out, scope, mint.cat(vals, dim=dim), dim)
    return out


def scatter_softmax(
    src: Tensor, index: Tensor, dim: int = -1, dim_size: Optional[int] = None
) -> Tensor:
    max_value_per_index = scatter(src, index, dim=dim, dim_size=dim_size)
    ix = broadcast_to(index, src, dim)
    max_per_src_element = ops.gather_nd(
        max_value_per_index, ix).reshape(src.shape)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = mint.exp(recentered_scores)

    sum_per_index = scatter(
        recentered_scores_exp, index, dim, dim_size=dim_size, reduce="sum"
    )
    normalizing_constants = ops.gather_nd(sum_per_index, ix).reshape(src.shape)
    return mint.div(recentered_scores_exp, normalizing_constants)


def scatter_log_softmax(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    eps: float = 1e-12,
    dim_size: Optional[int] = None,
) -> Tensor:
    max_value_per_index = scatter(
        src, index, dim=dim, dim_size=dim_size, reduce="sum")
    ix = broadcast_to(index, src, dim)
    max_per_src_element = ops.gather_nd(
        max_value_per_index, ix).reshape(src.shape)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = mint.exp(recentered_scores)

    sum_per_index = scatter(
        recentered_scores_exp, index, dim, dim_size=dim_size, reduce="sum"
    )
    normalizing_constants = ops.gather_nd(mint.log(sum_per_index + eps), ix).reshape(
        src.shape
    )
    return recentered_scores - normalizing_constants


def group_argsort(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    num_groups: Optional[int] = None,
    descending: bool = False,
    return_consecutive: bool = False,
) -> Tensor:
    r"""Returns the indices that sort the tensor :obj:`src` along a given
    dimension in ascending order by value.
    In contrast to :meth:`mindspore.argsort`, sorting is performed in groups
    according to the values in :obj:`index`.

    Args:
        src (Tensor): The source tensor.
        index (Tensor): The index tensor.
        dim (int, optional): The dimension along which to index.
            (default: :obj:`0`)
        num_groups (int, optional): The number of groups.
            (default: :obj:`None`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
        return_consecutive (bool, optional): If set to :obj:`True`, will not
            offset the output to start from :obj:`0` for each group.
            (default: :obj:`False`)
        stable (bool, optional): Controls the relative order of equivalent
            elements. (default: :obj:`False`)

    Example:
        >>> src = Tensor([0, 1, 5, 4, 3, 2, 6, 7, 8])
        >>> index = Tensor, 1, 1, 1, 1, 2, 2, 2])
        >>> group_argsort(src, index)
        tensor([0, 1, 3, 2, 1, 0, 0, 1, 2])
    """
    # Only implemented under certain conditions for now :(
    assert src.dim() == 1 and index.dim() == 1
    assert dim == 0 or dim == -1
    assert src.numel() == index.numel()

    if src.numel() == 0:
        return mint.zeros_like(src)

    # Normalize `src` to range [0, 1]:
    src = src - src.min()
    src = src / src.max()

    # Compute `grouped_argsort`:
    src = src - 2 * index if descending else src + 2 * index
    perm = src.argsort(descending=descending)
    out = 0 - mint.ones_like(index)
    out[perm] = mint.arange(index.numel())

    if return_consecutive:
        return out

    # Compute cumulative sum of number of entries with the same index:
    count = scatter(
        mint.ones_like(index), index, dim=dim, dim_size=num_groups, reduce="sum"
    )
    ptr = cumsum(count)

    return out - ptr[index]


def group_cat(
    tensors: Union[List[Tensor], Tuple[Tensor, ...]],
    indices: Union[List[Tensor], Tuple[Tensor, ...]],
    axis: int = 0,
    return_index: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Concatenates the given sequence of tensors :obj:`tensors` in the given
    dimension :obj:`dim`.
    Different from :meth:`ms.cat`, values along the concatenating dimension
    are grouped according to the indicies defined in the :obj:`index` tensors.
    All tensors must have the same shape (except in the concatenating
    dimension).

    Args:
        tensors ([Tensor]): Sequence of tensors.
        indices ([Tensor]): Sequence of index tensors.
        dim (int, optional): The dimension along which the tensors are
            concatenated. (default: :obj:`0`)
        return_index (bool, optional): If set to :obj:`True`, will return the
            new index tensor. (default: :obj:`False`)

    Example:
        >>> x1 = ms.Tensor([[0.2716, 0.4233],
        ...                    [0.3166, 0.0142],
        ...                    [0.2371, 0.3839],
        ...                    [0.4100, 0.0012]])
        >>> x2 = ms.Tensor([[0.3752, 0.5782],
        ...                    [0.7757, 0.5999]])
        >>> index1 = ms.Tensor([0, 0, 1, 2])
        >>> index2 = ms.Tensor([0, 2])
        >>> scatter_concat([x1,x2], [index1, index2], axis=0)
        tensor([[0.2716, 0.4233],
                [0.3166, 0.0142],
                [0.3752, 0.5782],
                [0.2371, 0.3839],
                [0.4100, 0.0012],
                [0.7757, 0.5999]])
    """
    assert len(tensors) == len(indices)
    index, perm = mint.sort(mint.cat(indices))
    out = mint.cat(tensors, dim=axis)[perm]
    return (out, index) if return_index else out


def scatter_concat(
    tensors: Union[List[Tensor], Tuple[Tensor, ...]],
    indices: Union[List[Tensor], Tuple[Tensor, ...]],
    axis: int = 0,
    return_index: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Concatenates the given sequence of tensors :obj:`tensors` in the given
    dimension :obj:`dim`.
    Different from :meth:`ops.cat`, values along the concatenating dimension
    are grouped according to the indicies defined in the :obj:`index` tensors.
    All tensors must have the same shape (except in the concatenating
    dimension).

    Args:
        tensors ([Tensor]): Sequence of tensors.
        indices ([Tensor]): Sequence of index tensors.
        dim (int, optional): The dimension along which the tensors are
            concatenated. (default: :obj:`0`)
        return_index (bool, optional): If set to :obj:`True`, will return the
            new index tensor. (default: :obj:`False`)

    Example:
        >>> x1 = Tensor([[0.2716, 0.4233],
        ...                    [0.3166, 0.0142],
        ...                    [0.2371, 0.3839],
        ...                    [0.4100, 0.0012]])
        >>> x2 = Tensor([[0.3752, 0.5782],
        ...                    [0.7757, 0.5999]])
        >>> index1 = Tensor([0, 0, 1, 2])
        >>> index2 = Tensor
        >>> scatter_concat([x1,x2], [index1, index2], dim=0)
        tensor([[0.2716, 0.4233],
                [0.3166, 0.0142],
                [0.3752, 0.5782],
                [0.2371, 0.3839],
                [0.4100, 0.0012],
                [0.7757, 0.5999]])
    """
    assert len(tensors) == len(indices)
    index = mint.cat(indices)
    perm = index.argsort()
    index = index[perm]
    out = mint.cat(tensors, dim=axis)[perm]
    return (out, index) if return_index else out
