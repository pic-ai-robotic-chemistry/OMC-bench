import mindspore as ms
from typing import Union 
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor

def cumsum(x: ms.Tensor, axis: int = 0) -> ms.Tensor:
    r"""Returns the cumulative sum of elements of :obj:`x`.
    In contrast to :meth:`mindspore.cumsum`, prepends the output with zero.

    Args:
        x (Tensor): The input tensor.
        axis (int, optional): The dimension to do the operation over.
            (default: :obj:`0`)

    Example:
        >>> x = Tensor([2, 4, 1])
        >>> cumsum(x)
        tensor([0, 2, 6, 7])

    """
    size = list(x.shape)
    size[axis] = 1
    pad_front = mint.zeros(size, dtype=x.dtype)
    x_cum = mint.cumsum(x, dim=axis)
    out = mint.cat([pad_front, x_cum], dim=axis)
    return out


def cumsum_np(x: np.ndarray, axis: int = 0) -> np.ndarray:
    temp = np.cumsum(x, axis=axis)
    first_shape = list(x.shape)
    first_shape[axis] = 1
    first = np.zeros(first_shape, dtype=temp.dtype)
    out = np.concatenate([first,temp])
    return out


def broadcast_to(index: Tensor, src: Tensor, dim: int, is_dense=False) -> Tensor:
    r"""Broadcat the index tensor to obtain the detailed information for the scatter operators.
    :param src: The source tensor.
    :param index: The indices of elements to scatter.
    :param dim: The dim along which to index.
    :return: the indices with detailed information.
    """
    if dim > src.ndim-1:
        raise ValueError(f"`dim` must lay between 0 and {src.ndim-1}")
    dim = src.dim() + dim if dim < 0 else dim
    index = index.expand_as(swapaxes(src, -1, dim))
    index = swapaxes(index, -1, dim)
    if is_dense:
        return index

    idx = (index == index).nonzero()
    index = index.view(-1)
    idx[:, dim] = index
    return idx


def swapaxes(tensor: Tensor, dim0, dim1):
    if dim0 == dim1 or dim1 == dim0 - tensor.ndim or dim0 == dim1 - tensor.ndim or tensor.ndim < 2:
        return tensor
    else:
        return tensor.swapaxes(dim0, dim1)


def index_fill(tensor, index, value, dim=0):
    tensor = swapaxes(tensor, 0, dim) 
    tensor[index] = swapaxes(value, 0, dim)
    return swapaxes(tensor, 0, dim)


def index_select(tensor, mask, dim=0):
    tensor = swapaxes(tensor, 0, dim)
    out = tensor[mask]
    return swapaxes(out, 0, dim)
