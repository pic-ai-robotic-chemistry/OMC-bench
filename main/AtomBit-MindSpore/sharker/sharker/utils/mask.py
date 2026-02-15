from typing import Optional

import mindspore as ms
from mindspore import mint
import numpy as np


def mask_select(src: ms.Tensor, axis: int, mask: ms.Tensor) -> ms.Tensor:
    r"""Returns a new tensor which masks the :obj:`src` tensor along the
    dimension :obj:`dim` according to the boolean mask :obj:`mask`.

    Args:
        src (Tensor): The input tensor.
        dim (int): The dimension in which to mask.
        mask (mindspore.BoolTensor): The 1-D tensor containing the binary mask to
            index with.
    """
    assert mask.dim() == 1

    assert src.shape[axis] == mask.numel()
    axis += src.dim() if axis < 0 else axis
    assert axis >= 0 and axis < src.dim()

    # Applying a 1-dimensional mask in the first dimension is significantly
    # faster than broadcasting the mask and utilizing `masked_select`.
    # As such, we transpose in the first dimension, perform the masking, and
    # then transpose back to the original shape.
    idx = mint.nonzero(mask).reshape(-1)
    if axis != 0:
        out = mint.index_select(src, 1, idx)
    else:
        out = mint.index_select(src, 0, idx)

    return out

def mask_select_np(src: np.ndarray, axis: int, mask: np.ndarray) -> np.ndarray:
    assert mask.ndim == 1

    assert src.shape[axis] == mask.size
    axis += src.ndim if axis < 0 else axis
    assert axis >= 0 and axis < src.ndim

    # Applying a 1-dimensional mask in the first dimension is significantly
    # faster than broadcasting the mask and utilizing `masked_select`.
    # As such, we transpose in the first dimension, perform the masking, and
    # then transpose back to the original shape.
    src = src.transpose(axis, 0) if axis != 0 else src
    out = src[mask]
    out = out.transpose(axis, 0) if axis != 0 else out

    return out


def index_to_mask(index: ms.Tensor, size: Optional[int] = None) -> ms.Tensor:
    r"""Converts indices to a mask representation.

    Args:
        index (Tensor): The indices.
        size (int, optional): The size of the mask. If set to :obj:`None`, a
                              minimal sized output mask is returned.

    Example:
        >>> index = ms.Tensor([1, 3, 5])
        >>> index_to_mask(index)
        tensor([False,  True, False,  True, False,  True])

        >>> index_to_mask(index, size=7)
        tensor([False,  True, False,  True, False,  True, False])
    """
    index = index.view(-1).astype(ms.int32)
    size = mint.max(index) + 1 if size is None else size
    mask = mint.zeros(size)
    mask = ms.ops.index_fill(mask, 0, index, True).astype(ms.bool_)

    return mask

def index_to_mask_np(index: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    index = index.reshape(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = np.zeros(size)
    mask[index] = True
    return mask.astype(np.bool_)


def mask_to_index(mask: ms.Tensor) -> ms.Tensor:
    r"""Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.

    Example:
        >>> mask = Tensore, True, False])
        >>> mask_to_index(mask)
        tensor([1])
    """
    return mint.nonzero(mask).view(-1)

def mask_to_index_np(mask: np.array) -> np.array:
    return mint.nonzero(mask).view(-1)
