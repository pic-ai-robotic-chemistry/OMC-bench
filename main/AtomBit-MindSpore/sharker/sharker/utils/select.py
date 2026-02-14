from typing import Any, List, Union

import numpy as np
import mindspore as ms
from mindspore import ops, mint

from .mask import mask_select, mask_select_np


def select(
    src: Union[ms.Tensor, np.ndarray, List[Any]],
    index_or_mask: Union[ms.Tensor, np.ndarray],
    axis: int,
) -> Union[ms.Tensor, np.ndarray, List[Any]]:
    r"""Selects the input tensor or input list according to a given index or
    mask vector.

    Args:
        src (Tensor or list): The input tensor or list.
        index_or_mask (Tensor): The index or mask vector.
        axis (int): The dimension along which to select.
    """
    if isinstance(src, ms.Tensor):
        if index_or_mask.dtype == ms.bool_:
            return mask_select(src, axis=axis, mask=index_or_mask)
        return mint.index_select(src, dim=axis, index=index_or_mask)

    if isinstance(src, np.ndarray):
        if index_or_mask.dtype == np.bool_:
            return mask_select_np(src, axis=axis, mask=index_or_mask)
        return np.take(src, index_or_mask, axis=axis)

    if isinstance(src, (tuple, list)):
        if axis != 0:
            raise ValueError("Cannot select along dimension other than 0")
        if index_or_mask.dtype == ms.bool_:
            return [src[i] for i, m in enumerate(index_or_mask) if m]
        return [src[i] for i in index_or_mask]

    raise ValueError(f"Encountered invalid input type (got '{type(src)}')")


def narrow(
    src: Union[ms.Tensor, List[Any]], axis: int, start: int, length: int
) -> Union[ms.Tensor, List[Any]]:
    r"""Narrows the input tensor or input list to the specified range.

    Args:
        src (Tensor or list): The input tensor or list.
        axis (int): The dimension along which to narrow.
        start (int): The starting dimension.
        length (int): The distance to the ending dimension.
    """
    if isinstance(src, ms.Tensor):
        return mint.narrow(src, axis, start, length)
    if isinstance(src, np.ndarray):
        if axis == 0:
            return src[start:start+length]
        else:
            src = src.swapaxes(0, axis)
            out = src[start:start+length]
            return out.swapaxes(0, axis)
    if isinstance(src, list):
        if axis != 0:
            raise ValueError("Cannot narrow along dimension other than 0")
        return src[start: start + length]

    raise ValueError(f"Encountered invalid input type (got '{type(src)}')")
