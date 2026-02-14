from copy import copy
from typing import Dict, Optional, Tuple, Union

import numpy as np
import mindspore as ms

from mindspore import Tensor, COOTensor, CSRTensor


def maybe_num_nodes(
    edge_index: Union[Tensor, Tuple[Tensor, Tensor]],
    num_nodes: Optional[int] = None,
) -> int:
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    elif isinstance(edge_index, np.ndarray):
        return int(edge_index.max()) + 1 if edge_index.size > 0 else 0
    elif isinstance(edge_index, tuple):
        if isinstance(edge_index[0], Tensor):
            return max(
                int(edge_index[0].max()) + 1 if edge_index[0].numel() > 0 else 0,
                int(edge_index[1].max()) + 1 if edge_index[1].numel() > 0 else 0,
            )
        elif isinstance(edge_index[0], np.ndarray):
            return max(
                int(edge_index[0].max()) + 1 if edge_index[0].size > 0 else 0,
                int(edge_index[1].max()) + 1 if edge_index[1].size > 0 else 0,
            )
    elif isinstance(edge_index, (COOTensor, CSRTensor)):
        return max(edge_index.shape[0], edge_index.shape[1])
    else:
        raise NotImplementedError


def maybe_num_nodes_dict(
    edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    num_nodes_dict: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    num_nodes_dict = {} if num_nodes_dict is None else copy(num_nodes_dict)

    found_types = list(num_nodes_dict.keys())

    for keys, edge_index in edge_index_dict.items():

        key = keys[0]
        if key not in found_types:
            N = int(edge_index[0].max() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        key = keys[-1]
        if key not in found_types:
            N = int(edge_index[1].max() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

    return num_nodes_dict
