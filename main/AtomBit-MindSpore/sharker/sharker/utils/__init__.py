r'''Utility package (slim build: only symbols needed for Graph + Dataloader).'''

from ._scatter import scatter, group_argsort, group_cat, scatter_concat
from .functions import cumsum, cumsum_np, swapaxes, index_fill, index_select, broadcast_to
from .num_nodes import maybe_num_nodes
from .coalesce import coalesce, coalesce_np
from .sort_edge_index import sort_edge_index, sort_edge_index_np
from .undirected import is_undirected, to_undirected
from .loop import (
    contains_self_loops,
    remove_self_loops,
    segregate_self_loops,
    add_self_loops,
    add_remaining_self_loops,
    get_self_loop_attr,
)
from .isolated import contains_isolated_nodes, remove_isolated_nodes
from .mask import mask_select, index_to_mask, mask_to_index
from .select import select, narrow

__all__ = [
    'scatter', 'group_argsort', 'group_cat', 'scatter_concat',
    'cumsum', 'cumsum_np', 'swapaxes', 'index_fill', 'index_select', 'broadcast_to',
    'maybe_num_nodes',
    'coalesce', 'coalesce_np',
    'sort_edge_index', 'sort_edge_index_np',
    'is_undirected', 'to_undirected',
    'contains_self_loops', 'remove_self_loops', 'segregate_self_loops',
    'add_self_loops', 'add_remaining_self_loops', 'get_self_loop_attr',
    'contains_isolated_nodes', 'remove_isolated_nodes',
    'mask_select', 'index_to_mask', 'mask_to_index',
    'select', 'narrow',
]
