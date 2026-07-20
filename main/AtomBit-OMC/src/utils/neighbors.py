from __future__ import annotations

from typing import Any

try:
    from matscipy.neighbours import neighbour_list as _neighbor_list_impl
except ImportError:  # pragma: no cover
    from ase.neighborlist import neighbor_list as _neighbor_list_impl


def neighbor_list(quantities: str, atoms: Any, cutoff: float, *args, **kwargs):
    """ASE-compatible neighbor list wrapper backed by matscipy when available."""

    return _neighbor_list_impl(quantities, atoms, cutoff, *args, **kwargs)
