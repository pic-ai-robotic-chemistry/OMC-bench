import importlib
import os
from typing import Optional

from torch.utils.cpp_extension import load


def env_flag_is_true(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def load_prebuilt_or_jit(
    module_name: str,
    fallback_name: str,
    sources: list[str],
    extra_cflags: Optional[list[str]] = None,
    extra_cuda_cflags: Optional[list[str]] = None,
):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if env_flag_is_true("ATOMBIT_DISABLE_OPS_JIT"):
            raise

    return load(
        name=fallback_name,
        sources=sources,
        verbose=False,
        extra_cflags=extra_cflags or [],
        extra_cuda_cflags=extra_cuda_cflags or [],
    )
