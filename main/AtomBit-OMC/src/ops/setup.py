import os
import shutil
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


OPS_DIR = Path(__file__).resolve().parent
ROOT = OPS_DIR.parent.parent


def _pick_host_compiler() -> str | None:
    """
    Prefer a real system compiler over conda's cross-compiler wrapper.
    Override order:
      1. ATOMBIT_NVCC_CCBIN
      2. CXX
      3. system g++ / c++
    """
    for env_name in ("ATOMBIT_NVCC_CCBIN", "CUDAHOSTCXX", "CXX"):
        candidate = os.environ.get(env_name)
        if candidate:
            return candidate

    for candidate in (
        "/usr/bin/g++",
        "/bin/g++",
        "/usr/bin/c++",
        "/bin/c++",
    ):
        if os.path.exists(candidate):
            return candidate

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    for compiler_name in ("g++", "c++"):
        candidate = shutil.which(compiler_name)
        if candidate and (not conda_prefix or not candidate.startswith(conda_prefix)):
            return candidate

    for compiler_name in ("g++", "c++"):
        candidate = shutil.which(compiler_name)
        if candidate:
            return candidate

    return None


DEFAULT_CC = "/usr/bin/gcc"
DEFAULT_CXX = "/usr/bin/g++"

os.environ.setdefault("CC", DEFAULT_CC)
os.environ.setdefault("CXX", DEFAULT_CXX)
os.environ.setdefault("CUDAHOSTCXX", DEFAULT_CXX)
os.environ.setdefault("ATOMBIT_NVCC_CCBIN", DEFAULT_CXX)

HOST_COMPILER = _pick_host_compiler() or os.environ["ATOMBIT_NVCC_CCBIN"]


class InplaceBuildExtension(BuildExtension):
    """
    Force `build_ext --inplace` to place compiled extensions into the source tree
    under `<repo>/src/...`, even though this setup.py lives in `src/ops`.
    """

    def get_ext_fullpath(self, ext_name):
        fullname = self.get_ext_fullname(ext_name)
        filename = self.get_ext_filename(fullname)

        if self.inplace:
            relative_path = Path(*fullname.split("."))
            return str(ROOT / relative_path.parent / Path(filename).name)

        return super().get_ext_fullpath(ext_name)


def make_cuda_extension(name: str, api_cpp: str, kernel_cu: str) -> CUDAExtension:
    nvcc_flags = ["-O3"]
    if HOST_COMPILER:
        nvcc_flags.extend(["-ccbin", HOST_COMPILER])

    return CUDAExtension(
        name=name,
        sources=[
            str(OPS_DIR / api_cpp),
            str(OPS_DIR / kernel_cu),
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": nvcc_flags,
        },
    )


setup(
    name="atombit-ops",
    packages=find_packages(where=str(ROOT)),
    package_dir={"": str(ROOT)},
    ext_modules=[
        make_cuda_extension(
            "src.ops._gating_proj_cuda",
            "gating_proj_api.cpp",
            "gating_proj_kernel.cu",
        ),
        make_cuda_extension(
            "src.ops._mat_mul_sym_cuda",
            "mat_mul_sym_api.cpp",
            "mat_mul_sym_kernel.cu",
        ),
        make_cuda_extension(
            "src.ops._outer_sym_detrace_cuda",
            "outer_sym_detrace_api.cpp",
            "outer_sym_detrace_kernel.cu",
        ),
    ],
    cmdclass={"build_ext": InplaceBuildExtension},
)
