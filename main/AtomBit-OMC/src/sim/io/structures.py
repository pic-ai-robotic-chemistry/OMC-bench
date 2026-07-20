import io
from pathlib import Path

import ase.io
from ase import Atoms


_FORMAT_ALIASES = {
    "cif": "cif",
    "xyz": "extxyz",
    "extxyz": "extxyz",
    "vasp": "vasp",
    "poscar": "vasp",
}

_SUFFIX_TO_FORMAT = {
    ".cif": "cif",
    ".xyz": "extxyz",
    ".extxyz": "extxyz",
    ".vasp": "vasp",
    ".poscar": "vasp",
}


def normalize_structure_format(structure_format: str) -> str:
    fmt = structure_format.lower().strip()
    if fmt not in _FORMAT_ALIASES:
        raise ValueError(f"Unsupported structure_format: {structure_format}")
    return _FORMAT_ALIASES[fmt]


def infer_structure_format_from_path(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in _SUFFIX_TO_FORMAT:
        raise ValueError(f"Unsupported file suffix: {suffix}")
    return _SUFFIX_TO_FORMAT[suffix]


def parse_structure_text(structure_text: str, structure_format: str) -> Atoms:
    fmt = normalize_structure_format(structure_format)
    return ase.io.read(io.StringIO(structure_text), format=fmt)


def serialize_atoms(atoms: Atoms, structure_format: str = "cif") -> str:
    fmt = normalize_structure_format(structure_format)
    text_buffer = io.StringIO()
    try:
        ase.io.write(text_buffer, atoms, format=fmt)
        return text_buffer.getvalue()
    except TypeError:
        byte_buffer = io.BytesIO()
        ase.io.write(byte_buffer, atoms, format=fmt)
        return byte_buffer.getvalue().decode("utf-8")
