from setuptools import setup, find_packages

core_requires = [
    "ase>=3.24",
    "phonopy>=2.43",
    "numpy>=1.26",
    "pandas>=2.2",
    "scipy>=1.15",
    "tqdm>=4.67",
    "spglib>=2.6",
    "pymatgen>=2025.5",
    "matplotlib>=3.10",
    "scikit-learn>=1.6",
    "joblib>=1.3",
    "h5py>=3.10",
    "pyyaml>=6.0",
    "seekpath>=2.1",
    "fsspec>=2024.3",
]

extras = {
    "mlip": [
        "torch>=2.6",
        "torch-geometric>=2.5",
        "torch-ema>=0.3",
        "e3nn>=0.4",
        "mace-torch>=0.3.12",
    ],
    "mindspore": [
        "mindspore>=2.2",
    ],
    "analysis": [
        "dscribe>=2.1",
        "matscipy>=1.1",
    ],
}
extras["core"] = core_requires

setup(
    name="OMC-bench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=core_requires,
    extras_require=extras,
    python_requires=">=3.9",
)
