from setuptools import setup, find_packages

setup(
    name="Omatbench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
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
        "mace-torch>=0.3.12",
        "torch>=2.6",
        "e3nn>=0.4",
        "dscribe>=2.1",
        "matscipy>=1.1",
    ],
    python_requires=">=3.9",
)
