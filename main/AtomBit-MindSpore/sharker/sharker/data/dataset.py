import copy
import os
import re
import sys
import warnings
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import mindspore as ms
from mindspore import ops, mint

from .graph import Graph
from ..io import fs

IndexType = Union[slice, ms.Tensor, np.ndarray, Sequence]
MISSING = "???"


class Dataset:
    r"""Dataset base class for creating graph datasets.
    See `here <https://sharker.readthedocs.io/en/latest/tutorial/
    create_dataset.html>`__ for the accompanying tutorial.

    Args:
        root (str, optional): Root directory where the dataset should be saved.
            (optional: :obj:`None`)
        transform (callable, optional): A function/transform that takes in a
            :class:`~sharker.data.Graph` or
            :class:`~sharker.data.HeteroGraph` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~sharker.data.Graph` or
            :class:`~sharker.data.HeteroGraph` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~sharker.data.Graph` or
            :class:`~sharker.data.HeteroGraph` object and returns a
            boolean value, indicating whether the data object should be
            included in the final dataset. (default: :obj:`None`)
        log (bool, optional): Whether to print any console output while
            downloading and processing the dataset. (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        """
        raise NotImplementedError

    def download(self) -> None:
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self) -> None:
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def len(self) -> int:
        r"""Returns the number of data objects stored in the dataset."""
        raise NotImplementedError

    def get(self, idx: int) -> Graph:
        r"""Gets the data object at index :obj:`idx`."""
        raise NotImplementedError

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = os.path.expanduser(fs.normpath(root))

        self.root = root or MISSING
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.log = log
        self._indices: Optional[Sequence] = None
        self.force_reload = force_reload

        if self.has_download:
            self._download()

        if self.has_process:
            self._process()

    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = self[0]
        # Do not fill cache for `InMemoryDataset`:
        if hasattr(self, "_data_list") and self._data_list is not None:
            self._data_list[0] = None
        data = data[0] if isinstance(data, tuple) else data
        if hasattr(data, "num_node_features"):
            return data.num_node_features
        raise AttributeError(
            f"'{data.__class__.__name__}' object has no "
            f"attribute 'num_node_features'"
        )

    @property
    def num_features(self) -> int:
        r"""Returns the number of features per node in the dataset.
        Alias for :py:attr:`~num_node_features`.
        """
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the dataset."""
        data = self[0]
        # Do not fill cache for `InMemoryDataset`:
        if hasattr(self, "_data_list") and self._data_list is not None:
            self._data_list[0] = None
        data = data[0] if isinstance(data, tuple) else data
        if hasattr(data, "num_edge_features"):
            return data.num_edge_features
        raise AttributeError(
            f"'{data.__class__.__name__}' object has no "
            f"attribute 'num_edge_features'"
        )

    def _infer_num_classes(self, y: Optional[ms.Tensor]) -> int:
        if y is None:
            return 0
        elif y.numel() == y.shape[0] and not ops.is_floating_point(y):
            return int(y.max()) + 1
        elif y.numel() == y.shape[0] and ops.is_floating_point(y):
            num_classes = ops.unique(y)[0].numel()
            if num_classes > 2:
                warnings.warn(
                    "Found floating-point labels while calling "
                    "`dataset.num_classes`. Returning the number of "
                    "unique elements. Please make sure that this "
                    "is expected before proceeding."
                )
            return num_classes
        else:
            return y.shape[-1]

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        # We iterate over the dataset and collect all labels to determine the
        # maximum number of classes. Importantly, in rare cases, `__getitem__`
        # may produce a tuple of data objects (e.g., when used in combination
        # with `RandomLinkSplit`, so we take care of this case here as well:
        data_list = _flatten([data for data in self])
        if "y" in data_list[0] and isinstance(data_list[0].y, ms.Tensor):
            y = mint.cat([data.y for data in data_list if "y" in data], dim=0)
        else:
            y = ms.Tensor([data.y for data in data_list if "y" in data])

        # Do not fill cache for `InMemoryDataset`:
        if hasattr(self, "_data_list") and self._data_list is not None:
            self._data_list = self.len() * [None]
        return self._infer_num_classes(y)

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading.
        """
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [os.path.join(self.raw_dir, f) for f in to_list(files)]

    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing.
        """
        files = self.processed_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [os.path.join(self.processed_dir, f) for f in to_list(files)]

    @property
    def has_download(self) -> bool:
        r"""Checks whether the dataset defines a :meth:`download` method."""
        return overrides_method(self.__class__, "download")

    def _download(self):
        if len(self.raw_paths) != 0 and all([fs.exists(f) for f in self.raw_paths]):
            return

        fs.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    @property
    def has_process(self) -> bool:
        r"""Checks whether the dataset defines a :meth:`process` method."""
        return overrides_method(self.__class__, "process")

    def _process(self):

        if not self.force_reload:
            if len(self.processed_paths) != 0 and all(
                [fs.exists(f) for f in self.processed_paths]
            ):
                return

        if self.log and "pytest" not in sys.modules:
            print("Processing...", file=sys.stderr)

        fs.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        path = os.path.join(self.processed_dir, "pre_transform.pt")
        fs.pickle_save(_repr(self.pre_transform), path)
        path = os.path.join(self.processed_dir, "pre_filter.pt")
        fs.pickle_save(_repr(self.pre_filter), path)

        if self.log and "pytest" not in sys.modules:
            print("Done!", file=sys.stderr)

    def __len__(self) -> int:
        r"""The number of examples in the dataset."""
        return len(self.indices())

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ) -> Union["Dataset", Graph]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices.
        """
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, ms.Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data

        else:
            return self.index_select(idx)

    def __iter__(self) -> Iterator[Graph]:
        for i in range(len(self)):
            yield self[i]

    def index_select(self, idx: IndexType) -> "Dataset":
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`Tensor` or :obj:`np.ndarray` of type
        long or bool.
        """
        indices = self.indices()

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            indices = indices[idx]

        elif isinstance(idx, ms.Tensor) and idx.dtype == ms.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, ms.Tensor) and idx.dtype == ms.bool_:
            idx = idx.flatten().nonzero()
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, Tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')"
            )

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def shuffle(
        self,
        return_perm: bool = False,
    ) -> Union["Dataset", Tuple["Dataset", ms.Tensor]]:
        r"""Randomly shuffles the examples in the dataset.

        Args:
            return_perm (bool, optional): If set to :obj:`True`, will also
                return the random permutation used to shuffle the dataset.
                (default: :obj:`False`)
        """
        perm = ops.shuffle(mint.arange(len(self)))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ""
        return f"{self.__class__.__name__}({arg_repr})"

    def get_summary(self) -> Any:
        r"""Collects summary statistics for the dataset (removed in slim build)."""
        raise NotImplementedError("Summary support was removed in this build.")

    def print_summary(self) -> None:
        r"""Prints summary statistics of the dataset to the console (removed in slim build)."""
        raise NotImplementedError("Summary support was removed in this build.")

    def to_datapipe(self) -> Any:
        r"""Converts the dataset into a DatasetAdapter (removed in slim build)."""
        raise NotImplementedError("Datapipe support was removed in this build.")


def overrides_method(cls, method_name: str) -> bool:
    if method_name in cls.__dict__:
        return True

    out = False
    for base in cls.__bases__:
        if base is not Dataset and base is not object:
            out |= overrides_method(base, method_name)
    return out


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def _repr(obj: Any) -> str:
    if obj is None:
        return "None"
    return re.sub("(<.*?)\\s.*(>)", r"\1\2", str(obj))


def _flatten(data_list: Iterable[Any]) -> List[Graph]:
    outs: List[Graph] = []
    for data in data_list:
        if isinstance(data, Graph):
            outs.append(data)
        elif isinstance(data, (tuple, list)):
            outs.extend(_flatten(data))
        elif isinstance(data, dict):
            outs.extend(_flatten(data.values()))
    return outs
