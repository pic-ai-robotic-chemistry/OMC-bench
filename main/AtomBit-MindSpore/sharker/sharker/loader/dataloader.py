import json
import math
import time
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore._c_dataengine as cde
import mindspore.dataset.engine.offload as offload
import weakref
from mindspore import log as logger
from mindspore import ops
from mindspore import log as logger
from mindspore.dataset.core.config import get_debug_mode
from mindspore.dataset.engine.datasets import Dataset, BatchDataset
from mindspore.dataset.engine.datasets_user_defined import GeneratorDataset
from mindspore.dataset.engine.iterators import Iterator, DummyIterator, _transform_md_to_output
from mindspore.dataset.engine.validators import check_dict_iterator
from typing import Any, List, Optional, Sequence, Union

from ..data import Graph, Batch

ITERATORS_LIST = list()


def _unset_iterator_cleanup():
    global _ITERATOR_CLEANUP
    _ITERATOR_CLEANUP = False

def collate_default(*kwargs):
    data_list = [x['graph'] for x in kwargs[0]]
    col1 = Batch.from_data_list(data_list, return_tensor=False)
    return {'graph':col1}

from collections.abc import Sequence

class GraphDictView(Sequence):
    """把一个 indexable dataset/list 伪装成 list[{'graph': ...}]，但不 materialize。"""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # 支持 sampler 传入 “一个 batch 的 index 列表”（例如 BinPackingSampler 会 yield list[int]）
        if isinstance(idx, (list, tuple, np.ndarray, ms.Tensor)):
            # ms.Tensor 可能是标量，也可能是 1-D index 列表
            if isinstance(idx, ms.Tensor):
                idx = idx.asnumpy().tolist()
            # numpy 标量兼容
            if isinstance(idx, np.ndarray) and idx.ndim == 0:
                idx = int(idx)
            if isinstance(idx, (list, tuple, np.ndarray)):
                return {'graph': [self.base[int(i)] for i in idx]}
        return {'graph': self.base[idx]}  # 关键：按需取

    def __iter__(self):
        # 可选：让 iter 也按需 yield，避免某些路径走 __getitem__ 太慢
        for i in range(len(self.base)):
            yield {'graph': self.base[i]}



class GraphBatchFromSampler:
    """
    将 batch sampler（yield idx 或 list[idx]）包装成可迭代 source：
    每次迭代产出一个 row：{'graph': Graph} 或 {'graph': list[Graph]}。

    目的：绕开 MindSpore Sampler 侧可能对 list/iterable idx 的“展开”行为，
    从而保证 BinPackingSampler 的 [idx...] 一定对应一个 batch。
    """
    def __init__(self, base, sampler):
        self.base = base
        self.sampler = sampler

    def __iter__(self):
        for idx in self.sampler:
            if isinstance(idx, ms.Tensor):
                idx = idx.asnumpy().tolist()
            if isinstance(idx, np.ndarray):
                if idx.ndim == 0:
                    idx = int(idx)
                else:
                    idx = idx.tolist()

            if isinstance(idx, (list, tuple)):
                yield {'graph': [self.base[int(i)] for i in idx]}
            else:
                yield {'graph': self.base[int(idx)]}

    def __len__(self):
        return len(self.sampler)


class Dataloader(GeneratorDataset):
    """
    A Dataloader that generates data from Python by invoking Python data source each epoch.

    If the type in source contains Graph, the column name will be automatically recognized.
    Otherwisw the column names and column types of generated dataset depend on Python data defined by users.

    Args:
        source (Union[Callable, Iterable, Random Accessible]):
            A generator callable object, an iterable Python object or a random accessible Python object.
            Callable source is required to return a tuple of NumPy arrays as a row of the dataset on source().next().
            Iterable source is required to return a tuple of NumPy arrays as a row of the dataset on
            iter(source).next().
            Random accessible source is required to return a tuple of NumPy arrays as a row of the dataset on
            source[idx].
        column_names (Union[str, list[str]], optional): List of column names of the dataset. Default: ``None`` .
            Users are required to provide either column_names or schema.
        column_types (list[mindspore.dtype], optional): List of column data types of the dataset. Default: ``None`` .
            If provided, sanity check will be performed on generator output.
        schema (Union[str, Schema], optional): Data format policy, which specifies the data types and shapes of the data
            column to be read. Both JSON file path and objects constructed by :class:`mindspore.dataset.Schema` are
            acceptable. Default: ``None`` .
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: ``None`` , all images.
        num_parallel_workers (int, optional): Number of worker threads/subprocesses used to
            fetch the dataset in parallel. Default: ``1``.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is required.
            Default: ``None`` , expected order behavior shown in the table below.
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset. Random accessible
            input is required. Default: ``None`` , expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: ``None`` .
            Random accessible input is required. When this argument is specified, `num_samples` reflects the maximum
            sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None`` .
            This argument must be specified only when `num_shards` is also specified.
            Random accessible input is required.
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy. Default: ``True``.
        max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory
            allocation to copy data between processes, the total occupied shared memory will increase as
            ``num_parallel_workers`` and :func:`mindspore.dataset.config.set_prefetch_size` increase. If set to -1,
            shared memory will be dynamically allocated with the actual size of data. This is only used if
            ``python_multiprocessing`` is set to True. Default: 16.
        """

    def __init__(self, source, column_names=['graph'], column_types=None, schema=None, num_samples=None,
                 num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None,
                 python_multiprocessing=False, max_rowsize=-1, max_node_per_batch=None, max_graph_per_batch=None,
                 prefetch_factor=None):
        self.dataset_is_graph = False
        if not callable(source) and isinstance(source[0], Graph):
            self.dataset_is_graph = True
        self._output_numpy = False
        if self.dataset_is_graph:
            self._output_numpy = True
            # 传了 sampler（例如 BinPackingSampler，yield list[int]）时：
            # 用 Python source 直接按 batch 拉取，避免 MindSpore sampler 展开 batch。
            if sampler is not None:
                source = GraphBatchFromSampler(source, sampler)
            else:
                source = GraphDictView(source)

        self.iterator = []
        # 保存 sampler 引用：后续 __len__ 可以动态反映当前 epoch 的 batch 数（BinPackingSampler.set_epoch 会改变它）
        self._sampler = sampler
        # num_samples 仅作为 fallback / 日志用途，不要在这里调用 len(sampler)，否则可能与 epoch 绑定导致误判为 0
        self.num_samples = len(source) if num_samples is None else num_samples

        # PyTorch 风格 prefetch_factor: 每个 worker 预取的 batch/样本数量
        # MindSpore 用全局 prefetch_size（单位：row），这里用 prefetch_factor * num_parallel_workers 做近似映射
        if prefetch_factor is not None:
            try:
                pf = int(prefetch_factor)
                nwp = int(num_parallel_workers) if num_parallel_workers is not None else 1
                if pf > 0:
                    ds.config.set_prefetch_size(max(1, pf * max(1, nwp)))
            except Exception as e:
                logger.warning(f"Failed to apply prefetch_factor={prefetch_factor}: {e}")

        # 如果我们走 GraphBatchFromSampler，就不再把 sampler 传给 MindSpore（避免 batch 被展开）
        ms_sampler = None if (self.dataset_is_graph and sampler is not None) else sampler
        ms_num_samples = None if ms_sampler is not None else self.num_samples
        super().__init__(source, column_names=column_names, column_types=column_types, schema=schema,
                         num_samples=ms_num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=ms_sampler,
                         num_shards=num_shards, shard_id=shard_id,
                         python_multiprocessing=python_multiprocessing, max_rowsize=max_rowsize)
        self.max_node_per_batch = max_node_per_batch
        self.max_graph_per_batch = max_graph_per_batch

    def __iter__(self):
        """Create an iterator over the dataset."""
        if not self.iterator:
            self.iterator = self.create_dict_iterator(num_epochs=-1, output_numpy=self._output_numpy)
        return self.iterator
    
    def __len__(self):
        # 动态 batch：len(loader) 定义为 “number of batches”，直接从 sampler 读取（跟随 set_epoch 变化）
        if self._sampler is not None:
            return len(self._sampler)
        # 没有 sampler 的情况下，回退为 dataset 的 row 数（与 MindSpore GeneratorDataset 语义一致）
        return self.num_samples

    def batch(self, batch_size, drop_remainder=False, num_parallel_workers=None,
              per_batch_map=collate_default, input_columns=["graph"], output_columns=["graph"], **kwargs):
        """
        Combine batch_size number of consecutive rows into batch which apply per_batch_map or collate_fn to the samples first.

        If the type in source contains Graph, all the elements within that column do not need to have the same shape.
        Otherwise, for any column, all the elements within that column must have the same shape.

        Refer to the following figure for the execution process:

        .. image:: batch_en.png

        Note:
            The order of using repeat and batch reflects the number of batches and (er_batch_map or collate_fn).
            It is recommended that the repeat operation applied after the batch operation finished.

        Args:
            batch_size (Union[int, Callable]): The number of rows each batch is created with. An
                int or callable object which takes exactly 1 parameter, BatchInfo.
            drop_remainder (bool, optional): Determines whether or not to drop the last block
                whose data row number is less than batch size. Default: ``False`` . If ``True`` ,
                and if there are less than `batch_size` rows available to make the last batch,
                then those rows will be dropped and not propagated to the child node.
            num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel.
                Default: ``None`` .
            **kwargs:

                - per_batch_map (Callable[[List[numpy.ndarray], ..., List[numpy.ndarray], BatchInfo], \
                  (List[numpy.ndarray], ..., List[numpy.ndarray])], optional): Per batch map callable.
                  Default: ``None``.
                  A callable which takes (List[numpy.ndarray], ..., List[numpy.ndarray], BatchInfo) as input parameters.
                  Each list[numpy.ndarray] represents a batch of numpy.ndarray on a given column. The number of lists
                  should match with the number of entries in input_columns. The last parameter of the callable should
                  always be a BatchInfo object. Per_batch_map should return
                  (list[numpy.ndarray], list[numpy.ndarray], ...). The length of each list in output should be the same
                  as the input. output_columns is required if the number of output lists is different from input.

                - input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of
                  the list should match with signature of `per_batch_map` callable. Default: ``None`` .

                - output_columns (Union[str, list[str]], optional): List of names assigned to the columns
                  outputted by the last operation. This parameter is mandatory if len(input_columns) !=
                  len(output_columns). The size of this list must match the number of output
                  columns of the last operation. Default: ``None`` , output columns will have the same
                  name as the input columns, i.e., the columns will be replaced.

                - python_multiprocessing (bool, optional): Parallelize Python function `per_batch_map` with
                  multi-processing or multi-threading mode, ``True`` means multi-processing,
                  ``False`` means multi-threading If `per_batch_map` is a I/O bound task, use
                  multi-threading mode. If `per_batch_map` is a CPU bound task, it is recommended to use
                  multi-processing mode. Default: ``False`` , use python multi-threading mode.

                - max_rowsize(Union[int, list[int]], optional): Maximum size of row in MB that is used for shared memory
                  allocation to copy data between processes, the total occupied shared memory will increase as
                  ``num_parallel_workers`` and :func:`mindspore.dataset.config.set_prefetch_size` increase. If set
                  to -1, shared memory will be dynamically allocated with the actual size of data. This is only used if
                  ``python_multiprocessing`` is set to True. If it is an int value, it represents
                  ``input_columns`` and ``output_columns`` use this value as the unit to create shared memory.
                  If it is a list, the first element represents the ``input_columns`` use this value as the unit to
                  create shared memory, and the second element represents ``output_columns`` use this value as the unit
                  to create shared memory. Default: 16.

                - collate_fn (Callable, optional): merges a list of samples to form a mini-batch of Tensor(s).
                  Used when using batched loading from a map-style dataset.
        Returns:
            CustomBatchDataset, a new dataset with the above operation applied.
        """
        if (not self.dataset_is_graph):
            per_batch_map = None
            input_columns = None
            output_columns = None
        return CustomBatchDataset(self, batch_size, drop_remainder, self.dataset_is_graph, num_parallel_workers,
                                  per_batch_map, input_columns, output_columns, **kwargs)

    def create_dict_iterator(self, num_epochs=-1, output_numpy=False, do_copy=True):
        """
        Create an CustomDictIterator over the dataset that yields samples of type dict,
        while the key is the column name and the value is the data.

        Args:
            num_epochs (int, optional): The number of epochs to iterate over the entire dataset.
                Default: ``-1`` , the dataset can be iterated indefinitely.
            output_numpy (bool, optional): Whether to keep the output data as NumPy ndarray, or
                convert it to Tensor. Default: ``False`` .
            do_copy (bool, optional): Whether to copy the data when converting output to Tensor,
                or reuse the buffer for better performance, only works when `output_numpy` is ``False`` .
                Default: ``True`` .

        Returns:
            Iterator, a dataset iterator that yields samples of type dict.
        """
        if output_numpy is None:
            output_numpy = False
        if Dataset._noop_mode():
            return DummyIterator(self, 'dict', output_numpy)
        return CustomDictIterator(self, num_epochs, output_numpy, do_copy)


class CustomDictIterator(Iterator):
    """
    The derived class of Iterator with dict type.
    """

    def __init__(self, dataset, num_epochs=-1, output_numpy=False, do_copy=True):
        start_init = time.time()
        super().__init__(dataset, num_epochs=num_epochs, output_numpy=output_numpy, do_copy=do_copy)
        self.dataset_type = None
        self.dataset_is_graph = dataset.dataset_is_graph
        self.max_node_per_batch = getattr(dataset, "max_node_per_batch", None)
        self.max_graph_per_batch = getattr(dataset, "max_graph_per_batch", None)

        self.__ori_dataset = dataset

        self.ir_tree, self.dataset = dataset.create_ir_tree()

        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
        if dataset.get_init_step() == 0:
            init_step = 0
            dataset_size = -1
        else:
            init_step = dataset.get_init_step()
            dataset_size = dataset.get_dataset_size()
        if get_debug_mode():
            consumer = cde.PythonPullBasedIteratorConsumer(num_epochs)
            consumer.Init(self.ir_tree)
        else:
            consumer = cde.PythonIteratorConsumer(num_epochs)
            consumer.Init(self.ir_tree, init_step, dataset_size)
        self._runtime_context.AssignConsumer(consumer)
        self._iterator = self._runtime_context.GetConsumer()
        self._output_numpy = output_numpy
        self._do_copy = do_copy
        self.__index = 0
        self.last_step_end = False
        self.offload_model = None
        json_offload = json.loads(consumer.GetOffload())
        # See if GetOffload identified any operations set to be offloaded.
        if json_offload is not None:
            offload.check_concat_zip_dataset(self.__ori_dataset)
            self.offload_model = offload.GetOffloadModel(consumer, self.__ori_dataset.get_col_names())

        ITERATORS_LIST.append(weakref.ref(self))
        _unset_iterator_cleanup()

    def __next__(self):
        """
        This is the implementation of the __next__() method for an iterator object in Python.
        If the dataset type is 'Graph', it will call self._get_next() depends on the batch size, it then applies a collater
        function if available to combine the data into a batch and returns the batch or data list.
        If there is no collater function, it returns the data list. If there is no more data to iterate over,
        it raises a StopIteration exception.
        """
        if not self._runtime_context:
            logger.warning("Iterator does not have a running C++ pipeline." +
                           "It might because Iterator stop() had been called, or C++ pipeline crashed silently.")
            raise RuntimeError("Iterator does not have a running C++ pipeline.")
        # Note offload is applied inside _get_next() if applicable since get_next converts to output format
        data = self._get_next()
        if not data:
            if self.__index == 0:
                logger.warning("No records available.")
            if self.__ori_dataset.dataset_size is None:
                self.__ori_dataset.dataset_size = self.__index
            self.__index = 0
            raise StopIteration
        self.__index += 1

        # 非 Graph 数据集：保持原行为（返回单条记录，不做 Batch/Collate）
        if not self.dataset_is_graph:
            return data

        # Graph 数据集：
        # - 如果 sampler 返回 list[int]，GraphDictView 会把它映射成 list[Graph]（即一个 batch）
        # - 如果 sampler 返回 int，则拿到单个 Graph，视作 batch_size=1
        batch_list = list(data) if isinstance(data, (list, tuple)) else [data]
        out = Batch.from_data_list(batch_list, return_tensor=False)
        out = out.tensor()
        return out

    def _get_next(self):
        """
        Returns the next record in the dataset as dictionary, and convert the dictionary back to Graph class if required.

        Returns:
            Dict, the next record in the dataset.
        """
        try:
            if self.offload_model is None:
                if self.dataset_is_graph:  #### Graph version
                    data_dict = {}
                    start_time = time.time()
                    for t in self._iterator.GetNextAsList():
                        data_dict = _transform_md_to_output(t,self._output_numpy,self._do_copy)
                    if data_dict:
                        return data_dict['graph']
                    else:
                        return None
                else:
                    return [_transform_md_to_output(t,self._output_numpy,self._do_copy) for k, t in self._iterator.GetNextAsMap().items()]

            data = [self._transform_md_to_tensor(t) for t in self._iterator.GetNextAsList()]
            if data:
                data = offload.apply_offload_iterators(data, self.offload_model)
                # Create output dictionary after offload
                out_data = {}
                for i, col in enumerate(self.get_col_names()):
                    out_data[col] = self._transform_tensor_to_output(data[i])
                data = out_data
            return data
        except RuntimeError as err:
            err_info = str(err)
            if err_info.find("Out of memory") >= 0 or err_info.find("MemoryError") >= 0:
                logger.critical("Memory error occurred, process will exit.")
                os.kill(os.getpid(), signal.SIGKILL)
            raise err


class CustomBatchDataset(BatchDataset):
    """
    The result of applying Batch operation to the input dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be batched.
        batch_size (Union[int, function]): The number of rows each batch is created with. An
            int or callable which takes exactly 1 parameter, BatchInfo.
        drop_remainder (bool, optional): Determines whether or not to drop the last
            possibly incomplete batch. Default: ``False``. If True, and if there are less
            than batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel. Default: ``None``.
        per_batch_map (callable, optional): Per batch map callable. A callable which takes
            (list[Tensor], list[Tensor], ..., BatchInfo) as input parameters. Each list[Tensor] represents a batch of
            Tensors on a given column. The number of lists should match with number of entries in input_columns. The
            last parameter of the callable must always be a BatchInfo object.
        input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of the list must
            match with signature of per_batch_map callable.
        output_columns (Union[str, list[str]], optional): List of names assigned to the columns outputted by
            the last operation. This parameter is mandatory if len(input_columns) !=
            len(output_columns). The size of this list must match the number of output
            columns of the last operation. Default: ``None``, output columns will have the same
            name as the input columns, i.e., the columns will be replaced.
        max_rowsize(Union[int, list[int]], optional): Maximum size of row in MB that is used for shared memory
            allocation to copy data between processes, the total occupied shared memory will increase as
            ``num_parallel_workers`` and :func:`mindspore.dataset.config.set_prefetch_size` increase. If set to -1,
            shared memory will be dynamically allocated with the actual size of data. This is only used if
            ``python_multiprocessing`` is set to True. If it is an int value, it represents
            ``input_columns`` and ``output_columns`` use this value as the unit to create shared memory.
            If it is a list, the first element represents the ``input_columns`` use this value as the unit to
            create shared memory, and the second element represents ``output_columns`` use this value as the unit
            to create shared memory. Default: 16.
        collate_fn (Callable, optional): merges a list of samples to form a mini-batch of Tensor(s).
            Used when using batched loading from a map-style dataset.
    """

    def __init__(self, input_dataset, batch_size, drop_remainder=False, dataset_is_graph=True, num_parallel_workers=None,
                 per_batch_map=None, input_columns=None, output_columns=None, python_multiprocessing=False,
                 max_rowsize=16, collate_fn=None, max_node_per_batch=None, max_graph_per_batch=None):

        self.dataset_is_graph = dataset_is_graph
        self._output_numpy = False
        if self.dataset_is_graph:
            self._output_numpy = True
        self.iterator = []
        super().__init__(input_dataset, batch_size, drop_remainder=drop_remainder,
                         num_parallel_workers=num_parallel_workers, per_batch_map=per_batch_map,
                         input_columns=input_columns, output_columns=output_columns,
                         python_multiprocessing=python_multiprocessing, max_rowsize=max_rowsize)
        self.max_node_per_batch = max_node_per_batch
        self.max_graph_per_batch = max_graph_per_batch
        self.real_batch_size = batch_size
        self.batch_size = 1

    def __len__(self):
        return self.get_dataset_size()

    def __iter__(self):
        """Create an iterator over the dataset."""
        if not self.iterator:
            self.iterator = self.create_dict_iterator(num_epochs=-1, output_numpy=self._output_numpy)
        return self.iterator

    def create_dict_iterator(self, num_epochs=-1, output_numpy=False, do_copy=True):
        """
        Create an CustomDictIterator over the dataset that yields samples of type dict,
        while the key is the column name and the value is the data.

        Args:
            num_epochs (int, optional): The number of epochs to iterate over the entire dataset.
                Default: ``-1`` , the dataset can be iterated indefinitely.
            output_numpy (bool, optional): Whether to keep the output data as NumPy ndarray, or
                convert it to Tensor. Default: ``False`` .
            do_copy (bool, optional): Whether to copy the data when converting output to Tensor,
                or reuse the buffer for better performance, only works when `output_numpy` is ``False`` .
                Default: ``True`` .

        Returns:
            Iterator, a dataset iterator that yields samples of type dict.
        """
        if output_numpy is None:
            output_numpy = False
        if Dataset._noop_mode():
            return DummyIterator(self, 'dict', output_numpy)
        return CustomDictIterator(self, num_epochs, output_numpy, do_copy)
