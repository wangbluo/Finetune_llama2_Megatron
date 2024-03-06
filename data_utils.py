import json
import random
import os
import io
from typing import Iterator, Optional, Dict, Iterator, List, Optional, Sequence, Union, Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from datasets import dataset_dict
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler
import math
import torch.distributed as dist
from dataclasses import dataclass
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from copy import deepcopy

DatasetType = Union[Dataset, ConcatDataset, dataset_dict.Dataset]
PathType = Union[str, os.PathLike]

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor.data
    tensor.div_(dist.get_world_size())
    return tensor

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def tokenize_batch_for_finetune(batch, tokenizer: Optional[LlamaTokenizer] = None, max_length: int = 2048):
    texts = [sample["prompt"] + sample["completion"] for sample in batch]
    data = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    data = {k: v.cuda() for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data

def tokenize_batch_for_finetune_tp(batch, tokenizer: Optional[LlamaTokenizer] = None, max_length: int = 2048):
    texts = batch["instruction"] + batch["output"]
    data = tokenizer(texts)
    data["input_ids"] = torch.LongTensor(data["input_ids"])
    data["attention_mask"] = torch.LongTensor(data["attention_mask"]) 
    data["labels"] = deepcopy(data["input_ids"])
    return data

class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: DatasetType,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        use_tp: Optional[bool] = False,
    ) -> None:
        if not use_tp:
            super().__init__(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
        else:
            # adapted from https://github.com/pytorch/pytorch/blob/4979f9c0d72490970e2019bb1d2284f83d93f76b/torch/utils/data/distributed.py#L62
            # TODO: support pp later
            num_replicas = 1
            if rank is None:
                rank = dist.get_rank()
            if rank < 0:
                raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, 0]")
            self.dataset = dataset
            self.num_replicas = num_replicas
            self.rank = rank
            self.epoch = 0
            self.drop_last = drop_last
            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.num_samples = math.ceil(
                    (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
                )
            else:
                self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
            self.total_size = self.num_samples * self.num_replicas
            self.shuffle = shuffle
            self.seed = seed
        self.start_index = 0
        self.use_tp = use_tp

    def __iter__(self) -> Iterator:
        if self.use_tp:
            # TODO Add support for tp_group not equal to 1
            pass
            # adpated from https://github.com/pytorch/pytorch/blob/4979f9c0d72490970e2019bb1d2284f83d93f76b/torch/utils/data/distributed.py#L96
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            else:
                indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[: self.total_size]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[
                : self.total_size : self.num_replicas
            ]  # num_replicas=tp_group=1, we only support tp_group==1 for now
            assert len(indices) == self.num_samples

            return iter(indices)

        else:
            iterator = super().__iter__()
            indices = list(iterator)
            indices = indices[self.start_index :]
            return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index

@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate instances for supervised dataset.
    Each instance is a tokenized dictionary with fields
    `input_ids`(List[int]), `labels`(List[int]) and `sequence`(str).
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 4096
    ignore_index: int = -100

    def __call__(self, instances: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """

        Args:
            instances (`Sequence[Dict[str, List[int]]]`):
                Mini-batch samples, each sample is stored in an individual dictionary.

        Returns:
            (`Dict[str, torch.Tensor]`): Contains the following `torch.Tensor`:
                `input_ids`: `torch.Tensor` of shape (bsz, max_len);
                `attention_mask`: `torch.BoolTensor` of shape (bsz, max_len);
                `labels`: `torch.Tensor` of shape (bsz, max_len), which contains `IGNORE_INDEX`.
        """
        assert isinstance(self.tokenizer.pad_token_id, int) and self.tokenizer.pad_token_id >= 0, (
            f"`{self.tokenizer.__class__.__name__}.pad_token_id` must be a valid non-negative integer index value, "
            f"but now `{self.tokenizer.pad_token_id}`"
        )

        batch_input_ids = [
            torch.LongTensor(instance["input_ids"][: self.max_length])
            if len(instance["input_ids"]) > self.max_length
            else torch.LongTensor(instance["input_ids"])
            for instance in instances
        ]
        batch_labels = [
            torch.LongTensor(instance["labels"][: self.max_length])
            if len(instance["labels"]) > self.max_length
            else torch.LongTensor(instance["labels"])
            for instance in instances
        ]
        if self.tokenizer.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                sequences=batch_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )  # (bsz, max_len)
            labels = torch.nn.utils.rnn.pad_sequence(
                sequences=batch_labels,
                batch_first=True,
                padding_value=self.ignore_index,
            )  # (bsz, max_len)
            # pad to max
            to_pad = self.max_length - input_ids.size(1)
            input_ids = F.pad(input_ids, (0, to_pad), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, (0, to_pad), value=self.ignore_index)
        elif self.tokenizer.padding_side == "left":
            reversed_input_ids = [seq.flip(dims=(0,)) for seq in batch_input_ids]
            reversed_input_ids = torch.nn.utils.rnn.pad_sequence(
                sequences=reversed_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )  # (bsz, max_len)
            input_ids = torch.flip(reversed_input_ids, dims=(1,))  # (bsz, max_len)
            reversed_labels = [seq.flip(dims=(0,)) for seq in batch_labels]
            reversed_labels = torch.nn.utils.rnn.pad_sequence(
                sequences=reversed_labels,
                batch_first=True,
                padding_value=self.ignore_index,
            )  # (bsz, max_len)
            labels = torch.flip(reversed_labels, dims=(1,))  # (bsz, max_len)
        else:
            raise RuntimeError(
                f"`{self.tokenizer.__class__.__name__}.padding_side` can only be `left` or `right`, "
                f"but now `{self.tokenizer.padding_side}`"
            )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)  # `torch.BoolTensor`, (bsz, max_len)

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

def setup_distributed_dataloader(
    dataset: DatasetType,
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 1024,
    drop_last: bool = False,
    pin_memory: bool = False,
    num_workers: int = 0,
    collate_fn: Callable[[Sequence[Dict[str, Union[str, List[int]]]]], Dict[str, torch.Tensor]] = None,
    process_group: Optional[ProcessGroup] = None,
    use_tp: Optional[bool] = False,
    **kwargs,
) -> DataLoader:
    """
    Setup dataloader for distributed training.
    """
    _kwargs = kwargs.copy()
    process_group = process_group or _get_default_group()
    sampler = StatefulDistributedSampler(
        dataset=dataset,
        num_replicas=process_group.size() if not use_tp else 1,
        rank=process_group.rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        use_tp=use_tp,
    )

    # Deterministic dataloader
    def seed_worker(worker_id: int) -> None:
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        **_kwargs,
    )
