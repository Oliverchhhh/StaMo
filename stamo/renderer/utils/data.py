import json
import math
import os
import random

import jsonlines
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def complex_to_device(complex, device, non_blocking=False):
    if complex is None:
        return complex
    if isinstance(complex, torch.Tensor):
        return complex.to(device, non_blocking=non_blocking)
    elif isinstance(complex, dict):
        return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
    elif isinstance(complex, list) or isinstance(complex, tuple):
        return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
    elif (
        isinstance(complex, str) or isinstance(complex, bytes) or isinstance(complex, int) or isinstance(complex, float)
    ):
        return complex
    else:
        raise ValueError("Unsupported complex", complex)


def fp32_to_fp16(batch):
    # deepspeed does not auto cast inputs.
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.half)
    elif isinstance(batch, list):
        new_batch = [fp32_to_fp16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(fp32_to_fp16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def fp32_to_bf16(batch):
    # deepspeed does not auto cast inputs.
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.bfloat16)
    elif isinstance(batch, list):
        new_batch = [fp32_to_bf16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(fp32_to_bf16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: fp32_to_bf16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def collate_fn(inputs):
    images = torch.stack([input["image"] for input in inputs])
    return {"images": images}


def get_loader_info(dataset, epochs, bsz):
    images_per_gpu = bsz
    images_per_batch = bsz * overwatch.world_size()
    iter_per_ep = len(dataset) // overwatch.world_size()
    num_iters = iter_per_ep * epochs
    loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
    return loader_info


class ImageData(Dataset):
    def __init__(self, metadata_path, flip_p, img_size: int = 224):
        self.flip_p = flip_p

        self.metadata = []
        with open(metadata_path, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                self.metadata.append(item["image"])

        self.length = len(self.metadata)

        overwatch.info(f"{self.length} data loaded from {metadata_path}")

        self.transforms = T.Compose(
            [
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ]
        )

    def preprocess_train(self, image):
        if torch.rand(1) < self.flip_p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.convert("RGB")
        image = self.transforms(image)
        return image

    def add(self, metadata_path):
        with open(metadata_path, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                self.metadata.append(item["image"])
        self.length = len(self.metadata)
        overwatch.info(f"{self.length} data loaded from {metadata_path}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                image = Image.open(self.metadata[idx])
                image = self.preprocess_train(image)
                inputs = {"image": image}
                break
            except Exception:
                overwatch.warning(f"read {self.metadata[idx]} error")
                idx = random.randint(0, self.length - 1)
        return inputs


class InfiniteDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        """
        无限循环分布式采样器。
        :param dataset: 数据集
        :param num_replicas: 总共的设备数量
        :param rank: 当前设备的 rank
        :param shuffle: 是否随机打乱
        """
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self._epoch = 0

    def __iter__(self):
        """
        无限循环返回索引。
        """
        while True:
            self.set_epoch(self._epoch)
            self._epoch += 1
            indices = super().__iter__()
            yield from indices

    def __len__(self):
        return len(self.dataset)


class InfiniteMultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, sample_per_dataset, shuffle=True):
        """
        多任务批量采样器，支持 Lightning 的分布式模式。
        :param datasets: 多个数据集的列表
        :param batch_size: 每个 batch 的大小
        :param drop_last: 是否丢弃最后一个不足 batch_size 的 batch
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_datasets = len(self.datasets)
        self.samples_per_dataset = sample_per_dataset
        # self.remaining_samples = batch_size % self.num_datasets
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]

        self.cumulative_sizes = [0] + self.dataset_lengths

        for i in range(1, len(self.cumulative_sizes)):
            self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

        self.cur_idx = 0

        # 为每个数据集创建无限采样器
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.samplers = [
            InfiniteDistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=shuffle)
            for dataset in datasets
        ]
        self.iterators = [iter(sampler) for sampler in self.samplers]

    def __iter__(self):
        """
        无限生成每个 batch 的样本索引。
        """
        while True:
            batch = []
            for i in range(len(self.iterators)):
                iterator = self.iterators[i]
                for _ in range(self.samples_per_dataset[i]):
                    batch.append(next(iterator) + self.cumulative_sizes[i])
            yield batch

    def __len__(self):
        return sum(self.dataset_lengths)


class FiniteMultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, sample_per_dataset, drop_last=False, shuffle=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.samples_per_dataset = sample_per_dataset
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = [0] + self.dataset_lengths

        for i in range(1, len(self.cumulative_sizes)):
            self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

        self.drop_last = drop_last
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1

        # 初始化标准分布式采样器
        self.samplers = [
            DistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=shuffle)
            for dataset in datasets
        ]
        self.iterators = [iter(sampler) for sampler in self.samplers]
        # 计算每个数据集还剩多少样本
        self.remaining_samples = [len(sampler) for sampler in self.samplers]

    def __iter__(self):
        iterators = [iter(sampler) for sampler in self.samplers]
        remaining_samples = self.remaining_samples.copy()

        while sum(remaining_samples) > 0:
            batch = []
            for i, iterator in enumerate(iterators):
                num_samples = min(self.samples_per_dataset[i], remaining_samples[i])
                for _ in range(num_samples):
                    try:
                        idx = next(iterator)
                        batch.append(idx + self.cumulative_sizes[i])
                        remaining_samples[i] -= 1
                    except StopIteration:
                        remaining_samples[i] = 0
                        break  # 当前dataset采样完毕

            if len(batch) == 0:
                break

            # 根据drop_last判断batch大小
            if self.drop_last and len(batch) < self.batch_size:
                break

            yield batch

    def __len__(self):
        # 总的batch数量（近似值）
        total_samples = sum(self.dataset_lengths)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


class MultiDatasetWrapper(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_lengths = [len(ds) for ds in datasets]

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, index):
        cumulative_sizes = 0
        for dataset, length in zip(self.datasets, self.dataset_lengths):
            if index < cumulative_sizes + length:
                return dataset[index - cumulative_sizes]
            cumulative_sizes += length
        raise IndexError("Index out of range")


def load_unsampler_datasets_from_json(
    json_path,
    flip_p,
    img_size,
    local_batch_size,
    num_workers=8,
    is_infinite=True,
    shuffle=True,
    drop_last=False,
):
    with open(json_path, "r") as f:
        config = json.load(f)

    dataset_paths = config["datasets"]
    dataset_path = os.path.join(os.path.dirname(json_path), dataset_paths[0])
    dataset = ImageData(dataset_path, flip_p=flip_p, img_size=img_size)

    for dataset_path in dataset_paths[1:]:
        dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
        dataset.add(dataset_path)

    rank = dist.get_rank() if dist.is_initialized() else 0
    num_replicas = dist.get_world_size() if dist.is_initialized() else 1

    if is_infinite:
        sampler = InfiniteDistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            sampler=sampler,
            drop_last=drop_last,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    return dataloader


def load_multi_datasets_form_json(
    json_path,
    flip_p,
    img_size,
    local_batch_size,
    num_workers=8,
    is_infinite=True,
    shuffle=True,
    drop_last=False,
    make_single_dataset=False,
):
    if make_single_dataset:
        return load_unsampler_datasets_from_json(
            json_path,
            flip_p,
            img_size,
            local_batch_size,
            num_workers,
            is_infinite,
            shuffle,
            drop_last,
        )
    with open(json_path, "r") as f:
        config = json.load(f)

    dataset_paths = config["datasets"]
    ratios = config["ratios"]
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    assert len(ratios) == len(dataset_paths), "Each dataset must have a corresponding ratio"

    datasets = []

    for dataset_path in dataset_paths:
        dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
        datasets.append(ImageData(dataset_path, flip_p=flip_p, img_size=img_size))

    sample_per_dataset = [max(1, math.floor(r * local_batch_size)) for r in ratios]

    total = sum(sample_per_dataset)
    if total < local_batch_size:
        sample_per_dataset[-1] += local_batch_size - total
    elif total > local_batch_size:
        sample_per_dataset[-1] -= total - local_batch_size

    wrapped_dataset = MultiDatasetWrapper(datasets)

    if is_infinite:
        batch_sampler = InfiniteMultiTaskBatchSampler(
            datasets, local_batch_size, sample_per_dataset=sample_per_dataset, shuffle=shuffle
        )
    else:
        batch_sampler = FiniteMultiTaskBatchSampler(
            datasets, local_batch_size, sample_per_dataset=sample_per_dataset, shuffle=shuffle, drop_last=drop_last
        )

    dataloader = DataLoader(
        wrapped_dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )

    return dataloader


if __name__ == "__main__":
    import torchvision.transforms as T

    dataset = ImageData(metadata_path="jsons/eval_VLA.jsonl", flip_p=0.5)

    dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
    data = next(iter(dataloader))
    print("image shape:", data["images"].shape)

    dataloader = load_multi_datasets_form_json(
        json_path="jsons/eval_VLA.json",
        flip_p=0,
        img_size=224,
        local_batch_size=4,
        num_workers=0,
        is_infinite=False,
        shuffle=False,
        drop_last=False,
        make_single_dataset=True,
    )

    data = next(iter(dataloader))
    print("image shape:", data["images"].shape)
