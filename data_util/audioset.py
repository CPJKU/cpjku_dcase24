import os
import numpy as np
import datasets
from tqdm import tqdm

from data_util.transforms import (
    Mp3DecodeTransform,
    SequentialTransform,
    AudioArrayTransform,
)
from data_util.utils import catchtime

import torch
from torch.utils.data.sampler import Sampler

from torch.utils.data import (
    Dataset as TorchDataset,
    ConcatDataset,
    DistributedSampler,
    WeightedRandomSampler,
)

logger = datasets.logging.get_logger(__name__)

data_split_seed = 42
validation_samples_per_dataset = 1000


def init_hf_config(max_shard_size="2GB", verbose=True, in_mem_max=None):
    datasets.config.MAX_SHARD_SIZE = max_shard_size
    if verbose:
        datasets.logging.set_verbosity_info()
    if in_mem_max is not None:
        datasets.config.IN_MEMORY_MAX_SIZE = in_mem_max


def get_hf_local_path(path, local_datasets_path=None):
    if local_datasets_path is None:
        local_datasets_path = os.environ.get(
            "HF_DATASETS_LOCAL",
            os.path.join(os.environ.get("HF_DATASETS_CACHE"), "../local"),
        )
    path = os.path.join(local_datasets_path, path)
    return path


def cv_transform(sample):
    sample["sampling_rate"] = [sample["audio"][0]["sampling_rate"]]
    sample["filename"] = [sample["audio"][0]["path"]]
    sample["audio"] = [sample["audio"][0]["array"]]
    sample["dataset"] = ["common_voice"]
    sample["domain"] = ["speech"]

    del sample["client_id"]
    del sample["path"]
    del sample["sentence"]
    del sample["up_votes"]
    del sample["down_votes"]
    del sample["age"]
    del sample["gender"]
    del sample["accent"]
    del sample["locale"]
    del sample["segment"]
    del sample["variant"]
    return sample


def audioset_transform(sample):
    # del sample["target"]
    sample["dataset"] = ["audioset"]
    sample["domain"] = ["env_sounds"]
    return sample


def unsqueeze_mono_transform(sample):  # size: 8000 --> 1, 8000
    sample["audio"] = [a.unsqueeze(0) for a in sample["audio"]]
    return sample

def target_transform(sample):
    sample["target"] = [torch.tensor(t) for t in sample["target"]]
    return sample

def filename_transform(sample):
    sample["filename"] = [name.replace(".mp3", "").split("Y", 1)[1] for name in sample["filename"]]
    return sample


class MixupDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            batch1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            batch2 = self.dataset[idx2]
            x1, x2 = batch1['audio'], batch2['audio']
            y1, y2 = batch1['target'], batch2['target']
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1-x1.mean()
            x2 = x2-x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            batch1['audio'] = x
            batch1['target'] = (y1 * l + y2 * (1. - l))
            return batch1
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class AugmentDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, shift_range=4000, gain=7):
        self.dataset = dataset
        self.shift_range = shift_range
        self.gain = gain
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        batch = self.dataset[index]
        if self.shift_range:
            x = batch['audio']
            sf = int(np.random.random_integers(-self.shift_range, self.shift_range))
            batch['audio'] = x.roll(sf, 0)
        if self.gain:
            gain = torch.randint(self.gain * 2, (1,)).item() - self.gain
            amp = 10 ** (gain / 20)
            batch['audio'] = batch['audio'] * amp
        return batch

    def __len__(self):
        return len(self.dataset)


def get_training_dataset(
    audio_length=10.0,
    sample_rate=32000,
    return_ds_info=None,
    wavmix=False,
    augment=False,
    decode=True
):
    init_hf_config()

    def no_decode(b):
        b['audio'] = [torch.zeros(10)]
        b["sampling_rate"] = [32000]
        return b

    mp3_decode_transform = Mp3DecodeTransform(
        sample_rate=sample_rate, max_length=audio_length, debug_info_key="filename"
    ) if decode else no_decode

    ds_list = []

    with catchtime("Loading audioset2m"):
        as_ds = datasets.load_from_disk(get_hf_local_path("audioset2m"))
    as_transforms = [
        audioset_transform,
        mp3_decode_transform,
        target_transform,
        filename_transform  # for loading correct ensemble predictions
    ]
    as_ds.set_transform(SequentialTransform(as_transforms))
    ds_list.append(as_ds["balanced_train"])
    ds_list.append(as_ds["unbalanced_train"])
    dataset = torch.utils.data.ConcatDataset(ds_list)
    if wavmix:
        dataset = MixupDataset(dataset)
    if augment:
        dataset = AugmentDataset(dataset)
    return dataset


def get_validation_dataset(
    audio_length=10.0,
    sample_rate=32000
):
    init_hf_config()
    ds_list = []
    decode_transform = Mp3DecodeTransform(
        sample_rate=sample_rate, max_length=audio_length, debug_info_key="filename"
    )

    with catchtime(f"Loading audioset:"):
        as_ds = datasets.load_from_disk(get_hf_local_path("audioset2m"))
    as_transforms = [
        audioset_transform,
        decode_transform,
        target_transform
    ]
    as_ds.set_transform(SequentialTransform(as_transforms))
    as_ds_eval = (
        as_ds["eval"]
    )
    ds_list.append(as_ds_eval)

    dataset = torch.utils.data.ConcatDataset(ds_list)

    logger.info(
        "\n".join(
            [
                "length of validation dataset: ",
                str(len(dataset)),
                "validation samples per dataset: ",
                str(validation_samples_per_dataset),
            ]
        )
    )
    return dataset


def get_weighted_sampler(
    samples_weights,
    epoch_len=100_000,
    sampler_replace=False,
):
    num_nodes = int(os.environ.get("WORLD_SIZE", 1))
    ddp = int(os.environ.get("DDP", 1))
    num_nodes = max(ddp, num_nodes)
    rank = int(os.environ.get("NODE_RANK", 0))
    return DistributedSamplerWrapper(
        sampler=WeightedRandomSampler(
            samples_weights, num_samples=epoch_len, replacement=sampler_replace
        ),
        dataset=range(epoch_len),
        num_replicas=num_nodes,
        rank=rank,
    )


def get_ft_cls_balanced_sample_weights(dataset, sample_weight_offset=100, sample_weight_sum=True,
                                       save_file="cache/weights.pt"):
    """
    :return: float tenosr of shape len(full_training_set) representing the weights of each sample.
    """
    # the order of balanced_train_hdf5,unbalanced_train_hdf5 is important.
    # should match get_full_training_set
    if os.path.exists(save_file):
        return torch.load(save_file)

    all_y = []
    for sample in dataset:
        target = sample['target']
        all_y.append(target)
    all_y = torch.stack(all_y)
    per_class = all_y.long().sum(0).float().reshape(1, -1)  # frequencies per class

    per_class = sample_weight_offset + per_class  # offset low freq classes
    if sample_weight_offset > 0:
        print(f"Warning: sample_weight_offset={sample_weight_offset} minnow={per_class.min()}")
    per_class_weights = 1000. / per_class
    all_weight = all_y * per_class_weights
    # print(all_weight.shape)
    # print(all_weight[1510])
    if sample_weight_sum:
        print("\nsample_weight_sum\n")
        all_weight = all_weight.sum(dim=1)
    else:
        all_weight, _ = all_weight.max(dim=1)

    torch.save(all_weight, save_file)
    return all_weight


class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
        self, sampler, dataset, num_replicas=None, rank=None, shuffle: bool = True
    ):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle
        )
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch < 2:
            logger.info(
                f"\n DistributedSamplerWrapper (rank {self.rank}) :  {indices[:3]} \n\n"
            )
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)


class BatchwiseDistributedSamplerWrapper(DistributedSampler):
    def __init__(
        self,
        sampler,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle: bool = True,
        batch_size=32,
    ):
        super(BatchwiseDistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle
        )
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()

        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch < 2:
            logger.info(
                f"\n BatchwiseDistributedSamplerWrapper (rank {self.rank}) All({len(indices)}):  {indices[:3]} \n\n"
            )

        idx_start = self.rank * (self.total_size / self.num_replicas)
        idx_end = (self.rank + 1) * (self.total_size / self.num_replicas)

        # ensure that a whole batch always belongs to one rank
        idx_start = int(idx_start // self.batch_size) * self.batch_size
        idx_end = int(idx_end // self.batch_size) * self.batch_size
        indices = indices[idx_start:idx_end]
        if self.epoch < 2:
            logger.info(
                f"\n B DistributedSamplerWrapper (rank {self.rank}) [{idx_start}:{idx_end}] indices:  {indices[:3]} \n\n"
            )
        return iter(indices)


class ValidationDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle: bool = False,
        drop_last: bool = True,
    ):
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        num_nodes = int(os.environ.get("WORLD_SIZE", 1))
        ddp = int(os.environ.get("DDP", 1))
        num_nodes = max(ddp, num_nodes)
        if num_replicas is None:
            num_replicas = num_nodes
        if rank is None:
            rank = int(os.environ.get("NODE_RANK", 0))

        super(ValidationDistributedSampler, self).__init__(
            dataset, num_replicas, rank, shuffle, drop_last=drop_last
        )
        assert not shuffle, "Validation sampler should not be shuffled."

    def __iter__(self):
        rank0_indices = (
            list(range(0, 10)) + list(range(1000, 1010)) + list(range(2000, 2010))
        )
        indices = list(x for x in range(self.total_size) if x not in rank0_indices)
        indices = rank0_indices + indices
        cutsize = len(rank0_indices)
        inds = [
            indices[i * cutsize : i * cutsize + cutsize]
            for i in range(self.num_replicas)
        ]
        indices = indices[self.num_replicas * cutsize :]
        for i in range(self.num_replicas):
            inds[i].extend(indices[i : len(indices) : self.num_replicas])
        if self.epoch < 2:
            logger.info(
                f"\n ValidatoinDistributedSampler epoch {self.epoch} (rank {self.rank}, nodes {self.num_replicas}) :  {inds[self.rank][:3]} \n\n"
            )
        return iter(inds[self.rank])


if __name__ == "__main__":
    # print(get_training_dataset()[0])
    # with catchtime(f"Loading get_validation_dataset"):
    #     ds = get_validation_dataset()

    # for nums in [1, 2, 4, 8]:
    #     s = set()
    #     for r in range(nums):
    #         s1 = ValidationDistributedSampler(ds, num_replicas=nums, rank=r)
    #         s1 = list(s1)
    #         s.update(s1)
    #         assert len(s1) == len(ds) // nums, f"wrong length {r}/{nums}"
    #     assert len(s) == len(ds), f"wrong length {nums}"
    import logging
    import sys

    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.INFO)
    h1 = logging.StreamHandler(sys.stdout)
    root_logger.addHandler(h1)

    concat_ds, ds_info = get_training_dataset(
        1.0, return_ds_info=True, use_new_training_dataset=True
    )
    batch_size = 12
    os.environ["WORLD_SIZE"] = "2"
    os.environ["NODE_RANK"] = "0"

    for r in range(2):
        s1 = get_balanced_domain_sampler(
            concat_ds, ds_info, batch_size, epoch_len=100_000
        )
        s1 = list(s1)
