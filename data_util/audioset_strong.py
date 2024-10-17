import os
import numpy as np
import datasets
from tqdm import tqdm
import pandas as pd


from data_util.transforms import (
    Mp3DecodeTransform,
    SequentialTransform,
    AudioArrayTransform,
)
from data_util.utils import catchtime

import torch
from torch.utils.data.sampler import Sampler
import hffastup

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


def audioset_transform(sample):
    # del sample["target"]
    sample["dataset"] = ["audioset"]
    sample["domain"] = ["env_sounds"]
    return sample


def merge_overlapping_events(sample):
    events = pd.DataFrame(sample['events'][0])
    events = events.sort_values(by='onset')
    sample['events'] = [None]

    for l in events['event_label'].unique():
        rows = []
        for i, r in events.loc[events['event_label'] == l].iterrows():
            if len(rows) == 0 or rows[-1]['offset'] < r['onset']:
                rows.append(r)
            else:
                onset = min(rows[-1]['onset'], r['onset'])
                offset = max(rows[-1]['offset'], r['offset'])
                rows[-1]['onset'] = onset
                rows[-1]['offset'] = offset
        if sample["events"][0] is None:
            sample['events'][0] = pd.DataFrame(rows)
        else:
            sample["events"][0] = pd.concat([sample['events'][0], pd.DataFrame(rows)])

    return sample

def strong_label_transform(sample, strong_label_encoder=None):
    assert strong_label_encoder is not None
    events = pd.DataFrame(sample['events'][0])
    sample["strong"] = [strong_label_encoder.encode_strong_df(events).T]
    sample["gt_string"] = ["++".join([";;".join([str(e[0]), str(e[1]), e[2]]) for e in zip(sample['events'][0]['onset'], sample['events'][0]['offset'], sample['events'][0]['event_label'])])]
    del sample["events"]
    return sample


def unsqueeze_mono_transform(sample):  # size: 8000 --> 1, 8000
    sample["audio"] = [a.unsqueeze(0) for a in sample["audio"]]
    return sample


def target_transform(sample):
    # sample["target"] = [torch.tensor(t) for t in sample["target"]]
    del sample["labels"]
    del sample["label_ids"]
    return sample


def filename_transform(sample):
    sample["filename"] = [name.replace(".mp3", "").split("Y", 1)[1] for name in sample["filename"]]
    return sample


def get_training_dataset(
    label_encoder,
    audio_length=10.0,
    sample_rate=32000,
    wavmix=False,
    augment=False
):
    assert wavmix is not True
    assert augment is not True

    init_hf_config()

    decode_transform = Mp3DecodeTransform(
        sample_rate=sample_rate, max_length=audio_length, debug_info_key="filename"
    )

    ds_list = []

    with catchtime("Loading audioset_strong"):
        as_ds = datasets.load_from_disk(get_hf_local_path("audioset_strong"))
        as_ds_info = as_ds["eval"].info

    # label encode transformation
    if label_encoder is not None:
        # build class id - class label mapping from description string
        metadata = as_ds_info.description.split('++')[1:]
        metadata = {
            metadata[0]: dict([m.split('::') for m in metadata[1].split(';;')]),
            metadata[2]: dict([m.split('::') for m in metadata[3].split(';;')]),
        }
        # set list of label names to be encoded
        label_encoder.labels = sorted(list(metadata['labels_strong'].values()))
        encode_label_fun = lambda x: strong_label_transform(x, strong_label_encoder=label_encoder)
    else:
        encode_label_fun = lambda x: x

    as_transforms = [
        audioset_transform,
        decode_transform,
        merge_overlapping_events,
        encode_label_fun,
        target_transform
    ]

    as_ds.set_transform(SequentialTransform(as_transforms))

    ds_list.append(as_ds["balanced_train"])
    ds_list.append(as_ds["unbalanced_train"])
    dataset = torch.utils.data.ConcatDataset(ds_list)

    # dataset = CacheAudios(dataset)

    if augment:
        dataset = AugmentDataset(dataset)

    return dataset


def get_validation_dataset(
    label_encoder,
    audio_length=10.0,
    sample_rate=32000
):
    init_hf_config()
    ds_list = []

    decode_transform = Mp3DecodeTransform(
        sample_rate=sample_rate, max_length=audio_length, debug_info_key="filename"
    )

    with catchtime(f"Loading audioset:"):
        as_ds = datasets.load_from_disk(get_hf_local_path("audioset_strong"))
        as_ds_info = as_ds["eval"].info

    # label encode transformation
    if label_encoder is not None:
        # build class id - class label mapping from description string
        metadata = as_ds_info.description.split('++')[1:]
        metadata = {
            metadata[0]: dict([m.split('::') for m in metadata[1].split(';;')]),
            metadata[2]: dict([m.split('::') for m in metadata[3].split(';;')]),
        }
        # set list of label names to be encoded
        label_encoder.labels = sorted(list(metadata['labels_strong'].values()))
        encode_label_fun = lambda x: strong_label_transform(x, strong_label_encoder=label_encoder)
    else:
        encode_label_fun = lambda x: x

    as_transforms = [
        audioset_transform,
        decode_transform,
        merge_overlapping_events,
        encode_label_fun,
        target_transform
    ]
    as_ds.set_transform(SequentialTransform(as_transforms))
    as_ds_eval = (
        as_ds["eval"]
    )
    ds_list.append(as_ds_eval)

    dataset = torch.utils.data.ConcatDataset(ds_list)

    # dataset = CacheAudios(dataset)

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

from tqdm import tqdm

def get_balanced_sample_weights(
        dataset,
        encoder,
        p=0.0,
        n=0.0,
        sample_weight_sum=False,
        save_file="cache/weights_strong.pt"
):
    """
    :return: float tenosr of shape len(full_training_set) representing the weights of each sample.
    """
    # the order of balanced_train_hdf5,unbalanced_train_hdf5 is important.
    # should match get_full_training_set

    if os.path.exists(save_file):
        all_y = torch.load(save_file)
    else:
        def parse_gt_string(gt):
            events = [event.split(";;") for event in gt.split("++")]
            return [[float(event[0]), float(event[1]), event[2]] for event in events]
        all_y = []
        for sample in dataset:
            target = torch.zeros(len(encoder.labels))
            events = parse_gt_string(sample['gt_string'])
            for e in events:
                target[encoder.labels.index(e[2])] = 1
            all_y.append(target)
        all_y = torch.stack(all_y)
        torch.save(all_y, save_file)

    per_class = all_y.long().sum(0).float().reshape(1, -1)  # frequencies per class

    N = len(all_y)
    per_class_virtual = (per_class + 1 + p*N) / (N + N * p + N * n)
    per_class_weights = 1 / per_class_virtual
    all_weight = all_y * per_class_weights

    if sample_weight_sum:
        print("\nsample_weight_sum\n")
        all_weight = all_weight.sum(dim=1)
    else:
        all_weight, _ = all_weight.max(dim=1)

    # torch.save(all_weight, save_file)
    return all_weight

from data_util.audioset import DistributedSamplerWrapper
def get_weighted_sampler(
    dataset,
    encoder,
    epoch_len=None,
    sampler_replace=False,
    n=0.0,
    p=0.0
):
    if epoch_len is None:
        epoch_len = len(dataset)
    samples_weights = get_balanced_sample_weights(dataset, encoder, p=p, n=n)
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

if __name__ == "__main__":

    import logging
    import sys

    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.INFO)
    h1 = logging.StreamHandler(sys.stdout)
    root_logger.addHandler(h1)

    from helpers.t4_encoder import ManyHotEncoder
    encoder = ManyHotEncoder([], 10., 1024, 160, net_pooling=4, fs=16_000)

    d = get_training_dataset(
        encoder, audio_length=10.0, sample_rate=16_000
    )

    weights = get_balanced_sample_weights(
            d,
            encoder
    )

    print(d[0]['audio'].numpy())
    ground_truth = {}
    audio_durations = {}

    for s in d:
        gt_string = s["gt_string"]
        f = s["filename"]
        if f in ground_truth:
            continue
        else:
            events = [e.split(";;") for e in gt_string.split("++")]
            events = [(float(e[0]), float(e[1]), e[2]) for e in events]
            audio_durations[f] = (s['audio'].shape[0] / s['sampling_rate'])
            ground_truth[f] = events

    from helpers.t4_metrics import (
        batched_decode_preds,
        compute_per_intersection_macro_f1,
        compute_psds_from_operating_points,
        compute_psds_from_scores,
        log_sedeval_metrics
    )

    import sed_scores_eval

    # drop audios without events
    ground_truth = {
        audio_id: gt for audio_id, gt in ground_truth.items()
        if len(gt) > 0 and audio_id in audio_durations
    }
    audio_durations = {
        audio_id: audio_durations[audio_id]
        for audio_id in ground_truth
    }

    random_prediction = [[i * 0.04, (i + 1) * 0.04] + [0 for j in range(456)] for i in range(250)]
    df = pd.DataFrame(random_prediction, columns=['onset', 'offset'] + encoder.labels)

    scores = {k: df for k in ground_truth}

    sed_scores_eval.segment_based.fscore(
        scores, ground_truth, audio_durations, 0.5, segment_length=1.0, num_jobs=4
    )

    intersection_f1_macro_student = compute_per_intersection_macro_f1(
        scores,
        pd.DataFrame([(f, e[0], e[1], e[2]) for f in ground_truth for e in ground_truth[f]], columns=['filename', 'onset', 'offset', 'event_label']),
        pd.DataFrame([(f, audio_durations[f]) for f in audio_durations], columns=['filename', 'duration']),
    )


    psds = compute_psds_from_scores(
            scores,
            ground_truth,
            audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )

    print(psds)

    #compute_per_intersection_macro_f1

    #compute_per_intersection_macro_f1

    #pd.DataFrame([(f, e[0], e[1], e[2]) for f in ground_truth for e in ground_truth[f]],
    #             columns=['filename', 'onset', 'offset', 'event_label']),