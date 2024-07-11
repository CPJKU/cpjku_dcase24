from torch.utils.data import Dataset
from torchaudio.transforms import FFTConvolve
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import torchaudio
import random
import torch
import glob
import h5py
from pathlib import Path
import pathlib
from copy import deepcopy

from datasets.samplers import ConcatDatasetBatchSampler
from datasets.classes_dict import classes_labels_desed, classes_labels_maestro_real, maestro_desed_alias, desed_maestro_alias
from configs import DIRS_PATH


def get_training_dataset(
    encoder,
    paths,
    audio_length=10.0,
    sample_rate=16000,
    weak_split=0.9,
    maestro_split=0.9,
    seed=42,
    use_desed_maestro_alias=True,
    use_maestro_desed_alias=True,
    include_external_strong=True,
    exclude_overlapping=True,
    use_pseudo_labels=True,
    wavmix_p=0.0,
    wavmix_target="strong",
    dir_p=0.0,
    dir_desed_strong=False,
    dir_weak=False,
    dir_unlabeled=False,
    dir_maestro=False,
    include_desed_test=False,
    include_maestro_test=False,
):
    if sample_rate == 16000:
        sr_key = ""
    else:
        print(f"Resample Training data to sr: {sample_rate}")
        sr_key = "_44k"

    mask_events_desed = set(classes_labels_desed.keys())
    if use_desed_maestro_alias:
        mask_events_desed = mask_events_desed.union(set(["cutlery and dishes", "people talking"]))

    if use_maestro_desed_alias:
        mask_events_maestro_real = (set(classes_labels_maestro_real.keys()).union(
            set(["Speech", "Dishes"])))
    else:
        mask_events_maestro_real = set(classes_labels_maestro_real.keys())

    synth_tsv = pd.read_csv(paths["synth_tsv"], sep="\t")
    if use_desed_maestro_alias:
        synth_tsv = process_tsvs(synth_tsv, desed_maestro_alias)

    # synthetic strongly labeled dataset
    synth_set = StronglyAnnotatedSet(
        paths["synth_folder" + sr_key],
        synth_tsv,
        encoder,
        pad_to=audio_length,
        fs=sample_rate,
        pseudo_labels_hdf5_file=paths["pseudo_labels"].format("synth_train") if use_pseudo_labels else None,
        mask_events_other_than=mask_events_desed
    )

    assert len(synth_set) == 10_000

    strong_tsv = pd.read_csv(paths["strong_tsv"], sep="\t")
    if use_desed_maestro_alias:
        strong_tsv = process_tsvs(strong_tsv, desed_maestro_alias)

    # real strongly labeled data
    strong_set = StronglyAnnotatedSet(
        paths["strong_folder" + sr_key],
        strong_tsv,
        encoder,
        pad_to=audio_length,
        fs=sample_rate,
        exclude_files=paths["strong_tsv_exclude"] if exclude_overlapping else None,
        pseudo_labels_hdf5_file=paths["pseudo_labels"].format("strong_train") if use_pseudo_labels else None,
        mask_events_other_than=mask_events_desed
    )

    if exclude_overlapping:
        num_exclude = len(set(pd.read_csv(paths["strong_tsv_exclude"], sep="\t")["filename"]))
    else:
        num_exclude = 0
    assert len(strong_set) == 3_470 - num_exclude

    # external strong data
    if include_external_strong:
        external_strong_tsv = pd.read_csv(paths["external_strong_train_tsv"], sep="\t")
        if use_desed_maestro_alias:
            external_strong_tsv = process_tsvs(external_strong_tsv, desed_maestro_alias)

        external_strong_set = StronglyAnnotatedSet(
            paths["external_strong_folder" + sr_key],
            external_strong_tsv,
            encoder,
            pad_to=audio_length,
            fs=sample_rate,
            pseudo_labels_hdf5_file=paths["pseudo_labels"].format("external_strong_train") if use_pseudo_labels else None,
            mask_events_other_than=mask_events_desed
        )

        strong_set = torch.utils.data.ConcatDataset([strong_set, external_strong_set])
        print("Using external strong set!!")

    if include_desed_test:
        test_tsv = pd.read_csv(paths["test_tsv"], sep="\t")
        if use_desed_maestro_alias:
            test_tsv = process_tsvs(test_tsv, desed_maestro_alias)

        desed_devtest_dataset = StronglyAnnotatedSet(
            paths["test_folder" + sr_key],
            test_tsv,
            encoder,
            pad_to=audio_length,
            fs=sample_rate,
            pseudo_labels_hdf5_file=paths["pseudo_labels"].format("devtest") if use_pseudo_labels else None,
            mask_events_other_than=mask_events_desed
        )

        strong_set = torch.utils.data.ConcatDataset([strong_set, desed_devtest_dataset])
        print("Using desed test set for training!!")

    # weakly labeled data
    weak_df = pd.read_csv(paths["weak_tsv"], sep="\t")
    assert len(weak_df) == 1_578
    train_weak_df = weak_df.sample(
        frac=weak_split,
        random_state=seed,
    )

    train_weak_df = train_weak_df.reset_index(drop=True)
    if use_desed_maestro_alias:
        train_weak_df = process_tsvs_weak(train_weak_df, alias_map=desed_maestro_alias)
    weak_set = WeakSet(
        paths["weak_folder" + sr_key],
        train_weak_df,
        encoder,
        pad_to=audio_length,
        fs=sample_rate,
        exclude_files=paths["weak_tsv_exclude"] if exclude_overlapping else None,
        pseudo_labels_hdf5_file=paths["pseudo_labels"].format("weak_train") if use_pseudo_labels else None,
        mask_events_other_than=mask_events_desed
    )

    if exclude_overlapping:
        expected_weak_len = len(
            set(train_weak_df["filename"]).difference(
                set(pd.read_csv(paths["weak_tsv_exclude"], sep="\t")["filename"])
            )
        )
    else:
        expected_weak_len = len(train_weak_df)
    assert expected_weak_len == len(weak_set)

    unlabeled_set = UnlabeledSet(
        paths["unlabeled_folder" + sr_key],
        encoder,
        pad_to=audio_length,
        fs=sample_rate,
        exclude_files=paths["unlabeled_tsv_exclude"] if exclude_overlapping else None,
        pseudo_labels_hdf5_file=paths["pseudo_labels"].format("unlabeled_train") if use_pseudo_labels else None,
        mask_events_other_than=mask_events_desed
    )

    if exclude_overlapping:
        num_exclude = len(set(pd.read_csv(paths["unlabeled_tsv_exclude"], sep="\t")["filename"]))
    else:
        num_exclude = 0

    assert len(unlabeled_set) == 14_412 - num_exclude

    # maestro
    maestro_real_tsv = pd.read_csv(paths["real_maestro_train_tsv"], sep="\t")
    maestro_real_train_df, maestro_real_valid_df = split_maestro(maestro_real_tsv, split=maestro_split, seed=seed)

    if use_maestro_desed_alias:
        maestro_real_train_df = process_tsvs(maestro_real_train_df, alias_map=maestro_desed_alias)
    maestro_real_train_set = StronglyAnnotatedSet(
        paths["real_maestro_train_folder"],
        maestro_real_train_df,
        encoder,
        pad_to=audio_length,
        fs=sample_rate,
        pseudo_labels_hdf5_file=paths["pseudo_labels"].format("maestro_real_train") if use_pseudo_labels else None,
        mask_events_other_than=mask_events_maestro_real,
    )

    assert len(set(maestro_real_train_df["filename"])) + len(set(maestro_real_valid_df["filename"])) == \
           len(set(maestro_real_tsv["filename"]))
    assert len(maestro_real_train_set) == len(set(maestro_real_train_df["filename"]))

    if include_maestro_test:
        maestro_real_devtest_df = pd.read_csv(paths["real_maestro_val_tsv"], sep="\t")
        if use_maestro_desed_alias:
            maestro_real_devtest_df = process_tsvs(maestro_real_devtest_df, alias_map=maestro_desed_alias)

        # optionally we can map to desed some maestro classes
        maestro_real_devtest = StronglyAnnotatedSet(
            paths["real_maestro_val_folder"],
            maestro_real_devtest_df,
            encoder,
            pad_to=audio_length,
            fs=sample_rate,
            pseudo_labels_hdf5_file=paths["pseudo_labels"].format("maestro_real_dev") if use_pseudo_labels else None,
            mask_events_other_than=mask_events_maestro_real,
        )

        maestro_real_train_set = torch.utils.data.ConcatDataset([maestro_real_train_set, maestro_real_devtest])
        print("Using maestro test set for training!!")

    if wavmix_p > 0:
        print("Using Wavmix!!")
        if wavmix_target == "strong":
            strong_set = MixupDataset(strong_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
            synth_set = MixupDataset(synth_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
            maestro_real_train_set = MixupDataset(maestro_real_train_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
        elif wavmix_target == "labeled":
            strong_set = MixupDataset(strong_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
            synth_set = MixupDataset(synth_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
            weak_set = MixupDataset(weak_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
            maestro_real_train_set = MixupDataset(maestro_real_train_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
        elif wavmix_target == "all":
            strong_set = MixupDataset(strong_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
            synth_set = MixupDataset(synth_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
            weak_set = MixupDataset(weak_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
            unlabeled_set = MixupDataset(unlabeled_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
            maestro_real_train_set = MixupDataset(maestro_real_train_set, rate=wavmix_p, use_pseudo_labels=use_pseudo_labels)
        else:
            ValueError(f"Unknown wavemix target: {wavmix_target}")

    if dir_p > 0:
        print("Using DIR augment!!")
        if dir_desed_strong:
            strong_set = DIRDataset(strong_set, dir_p=dir_p, sample_rate=sample_rate)
            synth_set = DIRDataset(synth_set, dir_p=dir_p, sample_rate=sample_rate)

        if dir_weak:
            weak_set = DIRDataset(weak_set, dir_p=dir_p, sample_rate=sample_rate)

        if dir_unlabeled:
            unlabeled_set = DIRDataset(unlabeled_set, dir_p=dir_p, sample_rate=sample_rate)

        if dir_maestro:
            maestro_real_train_set = DIRDataset(maestro_real_train_set, dir_p=dir_p, sample_rate=sample_rate)

    tot_train_data = [maestro_real_train_set, strong_set, synth_set, weak_set, unlabeled_set]
    train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

    return train_dataset, tot_train_data


def get_validation_dataset(
    encoder,
    paths,
    audio_length=10.0,
    sample_rate=16000,
    weak_split=0.9,
    maestro_split=0.9,
    seed=42,
    use_maestro_desed_alias=True
):
    if sample_rate == 16000:
        sr_key = ""
    else:
        print(f"Resample Validation data to sr: {sample_rate}")
        sr_key = "_44k"

    mask_events_desed = set(classes_labels_desed.keys())
    if use_maestro_desed_alias:
        mask_events_maestro_real = (set(classes_labels_maestro_real.keys()).union(
            set(["Speech", "Dishes"])))
    else:
        mask_events_maestro_real = set(classes_labels_maestro_real.keys())

    synth_val = StronglyAnnotatedSet(
        paths["synth_val_folder" + sr_key],
        paths["synth_val_tsv"],
        encoder,
        return_filename=True,
        pad_to=audio_length,
        fs=sample_rate,
        mask_events_other_than=mask_events_desed
    )

    assert len(synth_val) == 2500

    external_strong_tsv = pd.read_csv(paths["external_strong_val_tsv"], sep="\t")
    external_strong_val_set = StronglyAnnotatedSet(
        paths["external_strong_folder" + sr_key],
        external_strong_tsv,
        encoder,
        return_filename=True,
        pad_to=audio_length,
        fs=sample_rate,
        mask_events_other_than=mask_events_desed
    )

    weak_df = pd.read_csv(paths["weak_tsv"], sep="\t")
    train_weak_df = weak_df.sample(
        frac=weak_split,
        random_state=seed,
    )
    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)

    weak_val = WeakSet(
        paths["weak_folder" + sr_key],
        valid_weak_df,
        encoder,
        pad_to=audio_length,
        return_filename=True,
        fs=sample_rate,
        mask_events_other_than=mask_events_desed
    )

    assert len(weak_val) == len(weak_df) - len(train_weak_df)

    maestro_real_tsv = pd.read_csv(paths["real_maestro_train_tsv"], sep="\t")
    maestro_real_train_df, maestro_real_valid_df = split_maestro(maestro_real_tsv, split=maestro_split, seed=seed)

    maestro_real_valid_set = StronglyAnnotatedSet(
        paths["real_maestro_train_folder"],
        maestro_real_valid_df,
        encoder,
        pad_to=audio_length,
        fs=sample_rate,
        return_filename=True,
        mask_events_other_than=mask_events_maestro_real,
    )

    assert len(maestro_real_valid_set) == len(set(maestro_real_valid_df["filename"]))

    valid_dataset = torch.utils.data.ConcatDataset([synth_val, external_strong_val_set, weak_val, maestro_real_valid_set])
    return valid_dataset


def get_test_dataset(
    encoder,
    paths,
    audio_length=10.0,
    sample_rate=16000,
    use_maestro_desed_alias=True
):
    if sample_rate == 16000:
        sr_key = ""
    else:
        print(f"Resample Test data to sr: {sample_rate}")
        sr_key = "_44k"

    mask_events_desed = set(classes_labels_desed.keys())
    if use_maestro_desed_alias:
        mask_events_maestro_real = (set(classes_labels_maestro_real.keys()).union(
            set(["Speech", "Dishes"])))
    else:
        mask_events_maestro_real = set(classes_labels_maestro_real.keys())

    desed_devtest_dataset = StronglyAnnotatedSet(
        paths["test_folder" + sr_key],
        paths["test_tsv"],
        encoder,
        return_filename=True,
        pad_to=audio_length,
        mask_events_other_than=mask_events_desed
    )

    # optionally we can map to desed some maestro classes
    maestro_real_devtest = StronglyAnnotatedSet(
        paths["real_maestro_val_folder"],
        paths["real_maestro_val_tsv"],
        encoder,
        return_filename=True,
        pad_to=audio_length,
        mask_events_other_than=mask_events_maestro_real,
    )

    external_strong_tsv = pd.read_csv(paths["external_strong_val_tsv"], sep="\t")
    external_strong_val_set = StronglyAnnotatedSet(
        paths["external_strong_folder" + sr_key],
        external_strong_tsv,
        encoder,
        return_filename=True,
        pad_to=audio_length,
        mask_events_other_than=mask_events_desed
    )

    devtest_dataset = torch.utils.data.ConcatDataset([external_strong_val_set, desed_devtest_dataset, maestro_real_devtest])
    return devtest_dataset


def split_maestro(maestro_dev_df, split=0.9, seed=42):
    np.random.seed(seed)
    split_f = split
    for indx, scene_name in enumerate(['cafe_restaurant', 'city_center', 'grocery_store',
                                       'metro_station', 'residential_area']):
        mask = maestro_dev_df["filename"].apply(lambda x: "_".join(x.split("_")[:-1])) == scene_name
        filenames = maestro_dev_df[mask]["filename"].apply(lambda x: x.split("-")[0]).unique()
        np.random.shuffle(filenames)

        pivot = int(split_f*len(filenames))
        filenames_train = filenames[:pivot]
        filenames_valid = filenames[pivot:]
        if indx == 0:
            mask_train = maestro_dev_df["filename"].apply(lambda x: x.split("-")[0]).isin(filenames_train)
            mask_valid = maestro_dev_df["filename"].apply(lambda x: x.split("-")[0]).isin(filenames_valid)
            train_split = maestro_dev_df[mask_train]
            valid_split = maestro_dev_df[mask_valid]
        else:
            mask_train = maestro_dev_df["filename"].apply(lambda x: x.split("-")[0]).isin(filenames_train)
            mask_valid = maestro_dev_df["filename"].apply(lambda x: x.split("-")[0]).isin(filenames_valid)
            train_split = pd.concat([train_split, maestro_dev_df[mask_train]], ignore_index=True)
            valid_split = pd.concat([valid_split, maestro_dev_df[mask_valid]], ignore_index=True)

    return train_split, valid_split


def process_tsvs(tsv, alias_map=None):
    if alias_map is None:
        return tsv
    else:
        orig_tsv = deepcopy(tsv)
        for k in alias_map.keys():
            assert k in tsv["event_label"].unique()
            if k in tsv["event_label"].unique():
                mask = tsv["event_label"] == k
                tsv.loc[mask, "event_label"] = alias_map[k]
                orig_tsv = pd.concat([orig_tsv, tsv[mask]])
        # use alias_map to duplicate events
        # for each entry in tsv create a new one with the classes map
        # e.g. dog_bark --> dog
    return orig_tsv


def process_tsvs_weak(tsv, alias_map=None):
    if alias_map is None:
        return tsv
    else:
        # Function to update event_labels based on alias_map
        def update_event_labels(event_labels, alias_map):
            labels = event_labels.split(',')
            new_labels = labels.copy()  # Copy of original labels to avoid modifying during iteration
            for label in labels:
                if label in alias_map:
                    new_labels.append(alias_map[label])
            return ','.join(new_labels)

        tsv['event_labels'] = tsv['event_labels'].apply(update_event_labels, alias_map=alias_map)
    return tsv


def to_mono(mixture, random_ch=False):
    if mixture.ndim > 1:  # multi channel
        if not random_ch:
            mixture = torch.mean(mixture, 0)
        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[0] - 1)
            mixture = mixture[indx]
    return mixture


def pad_audio(audio, target_len, fs):
    if audio.shape[-1] < target_len:
        audio = torch.nn.functional.pad(
            audio, (0, target_len - audio.shape[-1]), mode="constant"
        )

        padded_indx = [target_len / len(audio)]
        onset_s = 0.000

    elif len(audio) > target_len:

        rand_onset = random.randint(0, len(audio) - target_len)
        audio = audio[rand_onset:rand_onset + target_len]
        onset_s = round(rand_onset / fs, 3)

        padded_indx = [target_len / len(audio)]
    else:

        onset_s = 0.000
        padded_indx = [1.0]

    offset_s = round(onset_s + (target_len / fs), 3)
    return audio, onset_s, offset_s, padded_indx


def process_labels(df, onset, offset):
    df["onset"] = df["onset"] - onset
    df["offset"] = df["offset"] - onset

    df["onset"] = df.apply(lambda x: max(0, x["onset"]), axis=1)
    df["offset"] = df.apply(lambda x: min(10, x["offset"]), axis=1)

    df_new = df[(df.onset < df.offset)]

    return df_new.drop_duplicates()


def read_audio(file, multisrc, random_channel, pad_to, fs=16_000):
    mixture, orig_fs = torchaudio.load(file)

    if not multisrc:
        mixture = to_mono(mixture, random_channel)

    if orig_fs != fs:
        assert False, "We only have the full dataset in resampled 16 khz version!!!"

    if pad_to is not None:
        mixture, onset_s, offset_s, padded_indx = pad_audio(mixture, pad_to, fs)
    else:
        padded_indx = [1.0]
        onset_s = None
        offset_s = None

    mixture = mixture.float()
    return mixture, onset_s, offset_s, padded_indx


def get_sampler(
    datasets,
    batch_sizes=(8, 4, 4, 8, 16)
):
    samplers = [torch.utils.data.RandomSampler(x) for x in datasets]
    batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)
    return batch_sampler


class StronglyAnnotatedSet(Dataset):
    def __init__(
            self,
            audio_folder,
            tsv_file,
            encoder,
            pad_to=10.0,
            fs=16000,
            exclude_files=None,
            return_filename=False,
            random_channel=False,
            multisrc=False,
            pseudo_labels_hdf5_file=None,
            mask_events_other_than=None,
    ):

        self.encoder = encoder
        self.fs = fs
        self.pad_to = int(pad_to * fs)
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.pseudo_labels_hdf5_file = pseudo_labels_hdf5_file

        # we mask events that are incompatible with the current setting
        if mask_events_other_than is not None:
            # fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    # set to zero corresponding entry, invalid class for this dataset
                    # we will skip loss computation
                    self.mask_events_other_than[indx] = 0
        else:
            # keep all, no mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))

        if exclude_files:
            to_exclude = set(pd.read_csv(exclude_files, sep="\t")["filename"])
            excluded_count = []

        if type(tsv_file) == pd.DataFrame:
            tsv_entries = tsv_file
        else:
            tsv_entries = pd.read_csv(tsv_file, sep="\t")
        tsv_entries = tsv_entries.dropna()

        examples = {}
        for i, r in tsv_entries.iterrows():
            if exclude_files and r["filename"] in to_exclude:
                excluded_count.append(r["filename"])
                continue
            if r["filename"] not in examples.keys():
                confidence = 1.0 if "confidence" not in r.keys() else r["confidence"]
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": [], "confidence": confidence
                }
                if not np.isnan(r["onset"]):
                    confidence = 1.0 if "confidence" not in r.keys() else r["confidence"]
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"], "confidence": confidence
                        }
                    )
            else:
                if not np.isnan(r["onset"]):
                    confidence = 1.0 if "confidence" not in r.keys() else r["confidence"]
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"], "confidence": confidence
                        }
                    )

        if exclude_files:
            print(f"Excluded {len(set(excluded_count))} from real strong set!!")

        # we construct a dictionary for each example
        self.examples = examples
        self.examples_list = list(examples.keys())

        if self.pseudo_labels_hdf5_file is not None:
            # fetch dict of positions for each example
            self.ex2pseudo_idx = {}
            f = h5py.File(self.pseudo_labels_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2pseudo_idx[fname.decode("UTF-8")] = i
        self._opened_pseudo_hdf5 = None

    def __len__(self):
        return len(self.examples_list)

    @property
    def pseudo_hdf5_file(self):
        if self._opened_pseudo_hdf5 is None:
            self._opened_pseudo_hdf5 = h5py.File(self.pseudo_labels_hdf5_file, "r")
        return self._opened_pseudo_hdf5

    def __getitem__(self, item):
        c_ex = self.examples[self.examples_list[item]]
        mixture, onset_s, offset_s, padded_indx = read_audio(
            c_ex["mixture"], self.multisrc, self.random_channel, self.pad_to, fs=self.fs
        )

        # labels
        labels = c_ex["events"]

        # to steps
        labels_df = pd.DataFrame(labels)
        labels_df = process_labels(labels_df, onset_s, offset_s)

        # check if labels exists:
        if not len(labels_df):
            max_len_targets = self.encoder.n_frames
            strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        else:
            strong = self.encoder.encode_strong_df(labels_df)
            strong = torch.from_numpy(strong).float()

        out_args = [mixture, strong.transpose(0, 1), padded_indx]

        if self.return_filename:
            out_args.append(c_ex["mixture"])

        if self.pseudo_labels_hdf5_file is not None:
            name = Path(c_ex["mixture"]).stem
            index = self.ex2pseudo_idx[name]

            pseudo_strong = torch.from_numpy(np.stack(self.pseudo_hdf5_file["strong_logits"][index])).float()
            pseudo_strong = torch.sigmoid(pseudo_strong)
            out_args.append(pseudo_strong)

        if self.mask_events_other_than is not None:
            out_args.append(self.mask_events_other_than)
        assert len(out_args) == 5
        return out_args


class WeakSet(Dataset):
    def __init__(
            self,
            audio_folder,
            tsv_entries,
            encoder,
            pad_to=10.0,
            fs=16000,
            exclude_files=None,
            return_filename=False,
            random_channel=False,
            multisrc=False,
            pseudo_labels_hdf5_file=None,
            mask_events_other_than=None
    ):

        self.encoder = encoder
        self.fs = fs
        self.pad_to = int(pad_to * fs)
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.pseudo_labels_hdf5_file = pseudo_labels_hdf5_file
        self.mask_events_other_than = mask_events_other_than

        if exclude_files:
            to_exclude = set(pd.read_csv(exclude_files, sep="\t")["filename"])
            excluded_count = []

        if mask_events_other_than is not None:
            # fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    # set to zero corresponding entry, invalid class for this dataset
                    # we will skip loss computation
                    self.mask_events_other_than[indx] = 0
        else:
            # keep all, no mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))

        examples = {}
        for i, r in tsv_entries.iterrows():
            if exclude_files and r["filename"] in to_exclude:
                excluded_count.append(r["filename"])
                continue

            # check if file exists
            if not os.path.exists(os.path.join(audio_folder, r["filename"])):
                continue
            if r["filename"] not in examples.keys():
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": r["event_labels"].split(","),
                }

        if exclude_files:
            print(f"Excluded {len(set(excluded_count))} from weak set!!")

        self.examples = examples
        self.examples_list = list(examples.keys())

        if self.pseudo_labels_hdf5_file is not None:
            # fetch dict of positions for each example
            self.ex2pseudo_idx = {}
            f = h5py.File(self.pseudo_labels_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2pseudo_idx[fname.decode("UTF-8")] = i
        self._opened_pseudo_hdf5 = None

    def __len__(self):
        return len(self.examples_list)

    @property
    def pseudo_hdf5_file(self):
        if self._opened_pseudo_hdf5 is None:
            self._opened_pseudo_hdf5 = h5py.File(self.pseudo_labels_hdf5_file, "r")
        return self._opened_pseudo_hdf5

    def __getitem__(self, item):
        file = self.examples_list[item]
        c_ex = self.examples[file]

        mixture, _, _, padded_indx = read_audio(
            c_ex["mixture"], self.multisrc, self.random_channel, self.pad_to, fs=self.fs
        )

        # labels
        labels = c_ex["events"]
        # check if labels exists:
        max_len_targets = self.encoder.n_frames
        weak = torch.zeros(max_len_targets, len(self.encoder.labels))
        if len(labels):
            weak_labels = self.encoder.encode_weak(labels)
            weak[0, :] = torch.from_numpy(weak_labels).float()

        out_args = [mixture, weak.transpose(0, 1), padded_indx]

        if self.return_filename:
            out_args.append(c_ex["mixture"])

        if self.pseudo_labels_hdf5_file is not None:
            name = Path(c_ex["mixture"]).stem
            index = self.ex2pseudo_idx[name]

            pseudo_strong = torch.from_numpy(np.stack(self.pseudo_hdf5_file["strong_logits"][index])).float()
            pseudo_strong = torch.sigmoid(pseudo_strong)
            out_args.append(pseudo_strong)

        if self.mask_events_other_than is not None:
            out_args.append(self.mask_events_other_than)
        assert len(out_args) == 5
        return out_args


class UnlabeledSet(Dataset):
    def __init__(
            self,
            unlabeled_folder,
            encoder,
            pad_to=10.0,
            fs=16000,
            exclude_files=None,
            return_filename=False,
            random_channel=False,
            multisrc=False,
            pseudo_labels_hdf5_file=None,
            mask_events_other_than=None
    ):

        self.encoder = encoder
        self.fs = fs
        self.pad_to = int(pad_to * fs) if pad_to is not None else None
        self.examples = glob.glob(os.path.join(unlabeled_folder, "*.wav"))
        if exclude_files:
            orig_len = len(self.examples)
            base_path = "/".join(self.examples[0].split("/")[:-1])
            to_exclude = list(pd.read_csv(exclude_files, sep="\t")["filename"])
            to_exclude = [os.path.join(base_path, file) for file in to_exclude]
            self.examples = [x for x in self.examples if x not in to_exclude]
            print(f"Excluded {orig_len - len(self.examples)} from unlabeled set!!")
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.pseudo_labels_hdf5_file = pseudo_labels_hdf5_file
        self.mask_events_other_than = mask_events_other_than

        if mask_events_other_than is not None:
            # fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    # set to zero corresponding entry, invalid class for this dataset
                    # we will skip loss computation
                    self.mask_events_other_than[indx] = 0
        else:
            # keep all, no mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))

        if self.pseudo_labels_hdf5_file is not None:
            # fetch dict of positions for each example
            self.ex2pseudo_idx = {}
            f = h5py.File(self.pseudo_labels_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2pseudo_idx[fname.decode("UTF-8")] = i
        self._opened_pseudo_hdf5 = None

    def __len__(self):
        return len(self.examples)

    @property
    def pseudo_hdf5_file(self):
        if self._opened_pseudo_hdf5 is None:
            self._opened_pseudo_hdf5 = h5py.File(self.pseudo_labels_hdf5_file, "r")
        return self._opened_pseudo_hdf5

    def __getitem__(self, item):
        c_ex = self.examples[item]

        mixture, _, _, padded_indx = read_audio(
            c_ex, self.multisrc, self.random_channel, self.pad_to, fs=self.fs
        )

        max_len_targets = self.encoder.n_frames
        strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        out_args = [mixture, strong.transpose(0, 1), padded_indx]

        if self.return_filename:
            out_args.append(c_ex)

        if self.pseudo_labels_hdf5_file is not None:

            name = Path(c_ex).stem
            index = self.ex2pseudo_idx[name]

            pseudo_strong = torch.from_numpy(np.stack(self.pseudo_hdf5_file["strong_logits"][index])).float()
            pseudo_strong = torch.sigmoid(pseudo_strong)
            out_args.append(pseudo_strong)

        if self.mask_events_other_than is not None:
            out_args.append(self.mask_events_other_than)
        assert len(out_args) == 5
        return out_args


class MixupDataset(Dataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=0.2, rate=0.5, use_pseudo_labels=False):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        self.use_pseudo_labels = use_pseudo_labels
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            batch1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            batch2 = self.dataset[idx2]
            x1, x2 = batch1[0], batch2[0]
            y1, y2 = batch1[1], batch2[1]
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1-x1.mean()
            x2 = x2-x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            batch1[0] = x
            batch1[1] = (y1 * l + y2 * (1. - l))
            if self.use_pseudo_labels:
                # strong
                pl1, pl2 = batch1[3], batch2[3]
                batch1[3] = (pl1 * l + pl2 * (1. - l))

                # weak
                pl1, pl2 = batch1[4], batch2[4]
                batch1[4] = (pl1 * l + pl2 * (1. - l))
            return batch1
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class DIRDataset(Dataset):
    """ augment using DIRs
    """

    def __init__(self, dataset, dir_p, sample_rate):
        self.dataset = dataset
        self.dir_p = dir_p
        self.convolve = FFTConvolve(mode="valid")
        self.dirs = self.load_dirs(sample_rate)

    def __getitem__(self, index):
        if torch.rand(1) < self.dir_p:
            batch = self.dataset[index]
            audio = batch[0]
            dir_idx = int(np.random.randint(0, len(self.dirs)))
            dir = self.dirs[dir_idx]

            pad = dir.size(1) // 2
            audio_len = len(audio)
            with torch.no_grad():
                audio = F.pad(audio, (pad, pad))
                audio = self.convolve(audio.reshape(1, -1), dir)

            audio = audio.squeeze()[:audio_len]

            batch[0] = audio
            return batch
        return self.dataset[index]

    def load_dirs(self, sample_rate):
        all_paths = [path for path in pathlib.Path(os.path.expanduser(DIRS_PATH)).rglob('*.wav')]
        all_paths = sorted(all_paths)

        def process_func(dir_file):
            sig, orig_fs = torchaudio.load(dir_file)
            sig = to_mono(sig)
            sig = torchaudio.functional.resample(sig, orig_freq=orig_fs, new_freq=sample_rate,
                                           resampling_method="sinc_interp_kaiser")
            return sig.reshape(1, -1)

        return [process_func(p) for p in all_paths]

    def __len__(self):
        return len(self.dataset)


class WavDataset(torch.utils.data.Dataset):
    def __init__(self, folder, pad_to=10, fs=16000):
        self.fs = fs
        self.pad_to = pad_to * fs if pad_to is not None else None
        self.examples = glob.glob(os.path.join(folder, "*.wav"))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        c_ex = self.examples[item]
        mixture, _, _, padded_indx = read_audio(c_ex, False, False, self.pad_to)
        return mixture, Path(c_ex).stem
