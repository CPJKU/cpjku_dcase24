import functools
import os
import sys
from munch import DefaultMunch
import wandb
import random
import numpy as np
from copy import deepcopy
import pandas as pd
from pathlib import Path
from collections import defaultdict
import math

# torch, lightning, huggingface
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import autocast
from helpers.scaler import TorchScaler
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
import torchmetrics
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import transformers

# models
from models.atst.atst_model_wrapper import ATSTWrapper, ATSTMel
from models.wrapper import Task4CRNNEmbeddingsWrapper
from helpers.encoder import get_encoder

# data & augmentations
from datasets import t4_24_datasets
from helpers.workersinit import worker_init_fn
from helpers.augment import mixup, frame_shift, gain_augment, time_mask, feature_transformation, mixstyle, RandomResizeCrop
from datasets.classes_dict import classes_labels_desed, classes_labels_maestro_real, \
    classes_labels_maestro_real_eval

# config & logging
from configs import add_configs, PRETRAINED_AUDIOSET, DATASET_PATH
from helpers.utils import config_call, register_print_hooks
from sacred import Experiment
from sacred.config_helpers import CMD
from codecarbon import OfflineEmissionsTracker

# evaluation
import sed_scores_eval
from helpers.metrics import (
    batched_decode_preds,
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    compute_psds_from_scores,
    log_sedeval_metrics
)
from sed_scores_eval.base_modules.scores import create_score_dataframe, validate_score_dataframe
from helpers.postprocess import ClassWiseMedianFilter

ex = Experiment("dcase24_t4_desed")

# define datasets config
get_training_dataset = ex.command(
    t4_24_datasets.get_training_dataset,
    prefix="training",
    audio_length=10.0,
    sample_rate=16000,
    weak_split=0.9,
    maestro_split=0.9,
    seed=42
)

# define datasets config
get_validation_dataset = ex.command(
    t4_24_datasets.get_validation_dataset,
    prefix="validation",
    audio_length=10.0,
    sample_rate=16000,
    weak_split=0.9,
    maestro_split=0.9,
    seed=42
)

# define datasets config
get_test_dataset = ex.command(
    t4_24_datasets.get_test_dataset,
    prefix="test",
    audio_length=10.0,
    sample_rate=16000
)

get_sampler = ex.command(
    t4_24_datasets.get_sampler, prefix="training", batch_sizes=(7, 5, 5, 9, 9)
)

# Define loaders
get_train_loader = ex.command(
    DataLoader,
    prefix="training",
    static_args=dict(worker_init_fn=worker_init_fn),
    train=True,
    num_workers=16,
    shuffle=None,
)

get_validate_loader = ex.command(
    DataLoader,
    prefix="validation",
    static_args=dict(worker_init_fn=worker_init_fn),
    batch_size=20,
    validate=True,
    shuffle=False,
    drop_last=False,
    num_workers=16,
    dataset=CMD("/get_validation_dataset"),
)

get_test_loader = ex.command(
    DataLoader,
    prefix="test",
    static_args=dict(worker_init_fn=worker_init_fn),
    batch_size=20,
    test=True,
    shuffle=False,
    drop_last=False,
    num_workers=16,
    dataset=CMD("/get_test_dataset"),
)

# the config is automatically passed to the objects we create in the following
# this happens via the commands
Trainer = ex.command(L.Trainer, prefix="trainer")
many_hot_encoder = ex.command(get_encoder, prefix="encoder")

## atst
atst_mel = ex.command(ATSTMel, prefix="atst_mel")

## crnn
MelSpectrogram = ex.command(MelSpectrogram, prefix="mel")
Task4CRNNEmbeddingsWrapper = ex.command(Task4CRNNEmbeddingsWrapper, prefix="t4_wrapper")


@ex.config
def default_conf():
    cmd = " ".join(sys.argv)  # command line arguments
    process_id = os.getpid()
    debug_shapes = 2  # print shapes of in step, 0 = never, 1 = first step etc...
    audio_len = 10
    sample_rate = 16_000
    trainer = dict(
        max_epochs=250,
        devices=1,
        weights_summary="full",
        benchmark=True,
        num_sanity_val_steps=0,
        precision="16-mixed",  # NOTE: we train in mixed precision
        reload_dataloaders_every_epoch=True,
        default_root_dir="./outputs",
        accumulate_grad_batches=1,
        check_val_every_n_epoch=10
    )

    optimizer = dict(
        cnn_lr=0.0001,
        rnn_lr=0.001,
        pt_lr=0.0001,
        pt_lr_scale=1.0,
        pt_trainable_layers="all",
        adamw=False,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        schedule_mode="cos",
        num_warmup_steps=270
    )

    mel = dict(
        n_mels=128,
        n_fft=2048,
        hop_length=256,
        win_length=2048,
        sample_rate=sample_rate,
        f_min=0,
        f_max=8000,
        window_fn=torch.hamming_window,
        wkwargs={"periodic": False},
        power=1,
    )

    encoder = dict(
        audio_len=10,
        frame_len=2048,
        frame_hop=256,
        net_pooling=4,
        fs=sample_rate
    )

    atst_frame = dict(
        pretrained_name="atst_as_strong"
    )

    t4_wrapper = dict(
        name="Task4CRNNEmbeddingsWrapper",
        audioset_classes=527,
        no_wrapper=False,
        dropout=0.5,
        n_layers_RNN=2,
        n_in_channel=1,
        nclass=27,
        attention=True,
        n_RNN_cell=256,
        activation="cg",
        rnn_type="BGRU",
        kernel_size=[3, 3, 3, 3, 3, 3, 3],
        padding=[1, 1, 1, 1, 1, 1, 1],
        stride=[1, 1, 1, 1, 1, 1, 1],
        nb_filters=[16, 32, 64, 128, 128, 128, 128],
        pooling=[[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
        dropout_recurrent=0,
        use_embeddings=True,
        embedding_size=768,
        embedding_type="frame",
        aggregation_type="pool1d",
        model_init_id=None
    )

    training = dict(sample_rate=sample_rate,
                    include_external_strong=True,
                    exclude_overlapping=True,
                    dir_p=0.5,
                    dir_desed_strong=True,
                    dir_weak=True,
                    dir_unlabeled=True,
                    dir_maestro=True,
                    wavmix_p=0.5,
                    wavmix_target="strong",
                    use_desed_maestro_alias=True,
                    use_pseudo_labels=True
                    )
    validation = dict(sample_rate=sample_rate)
    test = dict(sample_rate=sample_rate)

    # maestro, strong real, strong synth, weak, unlabeled (ssl loss), pseudo strong loss (set to 0 in stage 2)
    # these loss weights are the results of a (painful) tuning process; they might be far from optimal
    loss_weights = (2, 1, 1, 1, 70, 0)
    ssl_scale_max = 1.0
    ema_factor = 0.999
    val_thresholds = [0.5]

    # the following dict holds all the paths necessary to load all the different parts of the dataset
    base_path = DATASET_PATH
    t4_paths = dict(
        synth_folder=os.path.join(base_path, "dcase_synth/audio/train/synthetic21_train/soundscapes_16k/"),
        synth_folder_44k=os.path.join(base_path, "dcase_synth/audio/train/synthetic21_train/soundscapes/"),
        synth_tsv=os.path.join(base_path, "dcase_synth/metadata/train/synthetic21_train/soundscapes.tsv"),
        strong_folder=os.path.join(base_path, "audio/train/strong_label_real_16k/"),
        strong_folder_44k=os.path.join(base_path, "audio/train/strong_label_real/"),
        strong_tsv=os.path.join(base_path, "metadata/train/audioset_strong.tsv"),
        external_strong_folder=os.path.join(base_path, "audio/external_strong_16k/"),
        external_strong_train_tsv=os.path.join(base_path, "metadata/external_audioset_strong_train.tsv"),
        external_strong_val_tsv=os.path.join(base_path, "metadata/external_audioset_strong_eval.tsv"),
        external_strong_dur=os.path.join(base_path, "metadata/external_audioset_strong_dur.tsv"),
        weak_folder=os.path.join(base_path, "audio/train/weak_16k/"),
        weak_folder_44k=os.path.join(base_path, "audio/train/weak/"),
        weak_tsv=os.path.join(base_path, "metadata/train/weak.tsv"),
        unlabeled_folder=os.path.join(base_path, "audio/train/unlabel_in_domain_16k/"),
        unlabeled_folder_44k=os.path.join(base_path, "audio/train/unlabel_in_domain/"),
        synth_val_folder=os.path.join(base_path, "audio/validation/synthetic21_validation/soundscapes_16k/"),
        synth_val_folder_44k=os.path.join(base_path,
                                          "dcase_synth/audio/validation/synthetic21_validation/soundscapes/"),
        synth_val_tsv=os.path.join(base_path, "dcase_synth/metadata/validation/synthetic21_validation/soundscapes.tsv"),
        synth_val_dur=os.path.join(base_path, "dcase_synth/metadata/validation/synthetic21_validation/durations.tsv"),
        test_folder=os.path.join(base_path, "audio/validation/validation_16k/"),
        test_folder_44k=os.path.join(base_path, "audio/validation/validation/"),
        test_tsv=os.path.join(base_path, "metadata/validation/validation.tsv"),
        test_dur=os.path.join(base_path, "metadata/validation/validation_durations.tsv"),
        synth_maestro_train=os.path.join(base_path, "audio/maestro_synth_train_16k"),
        synth_maestro_train_44k=os.path.join(base_path, "audio/maestro_synth_train"),
        synth_maestro_tsv=os.path.join(base_path, "metadata/maestro_synth_train.tsv"),
        real_maestro_train_folder=os.path.join(base_path, "audio/maestro_real_train_16k"),
        real_maestro_train_folder_44k=os.path.join(base_path, "audio/maestro_real_train"),
        real_maestro_train_tsv=os.path.join(base_path, "metadata/maestro_real_train.tsv"),
        real_maestro_val_folder=os.path.join(base_path, "audio/maestro_real_validation_16k"),
        real_maestro_val_folder_44k=os.path.join(base_path, "audio/maestro_real_validation"),
        real_maestro_val_tsv=os.path.join(base_path, "metadata/maestro_real_validation.tsv"),
        real_maestro_val_dur=os.path.join(base_path, "metadata/maestro_real_durations.tsv"),
        embeddings=os.path.join(base_path, "embeddings/beats/{}.hdf5"),
        pseudo_labels=os.path.join("resources", "pseudo-labels/{}.hdf5"),
        strong_tsv_exclude=os.path.join(base_path, "metadata/train/audioset_strong_exclude.tsv"),
        weak_tsv_exclude=os.path.join(base_path, "metadata/train/weak_exclude.tsv"),
        unlabeled_tsv_exclude=os.path.join(base_path, "metadata/train/unlabeled_exclude.tsv"),
        unlabeled_labels_csv=os.path.join(base_path,
                                          "metadata/train/unlabeled_set_with_task4_class_names_full_mapping.csv"),
    )
    median_window = [3, 9, 9, 5, 5, 5, 9, 7, 11, 9, 7, 3, 9, 13, 7, 1, 13, 3, 13, 7, 5, 5, 1, 13, 17, 13, 15]
    test_n_thresholds = 50

    gain_augment = 0
    filter_augment = dict(
        apply=1,
        p=0.8,
        n_transform=2,
        filter_db_range=(-6, 6),
        filter_bands=(3, 6),
        filter_minimum_bandwidth=6
    )

    freq_warp = dict(
        apply=1,
        p=0.5,
        include_maestro=False
    )

    time_augment = dict(
        apply_mask=True,
        apply_shift=False,
        mask_target="desed_strong",  # options: "desed strong", "strong", "all"
        shift_range=0.075,
        shift_target="desed",  # options: "all", "desed"
        min_mask_ratio=0.05,
        max_mask_ratio=0.3,
    )

    mix_augment = dict(
        apply_mixup=True,
        apply_mixstyle=True,
        mixup_p=0.5,
        mixstyle_p=0.5,
        mixstyle_alpha=0.3,
        mixup_unlabeled=True,
        mixup_desed_strong=True,
        mixup_maestro=True,
        mixup_weak=True,
        mixup_max_coef=True,
        mixup_desed_maestro=True,
        mixstyle_desed_strong=True,
        mixstyle_maestro=True,
        mixstyle_unlabeled=True,
        mixstyle_weak=True,
        mixstyle_desed_maestro=True
    )

    # only a small subset of the configs we tried, e.g.:
    # - focal loss types
    # - shift-invariant loss types
    # - balancing activate/inactive frames and classes via loss weights
    # however, we couldn't make these work
    loss = dict(
        maestro_type="BCELoss",
        pseudo_type="BCELoss",
        selfsup_type="MSELoss",
        harden_labels_maestro_p=0.0,
        harden_pseudo_labels_p=0.0
    )

    # the scaler as used in the baseline system
    scaler = dict(
        statistic="instance",
        normtype="minmax",
        dims=(2, 3)
    )

    # different settings regarding ssl and pseudo loss;
    # interestingly, there is no drastic effect in varying these
    ssl_loss_warmup_steps = 270
    ssl_no_class_mask = False
    include_maestro_ssl = True
    exclude_maestro_weak_ssl = False
    use_ict_loss = True

    atst_checkpoint = "atst_base.ckpt"


add_configs(ex)  # add common configurations


# capture the WandbLogger and prefix it with "wandb", this allows to use sacred to update WandbLogger config from the command line
@ex.command(prefix="wandb")
def get_wandb_logger(config, name=None, project="dcase23_task4", offline=False, tags=[]):
    rundir = Path(f"./outputs/{project}/")
    rundir.mkdir(parents=True, exist_ok=True)
    mode = "offline" if offline else "online"
    run = wandb.init(name=name, mode=mode, dir=rundir, project=project, config=config, tags=tags)
    run.define_metric("trainer/global_step")
    run.define_metric("*", step_metric="trainer/global_step", step_sync=True)
    logger = WandbLogger(
        name=name, offline=offline, dir=rundir, project=project, config=config, tags=tags
    )
    return logger


# capture optimizer, this allows to set function parameters via command line
@ex.command(prefix="optimizer")
def get_lr_scheduler(
        optimizer,
        num_training_steps,
        schedule_mode="cos",
        gamma: float = 0.999996,
        num_warmup_steps=270,
        lr_end=1e-7,
):
    if schedule_mode in {"exp"}:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    if schedule_mode in {"cosine", "cos"}:
        return transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    if schedule_mode in {"linear"}:
        print("Linear schedule!")
        return transformers.get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=1.0,
            lr_end=lr_end,
        )
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")


@ex.command(prefix="scaler")
def get_scaler(statistic="instance", normtype="minmax", dims=(2, 3)):
    if statistic == "instance":
        scaler = TorchScaler(
            "instance",
            normtype,
            dims,
        )

        return scaler
    else:
        raise NotImplementedError


def separate_params(model, arch):
    cnn_params = []
    rnn_params = []
    if arch == "frame_dymn":
        pt_params = [[], [], [], [], [], [], []]
    elif arch == "passt":
        pt_params = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    elif arch == "atst_frame":
        pt_params = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    elif arch == "beats":
        pt_params = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    else:
        raise ValueError("Unknown arch: ", arch)

    for k, p in model.named_parameters():
        if not 'as_model' in k or "as_model.head_linear_layer" in k:
            # it is either rnn or cnn
            if 'cnn' in k:
                cnn_params.append(p)
            else:
                rnn_params.append(p)
        elif arch == "frame_dymn":
            if 'in_c' in k:
                pt_params[0].append(p)
            elif 'out_c' in k or 'head' in k:
                pt_params[-1].append(p)
            elif 'layers.0' in k or 'layers.1' in k or 'layers.2' in k:
                pt_params[1].append(p)
            elif 'layers.3' in k or 'layers.4' in k or 'layers.5' in k:
                pt_params[2].append(p)
            elif 'layers.6' in k or 'layers.7' in k or 'layers.8' in k:
                pt_params[3].append(p)
            elif 'layers.9' in k or 'layers.10' in k or 'layers.11' in k:
                pt_params[4].append(p)
            elif 'layers.12' in k or 'layers.13' in k or 'layers.14' in k:
                pt_params[5].append(p)
            else:
                ValueError("Check layer-wise learning for frame-dymn!")
        elif arch == "passt":
            if 'conv_in' in k or 'pos_embed' in k or 'token' in k or 'patch_embed' in k:
                pt_params[0].append(p)
            elif 'blocks.0' in k:
                pt_params[1].append(p)
            elif 'blocks.1' in k:
                pt_params[2].append(p)
            elif 'blocks.2' in k:
                pt_params[3].append(p)
            elif 'blocks.3' in k:
                pt_params[4].append(p)
            elif 'blocks.4' in k:
                pt_params[5].append(p)
            elif 'blocks.5' in k:
                pt_params[6].append(p)
            elif 'blocks.6' in k:
                pt_params[7].append(p)
            elif 'blocks.7' in k:
                pt_params[8].append(p)
            elif 'blocks.8' in k:
                pt_params[9].append(p)
            elif 'blocks.9' in k:
                pt_params[10].append(p)
            elif 'blocks.10' in k:
                pt_params[11].append(p)
            elif 'blocks.11' in k:
                pt_params[12].append(p)
            elif 'model.norm' in k or 'head_linear' in k or 'rnn' in k or 'sigmoid' in k:
                pt_params[13].append(p)
            else:
                ValueError("Check layer-wise learning for frame-passt!")
        elif arch == "atst_frame":
            if "blocks.0." in k:
                pt_params[1].append(p)
            elif "blocks.1." in k:
                pt_params[2].append(p)
            elif "blocks.2." in k:
                pt_params[3].append(p)
            elif "blocks.3." in k:
                pt_params[4].append(p)
            elif "blocks.4." in k:
                pt_params[5].append(p)
            elif "blocks.5." in k:
                pt_params[6].append(p)
            elif "blocks.6." in k:
                pt_params[7].append(p)
            elif "blocks.7." in k:
                pt_params[8].append(p)
            elif "blocks.8" in k:
                pt_params[9].append(p)
            elif "blocks.9." in k:
                pt_params[10].append(p)
            elif "blocks.10." in k:
                pt_params[11].append(p)
            elif "blocks.11." in k:
                pt_params[12].append(p)
            elif ".norm_frame." in k or ".rnn." in k or 'head_linear' in k or 'sigmoid_dense' in k:
                pt_params[13].append(p)
            else:
                pt_params[0].append(p)
        elif arch == "beats":
            if ".layers.0." in k:
                pt_params[1].append(p)
            elif ".layers.1." in k:
                pt_params[2].append(p)
            elif ".layers.2." in k:
                pt_params[3].append(p)
            elif ".layers.3." in k:
                pt_params[4].append(p)
            elif ".layers.4." in k:
                pt_params[5].append(p)
            elif ".layers.5." in k:
                pt_params[6].append(p)
            elif ".layers.6." in k:
                pt_params[7].append(p)
            elif ".layers.7." in k:
                pt_params[8].append(p)
            elif ".layers.8." in k:
                pt_params[9].append(p)
            elif ".layers.9." in k:
                pt_params[10].append(p)
            elif ".layers.10." in k:
                pt_params[11].append(p)
            elif ".layers.11." in k:
                pt_params[12].append(p)
            elif ".norm_frame." in k or ".rnn." in k or 'head_linear' in k or 'sigmoid_dense' in k:
                pt_params[13].append(p)
            else:
                pt_params[0].append(p)
        else:
            ValueError("Check layer-wise learning implementation!")

    return cnn_params, rnn_params, list(reversed(pt_params))


@ex.command(prefix="optimizer")
def get_optimizer(
        model, arch, cnn_lr=0.0001, rnn_lr=0.001, pt_lr=0.0001, pt_lr_scale=1.0, pt_trainable_layers="all",
        adamw=False, weight_decay=1e-4, betas=[0.9, 0.999]
):
    cnn_params, rnn_params, pt_params = separate_params(model, arch)

    cnn_param_groups = [
        {"params": cnn_params, "lr": cnn_lr}
    ]

    rnn_param_groups = [
        {"params": rnn_params, "lr": rnn_lr}
    ]

    pt_trainable_params = []
    if pt_trainable_layers == "all":
        trainable_layers = len(pt_params)
    else:
        trainable_layers = pt_trainable_layers
    for i in range(trainable_layers):
        pt_trainable_params.append(pt_params[i])
        for p in pt_params[i]:
            p.requires_grad = True

    init_lr = pt_lr
    lr_scale = pt_lr_scale
    scale_lrs = [init_lr * (lr_scale ** i) for i in range(trainable_layers)]
    pt_param_groups = [{"params": pt_trainable_params[i], "lr": scale_lrs[i]} for i in
                       range(len(pt_trainable_params))]

    param_groups = cnn_param_groups + rnn_param_groups + pt_param_groups

    if adamw:
        print(f"\nUsing adamw weight_decay={weight_decay}!\n")
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay, betas=betas)
    return torch.optim.Adam(param_groups, betas=betas)


class T4Module(L.LightningModule):
    def __init__(
            self,
            config
    ):
        super(T4Module, self).__init__()
        config = DefaultMunch.fromDict(config)
        self.config = config
        self.construct_modules()
        for param in self.teacher.parameters():
            param.detach_()

        self.sample_rate = self.config.sample_rate
        self.loss_weights = self.config.loss_weights

        self.mix_config = self.config.mix_augment
        self.ta_config = self.config.time_augment
        self.fa_config = self.config.filter_augment

        self.gain_augment = self.config.gain_augment if "gain_augment" in self.config else 0
        self.gain_target = self.config.gain_target if "gain_target" in self.config else "all"

        self.median_filter = ClassWiseMedianFilter(self.config.median_window)

        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))
        self.fwarp_config = self.config.freq_warp

        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.encoder.labels),
            average="macro"
        )

        self.get_weak_teacher_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.encoder.labels),
            average="macro"
        )

        self.val_buffer_sed_scores_eval_student = {}
        self.val_buffer_sed_scores_eval_teacher = {}

        self.val_buffer_real_sed_scores_eval_student = {}
        self.val_buffer_real_sed_scores_eval_teacher = {}

        test_n_thresholds = self.config.test_n_thresholds
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )

        self.test_buffer_psds_eval_student = {k: pd.DataFrame() for k in test_thresholds}
        self.test_buffer_psds_eval_teacher = {k: pd.DataFrame() for k in test_thresholds}
        self.test_buffer_sed_scores_eval_student = {}
        self.test_buffer_sed_scores_eval_teacher = {}
        self.test_buffer_sed_scores_eval_unprocessed_student = {}
        self.test_buffer_sed_scores_eval_unprocessed_teacher = {}
        self.test_buffer_detections_thres05_student = pd.DataFrame()
        self.test_buffer_detections_thres05_teacher = pd.DataFrame()

    def construct_modules(self):
        arch = self.config["arch"]
        self.arch = arch

        scall = functools.partial(
            config_call, config=self.config
        )

        if arch == "atst_frame":
            self.transformer_mel = scall(atst_mel)
            transformer = ATSTWrapper(os.path.join(PRETRAINED_AUDIOSET, self.config['atst_checkpoint']))
            embed_dim = 768
        else:
            raise ValueError(f"Unknown arch={arch}")

        self.crnn_mel = scall(MelSpectrogram)
        self.scaler = get_scaler()

        transformer.arch = arch
        if self.config.t4_wrapper.name == "Task4CRNNEmbeddingsWrapper":
            self.student = Task4CRNNEmbeddingsWrapper(transformer, audioset_classes=527,
                                                      embedding_size=embed_dim, nclass=27,
                                                      pretrained_name=self.config[arch]["pretrained_name"],
                                                      model_init_mode="student")
        else:
            raise ValueError(f"Unknown head={self.config.head}")

        # teacher is loaded separately from checkpoint
        # IMPORTANT - we need a copy of net, otherwise the same object is used in both student and teacher
        # which can have unwanted side effects
        transformer = deepcopy(transformer)
        if self.config.t4_wrapper.name == "Task4CRNNEmbeddingsWrapper":
            self.teacher = Task4CRNNEmbeddingsWrapper(transformer, audioset_classes=527,
                                                      embedding_size=embed_dim, nclass=27,
                                                      pretrained_name=self.config[arch]["pretrained_name"])
        else:
            raise ValueError(f"Unknown head={self.config.head}")
        self.encoder = scall(many_hot_encoder)

        # losses
        self.strong_loss = nn.BCELoss()
        self.weak_loss = nn.BCELoss()

        if self.config.loss.maestro_type == "BCELoss":
            self.maestro_loss = nn.BCELoss()
        elif self.config.loss.maestro_type == "MSELoss":
            self.maestro_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown Maestro loss type: {self.config.loss.maestro_type}")

        if self.config.loss.selfsup_type == "both":
            bce = nn.BCELoss()
            mse = nn.MSELoss()
            self.selfsup_loss = lambda y_hat, y: (mse(y_hat, y) * 4 + bce(y_hat, y)) / 2
        elif self.config.loss.selfsup_type == "BCELoss":
            self.selfsup_loss = nn.BCELoss()
        elif self.config.loss.selfsup_type == "MSELoss":
            self.selfsup_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown Pseudo loss type: {self.config.loss.pseudo_type}")

        if self.config.loss.pseudo_type == "both":
            bce = nn.BCELoss()
            mse = nn.MSELoss()
            self.pseudo_label_loss = lambda y_hat, y: (mse(y_hat, y) * 4 + bce(y_hat, y)) / 2
        elif self.config.loss.pseudo_type == "BCELoss":
            self.pseudo_label_loss = nn.BCELoss()
        elif self.config.loss.pseudo_type == "MSELoss":
            self.pseudo_label_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown Pseudo loss type: {self.config.loss.pseudo_type}")

    def on_train_start(self) -> None:
        # for tracking energy consumption
        log_path = os.path.join(self.loggers[0].name, self.loggers[0].experiment.id)
        os.makedirs(os.path.join(log_path, "codecarbon"), exist_ok=True)
        self.tracker_train = OfflineEmissionsTracker(
            "DCASE Task 4 Stage 2",
            output_dir=os.path.join(log_path, "codecarbon"),
            output_file="emissions_stage1.csv",
            log_level="warning",
            country_iso_code="AUT",
            gpu_ids=[torch.cuda.current_device()],
        )
        self.tracker_train.start()

    def on_test_start(self) -> None:
        log_path = os.path.join(self.loggers[0].name, self.loggers[0].experiment.id)
        os.makedirs(os.path.join(log_path, "codecarbon"), exist_ok=True)
        self.tracker_devtest = OfflineEmissionsTracker(
            "DCASE Task 4 Stage 2, Test",
            output_dir=os.path.join(log_path, "codecarbon"),
            output_file="emissions_stage1_test.csv",
            log_level="warning",
            country_iso_code="AUT",
            gpu_ids=[torch.cuda.current_device()],
        )

        self.tracker_devtest.start()

    def on_train_end(self) -> None:
        # dump consumption
        self.tracker_train.stop()
        training_kwh = self.tracker_train._total_energy.kWh
        self.loggers[0].experiment.log({"train/tot_energy_kWh": torch.tensor(float(training_kwh))})

    def update_ema(self, alpha, global_step, model, ema_model):
        """ Update teacher model parameters

        Args:
            alpha: float, the factor to be used between each updated step.
            global_step: int, the current global step to be used.
            model: torch.Module, student model to use
            ema_model: torch.Module, teacher model to use
        """
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(ema_model.parameters(), model.parameters()):
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def detect(self, mel_feats, transformer_feats, model, classes_mask=None):
        x = model(
            self.scaler(
                self.take_log(mel_feats)
            ),
            transformer_feats,
            classes_mask=classes_mask
        )
        return x

    def training_step(self, batch, batch_idx):
        hooks = []
        if self.trainer.global_rank == 0:
            hooks = register_print_hooks(
                self, register_at_step=self.config.debug_shapes
            )

        audio, labels, padded_indxs, pseudo_strong, valid_class_mask = batch
        pseudo_strong = pseudo_strong.transpose(1, 2)
        audio = audio.float()
        valid_class_mask = valid_class_mask.bool()
        indx_maestro, indx_strong, indx_synth, indx_weak, indx_unlabelled = np.cumsum(self.config.training.batch_sizes)

        batch_num = audio.shape[0]
        # deriving masks for each dataset
        all_strong_mask = torch.zeros(batch_num).to(audio).bool()
        maestro_mask = torch.zeros(batch_num).to(audio).bool()
        desed_strong_mask = torch.zeros(batch_num).to(audio).bool()
        desed_synth_mask = torch.zeros(batch_num).to(audio).bool()
        desed_real_strong_mask = torch.zeros(batch_num).to(audio).bool()
        weak_mask = torch.zeros(batch_num).to(audio).bool()
        ssl_mask = torch.zeros(batch_num).to(audio).bool()
        unlabeled_mask = torch.zeros(batch_num).to(audio).bool()
        desed_mask = torch.zeros(batch_num).to(audio).bool()

        all_strong_mask[:indx_synth] = 1
        maestro_mask[:indx_maestro] = 1
        desed_strong_mask[indx_maestro:indx_synth] = 1
        desed_synth_mask[indx_strong:indx_synth] = 1
        desed_real_strong_mask[indx_maestro:indx_strong] = 1
        weak_mask[indx_synth:indx_weak] = 1
        ssl_mask[indx_maestro:] = 1
        unlabeled_mask[indx_weak:] = 1
        desed_mask[indx_maestro:] = 1

        # calculate ssl loss on Maestro as well
        if self.config.include_maestro_ssl:
            ssl_mask = torch.ones(batch_num).to(audio).bool()

        if self.config.only_maestro_ssl:
            ssl_mask[indx_maestro:] = 0

        # gain augment
        if self.gain_augment > 0:
            if self.gain_target == "all":
                audio = gain_augment(audio, gain=self.gain_augment)
            elif self.gain_target == "desed":
                audio[desed_mask] = gain_augment(audio[desed_mask], gain=self.gain_augment)
            else:
                raise ValueError(f"Unknown gain target: {self.gain_target}")

        with autocast(enabled=False, device_type='cuda'):
            audio = audio.float()
            sed_feats = self.crnn_mel(audio).unsqueeze(1)
            pt_feats = self.transformer_mel(audio)

        # deriving weak labels
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()

        if self.config.loss.harden_labels_maestro_p > random.random():
            labels_maestro_hard = torch.where(labels[maestro_mask] > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            labels_maestro_hard = labels_maestro_hard.to(labels.dtype)
            labels[maestro_mask] = labels_maestro_hard

        if self.config.loss.harden_pseudo_labels_p > random.random():
            pseudo_strong = torch.where(pseudo_strong > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            pseudo_strong = pseudo_strong.to(labels.dtype)

        if self.ta_config.apply_shift:
            if self.ta_config.shift_target == "all":
                features, pt_feats, labels, pseudo_strong = frame_shift(sed_feats, labels, embeddings=pt_feats,
                                                           pseudo_labels=pseudo_strong,
                                                           net_pooling=self.encoder.net_pooling,
                                                           shift_range=self.ta_config.shift_range)
            elif self.ta_config.shift_target == "desed":
                sed_feats[desed_mask], pt_feats[desed_mask], labels[desed_mask], pseudo_strong[desed_mask] = \
                    frame_shift(sed_feats[desed_mask], labels[desed_mask], embeddings=pt_feats[desed_mask],
                                pseudo_labels=pseudo_strong[desed_mask],
                                net_pooling=self.encoder.net_pooling,
                                shift_range=self.ta_config.shift_range)
            else:
                raise ValueError(f"Unknown shift target: {self.ta_config.shift_target}")

        if self.ta_config.apply_mask:
            if self.ta_config.mask_target == "desed_strong":
                sed_feats[desed_strong_mask], pt_feats[desed_strong_mask], labels[desed_strong_mask], pseudo_strong[desed_strong_mask] = \
                    time_mask(sed_feats[desed_strong_mask], labels[desed_strong_mask],
                              embeddings=pt_feats[desed_strong_mask], pseudo_labels=pseudo_strong[desed_strong_mask], net_pooling=self.encoder.net_pooling,
                              min_mask_ratio=self.ta_config.min_mask_ratio,
                              max_mask_ratio=self.ta_config.max_mask_ratio)
            elif self.ta_config.mask_target == "strong":
                sed_feats[all_strong_mask], pt_feats[all_strong_mask], labels[all_strong_mask], pseudo_strong[all_strong_mask] = \
                    time_mask(sed_feats[all_strong_mask], labels[all_strong_mask],
                              embeddings=pt_feats[all_strong_mask], pseudo_labels=pseudo_strong[all_strong_mask], net_pooling=self.encoder.net_pooling,
                              min_mask_ratio=self.ta_config.min_mask_ratio,
                              max_mask_ratio=self.ta_config.max_mask_ratio)
            elif self.ta_config.mask_target == "all":
                sed_feats, pt_feats, labels, pseudo_strong = \
                    time_mask(sed_feats,
                              labels,
                              embeddings=pt_feats,
                              pseudo_labels=pseudo_strong,
                              net_pooling=self.encoder.net_pooling,
                              min_mask_ratio=self.ta_config.min_mask_ratio,
                              max_mask_ratio=self.ta_config.max_mask_ratio)
            else:
                raise ValueError(f"Unknown value for 'mask_target': {self.ta_config.mask_target}")

        sed_feats_org = sed_feats.clone()
        pt_feats_org = pt_feats.clone()

        if self.mix_config.apply_mixup and self.mix_config.mixup_p > random.random():
            if self.mix_config.mixup_weak:
                sed_feats[weak_mask], pt_feats[weak_mask], labels_weak, pseudo_strong[
                    weak_mask], perm_weak, c_weak = mixup(
                    sed_feats[weak_mask], embeddings=pt_feats[weak_mask], targets=labels_weak, mixup_label_type="soft",
                    pseudo_strong=pseudo_strong[weak_mask],
                    return_mix_coef=True, max_coef=self.mix_config.mixup_max_coef
                )

            if self.mix_config.mixup_desed_maestro and self.mix_config.mixup_desed_strong \
                    and self.mix_config.mixup_maestro:
                sed_feats[all_strong_mask], pt_feats[all_strong_mask], labels[
                    all_strong_mask], valid_class_mask[all_strong_mask], pseudo_strong[
                    all_strong_mask], perm_strong, c_strong = mixup(
                    sed_feats[all_strong_mask], embeddings=pt_feats[all_strong_mask],
                    valid_class_mask=valid_class_mask[all_strong_mask],
                    targets=labels[all_strong_mask], pseudo_strong=pseudo_strong[all_strong_mask],
                    mixup_label_type="soft",
                    return_mix_coef=True, max_coef=self.mix_config.mixup_max_coef
                )

                perm_maestro = None
            else:
                if self.mix_config.mixup_desed_strong:
                    sed_feats[desed_strong_mask], pt_feats[desed_strong_mask], labels[desed_strong_mask], pseudo_strong[
                        desed_strong_mask], \
                    perm_strong, c_strong = mixup(
                        sed_feats[desed_strong_mask], embeddings=pt_feats[desed_strong_mask],
                        targets=labels[desed_strong_mask], pseudo_strong=pseudo_strong[desed_strong_mask],
                        mixup_label_type="soft",
                        return_mix_coef=True, max_coef=self.mix_config.mixup_max_coef
                    )

                if self.mix_config.mixup_maestro:
                    sed_feats[maestro_mask], pt_feats[maestro_mask], labels[maestro_mask], pseudo_strong[
                        maestro_mask], perm_maestro, c_maestro = mixup(
                        sed_feats[maestro_mask], embeddings=pt_feats[maestro_mask],
                        targets=labels[maestro_mask], pseudo_strong=pseudo_strong[maestro_mask],
                        mixup_label_type="soft",
                        return_mix_coef=True, max_coef=self.mix_config.mixup_max_coef
                    )

            if self.mix_config.mixup_unlabeled:
                sed_feats[unlabeled_mask], pt_feats[unlabeled_mask], labels[unlabeled_mask], pseudo_strong[
                    unlabeled_mask], \
                perm_unlabeled, c_unlabeled = mixup(
                    sed_feats[unlabeled_mask], embeddings=pt_feats[unlabeled_mask],
                    targets=labels[unlabeled_mask], pseudo_strong=pseudo_strong[unlabeled_mask],
                    mixup_label_type="soft",
                    return_mix_coef=True, max_coef=self.mix_config.mixup_max_coef
                )
        else:
            perm_weak = None

        if self.mix_config.apply_mixstyle and self.mix_config.mixstyle_p > random.random():
            if self.mix_config.mixstyle_weak:
                sed_feats[weak_mask] = mixstyle(sed_feats[weak_mask], self.mix_config.mixstyle_alpha)
                pt_feats[weak_mask] = mixstyle(pt_feats[weak_mask], self.mix_config.mixstyle_alpha)

            if self.mix_config.mixstyle_desed_maestro and self.mix_config.mixstyle_desed_strong and \
                    self.mix_config.mixstyle_maestro:
                sed_feats[all_strong_mask] = mixstyle(sed_feats[all_strong_mask], self.mix_config.mixstyle_alpha)
                pt_feats[all_strong_mask] = mixstyle(pt_feats[all_strong_mask], self.mix_config.mixstyle_alpha)
            else:
                if self.mix_config.mixstyle_desed_strong:
                    sed_feats[desed_strong_mask] = mixstyle(sed_feats[desed_strong_mask],
                                                            self.mix_config.mixstyle_alpha)
                    pt_feats[desed_strong_mask] = mixstyle(pt_feats[desed_strong_mask], self.mix_config.mixstyle_alpha)

                if self.mix_config.mixstyle_maestro:
                    sed_feats[maestro_mask] = mixstyle(sed_feats[maestro_mask], self.mix_config.mixstyle_alpha)
                    pt_feats[maestro_mask] = mixstyle(pt_feats[maestro_mask], self.mix_config.mixstyle_alpha)

            if self.mix_config.mixstyle_unlabeled:
                sed_feats[unlabeled_mask] = mixstyle(sed_feats[unlabeled_mask], self.mix_config.mixstyle_alpha)
                pt_feats[unlabeled_mask] = mixstyle(pt_feats[unlabeled_mask], self.mix_config.mixstyle_alpha)

        if self.fwarp_config.apply and self.fwarp_config.p > random.random():
            pt_feats = pt_feats.squeeze(1)
            pt_feats[weak_mask] = self.freq_warp(pt_feats[weak_mask])
            pt_feats[desed_strong_mask] = self.freq_warp(pt_feats[desed_strong_mask])
            pt_feats[unlabeled_mask] = self.freq_warp(pt_feats[unlabeled_mask])
            if self.fwarp_config.include_maestro:
                pt_feats[maestro_mask] = self.freq_warp(pt_feats[maestro_mask])
            pt_feats = pt_feats.unsqueeze(1)

        if self.fa_config.apply and self.fa_config.p > random.random():
            sed_feats, _ = feature_transformation(sed_feats, self.fa_config.n_transform,
                                                  self.fa_config.filter_db_range,
                                                  self.fa_config.filter_bands,
                                                  self.fa_config.filter_minimum_bandwidth)

            pt_feats, _ = feature_transformation(pt_feats, self.fa_config.n_transform,
                                                 self.fa_config.filter_db_range,
                                                 self.fa_config.filter_bands,
                                                 self.fa_config.filter_minimum_bandwidth)

        # mask labels for invalid datasets classes after mixup.
        labels = labels.masked_fill(~valid_class_mask[:, :, None].expand_as(labels), 0.0)
        labels_weak = labels_weak.masked_fill(
            ~valid_class_mask[weak_mask], 0.0)
        if self.config.pseudo_strong_no_class_mask is False:
            pseudo_strong = pseudo_strong.masked_fill(~valid_class_mask[:, :, None].expand_as(pseudo_strong), 0.0)

        # sed student forward
        (strong_preds_student, weak_preds_student), (
        strong_preds_student_all, weak_preds_student_all) = self.detect(
            sed_feats,
            pt_feats,
            self.student,
            classes_mask=(valid_class_mask, None)
        )

        with autocast(enabled=False, device_type='cuda'):
            # calculate loss with 32-bit precision
            strong_preds_student = strong_preds_student.float()
            labels = labels.float()
            labels_weak = labels_weak.float()

            # supervised loss on synthetic strong labels
            loss_desed_synth = self.strong_loss(
                strong_preds_student[desed_synth_mask],
                labels[desed_synth_mask],
            )

            # supervised loss on real strong labels
            loss_desed_real_strong = self.strong_loss(
                strong_preds_student[desed_real_strong_mask],
                labels[desed_real_strong_mask],
            )

            # supervised loss on maestro
            loss_maestro = self.maestro_loss(
                strong_preds_student[maestro_mask],
                labels[maestro_mask],
            )

            # supervised loss on weakly labelled
            loss_weak = self.weak_loss(
                weak_preds_student[weak_mask],
                labels_weak,
            )

        # total supervised loss
        strong_loss = (self.loss_weights[0] * loss_maestro +
                       self.loss_weights[1] * loss_desed_real_strong +
                       self.loss_weights[2] * loss_desed_synth) / 3
        weak_loss = (self.loss_weights[3] * loss_weak) / 2
        tot_loss_supervised = strong_loss + weak_loss

        with torch.no_grad():
            # teacher predictions based on unaugmented samples
            strong_preds_teacher, weak_preds_teacher = self.detect(
                sed_feats_org,
                pt_feats_org,
                self.teacher,
                classes_mask=None if self.config.ssl_no_class_mask else valid_class_mask
            )

        if self.config.ssl_no_class_mask:
            strong_preds_student = strong_preds_student_all
            weak_preds_student = weak_preds_student_all

        ssl_weight = min(
            self.global_step / self.config.ssl_loss_warmup_steps * self.loss_weights[4],
            self.loss_weights[4] * self.config.ssl_scale_max
        )

        # either ICT or Mean Teacher loss
        if perm_weak is not None and self.config.use_ict_loss:
            # the ICT loss it is

            # weak subset
            c_weak = torch.tensor(c_weak, dtype=strong_preds_teacher.dtype, device=strong_preds_teacher.device)
            strong_org, weak_org = strong_preds_teacher[weak_mask], weak_preds_teacher[weak_mask]
            c_shape_strong = (strong_org.size(0), 1, 1)
            c_shape_weak = (weak_org.size(0), 1)
            assert len(c_shape_strong) == len(strong_org.shape)
            assert len(c_shape_weak) == len(weak_org.shape)
            strong_mix = c_weak.view(c_shape_strong) * strong_org + \
                         (1 - c_weak.view(c_shape_strong)) * strong_org[perm_weak]
            weak_mix = c_weak.view(c_shape_weak) * weak_org + \
                       (1 - c_weak.view(c_shape_weak)) * weak_org[perm_weak]
            loss_ict_weak = self.selfsup_loss(strong_preds_student[weak_mask], strong_mix.clamp(0, 1).detach()) + \
                            self.selfsup_loss(weak_preds_student[weak_mask], weak_mix.clamp(0, 1).detach()) / 2

            # unlabeled subset
            c_unlabeled = torch.tensor(c_unlabeled, dtype=strong_preds_teacher.dtype,
                                       device=strong_preds_teacher.device)
            strong_org, weak_org = strong_preds_teacher[unlabeled_mask], weak_preds_teacher[unlabeled_mask]
            c_shape_strong = (strong_org.size(0), 1, 1)
            c_shape_weak = (weak_org.size(0), 1)
            strong_mix = c_unlabeled.view(c_shape_strong) * strong_org + \
                         (1 - c_unlabeled.view(c_shape_strong)) * strong_org[perm_unlabeled]
            weak_mix = c_unlabeled.view(c_shape_weak) * weak_org + \
                       (1 - c_unlabeled.view(c_shape_weak)) * weak_org[perm_unlabeled]
            loss_ict_unlabeled = self.selfsup_loss(strong_preds_student[unlabeled_mask],
                                                   strong_mix.clamp(0, 1).detach()) + \
                                 self.selfsup_loss(weak_preds_student[unlabeled_mask],
                                                   weak_mix.clamp(0, 1).detach()) / 2

            if perm_maestro is not None:
                # strong subset
                c_strong = torch.tensor(c_strong, dtype=strong_preds_teacher.dtype,
                                        device=strong_preds_teacher.device)
                strong_org, weak_org = strong_preds_teacher[desed_strong_mask], weak_preds_teacher[desed_strong_mask]
                c_shape_strong = (strong_org.size(0), 1, 1)
                c_shape_weak = (weak_org.size(0), 1)
                strong_mix = c_strong.view(c_shape_strong) * strong_org + \
                             (1 - c_strong.view(c_shape_strong)) * strong_org[perm_strong]
                weak_mix = c_strong.view(c_shape_weak) * weak_org + \
                           (1 - c_strong.view(c_shape_weak)) * weak_org[perm_strong]
                loss_ict_strong = self.selfsup_loss(strong_preds_student[desed_strong_mask],
                                                    strong_mix.clamp(0, 1).detach()) + \
                                  self.selfsup_loss(weak_preds_student[desed_strong_mask],
                                                    weak_mix.clamp(0, 1).detach()) / 2

                # maestro subset
                c_maestro = torch.tensor(c_maestro, dtype=strong_preds_teacher.dtype,
                                         device=strong_preds_teacher.device)
                strong_org, weak_org = strong_preds_teacher[maestro_mask], weak_preds_teacher[maestro_mask]
                c_shape_strong = (strong_org.size(0), 1, 1)
                c_shape_weak = (weak_org.size(0), 1)
                strong_mix = c_maestro.view(c_shape_strong) * strong_org + \
                             (1 - c_maestro.view(c_shape_strong)) * strong_org[perm_maestro]
                weak_mix = c_maestro.view(c_shape_weak) * weak_org + \
                           (1 - c_maestro.view(c_shape_weak)) * weak_org[perm_maestro]

                if self.config.exclude_maestro_weak_ssl:
                    weak_weight = 0
                else:
                    weak_weight = 1
                loss_ict_maestro = self.selfsup_loss(strong_preds_student[maestro_mask],
                                                     strong_mix.clamp(0, 1).detach()) + \
                                   self.selfsup_loss(weak_preds_student[maestro_mask],
                                                     weak_mix.clamp(0, 1).detach()) / 2 * weak_weight
            else:
                c_strong = torch.tensor(c_strong, dtype=strong_preds_teacher.dtype,
                                        device=strong_preds_teacher.device)
                # joint mixup of desed strong and maestro
                strong_org, weak_org = strong_preds_teacher[all_strong_mask], weak_preds_teacher[all_strong_mask]
                c_shape_strong = (strong_org.size(0), 1, 1)
                c_shape_weak = (weak_org.size(0), 1)
                strong_mix = c_strong.view(c_shape_strong) * strong_org + \
                             (1 - c_strong.view(c_shape_strong)) * strong_org[perm_strong]
                weak_mix = c_strong.view(c_shape_weak) * weak_org + \
                           (1 - c_strong.view(c_shape_weak)) * weak_org[perm_strong]

                if self.config.exclude_maestro_weak_ssl:
                    mask_events_desed = set(classes_labels_desed.keys())
                    if self.config.training.use_desed_maestro_alias:
                        mask_events_desed = mask_events_desed.union(set(["cutlery and dishes", "people talking"]))
                    classes_mask = torch.ones(len(self.encoder.labels))
                    for indx, cls in enumerate(self.encoder.labels):
                        if cls not in mask_events_desed:
                            classes_mask[indx] = 0
                    classes_mask = classes_mask.to(weak_mix).bool()
                else:
                    classes_mask = torch.ones(len(self.encoder.labels))
                    classes_mask = classes_mask.to(weak_mix).bool()

                loss_ict_strong = self.selfsup_loss(strong_preds_student[all_strong_mask],
                                                    strong_mix.clamp(0, 1).detach()) + \
                                  self.selfsup_loss(weak_preds_student[all_strong_mask][:, classes_mask],
                                                    weak_mix[:, classes_mask].clamp(0, 1).detach()) / 2
                loss_ict_maestro = loss_ict_strong

            loss_ict = (loss_ict_strong + loss_ict_weak + loss_ict_unlabeled + loss_ict_maestro) / 8
            tot_self_loss = loss_ict * ssl_weight / 4
            self.log("train/student/tot_ict_loss", loss_ict)
        else:
            # Meant Teacher loss
            strong_self_sup_loss = self.selfsup_loss(
                strong_preds_student[ssl_mask], strong_preds_teacher.detach()[ssl_mask]
            )

            if self.config.exclude_maestro_weak_ssl:
                ssl_mask_weak = torch.clone(ssl_mask)
                ssl_mask_weak[:indx_maestro] = 0
            else:
                ssl_mask_weak = torch.clone(ssl_mask)

            weak_self_sup_loss = self.selfsup_loss(
                weak_preds_student[ssl_mask_weak], weak_preds_teacher.detach()[ssl_mask_weak]
            )

            tot_self_loss = (strong_self_sup_loss + weak_self_sup_loss / 2) * ssl_weight
            self.log("train/student/weak_self_sup_loss", weak_self_sup_loss)
            self.log("train/student/strong_self_sup_loss", strong_self_sup_loss)

        # pseudo label loss
        with autocast(enabled=False, device_type='cuda'):
            strong_preds_student = strong_preds_student.float()
            pseudo_strong.float()

            loss_pseudo_desed = self.pseudo_label_loss(
                strong_preds_student[desed_mask],
                pseudo_strong[desed_mask]
            )

            loss_pseudo_maestro = self.pseudo_label_loss(
                strong_preds_student[maestro_mask],
                pseudo_strong[maestro_mask]
            )

            # weight pseudo label losses on desed and maestro according to their corresponding
            # supervised loss weight
            loss_pseudo_desed = loss_pseudo_desed * (self.loss_weights[1] + self.loss_weights[2] + self.loss_weights[3])
            loss_pseudo_maestro = loss_pseudo_maestro * self.loss_weights[0]
            pseudo_label_loss = (loss_pseudo_desed + loss_pseudo_maestro) * self.loss_weights[5]

        tot_loss = tot_loss_supervised + tot_self_loss + pseudo_label_loss

        cnn_lr = self.optimizers().param_groups[0]["lr"]
        rnn_lr = self.optimizers().param_groups[1]["lr"]
        pt_lr = self.optimizers().param_groups[-1]["lr"]

        self.log("train/student/loss_pseudo", pseudo_label_loss)
        self.log("train/student/loss_desed_synth", loss_desed_synth)
        self.log("train/student/loss_desed_real_strong", loss_desed_real_strong)
        self.log("train/student/loss_maestro", loss_maestro)
        self.log("train/student/loss_strong", strong_loss)
        self.log("train/student/loss_weak", weak_loss)
        self.log("train/student/tot_ssl", tot_self_loss, prog_bar=True)
        self.log("train/student/tot_supervised", tot_loss_supervised, prog_bar=True)
        self.log("train/cnn_lr", cnn_lr, prog_bar=True)
        self.log("train/rnn_lr", rnn_lr, prog_bar=True)
        self.log("train/pt_lr", pt_lr, prog_bar=True)
        self.log("train/ssl_weight", ssl_weight)
        self.log("train/epoch", self.current_epoch)

        for h in hooks:
            h.remove()

        return tot_loss

    def on_before_zero_grad(self, *args, **kwargs):
        # update EMA teacher
        self.update_ema(
            self.config.ema_factor,
            self.global_step,
            self.student,
            self.teacher,
        )

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        hooks = []
        if self.trainer.global_rank == 0:
            hooks = register_print_hooks(
                self, register_at_step=self.config.debug_shapes
            )

        audio, labels, padded_indxs, filenames, valid_class_mask = batch
        valid_class_mask = valid_class_mask.bool()

        # create mel spectrograms
        with autocast(enabled=False, device_type='cuda'):
            audio = audio.float()
            sed_feats = self.crnn_mel(audio).unsqueeze(1)
            pt_feats = self.transformer_mel(audio)

        bs = len(labels)

        strong_preds_student, weak_preds_student = self.detect(
            sed_feats,
            pt_feats,
            self.student,
            classes_mask=valid_class_mask
        )
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(
            sed_feats,
            pt_feats,
            self.teacher,
            classes_mask=valid_class_mask
        )

        # we derive masks for each dataset based on folders of filenames
        mask_weak = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.config.t4_paths['weak_folder'])) or
                    str(Path(x).parent)
                    == str(Path(self.config.t4_paths['weak_folder_44k']))
                    for x in filenames
                ]
            )
                .to(audio)
                .bool()
        )
        mask_strong = (
            torch.tensor(
                [
                    str(Path(x).parent) in [
                        str(Path(self.config.t4_paths['synth_val_folder'])),
                        str(Path(self.config.t4_paths['synth_val_folder_44k'])),
                        str(Path(self.config.t4_paths['real_maestro_train_folder'])),
                        str(Path(self.config.t4_paths['real_maestro_train_folder_44k']))
                    ]
                    for x in filenames
                ]
            )
                .to(audio)
                .bool()
        )
        mask_real = (
            torch.tensor(
                [
                    str(Path(x).parent) in [
                        str(Path(self.config.t4_paths['external_strong_folder'])),
                    ]
                    for x in filenames
                ]
            )
                .to(audio)
                .bool()
        )

        if torch.any(mask_weak):
            labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()

            with autocast(enabled=False, device_type='cuda'):
                weak_preds_student = weak_preds_student.float()
                weak_preds_teacher = weak_preds_teacher.float()
                labels_weak = labels_weak.float()

                loss_weak_student = self.weak_loss(
                    weak_preds_student[mask_weak], labels_weak
                )
                loss_weak_teacher = self.weak_loss(
                    weak_preds_teacher[mask_weak], labels_weak
                )

            self.log("val/student/loss_weak", loss_weak_student, batch_size=bs)
            self.log("val/teacher/loss_weak", loss_weak_teacher, batch_size=bs)

            # accumulate f1 score for weak labels
            self.get_weak_student_f1_seg_macro.update(
                weak_preds_student[mask_weak], labels_weak.long()
            )
            self.get_weak_teacher_f1_seg_macro.update(
                weak_preds_teacher[mask_weak], labels_weak.long()
            )

        if torch.any(mask_strong):
            with autocast(enabled=False, device_type='cuda'):
                labels = labels.float()
                strong_preds_student = strong_preds_student.float()
                strong_preds_teacher = strong_preds_teacher.float()
                labels = labels.float()

                loss_strong_student = self.strong_loss(
                    strong_preds_student[mask_strong], labels[mask_strong]
                )
                loss_strong_teacher = self.strong_loss(
                    strong_preds_teacher[mask_strong], labels[mask_strong]
                )

            self.log("val/student/loss_strong", loss_strong_student, batch_size=bs)
            self.log("val/teacher/loss_strong", loss_strong_teacher, batch_size=bs)

            filenames_strong = [
                x
                for x in filenames
                if str(Path(x).parent) in [
                    str(Path(self.config.t4_paths['synth_val_folder'])),
                    str(Path(self.config.t4_paths['synth_val_folder_44k'])),
                    str(Path(self.config.t4_paths['real_maestro_train_folder'])),
                    str(Path(self.config.t4_paths['real_maestro_train_folder_44k'])),
                ]
            ]

            (
                scores_unprocessed_student_strong,
                scores_postprocessed_student_strong,
                decoded_student_strong,
            ) = batched_decode_preds(
                strong_preds_student[mask_strong],
                filenames_strong,
                self.encoder,
                median_filter=self.median_filter,
                thresholds=[],
            )

            self.val_buffer_sed_scores_eval_student.update(
                scores_postprocessed_student_strong
            )

            (
                scores_unprocessed_teacher_strong,
                scores_postprocessed_teacher_strong,
                decoded_teacher_strong,
            ) = batched_decode_preds(
                strong_preds_teacher[mask_strong],
                filenames_strong,
                self.encoder,
                median_filter=self.median_filter,
                thresholds=[],
            )

            self.val_buffer_sed_scores_eval_teacher.update(
                scores_postprocessed_teacher_strong
            )

        if torch.any(mask_real):
            with autocast(enabled=False, device_type='cuda'):
                labels = labels.float()
                strong_preds_student = strong_preds_student.float()
                strong_preds_teacher = strong_preds_teacher.float()
                labels = labels.float()

                loss_strong_student = self.strong_loss(
                    strong_preds_student[mask_real], labels[mask_real]
                )
                loss_strong_teacher = self.strong_loss(
                    strong_preds_teacher[mask_real], labels[mask_real]
                )

            self.log("val/student/loss_real_strong", loss_strong_student, batch_size=bs)
            self.log("val/teacher/loss_real_strong", loss_strong_teacher, batch_size=bs)

            filenames_real = [
                x
                for x in filenames
                if str(Path(x).parent) in [
                    str(Path(self.config.t4_paths['external_strong_folder']))
                ]
            ]

            (
                scores_unprocessed_student_strong,
                scores_postprocessed_student_strong,
                decoded_student_strong,
            ) = batched_decode_preds(
                strong_preds_student[mask_real],
                filenames_real,
                self.encoder,
                median_filter=self.median_filter,
                thresholds=[],
            )

            self.val_buffer_real_sed_scores_eval_student.update(
                scores_postprocessed_student_strong
            )

            (
                scores_unprocessed_teacher_strong,
                scores_postprocessed_teacher_strong,
                decoded_teacher_strong,
            ) = batched_decode_preds(
                strong_preds_teacher[mask_real],
                filenames_real,
                self.encoder,
                median_filter=self.median_filter,
                thresholds=[],
            )

            self.val_buffer_real_sed_scores_eval_teacher.update(
                scores_postprocessed_teacher_strong
            )

        for h in hooks:
            h.remove()

    def on_validation_epoch_end(self):
        weak_student_f1_macro = self.get_weak_student_f1_seg_macro.compute()
        weak_teacher_f1_macro = self.get_weak_teacher_f1_seg_macro.compute()

        # synth dataset
        desed_ground_truth = sed_scores_eval.io.read_ground_truth_events(self.config.t4_paths['synth_val_tsv'])
        desed_audio_durations = sed_scores_eval.io.read_audio_durations(self.config.t4_paths['synth_val_dur'])

        # drop audios without events
        desed_ground_truth = {
            audio_id: gt for audio_id, gt in desed_ground_truth.items() if len(gt) > 0
        }
        desed_audio_durations = {
            audio_id: desed_audio_durations[audio_id] for audio_id in desed_ground_truth.keys()
        }

        keys = ['onset', 'offset'] + sorted(classes_labels_desed.keys())
        desed_scores = {
            clip_id: self.val_buffer_sed_scores_eval_student[clip_id][keys]
            for clip_id in desed_ground_truth.keys()
        }

        psds1_sed_scores_eval_student = compute_psds_from_scores(
            desed_scores,
            desed_ground_truth,
            desed_audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )

        intersection_f1_macro_thres05_student_sed_scores_eval = sed_scores_eval.intersection_based.fscore(
            desed_scores, desed_ground_truth, threshold=.5, dtc_threshold=.5, gtc_threshold=.5)[0]['macro_average']
        collar_f1_macro_thres05_student_sed_scores_eval = sed_scores_eval.collar_based.fscore(
            desed_scores, desed_ground_truth, threshold=.5, onset_collar=.2, offset_collar=.2, offset_collar_rate=.2)[
            0]['macro_average']
        desed_scores = {
            clip_id: self.val_buffer_sed_scores_eval_teacher[clip_id][keys]
            for clip_id in desed_ground_truth.keys()
        }
        psds1_sed_scores_eval_teacher = compute_psds_from_scores(
            desed_scores,
            desed_ground_truth,
            desed_audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )
        intersection_f1_macro_thres05_teacher_sed_scores_eval = sed_scores_eval.intersection_based.fscore(
            desed_scores, desed_ground_truth, threshold=.5, dtc_threshold=.5, gtc_threshold=.5)[0]['macro_average']
        collar_f1_macro_thres05_teacher_sed_scores_eval = sed_scores_eval.collar_based.fscore(
            desed_scores, desed_ground_truth, threshold=.5, onset_collar=.2, offset_collar=.2, offset_collar_rate=.2)[
            0]['macro_average']

        ############################################################################################################
        ############################################################################################################
        # external strong real
        external_ground_truth = sed_scores_eval.io.read_ground_truth_events(self.config.t4_paths['external_strong_val_tsv'])
        external_audio_durations = sed_scores_eval.io.read_audio_durations(self.config.t4_paths['external_strong_dur'])

        # drop audios without events
        external_ground_truth = {
            audio_id: gt for audio_id, gt in external_ground_truth.items() if len(gt) > 0
        }
        external_audio_durations = {
            audio_id: external_audio_durations[audio_id] for audio_id in external_ground_truth.keys()
        }

        keys = ['onset', 'offset'] + sorted(classes_labels_desed.keys())
        external_scores = {
            clip_id: self.val_buffer_real_sed_scores_eval_student[clip_id][keys]
            for clip_id in external_ground_truth.keys()
        }

        psds1_real_sed_scores_eval_student = compute_psds_from_scores(
            external_scores,
            external_ground_truth,
            external_audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )

        intersection_f1_macro_thres05_student_real_sed_scores_eval = sed_scores_eval.intersection_based.fscore(
            external_scores, external_ground_truth, threshold=.5, dtc_threshold=.5, gtc_threshold=.5)[0]['macro_average']
        collar_f1_macro_thres05_student_real_sed_scores_eval = sed_scores_eval.collar_based.fscore(
            external_scores, external_ground_truth, threshold=.5, onset_collar=.2, offset_collar=.2, offset_collar_rate=.2)[
            0]['macro_average']
        external_scores = {
            clip_id: self.val_buffer_real_sed_scores_eval_teacher[clip_id][keys]
            for clip_id in external_ground_truth.keys()
        }
        psds1_real_sed_scores_eval_teacher = compute_psds_from_scores(
            external_scores,
            external_ground_truth,
            external_audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )
        intersection_f1_macro_thres05_teacher_real_sed_scores_eval = sed_scores_eval.intersection_based.fscore(
            external_scores, external_ground_truth, threshold=.5, dtc_threshold=.5, gtc_threshold=.5)[0]['macro_average']
        collar_f1_macro_thres05_teacher_real_sed_scores_eval = sed_scores_eval.collar_based.fscore(
            external_scores, external_ground_truth, threshold=.5, onset_collar=.2, offset_collar=.2, offset_collar_rate=.2)[
            0]['macro_average']
        ############################################################################################################
        ############################################################################################################

        # maestro
        maestro_ground_truth = pd.read_csv(self.config.t4_paths['real_maestro_train_tsv'], sep="\t")
        maestro_ground_truth = maestro_ground_truth[maestro_ground_truth.confidence > .5]
        maestro_ground_truth = maestro_ground_truth[
            maestro_ground_truth.event_label.isin(classes_labels_maestro_real_eval)]
        maestro_ground_truth = {
            clip_id: events
            for clip_id, events in sed_scores_eval.io.read_ground_truth_events(maestro_ground_truth).items()
            if clip_id in self.val_buffer_sed_scores_eval_student
        }
        maestro_ground_truth = _merge_overlapping_events(maestro_ground_truth)
        maestro_audio_durations = {
            clip_id: sorted(events, key=lambda x: x[1])[-1][1]
            for clip_id, events in maestro_ground_truth.items()
        }
        event_classes_maestro_eval = sorted(classes_labels_maestro_real_eval)
        keys = ['onset', 'offset'] + event_classes_maestro_eval
        maestro_scores_student = {
            clip_id: self.val_buffer_sed_scores_eval_student[clip_id][keys]
            for clip_id in maestro_ground_truth.keys()
        }
        segment_f1_macro_optthres_student = sed_scores_eval.segment_based.best_fscore(
            maestro_scores_student, maestro_ground_truth, maestro_audio_durations,
            segment_length=1.,
        )[0]['macro_average']
        segment_mauc_student = sed_scores_eval.segment_based.auroc(
            maestro_scores_student, maestro_ground_truth, maestro_audio_durations,
            segment_length=1.,
        )[0]['mean']
        segment_mpauc_student = sed_scores_eval.segment_based.auroc(
            maestro_scores_student, maestro_ground_truth, maestro_audio_durations,
            segment_length=1., max_fpr=.1,
        )[0]['mean']
        maestro_scores_teacher = {
            clip_id: self.val_buffer_sed_scores_eval_teacher[clip_id][keys]
            for clip_id in maestro_ground_truth.keys()
        }
        segment_f1_macro_optthres_teacher = sed_scores_eval.segment_based.best_fscore(
            maestro_scores_teacher, maestro_ground_truth, maestro_audio_durations,
            segment_length=1.,
        )[0]['macro_average']
        segment_mauc_teacher = sed_scores_eval.segment_based.auroc(
            maestro_scores_teacher, maestro_ground_truth, maestro_audio_durations,
            segment_length=1.,
        )[0]['mean']
        segment_mpauc_teacher = sed_scores_eval.segment_based.auroc(
            maestro_scores_teacher, maestro_ground_truth, maestro_audio_durations,
            segment_length=1., max_fpr=.1,
        )[0]['mean']

        obj_metric = torch.tensor(segment_mpauc_student + psds1_sed_scores_eval_student +
                                  psds1_real_sed_scores_eval_student)

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/student/weak_f1_macro_thres05", weak_student_f1_macro)
        self.log("val/teacher/weak_f1_macro_thres05", weak_teacher_f1_macro)
        self.log("val/student/intersection_f1_macro_thres05",
                 intersection_f1_macro_thres05_student_sed_scores_eval)
        self.log("val/teacher/intersection_f1_macro_thres05",
                 intersection_f1_macro_thres05_teacher_sed_scores_eval)
        self.log("val/student/collar_f1_macro_thres05", collar_f1_macro_thres05_student_sed_scores_eval)
        self.log("val/teacher/collar_f1_macro_thres05", collar_f1_macro_thres05_teacher_sed_scores_eval)
        self.log("val/student/psds1", psds1_sed_scores_eval_student)
        self.log("val/teacher/psds1", psds1_sed_scores_eval_teacher)
        self.log("val/student/segment_f1_macro_thresopt", segment_f1_macro_optthres_student)
        self.log("val/student/segment_mauc", segment_mauc_student)
        self.log("val/student/segment_mpauc", segment_mpauc_student)
        self.log("val/teacher/segment_f1_macro_thresopt", segment_f1_macro_optthres_teacher)
        self.log("val/teacher/segment_mauc", segment_mauc_teacher)
        self.log("val/teacher/segment_mpauc", segment_mpauc_teacher)

        # metrics computed on external strong set
        self.log("val/student/psds1_real", psds1_real_sed_scores_eval_student)
        self.log("val/teacher/psds1_real", psds1_real_sed_scores_eval_teacher)
        self.log("val/student/intersection_f1_macro_thres05_real",
                 intersection_f1_macro_thres05_student_real_sed_scores_eval)
        self.log("val/teacher/intersection_f1_macro_thres05_real",
                 intersection_f1_macro_thres05_teacher_real_sed_scores_eval)
        self.log("val/student/collar_f1_macro_thres05_real", collar_f1_macro_thres05_student_real_sed_scores_eval)
        self.log("val/teacher/collar_f1_macro_thres05_real", collar_f1_macro_thres05_teacher_real_sed_scores_eval)

        # free the buffers
        self.val_buffer_sed_scores_eval_student = {}
        self.val_buffer_sed_scores_eval_teacher = {}

        self.val_buffer_real_sed_scores_eval_student = {}
        self.val_buffer_real_sed_scores_eval_teacher = {}

        self.get_weak_student_f1_seg_macro.reset()
        self.get_weak_teacher_f1_seg_macro.reset()

        return obj_metric

    def on_save_checkpoint(self, checkpoint):
        checkpoint["student"] = self.student.state_dict()
        checkpoint["teacher"] = self.teacher.state_dict()
        return checkpoint

    def test_step(self, batch, batch_idx):
        hooks = []
        if self.trainer.global_rank == 0:
            hooks = register_print_hooks(
                self, register_at_step=self.config.debug_shapes
            )

        audio, labels, padded_indxs, filenames, valid_class_mask = batch
        valid_class_mask = valid_class_mask.bool()

        # create mel spectrograms
        with autocast(enabled=False, device_type='cuda'):
            audio = audio.float()
            sed_feats = self.crnn_mel(audio).unsqueeze(1)
            pt_feats = self.transformer_mel(audio)

        bs = len(labels)

        strong_preds_student, weak_preds_student = self.detect(
            sed_feats,
            pt_feats,
            self.student,
            classes_mask=valid_class_mask
        )

        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(
            sed_feats,
            pt_feats,
            self.teacher,
            classes_mask=valid_class_mask
        )

        with autocast(enabled=False, device_type='cuda'):
            strong_preds_student = strong_preds_student.float()
            strong_preds_teacher = strong_preds_teacher.float()
            labels = labels.float()

            loss_strong_student = self.strong_loss(strong_preds_student, labels)
            loss_strong_teacher = self.strong_loss(strong_preds_teacher, labels)

        self.log("test/student/loss_strong", loss_strong_student, batch_size=bs)
        self.log("test/teacher/loss_strong", loss_strong_teacher, batch_size=bs)

        # compute psds
        (
            scores_unprocessed_student_strong,
            scores_postprocessed_student_strong,
            decoded_student_strong,
        ) = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.median_filter,
            thresholds=list(self.test_buffer_psds_eval_student.keys()) + [0.5],
        )

        self.test_buffer_sed_scores_eval_unprocessed_student.update(scores_unprocessed_student_strong)
        self.test_buffer_sed_scores_eval_student.update(
            scores_postprocessed_student_strong
        )

        for th in self.test_buffer_psds_eval_student.keys():
            self.test_buffer_psds_eval_student[th] = pd.concat(
                [self.test_buffer_psds_eval_student[th], decoded_student_strong[th]],
                ignore_index=True,
            )

        (
            scores_unprocessed_teacher_strong,
            scores_postprocessed_teacher_strong,
            decoded_teacher_strong,
        ) = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.median_filter,
            thresholds=list(self.test_buffer_psds_eval_teacher.keys()) + [0.5],
        )

        self.test_buffer_sed_scores_eval_unprocessed_teacher.update(scores_unprocessed_teacher_strong)
        self.test_buffer_sed_scores_eval_teacher.update(
            scores_postprocessed_teacher_strong
        )

        for th in self.test_buffer_psds_eval_teacher.keys():
            self.test_buffer_psds_eval_teacher[th] = pd.concat(
                [self.test_buffer_psds_eval_teacher[th], decoded_teacher_strong[th]],
                ignore_index=True,
            )

            # compute f1 score
        self.test_buffer_detections_thres05_student = pd.concat(
            [self.test_buffer_detections_thres05_student, decoded_student_strong[0.5]]
        )
        self.test_buffer_detections_thres05_teacher = pd.concat(
            [self.test_buffer_detections_thres05_teacher, decoded_teacher_strong[0.5]]
        )

        for h in hooks:
            h.remove()

    def on_test_epoch_end(self):
        empty_keys = []
        for k in self.test_buffer_psds_eval_student:
            if len(self.test_buffer_psds_eval_student[k]) == 0:
                empty_keys.append(k)

        for k in empty_keys:
            print(f"Threshold {k} leads to empty 'test_buffer_psds_eval_student'")
            del self.test_buffer_psds_eval_student[k]

        empty_keys = []
        for k in self.test_buffer_psds_eval_teacher:
            if len(self.test_buffer_psds_eval_teacher[k]) == 0:
                empty_keys.append(k)

        for k in empty_keys:
            print(f"Threshold {k} leads to empty 'test_buffer_psds_eval_teacher'")
            del self.test_buffer_psds_eval_teacher[k]

        # calculate the metrics
        # psds_eval
        psds1_student_psds_eval = compute_psds_from_operating_points(
            self.test_buffer_psds_eval_student,
            self.config.t4_paths["test_tsv"],
            self.config.t4_paths["test_dur"],
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=1
        )
        psds2_student_psds_eval = compute_psds_from_operating_points(
            self.test_buffer_psds_eval_student,
            self.config.t4_paths["test_tsv"],
            self.config.t4_paths["test_dur"],
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1
        )
        psds1_teacher_psds_eval = compute_psds_from_operating_points(
            self.test_buffer_psds_eval_teacher,
            self.config.t4_paths["test_tsv"],
            self.config.t4_paths["test_dur"],
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=1
        )
        psds2_teacher_psds_eval = compute_psds_from_operating_points(
            self.test_buffer_psds_eval_teacher,
            self.config.t4_paths["test_tsv"],
            self.config.t4_paths["test_dur"],
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1
        )

        # synth dataset
        intersection_f1_macro_thres05_student_psds_eval = compute_per_intersection_macro_f1(
            {"0.5": self.test_buffer_detections_thres05_student},
            self.config.t4_paths["test_tsv"],
            self.config.t4_paths["test_dur"]
        )
        intersection_f1_macro_thres05_teacher_psds_eval = compute_per_intersection_macro_f1(
            {"0.5": self.test_buffer_detections_thres05_teacher},
            self.config.t4_paths["test_tsv"],
            self.config.t4_paths["test_dur"]
        )
        # sed_eval
        collar_f1_macro_thres05_student = log_sedeval_metrics(
            self.test_buffer_detections_thres05_student,
            self.config.t4_paths["test_tsv"],
        )[0]
        collar_f1_macro_thres05_teacher = log_sedeval_metrics(
            self.test_buffer_detections_thres05_teacher,
            self.config.t4_paths["test_tsv"],
        )[0]

        # sed_scores_eval
        desed_ground_truth = sed_scores_eval.io.read_ground_truth_events(
            self.config.t4_paths["test_tsv"]
        )
        desed_audio_durations = sed_scores_eval.io.read_audio_durations(
            self.config.t4_paths["test_dur"]
        )

        # drop audios without events
        desed_ground_truth = {
            audio_id: gt for audio_id, gt in desed_ground_truth.items() if len(gt) > 0
        }
        desed_audio_durations = {
            audio_id: desed_audio_durations[audio_id]
            for audio_id in desed_ground_truth.keys()
        }
        keys = ['onset', 'offset'] + sorted(classes_labels_desed.keys())
        desed_scores = {
            clip_id: self.test_buffer_sed_scores_eval_student[clip_id][keys]
            for clip_id in desed_ground_truth.keys()
        }

        psds1_student_sed_scores_eval = compute_psds_from_scores(
            desed_scores,
            desed_ground_truth,
            desed_audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )
        psds2_student_sed_scores_eval = compute_psds_from_scores(
            desed_scores,
            desed_ground_truth,
            desed_audio_durations,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
        )

        intersection_f1_macro_thres05_student_sed_scores_eval = sed_scores_eval.intersection_based.fscore(
            desed_scores, desed_ground_truth, threshold=.5, dtc_threshold=.5, gtc_threshold=.5)[0]['macro_average']
        collar_f1_macro_thres05_student_sed_scores_eval = sed_scores_eval.collar_based.fscore(
            desed_scores, desed_ground_truth, threshold=.5, onset_collar=.2, offset_collar=.2,
            offset_collar_rate=.2)[
            0]['macro_average']

        desed_scores = {
            clip_id: self.test_buffer_sed_scores_eval_teacher[clip_id][keys]
            for clip_id in desed_ground_truth.keys()
        }
        psds1_teacher_sed_scores_eval = compute_psds_from_scores(
            desed_scores,
            desed_ground_truth,
            desed_audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )
        psds2_teacher_sed_scores_eval = compute_psds_from_scores(
            desed_scores,
            desed_ground_truth,
            desed_audio_durations,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
        )

        intersection_f1_macro_thres05_teacher_sed_scores_eval = sed_scores_eval.intersection_based.fscore(
            desed_scores, desed_ground_truth, threshold=.5, dtc_threshold=.5, gtc_threshold=.5)[0]['macro_average']
        collar_f1_macro_thres05_teacher_sed_scores_eval = sed_scores_eval.collar_based.fscore(
            desed_scores, desed_ground_truth, threshold=.5, onset_collar=.2, offset_collar=.2,
            offset_collar_rate=.2)[
            0]['macro_average']

        maestro_audio_durations = sed_scores_eval.io.read_audio_durations(
            self.config.t4_paths["real_maestro_val_dur"])
        maestro_ground_truth_clips = pd.read_csv(
            self.config.t4_paths["real_maestro_val_tsv"], sep="\t")
        maestro_ground_truth_clips = maestro_ground_truth_clips[maestro_ground_truth_clips.confidence > .5]
        maestro_ground_truth_clips = maestro_ground_truth_clips[
            maestro_ground_truth_clips.event_label.isin(classes_labels_maestro_real_eval)]
        maestro_ground_truth_clips = sed_scores_eval.io.read_ground_truth_events(maestro_ground_truth_clips)

        maestro_ground_truth = _merge_maestro_ground_truth(maestro_ground_truth_clips)
        maestro_audio_durations = {file_id: maestro_audio_durations[file_id] for file_id in
                                   maestro_ground_truth.keys()}

        maestro_scores_student = {
            clip_id: self.test_buffer_sed_scores_eval_student[clip_id]
            for clip_id in maestro_ground_truth_clips.keys()
        }
        maestro_scores_teacher = {
            clip_id: self.test_buffer_sed_scores_eval_teacher[clip_id]
            for clip_id in maestro_ground_truth_clips.keys()
        }
        segment_length = 1.
        event_classes_maestro = sorted(classes_labels_maestro_real)
        segment_scores_student = _get_segment_scores_and_overlap_add(
            frame_scores=maestro_scores_student,
            audio_durations=maestro_audio_durations,
            event_classes=event_classes_maestro,
            segment_length=segment_length,
        )

        segment_scores_teacher = _get_segment_scores_and_overlap_add(
            frame_scores=maestro_scores_teacher,
            audio_durations=maestro_audio_durations,
            event_classes=event_classes_maestro,
            segment_length=segment_length,
        )

        event_classes_maestro_eval = sorted(classes_labels_maestro_real_eval)
        keys = ['onset', 'offset'] + event_classes_maestro_eval
        segment_scores_student = {
            clip_id: scores_df[keys]
            for clip_id, scores_df in segment_scores_student.items()
        }
        segment_scores_teacher = {
            clip_id: scores_df[keys]
            for clip_id, scores_df in segment_scores_teacher.items()
        }

        segment_f1_macro_optthres_student = sed_scores_eval.segment_based.best_fscore(
            segment_scores_student, maestro_ground_truth, maestro_audio_durations,
            segment_length=segment_length,
        )[0]['macro_average']
        segment_mauc_student = sed_scores_eval.segment_based.auroc(
            segment_scores_student, maestro_ground_truth, maestro_audio_durations,
            segment_length=segment_length,
        )[0]['mean']
        segment_mpauc_student = sed_scores_eval.segment_based.auroc(
            segment_scores_student, maestro_ground_truth, maestro_audio_durations,
            segment_length=segment_length, max_fpr=.1,
        )[0]['mean']
        segment_f1_macro_optthres_teacher = sed_scores_eval.segment_based.best_fscore(
            segment_scores_teacher, maestro_ground_truth, maestro_audio_durations,
            segment_length=segment_length,
        )[0]['macro_average']
        segment_mauc_teacher = sed_scores_eval.segment_based.auroc(
            segment_scores_teacher, maestro_ground_truth, maestro_audio_durations,
            segment_length=segment_length,
        )[0]['mean']
        segment_mpauc_teacher = sed_scores_eval.segment_based.auroc(
            segment_scores_teacher, maestro_ground_truth, maestro_audio_durations,
            segment_length=segment_length, max_fpr=.1,
        )[0]['mean']

        results = {
            "test/student/psds1_psds_eval": psds1_student_psds_eval,
            "test/student/psds2_psds_eval": psds2_student_psds_eval,
            "test/teacher/psds1_psds_eval": psds1_teacher_psds_eval,
            "test/teacher/psds2_psds_eval": psds2_teacher_psds_eval,
            "test/student/intersection_f1_macro_thres05_psds_eval": intersection_f1_macro_thres05_student_psds_eval,
            "test/teacher/intersection_f1_macro_thres05_psds_eval": intersection_f1_macro_thres05_teacher_psds_eval,
            "test/student/collar_f1_macro_thres05_sed_eval": collar_f1_macro_thres05_student,
            "test/teacher/collar_f1_macro_thres05_sed_eval": collar_f1_macro_thres05_teacher,
            "test/student/psds1_sed_scores_eval": psds1_student_sed_scores_eval,
            "test/student/psds2_sed_scores_eval": psds2_student_sed_scores_eval,
            "test/teacher/psds1_sed_scores_eval": psds1_teacher_sed_scores_eval,
            "test/teacher/psds2_sed_scores_eval": psds2_teacher_sed_scores_eval,
            "test/student/intersection_f1_macro_thres05_sed_scores_eval": intersection_f1_macro_thres05_student_sed_scores_eval,
            "test/teacher/intersection_f1_macro_thres05_sed_scores_eval": intersection_f1_macro_thres05_teacher_sed_scores_eval,
            "test/student/collar_f1_macro_thres05_sed_scores_eval": collar_f1_macro_thres05_student_sed_scores_eval,
            "test/teacher/collar_f1_macro_thres05_sed_scores_eval": collar_f1_macro_thres05_teacher_sed_scores_eval,
            "test/student/segment_f1_macro_thresopt_sed_scores_eval": segment_f1_macro_optthres_student,
            "test/student/segment_mauc_sed_scores_eval": segment_mauc_student,
            "test/student/segment_mpauc_sed_scores_eval": segment_mpauc_student,
            "test/teacher/segment_f1_macro_thresopt_sed_scores_eval": segment_f1_macro_optthres_teacher,
            "test/teacher/segment_mauc_sed_scores_eval": segment_mauc_teacher,
            "test/teacher/segment_mpauc_sed_scores_eval": segment_mpauc_teacher,
            "test/student/rank_score": psds1_student_sed_scores_eval + segment_mpauc_student,
            "test/teacher/rank_score": psds1_teacher_sed_scores_eval + segment_mpauc_teacher
        }

        self.tracker_devtest.stop()
        test_kwh = self.tracker_devtest._total_energy.kWh
        self.log("test/tot_energy_kWh", torch.tensor(float(test_kwh)))

        for key in results.keys():
            self.log(key, results[key], prog_bar=True)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = get_optimizer(self.student, self.config["arch"])

        num_training_steps = self.trainer.estimated_stepping_batches

        print(
            f"INFO: expected num_training_steps={num_training_steps} in {self.config.trainer.max_epochs} epochs "
            f"killnum_nodes={self.config.num_nodes}"
        )
        scheduler = get_lr_scheduler(optimizer, num_training_steps)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]


def _merge_maestro_ground_truth(clip_ground_truth):
    ground_truth = defaultdict(list)
    for clip_id in clip_ground_truth:
        file_id, clip_onset_time, clip_offset_time = clip_id.rsplit('-', maxsplit=2)
        clip_onset_time = int(clip_onset_time) // 100
        ground_truth[file_id].extend([
            (clip_onset_time + event_onset_time, clip_onset_time + event_offset_time, event_class)
            for event_onset_time, event_offset_time, event_class in clip_ground_truth[clip_id]
        ])
    return _merge_overlapping_events(ground_truth)


def _merge_overlapping_events(ground_truth_events):
    for clip_id, events in ground_truth_events.items():
        per_class_events = defaultdict(list)
        for event in events:
            per_class_events[event[2]].append(event)
        ground_truth_events[clip_id] = []
        for event_class, events in per_class_events.items():
            events = sorted(events)
            merged_events = []
            current_offset = -1e6
            for event in events:
                if event[0] > current_offset:
                    merged_events.append(list(event))
                else:
                    merged_events[-1][1] = max(current_offset, event[1])
                current_offset = merged_events[-1][1]
            ground_truth_events[clip_id].extend(merged_events)
    return ground_truth_events


def _get_segment_scores_and_overlap_add(frame_scores, audio_durations, event_classes, segment_length=1.):
    """
    #>>> event_classes = ['a', 'b', 'c']
    #>>> audio_durations = {'f1': 201.6, 'f2':133.1, 'f3':326}
    #>>> frame_scores = {\
    #    f'{file_id}-{int(100*onset)}-{int(100*(onset+10.))}': create_score_dataframe(np.random.rand(156,3), np.arange(157.)*0.064, event_classes)\
    #    for file_id in audio_durations for onset in range(int((audio_durations[file_id]-9.)))\
    #}
    #>>> frame_scores.keys()
    #>>> seg_scores = _get_segment_scores_and_overlap_add(frame_scores, audio_durations, event_classes, segment_length=1.)
    #>>> [(key, validate_score_dataframe(value)[0][-3:]) for key, value in seg_scores.items()]
    """
    segment_scores_file = {}
    summand_count = {}
    keys = ['onset', 'offset'] + event_classes
    for clip_id in frame_scores:
        file_id, clip_onset_time, clip_offset_time = clip_id.rsplit('-', maxsplit=2)
        clip_onset_time = float(clip_onset_time) / 100
        clip_offset_time = float(clip_offset_time) / 100
        if file_id not in segment_scores_file:
            segment_scores_file[file_id] = np.zeros(
                (math.ceil(audio_durations[file_id] / segment_length), len(event_classes)))
            summand_count[file_id] = np.zeros_like(segment_scores_file[file_id])
        segment_scores_clip = _get_segment_scores(
            frame_scores[clip_id][keys], clip_length=(clip_offset_time - clip_onset_time), segment_length=1.)[
            event_classes].to_numpy()
        seg_idx = int(clip_onset_time // segment_length)
        segment_scores_file[file_id][seg_idx:seg_idx + len(segment_scores_clip)] += segment_scores_clip
        summand_count[file_id][seg_idx:seg_idx + len(segment_scores_clip)] += 1
    return {
        file_id: create_score_dataframe(
            segment_scores_file[file_id] / np.maximum(summand_count[file_id], 1),
            np.minimum(np.arange(0., audio_durations[file_id] + segment_length, segment_length),
                       audio_durations[file_id]),
            event_classes,
        ) for file_id in segment_scores_file
    }


def _get_segment_scores(scores_df, clip_length, segment_length=1.):
    """
    # >>> scores_arr = np.random.rand(156,3)
    # >>> timestamps = np.arange(157)*0.064
    # >>> event_classes = ['a', 'b', 'c']
    # >>> scores_df = create_score_dataframe(scores_arr, timestamps, event_classes)
    # >>> seg_scores_df = _get_segment_scores(scores_df, clip_length=10., segment_length=1.)
    """
    frame_timestamps, event_classes = validate_score_dataframe(scores_df)
    scores_arr = scores_df[event_classes].to_numpy()
    segment_scores = []
    segment_timestamps = []
    seg_onset_idx = 0
    seg_offset_idx = 0
    for seg_onset in np.arange(0., clip_length, segment_length):
        seg_offset = seg_onset + segment_length
        while frame_timestamps[seg_onset_idx + 1] <= seg_onset:
            seg_onset_idx += 1
        while seg_offset_idx < len(scores_arr) and frame_timestamps[seg_offset_idx] < seg_offset:
            seg_offset_idx += 1
        seg_weights = (
                np.minimum(frame_timestamps[seg_onset_idx + 1:seg_offset_idx + 1], seg_offset)
                - np.maximum(frame_timestamps[seg_onset_idx:seg_offset_idx], seg_onset)
        )
        segment_scores.append(
            (seg_weights[:, None] * scores_arr[seg_onset_idx:seg_offset_idx]).sum(0) / seg_weights.sum()
        )
        segment_timestamps.append(seg_onset)
    segment_timestamps.append(clip_length)
    return create_score_dataframe(np.array(segment_scores), np.array(segment_timestamps), event_classes)


@ex.command
def main(
        _run,
        _config,
        _log,
        _rnd,
        _seed,
        rank=0,
):
    logger = get_wandb_logger(_config)
    config = DefaultMunch.fromDict(_config)

    print("main() is running pid", os.getpid(), "in module main: ", __name__)

    module = T4Module(config)

    train_ds, datasets = get_training_dataset(module.encoder, config.t4_paths)
    train_sampler = get_sampler(datasets)
    train_loader = get_train_loader(dataset=train_ds, batch_sampler=train_sampler)

    val_ds = get_validation_dataset(module.encoder, config.t4_paths)
    validate_loader = get_validate_loader(dataset=val_ds)

    test_ds = get_test_dataset(module.encoder, config.t4_paths)
    test_loader = get_test_loader(dataset=test_ds)

    callbacks = [
        ModelCheckpoint(
            logger.log_dir,
            monitor="val/obj_metric",
            save_top_k=1,
            mode="max",
            save_last=True,
        ),
    ]

    trainer = Trainer(logger=logger, callbacks=callbacks)
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validate_loader,
    )

    best_path = trainer.checkpoint_callback.best_model_path
    print(f"best model: {best_path}")
    test_state_dict = torch.load(best_path)["state_dict"]
    module.load_state_dict(test_state_dict)
    trainer.test(module, dataloaders=test_loader)

    if rank == 0:
        wandb.finish()

    return 0  # great success



# this is the program's entry point
@ex.automain
def default_command():
    return main()
