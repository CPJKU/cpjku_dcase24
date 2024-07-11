import functools
import os
import sys
from munch import DefaultMunch
import transformers
import wandb
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from configs import add_configs
from helpers.hub_mixin import PyTorchModelHubMixin
from helpers.utils import config_call, register_print_hooks
from helpers.workersinit import worker_init_fn
from sacred import Experiment
from pathlib import Path
from sacred.config_helpers import CMD
import datasets
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torch import autocast

# T4 related imports
from models.bl.CRNN import CRNN
import models.jbt.jazz_beat_transformer as beat_tracker
from models import fpasst as passt
from models.frame_dymn.model import get_model as get_frame_dymn
from models.wrapper import Task4Wrapper, Task4AttentionWrapper, Task4RNNWrapper, head_function
from models import preprocess
from helpers.scaler import TorchScaler
from helpers.encoder import ManyHotEncoder
from data_util.t4_datasets import classes_labels
from data_util import t4_datasets
from copy import deepcopy
import torchmetrics
from helpers.t4_data_augm import mixup, frame_shift, gain_augment, time_mask, feature_transformation, mixstyle
from helpers.t4_losses import AsymmetricalFocalLoss, ShiftTolerantBCELoss
import pandas as pd
from helpers.t4_metrics import (
    batched_decode_preds,
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    compute_psds_from_scores,
    log_sedeval_metrics
)
import sed_scores_eval
from lightning.pytorch.callbacks import ModelCheckpoint
from data_util import audioset_strong

os.environ["HF_DATASETS_CACHE"] = "/share/hel/datasets/HF_datasets/cache/"

if 'LD_LIBRARY_PATH' in os.environ:
    del os.environ['LD_LIBRARY_PATH']

import socket

hostename = socket.gethostname()

if hostename in ['chili', 'mint']:
    DEBUG = True
else:
    DEBUG = False

ex = Experiment("dcase23_t4_bl", save_git_info=False)

# verbose logging
datasets.logging.set_verbosity_info()

# define datasets config
get_training_dataset = ex.command(
    audioset_strong.get_training_dataset if not DEBUG else audioset_strong.get_validation_dataset,
    prefix="training",
    audio_length=10.0,
    sample_rate=32000,
)

# define datasets config
get_validation_dataset = ex.command(
    audioset_strong.get_validation_dataset,
    prefix="validation",
    audio_length=10.0,
    sample_rate=32000,
)

# define datasets config
get_test_dataset = ex.command(
    audioset_strong.get_validation_dataset,
    prefix="test",
    audio_length=10.0,
    sample_rate=32000
)

# Define loaders
get_train_loader = ex.command(
    DataLoader,
    prefix="training",
    batch_size=20,
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

Trainer = ex.command(L.Trainer, prefix="trainer")

many_hot_encoder = ex.command(ManyHotEncoder, prefix="encoder")

# Define models

crnn_mel = ex.command(MelSpectrogram, prefix="crnn_mel")
crnn_net = ex.command(CRNN, prefix="crnn", crnn_pipeline=False)

## fPaSST
passt_mel = ex.command(preprocess.AugmentMelSTFT, prefix="passt_mel")
passt_net = ex.command(passt.get_model, prefix="passt")

## jazz-beat-transformer
jbt_net = ex.command(beat_tracker.BeatTransformer, prefix="jbt")

## frame dynamic net
frame_dymn_net = ex.command(get_frame_dymn, prefix="frame_dymn")

# wrapper
Task4Wrapper = ex.command(Task4Wrapper, prefix="t4_wrapper")
Task4AttentionWrapper = ex.command(Task4AttentionWrapper, prefix="t4_wrapper")
Task4RNNWrapper = ex.command(Task4RNNWrapper, prefix="t4_wrapper")


@ex.config
def default_conf():
    # ruff: noqa: F841
    cmd = " ".join(sys.argv)  # command line arguments

    passt_arch = "passt_arch" in cmd
    jbt_arch = "jbt_arch" in cmd
    crnn_arch = "crnn_arch" in cmd
    frame_dymn_arch = "frame_dymn" in cmd

    arch_sum = passt_arch + jbt_arch + crnn_arch + frame_dymn_arch

    if arch_sum != 1:
        raise ValueError("Please specify exactly one architecture to train.")

    if jbt_arch:
        # set for flash attention
        torch.backends.cuda.enable_flash_sdp(True)  # True
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # False
        torch.backends.cuda.enable_math_sdp(True)  # False  # TODO: check why this must be True

        # for A40
        torch.set_float32_matmul_precision('medium')
        print("Successfully set JBT_SEED and flash attention settings.")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if os.environ.get("SLURM_ARRAY_JOB_ID", False):
        slurm_job_id = (
                os.environ.get("SLURM_ARRAY_JOB_ID", "").strip()
                + "_"
                + os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
        )
    process_id = os.getpid()
    debug_shapes = 2  # print shapes of in step, 0 = never, 1 = first step etc...
    encoder = dict(
        labels=list(classes_labels.keys())
    )

    audio_len = 10
    watch_model = False
    trainer = dict(
        max_epochs=100,
        devices=1,
        weights_summary="full",
        benchmark=True,
        num_sanity_val_steps=0,
        precision="16-mixed",
        reload_dataloaders_every_epoch=True,
        default_root_dir="./outputs",
        accumulate_grad_batches=1,
        check_val_every_n_epoch=10
    )

    compile = False  # compile the model, requires pytorch >= 2.0
    optimizer = dict(
        lr=0.0001,
        adamw=False,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        schedule_mode="cos"
    )

    self_supervised_loss_weight = 2
    strong_loss_weight = 0.5
    ema_factor = 0.999
    val_thresholds = [0.5]
    obj_metric_synth_type = "intersection"

    base_path = "/share/hel/datasets/dcase/task4/dataset_complete/"

    median_window = 7

    seq_len = 250

    # in case of 'head=logits'
    head_fn_mode = "mean"

    test_n_thresholds = 50

    use_scaler = False

    repr_dropout_p = 0

    t4_wrapper = dict(
        name="Task4AttentionWrapper",
        seq_len=250
    )

    mix_augment = dict(
        apply_mixup=True,
        apply_mixstyle=False,
        mixup_p=0.5,
        mixstyle_p=0.2,
        mixstyle_alpha=0.3,
        mix_unlabeled=False,
        mixstyle_target="joint"  # options: "joint", "separate", "strong"
    )

    time_augment = dict(
        apply_mask=True,
        apply_shift=True,
        only_strong=True,
        shift_range=0.125,
        min_mask_ratio=0.05,
        max_mask_ratio=0.3,
    )

    loss = dict(
        type="BCELoss",
        afl_gamma=0,
        afl_zeta=1,
        tolerance=5,
        pos_weight=0.5,
        balance_active=False,
        balance_classes=False,
        balance_all=False
    )
    filter_augment = dict(
        apply=0,
        n_transform=2,
        filter_db_range=(-6, 6),
        filter_bands=(3, 6),
        filter_minimum_bandwidth=6
    )


add_configs(ex)  # add common configurations


# capture the WandbLogger and prefix it with "wandb", this allows to use sacred to update WandbLogger config from the command line
@ex.command(prefix="wandb")
def get_wandb_logger(config, name=None, project="dcase23_task4", rank0_only=True, offline=False, tags=[]):
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


@ex.command(prefix="optimizer")
def get_lr_scheduler(
        optimizer,
        num_training_steps,
        schedule_mode="exp",
        gamma: float = 0.999996,
        num_warmup_steps=4000,
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


@ex.command(prefix="optimizer")
def get_optimizer(
        model, lr=0.0001, lr_pt_scaler=1., adamw=True, weight_decay=0.01, betas=[0.9, 0.999]
):
    lr_pt = lr * lr_pt_scaler
    params = []
    for name, param in model.named_parameters():
        if name.startswith('model'):
            # the audioset pre-trained part
            params.append({'params': param, 'lr': lr_pt})
        else:
            params.append({'params': param, 'lr': lr})

    if adamw:
        print(f"\nUsing adamw weight_decay={weight_decay}!\n")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    return torch.optim.Adam(params, lr=lr, betas=betas)


@ex.command(prefix="scaler")
def get_scaler(get_train_loader, mel, statistic="instance", normtype="minmax", dims=(1, 2), savepath="./scaler.ckpt"):
    """ Scaler inizialization

    Raises:
        NotImplementedError: in case of not Implemented scaler

    Returns:
        TorchScaler: returns the scaler
    """

    assert statistic == "instance", f"Only statistic=instance implemented so far."
    if statistic == "instance":
        scaler = TorchScaler(
            "instance",
            normtype,
            dims,
        )

        return scaler
    elif statistic == "dataset":
        # we fit the scaler
        scaler = TorchScaler(
            "dataset",
            normtype,
            dims,
        )
    else:
        raise NotImplementedError

    if savepath is not None:
        if os.path.exists(savepath):
            scaler = torch.load(savepath)
            print(
                "Loaded Scaler from previous checkpoint from {}".format(
                    savepath
                )
            )
            return scaler

    train_loader = get_train_loader()
    scaler.fit(
        train_loader,
        transform_func=lambda x: take_log(mel(x[0])),
    )

    if savepath is not None:
        torch.save(scaler, savepath)
        print(
            "Saving Scaler from previous checkpoint at {}".format(
                savepath
            )
        )
        return scaler


def take_log(mels):
    """ Apply the log transformation to mel spectrograms.
    Args:
        mels: torch.Tensor, mel spectrograms for which to apply log.

    Returns:
        Tensor: logarithmic mel spectrogram of the mel spectrogram given as input
    """

    amp_to_db = AmplitudeToDB(stype="amplitude")
    amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
    return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code


class BL23Module(L.LightningModule, PyTorchModelHubMixin):
    def __init__(
            self,
            config
    ):
        super(BL23Module, self).__init__()
        config = DefaultMunch.fromDict(config)
        self.config = config
        self.construct_modules()
        for param in self.teacher.parameters():
            param.detach_()

        if self.config.arch in ["crnn"]:
            self.mel_postprocess_fn = take_log
        elif self.config.arch in ["passt", "frame_dymn"]:
            self.mel_postprocess_fn = lambda x: x
        elif self.config.arch in ["jbt"]:
            self.mel_postprocess_fn = nn.Identity()
        else:
            raise ValueError(f"No such mel type: {self.config.mel_type}")

        if self.config.use_scaler:
            self.scaler = get_scaler(self.train_dataloader, self.mel)
        else:
            self.scaler = nn.Identity()

        self.sample_rate = self.config.sample_rate

        self.distributed_mode = self.config.trainer.num_nodes > 1

        assert 1 >= self.config.strong_loss_weight >= 0

        if self.config.compile:
            # pt 2 magic
            print("\n\nCompiling the model pytorch 2... \n\n")
            self.student = torch.compile(self.student)
            self.teacher = torch.compile(self.teacher)

        self.head_fn_mode = config["head_fn_mode"]

        self.mixup_type = self.config.mixup
        self.mixup_p = self.config.mixup_p if "mixup_p" in self.config else 0.5

        self.mix_config = self.config.mix_augment
        self.ta_config = self.config.time_augment
        self.fa_config = self.config.filter_augment

        # representation dropout
        if self.config.repr_dropout_p > 0:
            self.repr_dropout = nn.Dropout2d(p=self.config.repr_dropout_p)
        else:
            self.repr_dropout = nn.Identity()

        self.gain_augment = self.config.gain_augment if "gain_augment" in self.config else 0

        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.encoder.labels),
            average="macro"
        )

        self.get_weak_teacher_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.encoder.labels),
            average="macro"
        )

        self.val_buffer_student = {
            k: pd.DataFrame() for k in self.config.val_thresholds
        }
        self.val_buffer_teacher = {
            k: pd.DataFrame() for k in self.config.val_thresholds
        }

        self.val_buffer_student_test = {
            k: pd.DataFrame() for k in self.config.val_thresholds
        }
        self.val_buffer_teacher_test = {
            k: pd.DataFrame() for k in self.config.val_thresholds
        }
        self.val_scores_student_postprocessed_buffer = {}
        self.val_scores_teacher_postprocessed_buffer = {}

        test_n_thresholds = self.config.test_n_thresholds
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )

        self.test_psds_buffer_student = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_teacher = {k: pd.DataFrame() for k in test_thresholds}
        self.decoded_student_05_buffer = pd.DataFrame()
        self.decoded_teacher_05_buffer = pd.DataFrame()
        self.test_scores_raw_buffer_student = {}
        self.test_scores_raw_buffer_teacher = {}
        self.test_scores_postprocessed_buffer_student = {}
        self.test_scores_postprocessed_buffer_teacher = {}

        self.val_ground_truth = {}
        self.val_duration = {}

    def construct_modules(self):
        arch = self.config["arch"]
        self.arch = arch

        scall = functools.partial(
            config_call, config=self.config
        )

        if arch == "passt":
            self.mel = scall(passt_mel)
            net = scall(passt_net)
            embed_dim = net.num_features
        elif arch == "jbt":
            self.mel = None
            net = scall(jbt_net)
            embed_dim = net.hidden_dim
        elif arch == "frame_dymn":
            self.mel = scall(passt_mel)
            net = scall(frame_dymn_net)
            embed_dim = net.lastconv_output_channels
        elif arch == "crnn":
            self.mel = scall(crnn_mel)
            net = scall(crnn_net)
            embed_dim = 2 * net.n_RNN_cell
        else:
            raise ValueError(f"Unknown arch={arch}")

        net.arch = arch
        if self.config.t4_wrapper.name == "Task4AttentionWrapper":
            self.student = Task4AttentionWrapper(net, audioset_classes=527, seq_len=self.config["seq_len"],
                                        embed_dim=embed_dim, n_classes=456, wandb_id=self.config[arch]["wandb_id"])
            self.head_function = lambda x, mode: x
        elif self.config.t4_wrapper.name == "Task4RNNWrapper":
            self.student = Task4RNNWrapper(net, audioset_classes=527, seq_len=self.config["seq_len"],
                                        embed_dim=embed_dim, n_classes=456, wandb_id=self.config[arch]["wandb_id"])
            self.head_function = lambda x, mode: x
        elif self.config.t4_wrapper.name == "Task4Wrapper":
            self.student = Task4Wrapper(net, audioset_classes=527, seq_len=self.config["seq_len"],
                                        embed_dim=embed_dim, n_classes=456, wandb_id=self.config[arch]["wandb_id"])
            self.head_function = head_function
        else:
            raise ValueError(f"Unknown head={self.config.head}")
        self.teacher = deepcopy(self.student)

        self.encoder = scall(many_hot_encoder)

        # losses
        if not self.config.t4_wrapper.name == "Task4Wrapper":
            if self.config.loss.type == "BCELoss":
                self.strong_loss = nn.BCELoss()
                self.weak_loss = nn.BCELoss()
            elif self.config.loss.type == "AsymmetricalFocalLoss":
                self.strong_loss = AsymmetricalFocalLoss(self.config.loss.afl_gamma, self.config.loss.afl_zeta)
                self.weak_loss = AsymmetricalFocalLoss(self.config.loss.afl_gamma, self.config.loss.afl_zeta)
            elif self.config.loss.type == "ShiftTolerantBCELoss":
                self.strong_loss = ShiftTolerantBCELoss(tolerance=self.config.loss.tolerance)
                self.weak_loss = nn.BCELoss()
            else:
                raise ValueError(f"No such loss type: {self.config.loss.type}")
            self.out_probs = True
        else:
            self.strong_loss = nn.BCEWithLogitsLoss()
            self.weak_loss = nn.BCEWithLogitsLoss()
            self.out_probs = False

        self.selfsup_loss = torch.nn.MSELoss()

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

    def detect(self, mel_feats, model):
        x = model(
            self.scaler(
                self.mel_postprocess_fn(mel_feats)
            )
        )
        return self.head_function(x, self.head_fn_mode)

    def training_step(self, batch, batch_idx):
        hooks = []
        if self.trainer.global_rank == 0:
            hooks = register_print_hooks(
                self, register_at_step=self.config.debug_shapes
            )

        audio, labels = batch['audio'], batch['strong']

        # gain augment
        if self.gain_augment > 0:
            audio = gain_augment(audio, gain=self.gain_augment)

        if self.mel:
            features = self.mel(audio)
        else:
            features = audio  # JBT: mel transform is in the model, these are still waveforms

        # frame_shift
        if self.ta_config.apply_shift:
            features, labels = frame_shift(features, labels, self.encoder.net_pooling, self.ta_config.shift_range)

        if self.mix_config.apply_mixup and self.mix_config.mixup_p > random.random():
            features, labels = mixup(
                features, labels, mixup_label_type="soft"
            )

        if self.mix_config.apply_mixstyle and self.mix_config.mixstyle_p > random.random():
            if self.mix_config.mixstyle_target == "separate":
                features = mixstyle(features, self.mix_config.mixstyle_alpha)
            elif self.mix_config.mixstyle_target == "joint":
                features = mixstyle(features, self.mix_config.mixstyle_alpha)
            elif self.mix_config.mixstyle_target == "strong":
                features = mixstyle(features, self.mix_config.mixstyle_alpha)
            elif self.mix_config.mixstyle_target == "labeled":
                features = mixstyle(features, self.mix_config.mixstyle_alpha)
            else:
                ValueError(f"Mixstyle target {self.mix_config.mixstyle_target} not known.")

        # representation dropout - for multi-resolution input (mutliple fft windows)
        features = self.repr_dropout(features)

        if self.ta_config.apply_mask:
            features, labels = time_mask(features, labels, self.encoder.net_pooling,
                              self.ta_config.min_mask_ratio, self.ta_config.max_mask_ratio)

        if self.fa_config.apply and self.fa_config.p > random.random():
            features_stud, features_teach = feature_transformation(features, self.fa_config.n_transform,
                                                                   self.fa_config.filter_db_range,
                                                                   self.fa_config.filter_bands,
                                                                   self.fa_config.filter_minimum_bandwidth)
        else:
            features_stud, features_teach = features, features

        # sed student forward
        strong_logits_student, weak_logits_student = self.detect(
            features_stud, self.student
        )


        with autocast(enabled=False, device_type='cuda'):
            strong_logits_student = strong_logits_student.float()
            labels = labels.float()

            # supervised loss on strong labels
            loss_strong = self.strong_loss(
                strong_logits_student, labels
            )

            # supervised loss on weakly labelled
            loss_weak = 0.0 # self.weak_loss(weak_logits_student[weak_mask], labels_weak)

        # total supervised loss
        tot_loss_supervised = self.config.strong_loss_weight * loss_strong + (1 - self.config.strong_loss_weight) * loss_weak
        tot_loss_supervised = tot_loss_supervised * 2  # to obtain same magnitude of loss as before weighting

        if self.config.self_supervised_loss_weight != 0:

            with torch.no_grad():
                strong_logits_teacher, weak_logits_teacher = self.detect(
                    features_teach, self.teacher
                )

                with autocast(enabled=False, device_type='cuda'):
                    strong_logits_teacher = strong_logits_teacher.float()
                    labels = labels.float()
                    loss_strong_teacher = self.strong_loss(
                        strong_logits_teacher, labels
                    )

                    loss_weak_teacher = 0 # self.weak_loss(weak_logits_teacher, labels_weak)

            if self.out_probs:
                strong_preds_student = strong_logits_student
                strong_preds_teacher = strong_logits_teacher

                weak_preds_student = weak_logits_student
                weak_preds_teacher = weak_logits_teacher
            else:
                strong_preds_student = torch.sigmoid(strong_logits_student).float()
                strong_preds_teacher = torch.sigmoid(strong_logits_teacher).float()

                weak_preds_student = torch.sigmoid(weak_logits_student).float()
                weak_preds_teacher = torch.sigmoid(weak_logits_teacher).float()

            strong_self_sup_loss = self.selfsup_loss(
                strong_preds_student, strong_preds_teacher.detach()
            )
            weak_self_sup_loss = 0.0 # self.selfsup_loss(weak_preds_student, weak_preds_teacher.detach())

            weight = self.config.self_supervised_loss_weight

            tot_self_loss = (strong_self_sup_loss + weak_self_sup_loss) * weight
        else:
            tot_self_loss = 0.0
            weight = self.config.self_supervised_loss_weight
            weak_self_sup_loss = 0.0
            strong_self_sup_loss = 0.0
            loss_strong_teacher = 0.0
            loss_weak_teacher = 0.0

        tot_loss = tot_loss_supervised + tot_self_loss

        self.log("train/student/loss_strong", loss_strong)
        self.log("train/student/loss_weak", loss_weak)
        self.log("train/teacher/loss_strong", loss_strong_teacher)
        self.log("train/teacher/loss_weak", loss_weak_teacher)
        self.log("train/student/tot_self_loss", tot_self_loss, prog_bar=True)
        self.log("train/ssl_weight", weight)
        self.log("train/student/tot_supervised", tot_loss_supervised, prog_bar=True)
        self.log("train/student/weak_self_sup_loss", weak_self_sup_loss)
        self.log("train/student/strong_self_sup_loss", strong_self_sup_loss)
        self.log("train/lr", self.optimizers().param_groups[-1]["lr"], prog_bar=True)
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

        audio, labels, filenames = batch['audio'], batch['strong'], batch['filename']
        bs = len(labels)

        for f, gt_string in zip(filenames, batch["gt_string"]):
            if f in self.val_ground_truth:
                continue
            else:
                events = [e.split(";;") for e in gt_string.split("++")]
                self.val_ground_truth[f.split(".")[0]] = [(float(e[0]), float(e[1]), e[2]) for e in events]
                self.val_duration[f.split(".")[0]] = (audio.shape[1] / batch['sampling_rate'][0]).item()

        # prediction for student
        if self.mel:
            mels = self.mel(audio)
        else:
            mels = audio  # JBT: mel transform is in the model, these are still waveforms

        strong_logits_student, weak_logits_student = self.detect(mels, self.student)
        # strong_logits_teacher, weak_logits_teacher = self.detect(mels, self.teacher)

        if self.out_probs:
            strong_preds_student = strong_logits_student.float()
            # strong_preds_teacher = strong_logits_teacher.float()
        else:
            strong_preds_student = torch.sigmoid(strong_logits_student).float()
            # strong_preds_teacher = torch.sigmoid(strong_logits_teacher).float()


        with autocast(enabled=False, device_type='cuda'):
            labels = labels.float()
            strong_logits_student = strong_logits_student.float()
            # strong_logits_teacher = strong_logits_teacher.float()

            loss_strong_student = self.strong_loss(
                strong_logits_student, labels
            )
            # loss_strong_teacher = self.strong_loss(
            #    strong_logits_teacher, labels
            #)

        self.log("val/student/loss_strong", loss_strong_student, batch_size=bs)
        # self.log("val/teacher/loss_strong", loss_strong_teacher, batch_size=bs)

        (
            scores_raw_student_strong, scores_postprocessed_student_strong,
            decoded_student_strong,
        ) = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.config.median_window,
            thresholds=list(self.val_buffer_student.keys()),
        )

        self.val_scores_student_postprocessed_buffer.update(
            scores_postprocessed_student_strong
        )

        for h in hooks:
            h.remove()

    def on_validation_epoch_end(self):

        # synth dataset
        ground_truth = self.val_ground_truth
        audio_durations = self.val_duration

        # drop the unused classes
        unused_classes = list(set(self.encoder.labels).difference(set([e[2] for f, events in ground_truth.items() for e in events])))
        for f, df in self.val_scores_student_postprocessed_buffer.items():
                df.drop(columns=list(unused_classes), axis=1, inplace=True)


        segment_based_pauroc = sed_scores_eval.segment_based.auroc(
            self.val_scores_student_postprocessed_buffer,
            ground_truth,
            audio_durations,
            max_fpr=0.1,
            segment_length=1.0,
            num_jobs=12,

        )

        segment_based_f1 = sed_scores_eval.segment_based.fscore(
            self.val_scores_student_postprocessed_buffer,
            ground_truth,
            audio_durations,
            0.5,
            segment_length=1.0,
            num_jobs=12
        )

        segment_based_f1_metric_micro = segment_based_f1[0]['micro_average']
        segment_based_f1_metric_macro = segment_based_f1[0]['macro_average']
        segment_based_pauroc_metric_macro = segment_based_pauroc[0]['mean']

        for k in segment_based_f1[0]:
            if k in ['macro_average', 'micro_average']:
                continue
            self.log(f"val/student/segment_f1_macro/{k}", segment_based_f1[0][k])
            self.log(f"val/student/segment_pauroc_macro/{k}", segment_based_pauroc[0][k])

        obj_metric = torch.tensor(segment_based_pauroc_metric_macro)

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/student/segment_f1_macro", segment_based_f1_metric_macro)
        self.log("val/student/segment_f1_micro", segment_based_f1_metric_micro)
        self.log("val/student/segment_pauroc_macro", segment_based_pauroc_metric_macro)

        # free the buffers
        self.val_buffer_student = {
            k: pd.DataFrame() for k in self.config.val_thresholds
        }
        self.val_buffer_teacher = {
            k: pd.DataFrame() for k in self.config.val_thresholds
        }
        self.val_scores_student_postprocessed_buffer = {}

        return obj_metric

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = get_optimizer(self.student)

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


@ex.command
def main(
        _run,
        _config,
        _log,
        _rnd,
        _seed,
        rank=0,
):
    logger = None
    if rank == 0 and _config["wandb"]["rank0_only"] and not DEBUG:
        # final experiment config is resolved by now
        logger = get_wandb_logger(_config)
    config = DefaultMunch.fromDict(_config)

    print("main() is running pid", os.getpid(), "in module main: ", __name__)

    module = BL23Module(config)

    train_ds = get_training_dataset(module.encoder)
    train_loader = get_train_loader(dataset=train_ds)

    val_ds = get_validation_dataset(module.encoder)
    validate_loader = get_validate_loader(dataset=val_ds)

    test_ds = get_test_dataset(module.encoder)
    test_loader = get_test_loader(dataset=test_ds)

    callbacks = [
        ModelCheckpoint(
            logger.log_dir,
            monitor="val/obj_metric",
            save_top_k=1,
            mode="max",
            save_last=True,
        )
    ] if not DEBUG else []

    trainer = Trainer(logger=logger, callbacks=callbacks)
    if DEBUG:
        trainer.validate(module, validate_loader)
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validate_loader,
    )

    # best_path = trainer.checkpoint_callback.best_model_path
    # print(f"best model: {best_path}")
    # test_state_dict = torch.load(best_path)["state_dict"]
    # module.load_state_dict(test_state_dict)
    # trainer.test(module, dataloaders=test_loader)

    if rank == 0:
        wandb.finish()

    return 0  # great success


def multiprocessing_run(rank, word_size, pernode=None):
    import socket

    print(
        "rank ",
        rank,
        os.getpid(),
        "hash=",
        hash("kk test"),
        " on node ",
        socket.gethostname(),
    )
    print("word_size ", word_size)
    if pernode is None:
        pernode = word_size
    print("Tasks per node = ", pernode)

    os.environ["NODE_RANK"] = str(rank)

    print("Sat os.environ['CUDA_VISIBLE_DEVICES']=", os.environ["CUDA_VISIBLE_DEVICES"])

    my_gpu = rank % pernode
    print("rank ", rank, " will get gpu ", my_gpu, " on node ", socket.gethostname())
    argv = sys.argv
    if "with" not in argv:
        argv = argv + ["with"]
    if rank != 0:
        print(f"Unobserved {os.getpid()} with rank {rank}")
        # only rank 0 is observed
        argv = argv + ["wandb.project=distributed_runs", "-u"]

    argv = argv + [
        f"trainer.num_nodes={word_size}",
        f"rank={rank}",
        f"trainer.strategy=ddp_find_unused_parameters_true",
        f"trainer.devices=[{my_gpu}]",
    ]
    print(argv)

    @ex.main
    def default_command():
        return main()

    ex.run_commandline(argv)


if __name__ == "__main__":
    # set DDP=2 forks two processes to run on two GPUs
    # the environment variable "DDP" define the number of processes to fork
    # With two 2x 2080ti you can train the full model to .47 in around 24 hours
    # you may need to set NCCL_P2P_DISABLE=1
    global word_size
    word_size = os.environ.get("DDP", None)
    DDP_SLURM = os.environ.get("DDP_SLURM", None)
    if DDP_SLURM:
        print("\n***SLLURM DDP MODE***\n\n")
        if "SLURM_NTASKS" in os.environ:
            del os.environ["SLURM_NTASKS"]
        if "SLURM_JOB_NAME" in os.environ:
            del os.environ["SLURM_JOB_NAME"]
        word_size = int(os.environ.get("WORLD_SIZE", None))
        print("word_size = ", word_size)
        pernode = int(os.environ.get("SLURM_NTASKS_PER_NODE", None))
        print("pernode = ", pernode)
        rank = int(os.environ.get("SLURM_PROCID", None))
        print("rank = ", rank)
        os.environ["PL_IN_DDP_SUBPROCESS"] = "1"
        print("I'm runing  with, pid=", os.getpid())
        multiprocessing_run(rank, word_size, pernode)
        exit(0)

    if word_size:
        if "SLURM_NTASKS" in os.environ:
            del os.environ["SLURM_NTASKS"]
        if "SLURM_JOB_NAME" in os.environ:
            del os.environ["SLURM_JOB_NAME"]
        word_size = int(word_size)
        print(f"\n\nDDP TRAINING WITH WORD_SIZE={word_size}\n\n")
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        # plz no collisions
        os.environ["MASTER_PORT"] = f"{9999 + random.randint(0, 9999)}"
        os.environ["PL_IN_DDP_SUBPROCESS"] = "1"
        os.environ["WORLD_SIZE"] = str(word_size)
        if word_size == 1:
            multiprocessing_run(0, word_size)
            exit(0)
        cpids = []
        for rank in range(word_size):
            pid = os.fork()
            if pid == 0:
                print("Child Forked, pid=", os.getpid())
                multiprocessing_run(rank, word_size)
                print("Child Done!, pid=", os.getpid())
                sys.exit(0)
            else:
                cpids.append(pid)

        # pid, exit_code = os.wait()
        pid, exit_code = os.waitpid(cpids[0], 0)
        print(f"rank 0 pid= {pid}, is done waiting with exit_code={exit_code}")

        sys.exit(0)

print("__main__ is running pid", os.getpid(), "in module main: ", __name__)


@ex.automain
def default_command():
    return main()
