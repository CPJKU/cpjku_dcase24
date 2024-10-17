import functools
import os
import sys
from munch import DefaultMunch
import torch
import lightning as L
import numpy as np
import transformers
import wandb

from configs import add_configs, PRETRAINED_MODELS
from helpers.utils import config_call
from helpers.workersinit import worker_init_fn
from sacred import Experiment
from pathlib import Path
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
import datasets
from torch.hub import download_url_to_file
import pickle
import torch.nn as nn
from torch import autocast

from data_util import audioset
from data_util import audioset_strong

from models.fpasst import fpasst as passt
from models import preprocess
from sklearn import metrics
from models.wrapper import AudiosetWrapper, Task4RNNWrapper

from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.callbacks import ModelCheckpoint
import sed_scores_eval

from helpers.augment import mixup, frame_shift, gain_augment, time_mask, feature_transformation, mixstyle
from helpers.metrics import batched_decode_predictions_parallel

from models.atst.atst_model_wrapper import ATSTWrapper, ATSTMel
from models.beats.BEATs_wrapper import BEATsWrapper

if 'LD_LIBRARY_PATH' in os.environ:
    del os.environ['LD_LIBRARY_PATH']

import socket

hostname = socket.gethostname()
# os.environ["HF_DATASETS_CACHE"] = '' # TODO: set this

DEBUG = False

ex = Experiment("audioset")

# verbose logging
datasets.logging.set_verbosity_info()

# define datasets config
get_weak_training_dataset = ex.command(
    audioset.get_training_dataset,
    prefix="training_weak",
    audio_length=10.0,
    wavmix=False,
    augment=True
)

# define datasets config
get_strong_training_dataset = ex.command(
    audioset_strong.get_training_dataset,
    prefix="training_strong",
    audio_length=10.0
)

get_weak_validation_dataset = ex.command(
    audioset.get_validation_dataset,
    prefix="validation_weak",
    audio_length=10.0,
)

get_strong_validation_dataset = ex.command(
    audioset_strong.get_validation_dataset,
    prefix="validation_strong",
    audio_length=10.0,
)

get_weighted_sampler_weak = ex.command(
    audioset.get_weighted_sampler, prefix="training_weak", epoch_len=100_000
)

#get_weighted_sampler_strong = ex.command(
#    audioset_strong.get_weighted_sampler, prefix="training_strong", epoch_len=10_000, n=0.0, p=0.0
#)

# Define loaders
get_train_loader = ex.command(
    DataLoader,
    prefix="training_weak",
    static_args=dict(worker_init_fn=worker_init_fn),
    batch_size=12,
    num_workers=16,
    shuffle=None,
)

get_strong_train_loader = ex.command(
    DataLoader,
    prefix="training_strong",
    static_args=dict(worker_init_fn=worker_init_fn),
    batch_size=12,
    num_workers=16,
    shuffle=True
)

get_weak_validate_loader = ex.command(
    DataLoader,
    prefix="validation_weak",
    static_args=dict(worker_init_fn=worker_init_fn),
    batch_size=20,
    num_workers=16
)

get_strong_validate_loader = ex.command(
    DataLoader,
    prefix="validation_strong",
    static_args=dict(worker_init_fn=worker_init_fn),
    batch_size=20,
    num_workers=16
)

# label encoder
from helpers.encoder import get_encoder
many_hot_encoder = ex.command(get_encoder, prefix="encoder")

Trainer = ex.command(L.Trainer, prefix="trainer")

mel = ex.command(preprocess.AugmentMelSTFT, prefix="passt_mel")
passt_net = ex.command(passt.get_model, prefix="passt")
atst_mel = ex.command(ATSTMel, prefix="atst_mel")

weak_wrapper = ex.command(AudiosetWrapper, prefix="weak_wrapper")
strong_wrapper = ex.command(Task4RNNWrapper, prefix="strong_wrapper")

@ex.config
def default_conf():
    # ruff: noqa: F841
    cmd = " ".join(sys.argv)  # command line arguments

    passt_arch = "passt_arch" in cmd
    jbt_arch = "jbt_arch" in cmd
    frame_dymn_arch = "frame_dymn" in cmd
    atst_arch = "atst_frame" in cmd
    beats_arch = "beats_arch" in cmd

    arch_sum = passt_arch + jbt_arch + frame_dymn_arch + atst_arch + beats_arch

    if arch_sum != 1:
        raise ValueError("Please specify exactly one architecture to train.")

    if "jbt_arch" in cmd:
        # set for flash attention
        # maybe try false true true
        # see: https://github.com/lucidrains/BS-RoFormer/blob/main/bs_roformer/attend.py
        torch.backends.cuda.enable_flash_sdp(True)  # True
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # False
        torch.backends.cuda.enable_math_sdp(True)  # False  # TODO: check why this must be True

        # for A40
        torch.set_float32_matmul_precision('medium')
        print("Successfully set JBT_SEED and flash attention settings.")

    weak_wrapper = dict(use_attention_head=True)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if os.environ.get("SLURM_ARRAY_JOB_ID", False):
        slurm_job_id = (
                os.environ.get("SLURM_ARRAY_JOB_ID", "").strip()
                + "_"
                + os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
        )
    process_id = os.getpid()
    debug_shapes = 2  # print shapes of in step, 0 = never, 1 = first step etc...
    watch_model = False
    trainer = dict(
        max_epochs=130,
        devices=1,
        weights_summary="full",
        benchmark=True,
        num_sanity_val_steps=0,
        precision="16-mixed",
        reload_dataloaders_every_epoch=True,
        default_root_dir="./outputs",
    )

    atst_frame = dict(
        pretrained_name="atst_as"
    )

    beats = dict(
        pretrained_name="beats_as"
    )

    sample_rate = 16_000

    training_weak = dict(
        sample_rate=sample_rate
    )

    training_strong = dict(
        sample_rate=sample_rate
    )

    validation_weak = dict(
        sample_rate=sample_rate
    )

    validation_strong = dict(
        sample_rate=sample_rate
    )

    compile = False  # compile the model, requires pytorch >= 2.0
    optimizer = dict(
        lr=0.0002,
        inital_lr=None,
        lr_pt=None,
        initial_lr_pt=None,
        schedule_mode="cos",
        num_warmup_steps=None
    )

    # for audioset kd
    as_urls = {
        "preds": "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/passt_enemble_logits_mAP_495.npy",
        "fname_to_index": "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/fname_to_index.pkl"
    }

    as_local = {
        "preds": "cache/passt_enemble_logits_mAP_495.npy",
        "fname_to_index": "cache/fname_to_index.pkl"
    }

    seq_len = 250

    # median window
    median_window = 12

    repr_dropout_p = 0.0

    # augmentations
    weak_augmentations = dict(
        use_mixup=True,
        mixup_alpha=0.3,
        gain_augment=5
    )

    strong_augmentations = dict(
        gain_augment=5,
        filter_augment=dict(
            apply=1,
            p=0.8,
            n_transform=2,
            filter_db_range=(-6, 6),
            filter_bands=(3, 6),
            filter_minimum_bandwidth=6
        ),
        time_augment=dict(
            apply_mask=True,
            apply_shift=True,
            shift_range=0.125,
            min_mask_ratio=0.05,
            max_mask_ratio=0.3,
        ),
        mix_augment=dict(
            apply_mixup=True,
            apply_mixstyle=True,
            mixup_p=0.5,
            mixstyle_p=0.2,
            mixstyle_alpha=0.3,
        )
    )

    # loss weights
    weak_supervised_loss_weight = 0.9
    weak_distillation_loss_weight = 0.1
    strong_supervised_loss_weight = 0.0

    atst_checkpoint = "atst_as.ckpt"
    beats_checkpoint = "beats_as.pt"

    skip_checkpoint = False




add_configs(ex)  # add common configurations


# capture the WandbLogger and prefix it with "wandb", this allows to use sacred to update WandbLogger config from the command line
@ex.command(prefix="wandb")
def get_wandb_logger(config, name=None, project="audioset", rank0_only=True, tags=[]):
    rundir = Path(f"./outputs/{project}/")
    rundir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(name=name, dir=rundir, project=project, config=config, tags=tags)
    run.define_metric("trainer/global_step")
    run.define_metric("*", step_metric="trainer/global_step", step_sync=True)
    logger = WandbLogger(
        name=name, dir=rundir, project=project, config=config, tags=tags
    )
    return logger


def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


@ex.command(prefix="optimizer")
def get_lr_scheduler(
        optimizer,
        num_training_steps,
        schedule_mode="exp",
        gamma: float = 0.999996,
        num_warmup_steps=20000,
        lr_end=2e-7,
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
        model, lr=0.0001, lr_pt=None, adamw=True, weight_decay=0.01, betas=(0.9, 0.999)
):
    pt_model_params = []
    ds_model_params = []
    n_scaled_layers = 0

    if lr_pt is None:
        lr_pt = lr

    for name, param in model.named_parameters():
        if name.startswith('net_weak.model'):
            # the audioset pre-trained part
            pt_model_params.append(param)
            n_scaled_layers += 1
        else:
            ds_model_params.append(param)
    print("Scaling lr for", n_scaled_layers)
    param_groups = [
        {'params': pt_model_params, 'lr': lr_pt}, # pretrained model
        {'params': ds_model_params, 'lr': lr}  # downstream model
    ]

    if adamw:
        print(f"\nUsing adamw weight_decay={weight_decay}!\n")
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=betas)
    return torch.optim.Adam(param_groups, lr=lr, betas=betas)


def my_mixup(size, alpha):
    rn_indices = torch.randperm(size)
    lambd = np.random.beta(alpha, alpha, size).astype(np.float32)
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.FloatTensor(lambd)
    # data = data * lam + data2 * (1 - lam)
    # targets = targets * lam + targets2 * (1 - lam)
    return rn_indices, lam


class BL23Module(L.LightningModule):
    def __init__(
            self,
            config
    ):
        super(BL23Module, self).__init__()
        config = DefaultMunch.fromDict(config)
        self.config = config
        self.construct_modules()

        # TODO: make this configurable and separate for weak and strong augmentations
        self.weak_augmentations = self.config.weak_augmentations
        self.strong_augmentations = self.config.strong_augmentations

        self.distributed_mode = self.config.trainer.num_nodes > 1

        if self.config.compile:
            # pt 2 magic
            print("\n\nCompiling the model pytorch 2... \n\n")
            self.net_weak = torch.compile(self.net_weak)
            self.net_strong = torch.compile(self.net_strong)

        print(self.net_weak)

        # TODO: move this into a separate caching routine that can generate labels from pretrained models
        # download ensemble predictions and meta data
        os.makedirs("cache", exist_ok=True)
        if not os.path.exists(self.config.as_local['preds']):
            # download file
            print("Download audioset ensemble predictions.")
            download_url_to_file(self.config.as_urls['preds'], self.config.as_local['preds'])
        if not os.path.exists(self.config.as_local['fname_to_index']):
            # download file
            print("Download audioset ensemble predictions mappings file.")
            download_url_to_file(self.config.as_urls['fname_to_index'], self.config.as_local['fname_to_index'])

        # build the corresponding mapping form file name to predictions
        as_ensemble_preds = np.load(self.config.as_local['preds'])
        as_ensemble_preds = torch.from_numpy(as_ensemble_preds).float()
        as_ensemble_preds = torch.sigmoid(as_ensemble_preds)
        as_ensemble_preds.requires_grad = False
        as_ensemble_preds = as_ensemble_preds
        with open(self.config.as_local['fname_to_index'], 'rb') as f:
            fname_to_index = pickle.load(f)
        self.filename_to_weak_predictions = {f: as_ensemble_preds[i] for f, i in fname_to_index.items()}

        self.weak_loss_fn = nn.BCELoss(reduction="none")
        self.weak_distillation_loss_fn = nn.BCELoss(reduction="none")

        self.strong_loss_fn = nn.BCELoss()

        # representation dropout
        if self.config.repr_dropout_p > 0:
            self.repr_dropout = nn.Dropout2d(p=self.config.repr_dropout_p)
        else:
            self.repr_dropout = nn.Identity()

        # pl 2 containers:

        # weak eval
        self.val_step_outputs_weak = []

        # strong eval
        self.val_predictions_strong = {}
        self.val_ground_truth = {}
        self.val_duration = {}

    def construct_modules(self):
        arch = self.config["arch"]

        scall = functools.partial(
            config_call, config=self.config
        )

        if arch == "passt":
            self.mel = scall(mel)
            net = scall(passt_net)
            embed_dim = net.num_features
        elif arch == "beats":
            net = BEATsWrapper(
                cfg_path=os.path.join(PRETRAINED_MODELS, self.config['beats_checkpoint']),
                output_tokens_per_timestep=self.config['output_tokens_per_timestep']
            )
            self.mel = net.preprocess
            embed_dim = 768*(8//self.config['output_tokens_per_timestep'])
        elif arch == "atst_frame":
            self.mel = scall(atst_mel)
            net = ATSTWrapper(os.path.join(PRETRAINED_MODELS, self.config['atst_checkpoint']))
            embed_dim = 768
        else:
            raise ValueError(f"Unknown arch={arch}")

        net.arch = arch

        self.net_weak = weak_wrapper(
            net,
            embed_dim=embed_dim,
            seq_len=self.config.seq_len,
            pretrained_name=self.config[arch]["wandb_id"]
        )

        self.net_strong = strong_wrapper(
            net,
            seq_len=self.config.seq_len,
            embed_dim=embed_dim,
            pretrained_name=self.config[arch]["wandb_id"]
        )

        self.encoder = scall(many_hot_encoder)

    def forward(self, x):
        strong, weak = self.net_weak(x)
        return strong, weak

    def forward_weak(self, batch):
        """
        Computes a complete forward pass with augmentations for a weakly labeled batch.
        Takes a batch (dictionary) as input and returns the same batch with additional keys.
        """
        x = batch["audio"]
        y = batch["target"]

        # sanity check
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Input data contains NaN or infinite values.")

        if self.training and self.weak_augmentations.gain_augment > 0:
                x = gain_augment(x, gain=self.weak_augmentations.gain_augment)

        # compute audio-visual features
        if self.mel:
            x = self.mel(x)

        # augmentations
        mixup_config = None
        if self.training:
            # MixUp inputs & targets
            batch_size = len(y)
            if self.weak_augmentations.use_mixup:
                mixup_config = my_mixup(batch_size, self.weak_augmentations.mixup_alpha)
                mixup_config = (mixup_config[0], mixup_config[1].to(x.device))
                permutation_indices, lam = mixup_config
                x = x * lam.reshape(batch_size, 1, 1, 1) + x[permutation_indices] * (
                            1. - lam.reshape(batch_size, 1, 1, 1))
                y = y * lam.reshape(batch_size, 1) + y[permutation_indices] * (1. - lam.reshape(batch_size, 1))

            # representation dropout for multi channel inputs
            x = self.repr_dropout(x)

            # TODO: add other augmentations

        # forward through network; use weak head
        y_hat_strong, y_hat = self.net_weak(x)

        # store things in batch for loss computation
        batch['y'] = y
        batch['y_hat'] = y_hat
        batch['y_hat_strong'] = y_hat_strong
        batch['mixup_config'] = mixup_config

        return batch

    def forward_strong(self, batch):
        """
        Computes a complete forward pass with augmentations for a weakly labeled batch.
        Takes a batch (dictionary) as input and returns the same batch with additional keys.
        """
        x = batch["audio"]

        # sanity check
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Input data contains NaN or infinite values.")

        if self.training and self.strong_augmentations.gain_augment > 0:
                x = gain_augment(x, gain=self.strong_augmentations.gain_augment)

        # compute audio-visual features
        if self.mel:
            x = self.mel(x)

        features = x
        labels = batch['strong']
        # augmentations
        if self.training:

            # rolling
            if self.strong_augmentations.time_augment.apply_shift:
                features, labels = frame_shift(
                    features,
                    labels,
                    net_pooling=self.encoder.net_pooling,
                    shift_range=self.strong_augmentations.time_augment.shift_range
                )

            # mixup
            if self.strong_augmentations.mix_augment.apply_mixup and self.strong_augmentations.mix_augment.mixup_p > random.random():
                features, labels = mixup(
                    features,
                    targets=labels,
                    mixup_label_type="soft"
                )

            # mixstyle
            if self.strong_augmentations.mix_augment.apply_mixstyle and self.strong_augmentations.mix_augment.mixstyle_p > random.random():
                features = mixstyle(
                    features,
                    alpha=self.strong_augmentations.mix_augment.mixstyle_alpha
                )

            # representation dropout - for multi-resolution input (mutliple fft windows)
            features = self.repr_dropout(features)

            # time masking
            if self.strong_augmentations.time_augment.apply_mask:
                features, labels = time_mask(
                    features,
                    labels,
                    net_pooling=self.encoder.net_pooling,
                    min_mask_ratio=self.strong_augmentations.time_augment.min_mask_ratio,
                    max_mask_ratio=self.strong_augmentations.time_augment.max_mask_ratio
                )
            # frequency masking
            if self.strong_augmentations.filter_augment.apply and self.strong_augmentations.filter_augment.p > random.random():
                features, _ = feature_transformation(
                    features,
                    self.strong_augmentations.filter_augment.n_transform,
                    self.strong_augmentations.filter_augment.filter_db_range,
                    self.strong_augmentations.filter_augment.filter_bands,
                    self.strong_augmentations.filter_augment.filter_minimum_bandwidth
                )

        # forward through network; use strong head
        y_hat_strong, y_hat = self.net_strong(features)

        # store things in batch for loss computation
        batch['y_hat'] = y_hat
        batch['y_hat_strong'] = y_hat_strong
        batch['y_strong'] = labels

        return batch


    def distillation_loss(self, y_hat, file_names, file_name_to_prediction, loss_fun, mixup_config=None):
        """
        TODO
        """
        # return 0 if weight is zero
        if self.config.weak_distillation_loss_weight <= 0:
            return torch.tensor(0., device=y_hat.device, dtype=y_hat.dtype)

        # get the teacher embeddings for each example in the batch
        y_hat_teacher = [file_name_to_prediction.get(f, None) for f in file_names]

        # check if all teacher embeddings are available
        assert all([p is not None for p in y_hat_teacher]), f"Some ensemble embeddings are not available."

        # get teacher embeddings
        y_hat_teacher = torch.stack([p for p in y_hat_teacher if p is not None]).to(y_hat.device)

        with autocast(enabled=False, device_type='cuda'):
            y_hat_teacher = y_hat_teacher.to(y_hat.device).float()
            y_hat = y_hat.float()
            batch_size = len(y_hat)
            if mixup_config is not None:
                permutation_indices, lam = mixup_config
                lam = lam.reshape(batch_size)
                distillation_loss = \
                    loss_fun(y_hat, y_hat_teacher).mean(dim=1) * lam + \
                    loss_fun(y_hat, y_hat_teacher[permutation_indices]).mean(dim=1) * (1. - lam)
            else:
                distillation_loss = loss_fun(y_hat, y_hat_teacher)

        distillation_loss = distillation_loss.mean()

        # weighting losses
        return distillation_loss

    def training_step(self, batch, batch_idx):

        self.update_lr()
        # check if both weak and strong batches are included

        if "strong" in batch and "weak" in batch:
            weak_batch = batch["weak"]
            strong_batch = batch["strong"]
        elif "strong" in batch:
            weak_batch = None
            strong_batch = batch["strong"]
        else:
            weak_batch = batch["weak"]
            strong_batch = None

        if weak_batch is not None:
            # forward the weak batch supervised
            weak_batch = self.forward_weak(weak_batch)

            # compute the weak supervised loss
            with autocast(enabled=False, device_type='cuda'):
                y_hat = weak_batch["y_hat"].float()
                y = weak_batch["y"].float()
                weak_supervised_loss = self.weak_loss_fn(y_hat, y).mean()

            # compute distillation loss
            weak_distillation_loss = self.distillation_loss(
                weak_batch["y_hat"],
                weak_batch["filename"],
                self.filename_to_weak_predictions,
                self.weak_distillation_loss_fn,
                weak_batch["mixup_config"]
            )
        else:
            weak_supervised_loss = torch.tensor(0., device=strong_batch["audio"].device, dtype=strong_batch["audio"].dtype)
            weak_distillation_loss = torch.tensor(0., device=strong_batch["audio"].device, dtype=strong_batch["audio"].dtype)

        # TODO: weak self-supervised loss with strong teacher
        # def forward_teacher ...
        if strong_batch is not None:
            strong_batch = self.forward_strong(strong_batch)

            # compute the weak supervised loss
            # with autocast(enabled=False, device_type='cuda'):
            with autocast(enabled=False, device_type='cuda'):
                y_hat_strong = strong_batch["y_hat_strong"].float()
                y_strong = strong_batch["y_strong"].float()

                strong_supervised_loss = self.strong_loss_fn(y_hat_strong, y_strong)
        else:
            strong_supervised_loss = torch.tensor(0., device=y_hat.device, dtype=y_hat.dtype)

        loss = (
                self.config.weak_supervised_loss_weight * weak_supervised_loss +
                self.config.weak_distillation_loss_weight * weak_distillation_loss +
                self.config.strong_supervised_loss_weight * strong_supervised_loss
        )

        # logging
        self.log('epoch', self.current_epoch)
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f'trainer/lr_optimizer_{i}', param_group['lr'])
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)
        self.log("train/weak_supervised_loss", weak_supervised_loss.detach().cpu())
        self.log("train/weak_distillation_loss", weak_distillation_loss.detach().cpu())
        self.log("train/strong_supervised_loss", strong_supervised_loss.detach().cpu())
        return loss

    def update_lr(self):
        if self.config['optimizer']['num_warmup_steps'] is not None:
            if self.config['optimizer']['num_warmup_steps'] == 0:
                warmup_weight = 1.0
            else:
                warmup_weight = min(self.global_step / self.config['optimizer']['num_warmup_steps'], 1.0)

            # update learning rate for new model
            if self.config['optimizer']['initial_lr'] is not None:
                lr = self.config['optimizer']['lr']
                initial_lr = self.config['optimizer']['initial_lr']
                delta = (lr - initial_lr) * warmup_weight
                self.trainer.optimizers[0].param_groups[1]['lr'] = initial_lr + delta

            # update learning rate for pretrained model
            if self.config['optimizer']['initial_lr_pt'] is not None:
                lr_pt = self.config['optimizer']['lr_pt']
                initial_lr_pt = self.config['optimizer']['initial_lr_pt']
                delta = (lr_pt - initial_lr_pt) * warmup_weight
                self.trainer.optimizers[0].param_groups[0]['lr'] = initial_lr_pt + delta

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        assert dataloader_idx in [0, 1], "Only weak and strong evaluation are supported."

        # distinguish between weak and strong eval sets
        if 'gt_string' not in batch:
            # weak evaluation
            batch = self.forward_weak(batch)
            y_hat = batch['y_hat']
            y = batch['y']
            nan_mask = torch.isnan(y_hat)
            y_hat = torch.nan_to_num(y_hat, nan=0.0)

            assert not torch.isnan(y_hat).any(), f"y_hat contains NaN values."
            assert not torch.isnan(y).any(), f"y contains NaN values."

            with autocast(enabled=False, device_type='cuda'):
                y = y.float()
                y_hat = y_hat.float()
                samples_loss = self.weak_loss_fn(y_hat, y)

            y_hat = torch.sigmoid(y_hat.detach())
            # self.log("validation.loss", loss, prog_bar=True, on_epoch=True, on_step=False)
            results = {
                "loss": samples_loss.detach(),
                "y_hat": y_hat.detach(),
                "target": y.detach(),
                "nan_mask": nan_mask.detach()
            }
            results = {k: v.cpu() for k, v in results.items()}
            self.val_step_outputs_weak.append(results)

        else:
            # eval strong
            # parse ground truth
            for f, gt_string in zip(batch["filename"], batch["gt_string"]):
                if f in self.val_ground_truth:
                    continue
                else:
                    events = [e.split(";;") for e in gt_string.split("++")]
                    self.val_ground_truth[f.split(".")[0]] = [(float(e[0]), float(e[1]), e[2]) for e in events]
                    self.val_duration[f.split(".")[0]] = (batch["audio"].shape[1] / batch["sampling_rate"][0]).item()

            batch = self.forward_strong(batch)
            y_hat_strong = batch['y_hat_strong']

            scores_postprocessed_student_strong = batched_decode_predictions_parallel(
                y_hat_strong.float(),
                batch['filename'],
                self.encoder,
                median_filter=self.config.median_window,
                n_jobs=12
            )

            self.val_predictions_strong.update(
                scores_postprocessed_student_strong
            )

    def compute_weak_metrics(self):
        if len(self.val_step_outputs_weak) == 0:
            return {}
        # list of dictionaries to dictionary of list
        outputs = {k: [] for k in self.val_step_outputs_weak[0]}
        for step_output in self.val_step_outputs_weak:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.cat(outputs[k])

        avg_loss = outputs['loss'].mean()
        out = outputs['y_hat']
        target = outputs['target']
        nan_count = outputs['nan_mask'].sum()

        if self.distributed_mode:
            out = self.all_gather(out)
            target = self.all_gather(target)
            nan_count = self.all_gather(nan_count)

        try:
            average_precision = metrics.average_precision_score(
                target.float().numpy(),
                out.float().numpy(),
                average=None
            )
        except ValueError:
            average_precision = np.array([np.nan] * out.shape[-1])

        try:
            roc = metrics.roc_auc_score(target.numpy(), out.numpy(), average=None)
        except ValueError:
            roc = np.array([np.nan] * out.shape[-1])

        logs = {
            "val/loss": torch.as_tensor(avg_loss).cuda(),
            "val/ap": torch.as_tensor(average_precision.mean()).cuda(),
            "val/roc": torch.as_tensor(roc.mean()).cuda(),
            "val/nan_count": torch.as_tensor(nan_count).cuda().float(),
        }

        return logs

    def compute_strong_metrics(self):
        if len(self.val_predictions_strong) == 0:
            return {}

        # synth dataset
        ground_truth = self.val_ground_truth
        audio_durations = self.val_duration

        # drop classes not present in val set
        unused_classes = list(
            set(self.encoder.labels).difference(set([e[2] for f, events in ground_truth.items() for e in events])))
        print("Total unused classes:", len(unused_classes), "out of", len(self.encoder.labels))
        for f, df in self.val_predictions_strong.items():
            df.drop(columns=list(unused_classes), axis=1, inplace=True)

        print("Computing pauc")
        segment_based_pauroc = sed_scores_eval.segment_based.auroc(
            self.val_predictions_strong,
            ground_truth,
            audio_durations,
            max_fpr=0.1,
            segment_length=1.0,
            num_jobs=1
        )

        print("Computing psds")
        psds1 = sed_scores_eval.intersection_based.psds(
            self.val_predictions_strong,
            ground_truth,
            audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
            num_jobs=1
        )

        print("Computing Proxy Targets")
        from data_util.t4.t4_24_classes_dict import as_proxy_classes_mapping
        proxy_classes = []
        for k, v in as_proxy_classes_mapping.items():
            proxy_classes.extend(v)
        proxy_classes = set(proxy_classes)

        proxy_ground_truth = {}
        proxy_audio_durations = {}
        for f in ground_truth:
            events = []
            for e in ground_truth[f]:
                if e[2] in proxy_classes:
                    events.append(e)
            if len(events) > 0:
                proxy_ground_truth[f] = events
                proxy_audio_durations[f] = audio_durations[f]

        unused_classes = list(set(self.encoder.labels).difference(set(unused_classes)).difference(proxy_classes))
        for f, df in self.val_predictions_strong.items():
            df.drop(columns=list(unused_classes), axis=1, inplace=True)

        proxy_psds1 = sed_scores_eval.intersection_based.psds(
            {f:d for f,d in self.val_predictions_strong.items() if f in proxy_ground_truth},
            proxy_ground_truth,
            proxy_audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
            num_jobs=1
        )

        proxy_segment_based_pauroc = sed_scores_eval.segment_based.auroc(
            {f:d for f,d in self.val_predictions_strong.items() if f in proxy_ground_truth},
            proxy_ground_truth,
            proxy_audio_durations,
            max_fpr=0.1,
            segment_length=1.0,
            num_jobs=1
        )

        logs = {
            **{f"val/strong/classwise/psds1_{k}": v for k, v in psds1[1].items()},
            "val/strong/psds1": psds1[0],
            "val/strong/psds1_macro_averaged": np.array([v for k, v in psds1[1].items()]).mean(),
            "val/strong/pauroc": segment_based_pauroc[0]['mean'],
            "val/obj_metric": psds1[0] + segment_based_pauroc[0]['mean'],
            "val/strong/psds1_proxy": proxy_psds1[0],
            "val/strong/pauroc_proxy": proxy_segment_based_pauroc[0]['mean'],
        }

        return logs

    def on_validation_epoch_end(self):

        # compute weak and strong metrics
        weak_metrics = self.compute_weak_metrics()
        self.log_dict(weak_metrics, sync_dist=True)

        strong_metrics = self.compute_strong_metrics()
        self.log_dict(strong_metrics, sync_dist=True)

        # empty output
        self.val_step_outputs_weak.clear()
        self.val_predictions_strong.clear()

    def configure_optimizers(self):

        optimizer = get_optimizer(self)

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
        watch_model=False,
        some_var=True,
        balanced_domain_sampler=False,
        use_new_training_dataset=True,
):
    logger = None
    if rank == 0 and _config["wandb"]["rank0_only"] and not DEBUG:
        # final experiment config is resolved by now
        logger = get_wandb_logger(_config)
    config = DefaultMunch.fromDict(_config)

    print("main() is running pid", os.getpid(), "in module main: ", __name__)

    module = BL23Module(config)
    # load training data sets
    # weak
    train_ds = get_weak_training_dataset()
    train_sampler = get_weighted_sampler_weak(audioset.get_ft_cls_balanced_sample_weights(train_ds))
    train_loader = get_train_loader(dataset=train_ds, sampler=train_sampler)
    # strong
    train_strong_ds = get_strong_training_dataset(module.encoder)

    #train_sampler_strong = get_weighted_sampler_strong(train_strong_ds, module.encoder)
    #train_strong_loader = get_strong_train_loader(dataset=train_strong_ds, sampler=

    train_strong_loader = get_strong_train_loader(dataset=train_strong_ds)
    # combine
    iterables = {"weak": train_loader, "strong": train_strong_loader}

    if _config["strong_supervised_loss_weight"] == 0:
        iterables = {"weak": train_loader}
    elif _config["weak_supervised_loss_weight"] == 0 and _config["weak_distillation_loss_weight"] == 0:
        iterables = {"strong": train_strong_loader}
    else:
        iterables = {"weak": train_loader, "strong": train_strong_loader}

    combined_train_loader = CombinedLoader(iterables, mode="min_size")

    # load validation data sets
    # weak
    val_ds = get_weak_validation_dataset()
    validation_sampler = audioset.ValidationDistributedSampler(val_ds)
    validate_loader = get_weak_validate_loader(dataset=val_ds, sampler=validation_sampler)
    # strong
    val_strong_ds = get_strong_validation_dataset(module.encoder)
    validation_strong_sampler = audioset.ValidationDistributedSampler(val_strong_ds)
    validate_strong_loader = get_strong_validate_loader(dataset=val_strong_ds, sampler=validation_strong_sampler)
    # combine
    if _config["strong_supervised_loss_weight"] == 0:
        combined_val_loader = validate_loader
    elif _config["weak_supervised_loss_weight"] == 0 and _config["weak_distillation_loss_weight"] == 0:
        combined_val_loader = validate_strong_loader
    else:
        combined_val_loader = [validate_strong_loader, validate_loader] # CombinedLoader(iterables, mode="min_size")

    callbacks = [
        ModelCheckpoint(
            logger.log_dir,
            monitor="val/obj_metric",
            save_top_k=1,
            mode="max",
            save_last=True,
        )
    ] if not _config["skip_checkpoint"] and not DEBUG else []

    # training routine ...
    trainer = Trainer(logger=logger, gradient_clip_val=0.5, callbacks=callbacks)
    # trainer.validate(module, dataloaders=combined_val_loader)
    trainer.fit(
        module,
        train_dataloaders=combined_train_loader,
        val_dataloaders=combined_val_loader,
    )
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
        import random

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
