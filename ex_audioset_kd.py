import functools
import os
import sys
from munch import DefaultMunch
import torch
import lightning as L
import numpy as np
import transformers
import wandb

from config_updates import add_configs
from helpers.utils import config_call
from helpers.workersinit import worker_init_fn
from sacred import Experiment
from pathlib import Path
from sacred.config_helpers import CMD
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
import datasets
from torch.hub import download_url_to_file
import pickle
import torch.nn as nn
from torch import autocast

from data_util import audioset

from models.frame_passt import fpasst as passt
from models import preprocess
from helpers.mixup import my_mixup
from sklearn import metrics
from models.frame_dymn.model import get_model as get_frame_dymn

import models.jbt.jazz_beat_transformer as beat_tracker
from models.wrapper import AudiosetWrapper

ex = Experiment("audioset")

# verbose logging
datasets.logging.set_verbosity_info()

# define datasets config
get_training_dataset = ex.command(
    audioset.get_training_dataset,
    prefix="training",
    audio_length=10.0,
    wavmix=False,
    augment=True
)

# define datasets config
get_validation_dataset = ex.command(
    audioset.get_validation_dataset,
    prefix="validation",
    audio_length=10.0,
)

get_weighted_sampler = ex.command(
    audioset.get_weighted_sampler, prefix="training", epoch_len=100_000
)

# Define loaders
get_train_loader = ex.command(
    DataLoader,
    prefix="training",
    static_args=dict(worker_init_fn=worker_init_fn),
    train=True,
    batch_size=12,
    num_workers=16,
    shuffle=None,
)

get_validate_loader = ex.command(
    DataLoader,
    prefix="validation",
    static_args=dict(worker_init_fn=worker_init_fn),
    validate=True,
    batch_size=20,
    num_workers=16,
    dataset=CMD("/get_validation_dataset"),
)
Trainer = ex.command(L.Trainer, prefix="trainer")

mel = ex.command(preprocess.AugmentMelSTFT, prefix="passt_mel")
passt_net = ex.command(passt.get_model, prefix="passt")
jbt_net = ex.command(beat_tracker.BeatTransformer, prefix="jbt")
frame_dymn_net = ex.command(get_frame_dymn, prefix="frame_dymn")

@ex.config
def default_conf():
    # ruff: noqa: F841
    cmd = " ".join(sys.argv)  # command line arguments

    passt_arch = "passt_arch" in cmd
    jbt_arch = "jbt_arch" in cmd
    frame_dymn_arch = "frame_dymn" in cmd

    arch_sum = passt_arch + jbt_arch + frame_dymn_arch

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
    mixup_alpha = 0.3
    compile = False  # compile the model, requires pytorch >= 2.0
    optimizer = dict(
        lr=0.0002,
        schedule_mode="cos"
    )

    kd_lambda = 0.1

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

    wrapper_name = "AudiosetWrapper"

    repr_dropout_p = 0


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
        params, lr, adamw=True, weight_decay=0.0001
):
    if adamw:
        print(f"\nUsing adamw weight_decay={weight_decay}!\n")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr)


class BL23Module(L.LightningModule):
    def __init__(
            self,
            config
    ):
        super(BL23Module, self).__init__()
        config = DefaultMunch.fromDict(config)
        self.config = config
        self.construct_modules()

        self.use_mixup = self.config.use_mixup or False
        self.mixup_alpha = self.config.mixup_alpha
        self.net.return_embed = True
        self.do_swa = False

        self.distributed_mode = self.config.trainer.num_nodes > 1

        if self.config.compile:
            # pt 2 magic
            print("\n\nCompiling the model pytorch 2... \n\n")
            self.net = torch.compile(self.net)

        print(self.net)

        os.makedirs("cache", exist_ok=True)
        if not os.path.exists(self.config.as_local['preds']):
            # download file
            print("Download audioset ensemble predictions.")
            download_url_to_file(self.config.as_urls['preds'], self.config.as_local['preds'])
        as_ensemble_preds = np.load(self.config.as_local['preds'])
        as_ensemble_preds = torch.from_numpy(as_ensemble_preds).float()
        as_ensemble_preds = torch.sigmoid(as_ensemble_preds)
        as_ensemble_preds.requires_grad = False
        self.as_ensemble_preds = as_ensemble_preds

        if not os.path.exists(self.config.as_local['fname_to_index']):
            # download file
            print("Download audioset ensemble predictions mappings file.")
            download_url_to_file(self.config.as_urls['fname_to_index'], self.config.as_local['fname_to_index'])
        with open(self.config.as_local['fname_to_index'], 'rb') as f:
            self.fname_to_index = pickle.load(f)

        self.loss_fn = nn.BCELoss(reduction="none")
        self.distillation_loss = nn.BCELoss(reduction="none")

        # representation dropout
        if self.config.repr_dropout_p > 0:
            self.repr_dropout = nn.Dropout2d(p=self.config.repr_dropout_p)
        else:
            self.repr_dropout = nn.Identity()

        # pl 2 containers:
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.eval_step_outputs = []

    def construct_modules(self):
        arch = self.config["arch"]

        scall = functools.partial(
            config_call, config=self.config
        )

        if arch == "passt":
            self.mel = scall(mel)
            net = scall(passt_net)
            embed_dim = net.num_features
        elif arch == "jbt":
            self.mel = None
            net = scall(jbt_net)
            embed_dim = net.hidden_dim
        elif arch == "frame_dymn":
            self.mel = scall(mel)
            net = scall(frame_dymn_net)
            embed_dim = net.lastconv_output_channels
        else:
            raise ValueError(f"Unknown arch={arch}")

        net.arch = arch
        if self.config.wrapper_name == "AudiosetWrapper":
            self.net = AudiosetWrapper(net, 527, embed_dim, seq_len=self.config.seq_len, use_attention_head=True,
                                       wandb_id=self.config[arch]["wandb_id"])
        else:
            raise ValueError(f"Unknown wrapper_name: {self.config.wrapper_name}")

    def forward(self, x):
        _, weak = self.net(x)
        return weak

    def mel_forward(self, x):
        x = self.mel(x)
        # x = x.unsqueeze(1)
        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x = batch["audio"]
        f = batch["filename"]
        y = batch["target"]

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Input data contains NaN or infinite values.")

        if self.mel:
            x = self.mel_forward(x)

        batch_size = len(y)

        rn_indices, lam = None, None
        if self.use_mixup:
            rn_indices, lam = my_mixup(batch_size, self.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(batch_size, 1, 1, 1) + \
                x[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))

        # representation dropout
        x = self.repr_dropout(x)

        y_hat = self.forward(x)

        with autocast(enabled=False, device_type='cuda'):
            y = y.float()
            y_hat = y_hat.float()

            if self.use_mixup:
                lam = lam.float()
                y_mix = y * lam.reshape(batch_size, 1) + \
                        y[rn_indices] * (1. - lam.reshape(batch_size, 1))

                samples_loss = self.loss_fn(
                    y_hat, y_mix)
                label_loss = samples_loss.mean()
                samples_loss = samples_loss.detach()
            else:
                samples_loss = self.loss_fn(
                    y_hat, y)
                label_loss = samples_loss.mean()
                samples_loss = samples_loss.detach()

        # knowledge distillation
        if self.config.kd_lambda > 0:
            # fetch the correct index in 'teacher_preds' for given filename
            # insert -1 for files not in fname_to_index (proportion of files successfully downloaded from
            # YouTube can vary for AudioSet)
            indices = torch.tensor(
                [self.fname_to_index[fname] if fname in self.fname_to_index else -1 for fname in f],
                dtype=torch.int64
            )
            # get indices of files we could not find the teacher predictions for
            unknown_indices = indices == -1
            y_soft_teacher = self.as_ensemble_preds[indices]
            y_soft_teacher = y_soft_teacher.to(y_hat.device)

            with autocast(enabled=False, device_type='cuda'):
                y_soft_teacher = y_soft_teacher.float()
                y_hat = y_hat.float()

                if self.use_mixup:
                    lam = lam.float()
                    soft_targets_loss = \
                        self.distillation_loss(y_hat, y_soft_teacher).mean(dim=1) * lam.reshape(batch_size) + \
                        self.distillation_loss(y_hat, y_soft_teacher[rn_indices]).mean(dim=1) \
                        * (1. - lam.reshape(batch_size))
                else:
                    soft_targets_loss = self.distillation_loss(y_hat, y_soft_teacher)

            # zero out loss for samples we don't have teacher predictions for
            soft_targets_loss[unknown_indices] = soft_targets_loss[unknown_indices] * 0
            soft_targets_loss = soft_targets_loss.mean()

            # weighting losses
            label_loss = self.config.kd_lambda * label_loss
            soft_targets_loss = (1 - self.config.kd_lambda) * soft_targets_loss
        else:
            soft_targets_loss = torch.tensor(0., device=label_loss.device, dtype=label_loss.dtype)

        # total loss is sum of lambda-weighted label and distillation loss
        loss = label_loss + soft_targets_loss

        results = {"loss": loss, }
        self.log('trainer/lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log('epoch', self.current_epoch)
        self.log("train/label_loss", label_loss.detach().cpu())
        self.log("train/soft_targets_loss", soft_targets_loss.detach().cpu())
        self.log("train/loss", loss.detach().cpu())
        return results['loss']

    def on_train_epoch_end(self) -> None:
        pass

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, f, y = batch
        if self.mel:
            x = self.mel_forward(x)

        strong, weak = self.forward(x)
        return f, weak

    def validation_step(self, batch, batch_idx):
        # REQUIRED
        x = batch["audio"]
        f = batch["filename"]
        y = batch["target"]
        if self.mel:
            x = self.mel_forward(x)

        results = {}
        model_name = [("", self.net)]
        if self.do_swa:
            model_name = model_name + [("swa_", self.net_swa)]
        for net_name, net in model_name:
            _, y_hat = net(x)

            nan_mask = torch.isnan(y_hat)
            y_hat = torch.nan_to_num(y_hat, nan=0.0)

            assert not torch.isnan(y_hat).any(), f"y_hat contains NaN values."
            assert not torch.isnan(y).any(), f"y contains NaN values."

            with autocast(enabled=False, device_type='cuda'):
                y = y.float()
                y_hat = y_hat.float()
                samples_loss = self.loss_fn(y_hat, y)

            out = torch.sigmoid(y_hat.detach())
            # self.log("validation.loss", loss, prog_bar=True, on_epoch=True, on_step=False)
            results = {**results, net_name + "loss": samples_loss,
                       net_name + "out": out, net_name + "target": y.detach(), "nan_mask": nan_mask.detach()}
        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.cat(outputs[k])

        model_name = [("", self.net)]
        if self.do_swa:
            model_name = model_name + [("swa_", self.net_swa)]
        for net_name, net in model_name:
            avg_loss = outputs[net_name + 'loss'].mean()
            out = outputs[net_name + 'out']
            target = outputs[net_name + 'target']
            nan_count = outputs[net_name + 'nan_mask'].sum()
            try:
                average_precision = metrics.average_precision_score(
                    target.float().numpy(), out.float().numpy(), average=None)
            except ValueError:
                average_precision = np.array([np.nan] * 527)
            try:
                roc = metrics.roc_auc_score(
                    target.numpy(), out.numpy(), average=None)
            except ValueError:
                roc = np.array([np.nan] * 527)
            logs = {"val/" + net_name + 'loss': torch.as_tensor(avg_loss).cuda(),
                    "val/" + net_name + 'ap': torch.as_tensor(average_precision.mean()).cuda(),
                    "val/" + net_name + 'roc': torch.as_tensor(roc.mean()).cuda(),
                    "val/" + net_name + 'nan_count': torch.as_tensor(nan_count).cuda().float(),
                    }
            self.log_dict(logs, sync_dist=True)
            if self.distributed_mode:
                allout = self.all_gather(out)
                alltarget = self.all_gather(target)

                average_precision = metrics.average_precision_score(
                    alltarget.reshape(-1, alltarget.shape[-1]).cpu().numpy(),
                    allout.reshape(-1, allout.shape[-1]).cpu().numpy(), average=None)
                if self.trainer.is_global_zero:
                    logs = {net_name + "allap": torch.as_tensor(average_precision.mean()).cuda(),
                            # 'step': torch.as_tensor(self.current_epoch).cuda()
                            }
                    self.log_dict(logs, sync_dist=False)
            else:
                self.log_dict(
                    {"val/" + net_name + "allap": logs["val/" + net_name + 'ap'],
                     # 'step': logs['step']
                     }
                    , sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = get_optimizer(self.net.parameters())

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
    if rank == 0 and _config["wandb"]["rank0_only"]:
        # final experiment config is resolved by now
        logger = get_wandb_logger(_config)
    config = DefaultMunch.fromDict(_config)

    print("main() is running pid", os.getpid(), "in module main: ", __name__)

    module = BL23Module(config)

    train_ds = get_training_dataset()
    train_sampler = get_weighted_sampler(audioset.get_ft_cls_balanced_sample_weights(train_ds))
    train_loader = get_train_loader(dataset=train_ds, sampler=train_sampler)

    val_ds = get_validation_dataset()
    validation_sampler = audioset.ValidationDistributedSampler(val_ds)
    validate_loader = get_validate_loader(dataset=val_ds, sampler=validation_sampler)

    trainer = Trainer(logger=logger, gradient_clip_val=0.5)
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validate_loader,
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
