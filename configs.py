import os

PRETRAINED_MODELS = "resources/pretrained_models"
DATASET_PATH = "/share/hel/datasets/dcase24_task4/dataset"
DIRS_PATH = "resources/dirs"

def add_configs(ex):
    """
    This functions add generic configuration for the experiments, such as mix-up, architectures, etc...
    """

    @ex.named_config
    def atst_frame_arch_16khz():
        arch = "atst_frame"
        audio_len = 10

        sample_rate = 16_000
        win_length = 1024
        n_fft = 1024
        hop_size = 160

        sequence_length = 250  # quadratic runtime (Transformer)

        # encoder for DCASE Task 4 dataset
        net_subsample = 4
        encoder = dict(
            audio_len=audio_len,
            frame_len=win_length,
            frame_hop=hop_size,
            net_pooling=net_subsample,
            fs=sample_rate,
        )

        atst_frame = dict(
            pretrained_name=None
        )


    @ex.named_config
    def audio_set_strong_pretraining():

        weak_supervised_loss_weight = 0.0
        weak_distillation_loss_weight = 0.0
        strong_supervised_loss_weight = 1.0

        training_weak = dict(
            batch_sizes=64
        )

        training_strong = dict(
            batch_size=64
        )

        validation_strong = dict(
            batch_size=128
        )

        validation_weak = dict(
            batch_size=128
        )

        trainer = dict(
            max_epochs=100,
            check_val_every_n_epoch=10
        )

        weak_wrapper = dict(
            audioset_classes=527,
            rnn_layers=4,
            rnn_dim=256, # increase this
            seq_len=250
        )
        strong_wrapper = dict(
            audioset_classes=527,
            n_classes=456,
            rnn_layers=4,
            rnn_dim=256,
            seq_len=250
        )

        optimizer = dict(
            adamw=False,
            weight_decay=0.0,
            lr=0.00002,
            lr_pt_scaler=1.0,
            num_warmup_steps=0
        )

        passt_mel = dict(
            fmin_aug_range=10,
            fmax_aug_range=2000,
            freqm=0,
            timem=0
        )

        # patchout tends to reduce the performance; overwrite default of 50
        passt = dict(
            frame_patchout=0,
            wandb_id='ngagmung'
        )

    pass


