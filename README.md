# CP Submission to DCASE'24 Task 4

This is the repo that will soon be holding the full code for our [DCASE 2024 Task 4](https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels-results) submission.

The setup is described in the paper [Multi-Iteration Multi-Stage Fine-Tuning of Transformers for Sound Event Detection with Heterogeneous Datasets](https://arxiv.org/abs/2407.12997).

The repository is currently **under construction** and will be cleaned for public use. It currently contains:
* Datasets and Dataloading
* ATST, fPaSST, BEATs
* Stage 1 Training
* Stage 2 Training
* Pre-Trained Model Checkpoints

We will further include:
* AudioSet strong pre-training
* cSEBBs postprocessing

## Data Setup

### AudioSet (strong)

1. Download the audio snippets by using the scripts provided by [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn).

2. Download the temporally-strong labels from the [official site](https://research.google.com/audioset/download_strong.html).

### DESED

Follow the official instructions [here](https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels). Missing files can be requested by contacting the DCASE task organizers.

Extra strongly-labeled data extracted from strongly-labeled AudioSet with classes mapped to DESED classes can be found [here](https://saoyear.github.io/post/downloading-real-waveforms-for-desed/).

### MAESTRO

Follow the official instructions [here](https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels).

### Summary

If you have collected all clips, you should have the following sets:

* **Synth**: 10,000 synthetic strongly-labeled clips
* **Synth_Val**: 2,500 synthetic strongly-labeled clips
* **Weak**: 1,578 weakly-labeled clips
* **Unlabeled**: 14,412 unlabeled clips
* **Strong**: 3,470 strongly-labeled real clips from AudioSet
* **External_Strong**: 7,383 external strongly-labeled real clips from AudioSet
* **Test**: 1,168 strongly-labeled real clips (for **testing**)
* **Real_MAESTRO_Train**: 7,503 real strongly-labeled clips from MAESTRO
* **Real_MAESTRO_Val**: 3,474 real strongly-labeled clips from MAESTRO (for **testing**)

Store the base path to the dataset in variable ```DATASET_PATH``` in the file [configs.py](configs.py). The structure of the 
dataset must match the structure in variable ```t4_paths``` in files [ex_stage1.py](ex_stage1.py) and [ex_stage2.py](ex_stage2.py).

## Environment Setup

1. If needed, create a new environment with python 3.9 and activate it:

```bash
conda create -n t4_24 python=3.9
conda activate t4_24
 ```

2. Install pytorch build that suits your system. For example:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or for cuda >= 12.1
pip3 install torch torchvision torchaudio 
```

3. Install Cython:

```
pip3 install Cython
```

4. Install the requirements:

 ```bash
pip3 install -r requirements.txt
 ```

5. Install minimp3 fork for faster mp3 decoding (needed for AudioSet training):

 ```bash
CFLAGS='-O3 -march=native' pip install https://github.com/f0k/minimp3py/archive/master.zip
 ```

6. Login to wandb:
Get token from <https://wandb.ai/authorize>

 ```bash
 wandb login
```

## Resources Setup

Pseudo-labels and pre-trained model checkpoints are available in [this GitHub release](https://github.com/CPJKU/cpjku_dcase24/releases/tag/files).

These files are intended to end up in the folder [resources](resources). We provide a script that is downloading all files at once from the release
and placing them in the [resources](resources) folder:

 ```bash
python download_resources.py
 ```

Alternatively, you can just download the files you need and want to work with.

## Example Commands Stage 1:

ATST:
```bash
python -m ex_stage1 with arch=atst_frame loss_weights="(0.5, 0.25, 0.12, 0.1, 0.1, 1.5)" trainer.max_epochs=200 optimizer.crnn_lr=0.0005 filter_augment.apply=0 mix_augment.apply_mixstyle=0 ssl_no_class_mask=1 wandb.name=s1.i1,atst
```

fPaSST:
```bash
python -m ex_stage1 with arch=fpasst loss_weights="(0.5, 0.25, 0.12, 0.1, 0.1, 1.5)" trainer.max_epochs=200 optimizer.crnn_lr=0.0005 filter_augment.apply=0 exclude_maestro_weak_ssl=1 t4_wrapper.embed_pool=int t4_wrapper.interpolation_mode=nearest-exact mix_augment.apply_mixstyle=0 passt_mel.fmax_aug_range=2000 wandb.name=s1.i1,pfasst
```

BEATs:
```bash
python -m ex_stage1 with arch=beats loss_weights="(0.5, 0.25, 0.08, 0.1, 0.1, 1.5)" trainer.max_epochs=200 optimizer.crnn_lr=0.0005 filter_augment.apply=0 t4_wrapper.no_wrapper=1 ssl_no_class_mask=1 trainer.accumulate_grad_batches=4 "training.batch_sizes=(3, 3, 3, 5, 5)" t4_wrapper.embed_pool=int t4_wrapper.interpolation_mode=nearest-exact wandb.name=s1.i1,beats
```

## Example Command: Stage 2 Training w. ATST

```bash
CUDA_VISIBLE_DEVICES=0 python -m ex_stage2 with arch=atst_frame trainer.accumulate_grad_batches=8 loss_weights="(12, 3, 0.25, 1, 60, 0)" t4_wrapper.model_init_id=atst_stage1 optimizer.pt_lr_scale=0.5 optimizer.cnn_lr=1e-5 optimizer.rnn_lr=1e-4 freq_warp.include_maestro=1 optimizer.adamw=1 optimizer.weight_decay=1e-3 wandb.name=s2.i1,atst
```
