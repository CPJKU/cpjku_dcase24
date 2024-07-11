# cpjku_dcase24

This is the repo that will soon be holding the full code for our [DCASE 2024 Task 4](https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels) submission.

The repository is currently **under construction** and will be cleaned for public use. It currently contains:
* Datasets and Dataloading
* Models
* Stage 1 Training

We will further include:
* Instructions how to get the datasets
* Stage 2 Training
* AudioSet strong pre-training
* Pre-Trained Models
* A cleaner version of the code

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

## Example Command: Stage 1 Training w. ATST

```bash
CUDA_VISIBLE_DEVICES=0 python -m ex_stage1 with arch=atst_frame loss_weights="(0.5, 0.25, 0.12, 0.1, 0.1, 1.5)" trainer.max_epochs=200 optimizer.crnn_lr=0.0005 filter_augment.apply=0 training.pseudo_labels_name=final mix_augment.apply_mixstyle=0 wandb.name=s1.i1,atst
```

## Example Command: Stage 2 Training w. ATST

