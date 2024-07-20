# CP Submission to DCASE'24 Task 4

This is the repo that holds the code for our top-ranked [DCASE 2024 Task 4](https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels-results) submission.

The setup is described in the paper [Multi-Iteration Multi-Stage Fine-Tuning of Transformers for Sound Event Detection with Heterogeneous Datasets](https://arxiv.org/abs/2407.12997).

The repository currently contains:
* Datasets and Dataloading
* ATST, fPaSST, BEATs
* Stage 1 Training
* Stage 2 Training
* Pre-Trained Model Checkpoints
* Pseudo-Labels 
* Commands for computing and storing predictions
* The command for evaluating stored predictions, including cSEBBs post-processing

In the near further, we will include:
* AudioSet strong pre-training

The codebase is still a bit messy (also due to the complex task setup). We will clarify and clean code pieces further based on requests.
Feel free to open an issue or drop me an email: florian.schmid@jku.at

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

## Example Commands Iteration 2, Stage 1:

All commands below are specified to fit on a GPU with 11-12 GB memory. If you have more GPU memory reduce ```accumulate_grad_batches``` and increase the batch size.

ATST:
```bash
python -m ex_stage1 with arch=atst_frame loss_weights="(0.5, 0.25, 0.12, 0.1, 0.1, 1.5)" trainer.max_epochs=200 optimizer.crnn_lr=0.0005 filter_augment.apply=0 mix_augment.apply_mixstyle=0 ssl_no_class_mask=1 wandb.name=s1.i2,atst
```

fPaSST:
```bash
python -m ex_stage1 with arch=fpasst loss_weights="(0.5, 0.25, 0.12, 0.1, 0.1, 1.5)" trainer.max_epochs=200 optimizer.crnn_lr=0.0005 filter_augment.apply=0 exclude_maestro_weak_ssl=1 t4_wrapper.embed_pool=int t4_wrapper.interpolation_mode=nearest-exact mix_augment.apply_mixstyle=0 passt_mel.fmax_aug_range=2000 wandb.name=s1.i2,pfasst
```

BEATs:
```bash
python -m ex_stage1 with arch=beats loss_weights="(0.5, 0.25, 0.08, 0.1, 0.1, 1.5)" trainer.max_epochs=200 optimizer.crnn_lr=0.0005 filter_augment.apply=0 t4_wrapper.no_wrapper=1 ssl_no_class_mask=1 trainer.accumulate_grad_batches=4 "training.batch_sizes=(3, 3, 3, 5, 5)" t4_wrapper.embed_pool=int t4_wrapper.interpolation_mode=nearest-exact wandb.name=s1.i2,beats
```

## Example Command Iteration 2, Stage 2:

ATST:

```bash
python -m ex_stage2 with arch=atst_frame trainer.accumulate_grad_batches=8 loss_weights="(12, 3, 0.25, 1, 60, 0)" t4_wrapper.model_init_id=atst_stage1 optimizer.pt_lr_scale=0.5 optimizer.cnn_lr=1e-5 optimizer.rnn_lr=1e-4 freq_warp.include_maestro=1 optimizer.adamw=1 optimizer.weight_decay=1e-3 wandb.name=s2.i2,atst
```
fPaSST:

```bash
python -m ex_stage2 with arch=fpasst trainer.accumulate_grad_batches=8 loss_weights="(12, 3, 0.25, 1, 60, 0)" t4_wrapper.model_init_id=passt_stage1 freq_warp.apply=0 optimizer.adamw=1 optimizer.weight_decay=1e-3 passt_mel.fmin_aug_range=1 passt_mel.fmax_aug_range=2000 optimizer.cnn_lr=5e-5 optimizer.rnn_lr=5e-4 exclude_maestro_weak_ssl=1 t4_wrapper.embed_pool=int t4_wrapper.interpolation_mode=nearest-exact wandb.name=s2.i2,passt
```

BEATs:
```bash
python -m ex_stage2 with arch=beats loss_weights="(12, 3, 0.25, 1, 60, 0)" trainer.accumulate_grad_batches=36 "training.batch_sizes=(3, 2, 2, 3, 3)" t4_wrapper.model_init_id=beats_stage1 freq_warp.include_maestro=1 optimizer.adamw=1 optimizer.weight_decay=1e-3 t4_wrapper.no_wrapper=1 optimizer.cnn_lr=5e-5 optimizer.rnn_lr=5e-4 t4_wrapper.embed_pool=int t4_wrapper.interpolation_mode=nearest-exact exclude_maestro_weak_ssl=1 wandb.name=s2.i2,beats
```

## Additional Infrastructure

We also provide the basic commands we used for computing and storing frame-wise predictions, ensembling predictions, applying post-processing, and preparing the submission.

```store_predictions``` is the command to compute and store frame-wise predictions of a model. These predictions are stored in form of 
hd5f files per dataset (the same format as pseudo-labels are stored). 

ATST:
```bash
python -m ex_stage2 store_predictions with arch=atst_frame t4_wrapper.model_init_id=atst_stage2
```

fPaSST:
```bash
python -m ex_stage2 store_predictions with arch=fpasst pretrained_name=passt_stage2
```

BEATs:
```bash
python -m ex_stage2 store_predictions with arch=beats t4_wrapper.no_wrapper=1 pretrained_name=beats_stage2
```

```ensemble_predictions``` combines multiple hdf5 files, resulting from the ```store_predictions``` command. This is useful when
submitting an ensemble, or when computing ensemble pseudo-labels to learn from in the 2nd iteration. 

Combine ATST, fPaSST and BEATs into an ensemble:
```bash
python -m ex_stage2 ensemble_predictions with pretrained_names='("atst_stage2", "passt_stage2", "beats_stage2")' out_name="ensemble_3"
```

Evaluate the stored predictions with cSEBBs postprocessing and generate submission csv:
```bash
python -m ex_stage2 eval_predictions with arch=atst_frame pretrained_name="ensemble_3"
```

The ```eval_predictions``` command leads to the following outcome printed to console:

maestro_real_dev mpAUC: 0.7375462583538749 

maestro_real_dev mpAUC_csebbs (should be the same): 0.7375462583538749

devtest psds1: 0.5480958592426525

devtest psds1_csebbs: 0.6406416548838889

eval_public psds1: 0.6308520777223674

eval_public psds1_csebbs: 0.7102015700340647

Storing predictions to resources/predictions/ensemble_3.




