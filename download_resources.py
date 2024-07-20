import os.path
import zipfile
from torch.hub import download_url_to_file

source_url = "https://github.com/CPJKU/cpjku_dcase24/releases/download/files/"
target_folder = "resources"


def download_zip(fname, subfolder=None):
    if subfolder is not None:
        tgt_folder = os.path.join(target_folder, subfolder)
    else:
        tgt_folder = target_folder
    file = fname
    file_zip = file + ".zip"

    source_file = os.path.join(source_url, file_zip)
    target_file = os.path.join(tgt_folder, file)
    target_file_zip = os.path.join(tgt_folder, file_zip)
    if not os.path.exists(target_file):
        if not os.path.isfile(target_file_zip):
            print(f"Downloading and extracting {fname}.")
            download_url_to_file(source_file, target_file_zip)
        with zipfile.ZipFile(target_file_zip, 'r') as zip_ref:
            zip_ref.extractall(tgt_folder)
        os.remove(target_file_zip)


def download_file(fname, subfolder=None):
    if subfolder is not None:
        tgt_folder = os.path.join(target_folder, subfolder)
    else:
        tgt_folder = target_folder
    src = os.path.join(source_url, fname)
    tgt = os.path.join(tgt_folder, fname)
    if not os.path.isfile(tgt):
        print("Download ", fname, ".")
        download_url_to_file(src, tgt)

# pseudo-labels
download_zip("pseudo-labels")

# device impulse responses
download_zip("dirs")

os.makedirs(os.path.join(target_folder, "pretrained_models"), exist_ok=True)

# external pre-trained checkpoint
download_file("beats_as.pt", subfolder="pretrained_models")

# external pre-trained checkpoint
download_file("atst_as.ckpt", subfolder="pretrained_models")

# AudioSet strong pre-trained checkpoints
download_file("atst_as_strong.ckpt", subfolder="pretrained_models")
download_file("passt_as_strong.ckpt", subfolder="pretrained_models")

# Stage 1 pre-trained checkpoints
download_file("atst_stage1.ckpt", subfolder="pretrained_models")
download_file("passt_stage1.ckpt", subfolder="pretrained_models")
download_file("beats_stage1.ckpt", subfolder="pretrained_models")

# Stage 2 pre-trained checkpoints
download_file("atst_stage2.ckpt", subfolder="pretrained_models")
download_file("passt_stage2.ckpt", subfolder="pretrained_models")
download_file("beats_stage2.ckpt", subfolder="pretrained_models")


os.makedirs(os.path.join(target_folder, "exclude"), exist_ok=True)

# for excluding overlapping files in data setup
download_file("unlabeled_exclude.tsv", subfolder="exclude")
download_file("weak_exclude.tsv", subfolder="exclude")
download_file("audioset_strong_exclude.tsv", subfolder="exclude")

os.makedirs(os.path.join(target_folder, "external_strong_split"), exist_ok=True)

# for the external strong split we used
download_file("external_audioset_strong_train.tsv", subfolder="external_strong_split")
download_file("external_audioset_strong_eval.tsv", subfolder="external_strong_split")
download_file("external_audioset_strong_dur.tsv", subfolder="external_strong_split")
