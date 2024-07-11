import os.path
import zipfile
from torch.hub import download_url_to_file

source_url = "https://github.com/CPJKU/cpjku_dcase24/releases/download/files/"
target_folder = "resources"


def download_zip(fname):
    file = fname
    file_zip = file + ".zip"

    source_file = os.path.join(source_url, file_zip)
    target_file = os.path.join(target_folder, file)
    target_file_zip = os.path.join(target_folder, file_zip)
    if not os.path.exists(target_file):
        if not os.path.isfile(target_file_zip):
            print(f"Downloading and extracting {fname}.")
            download_url_to_file(source_file, target_file_zip)
        with zipfile.ZipFile(target_file_zip, 'r') as zip_ref:
            zip_ref.extractall(target_folder)
        os.remove(target_file_zip)


def download_ckpt(fname):
    src = os.path.join(source_url, fname)
    tgt = os.path.join(target_folder, fname)
    if not os.path.isfile(tgt):
        print("Download ", fname, ": ")
        download_url_to_file(src, tgt)

# pseudo-labels
download_zip("pseudo-labels")

# device impulse responses
download_zip("dirs")

# external pre-trained checkpoints
download_ckpt("atst_base.ckpt")

# AudioSet pre-trained checkpoints
download_ckpt("atst_as_strong.ckpt")

# Stage 1 pre-trained checkpoints
download_ckpt("atst_stage1.ckpt")
