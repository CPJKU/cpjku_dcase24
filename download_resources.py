import os.path
import zipfile
from torch.hub import download_url_to_file

source_url = "https://github.com/CPJKU/cpjku_dcase24/releases/download/files/"
target_folder = "resources"

# pseudo-labels
file = "pseudo-labels.zip"
file_ext = "pseudo-labels"
source_file = os.path.join(source_url, file)
target_file = os.path.join(target_folder, file)
target_file_ext = os.path.join(target_folder, file_ext)
if not os.path.exists(target_file_ext):
    if not os.path.isfile(target_file):
        print("Downloading and extracting pseudo-labels.")
        download_url_to_file(source_file, target_file)
    with zipfile.ZipFile(target_file, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    os.remove(target_file)


def download_ckpt(fname):
    src = os.path.join(source_url, fname)
    tgt = os.path.join(target_folder, fname)
    if not os.path.isfile(tgt):
        print("Download ", fname, ": ")
        download_url_to_file(src, tgt)


# external pre-trained checkpoints
download_ckpt("atst_base.ckpt")

# AudioSet pre-trained checkpoints
download_ckpt("atst_as_strong.ckpt")

# Stage 1 pre-trained checkpoints
download_ckpt("atst_stage1.ckpt")
