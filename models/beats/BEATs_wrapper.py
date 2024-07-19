import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from models.beats.BEATs import BEATsModel, BEATsConfig, BEATs


class BEATsWrapper(nn.Module):
    def __init__(self, cfg_path):
        super().__init__()
        # load the pre-trained checkpoint
        checkpoint = torch.load(cfg_path)
        cfg = BEATsConfig(checkpoint["cfg"])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint["model"])
        self.model = BEATs_model
        self.ckpt = checkpoint

    def preprocess(self, x):
        mel = self.model.preprocess(x)
        mel = mel.unsqueeze(1).transpose(2, 3)
        return mel

    def forward(self, x):
        x = x.transpose(2, 3)
        features = self.model.extract_features(x, do_preprocess=False)[0]
        return features
