import os
import torch
from models.atst.audio_transformer import FrameASTModel

from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from models.atst.frequency_warping import RandomResizeCrop


class ATSTMel(torch.nn.Module):
    def __init__(self, freq_scale=1.0) -> None:
        super().__init__()
        self.mel_transform = MelSpectrogram(
            16000,
            f_min=60,
            f_max=7800,
            hop_length=160,
            win_length=1024,
            n_fft=1024,
            n_mels=64
        )
        self.amp_to_db = AmplitudeToDB(stype="power", top_db=80)
        self.scaler = MinMax(min=-79.6482, max=50.6842)
        if freq_scale == 1.0:
            self.rrc = torch.nn.Identity()
        else:
            self.rrc = RandomResizeCrop(freq_scale=(freq_scale, 1.0))

    def amp2db(self, spec):
        return self.amp_to_db(spec).clamp(min=-50, max=80)

    def forward(self, audio):
        with torch.autocast(device_type="cuda", enabled=False):
            spec = self.mel_transform(audio)
        spec = self.scaler(self.amp2db(spec))
        if self.training:
            self.rrc(spec)
        spec = spec.unsqueeze(1)
        return spec


class ATSTWrapper(torch.nn.Module):
    def __init__(self, atst_path, *args, atst_dropout=0.0, **kwargs, ) -> None:
        super().__init__()
        self.atst = FrameASTModel(atst_dropout=atst_dropout)

        self.load_atst(atst_path)
        self.fake_length = torch.tensor([1001])
        self.cls_embed = None


    def set_cls_embed(self, cls_embed):
        self.cls_embed = cls_embed

    def forward(self, spec, other_emb=None):

        atst_x = self.atst.get_intermediate_layers(
            spec,
            self.fake_length.to(spec).repeat(len(spec)),
            1,
            scene=False,
            other_emb=other_emb,
        )
        # atst_x = atst_x.transpose(1, 2)
        return atst_x

    def load_atst(self, path=None):
        if path is None:
            pre_path = "/share/rk8/shared/dcase24/as_models/atst_as2M.ckpt"
            assert os.path.exists(pre_path), "Please make sure you have a default path to load ATST. Please change this path to the atst_as2M.ckpt that you downloaded."
            path = pre_path    # Change path to the atst_as2M.ckpt the downloaded checkpoint from the home page.
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        atst_state_dict = {}
        for k, v in state_dict.items():
            if "model.teacher.encoder." in k:
                if "encoder.norm." in k:
                    new_k = k.replace("model.teacher.encoder.norm", "norm_frame")
                elif "cls_token" in k:
                    continue
                else:
                    new_k = k.replace("model.teacher.encoder.", "")
                atst_state_dict[new_k] = v
            # C2F
            if "encoder.encoder.frame_encoder." in k:
                new_k = k.replace("encoder.encoder.frame_encoder.", "")
                atst_state_dict[new_k] = v
                continue
            if "encoder.encoder.teacher_module." in k:
                continue
            # ATST-Frame
            if "encoder.encoder." in k:
                new_k = k.replace("encoder.encoder.", "")
                atst_state_dict[new_k] = v

        self.atst.load_state_dict(atst_state_dict, strict=True)


class CustomAudioTransform:
    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinMax(CustomAudioTransform):
    def __init__(self, min, max):
        self.min=min
        self.max=max
    def __call__(self,input):
        min_,max_ = None,None
        if self.min is None:
            min_ = torch.min(input)
            max_ = torch.max(input)
        else:
            min_ = self.min
            max_ = self.max
        input = (input - min_)/(max_- min_) * 2. - 1.
        return input


if __name__ == '__main__':
    atst = ATSTWrapper("atstframe_base_as2M.ckpt")

    atst(torch.zeros(1, 16000))
    print("Done")
