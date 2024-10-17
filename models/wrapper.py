import torch
import torch.nn as nn
import os
from torch import autocast
import torchaudio

from models.baseline.RNN import BidirectionalGRU
from models.baseline.CNN import CNN
from configs import PRETRAINED_MODELS


first_RUN = True


class AudiosetWrapper(nn.Module):
    def __init__(self, model, audioset_classes=527, embed_dim=768, seq_len=250,
                 use_attention_head=False, pretrained_name=None):
        super(AudiosetWrapper, self).__init__()
        self.model = model
        self.pretrained_name = pretrained_name

        self.seq_len = seq_len

        self.num_features = embed_dim
        self.audioset_classes = audioset_classes

        self.head_linear_layer = nn.Linear(self.num_features, self.audioset_classes)
        self.att_linear_layer = nn.Linear(self.num_features, self.audioset_classes)
        self.use_attention_head = use_attention_head

        if self.pretrained_name:
            self.load_model()

        if not self.use_attention_head:
            del self.att_linear_layer

    def load_model(self):
        pretrained_weights = torch.load(os.path.join(PRETRAINED_MODELS, self.pretrained_name + ".ckpt"),
                                        map_location="cpu")["state_dict"]
        state_dict_keys = pretrained_weights.keys()
        is_strong_pretrained = any([k.startswith('net_strong') for k in state_dict_keys])
        if is_strong_pretrained:
            is_atst = any([".atst." in k for k in state_dict_keys])
            if is_atst:
                pretrained_weights = {"model." + ".".join(k.split(".")[2:]): v for k, v in pretrained_weights.items() if
                                      k.startswith("net_strong.model")}
                assert len(set(self.state_dict().keys()).difference(set(pretrained_weights.keys()))) == 4, \
                    f"Something went with loading AudioSet strong pre-trained model. Expected: {list((self.state_dict().keys()))} but got: {list(pretrained_weights.keys())}. Difference: {set(self.state_dict().keys()).difference(set(pretrained_weights.keys()))}"
            else:
                if 'net_strong.model.cls_token' in pretrained_weights:
                    offset = 1 # use offset of one for models that were trained w/o AudioSet wrapper
                    allowed_dif = 4
                    print("Loading Passt trained without AudioSet Wrapper")
                else:
                    offset = 2
                    allowed_dif = 2
                pretrained_weights = {".".join(k.split(".")[offset:]): v for k, v in pretrained_weights.items() if
                      k.startswith("net_strong.model")}
                assert len(set(self.state_dict().keys()).difference(set(pretrained_weights.keys()))) == allowed_dif,\
                    f"Something went with loading AudioSet strong pre-trained model. Expected: {list((self.state_dict().keys()))} but got: {list(pretrained_weights.keys())}. Difference: {set(self.state_dict().keys()).difference(set(pretrained_weights.keys()))}"
            self.load_state_dict(pretrained_weights, strict=False)
        else:
            pretrained_weights = {k[4:]: v for k, v in pretrained_weights.items() if k[:4] == "net."}
            self.load_state_dict(pretrained_weights)
        print("Loaded model successfully. pretrained_name:", self.pretrained_name)

    def forward(self, x):
        x = self.model(x)

        if x.size(-2) != self.seq_len:
            # adapt seq len
            x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.seq_len).transpose(1, 2)

        logits = self.head_linear_layer(x)

        if self.use_attention_head:
            strong = torch.sigmoid(logits)
            sof = torch.softmax(self.att_linear_layer(x), dim=-1)
            sof = torch.clamp(sof, min=1e-7, max=1)

            weak = (strong * sof).sum(1) / sof.sum(1)
            return strong.transpose(1, 2), weak
        return logits


class Task4RNNWrapper(nn.Module):
    def __init__(self, model, audioset_classes=527, seq_len=250, embed_dim=768, rnn_type="BGRU",
                 rnn_dim=256, rnn_layers=2, n_classes=10, pretrained_name=None):
        super(Task4RNNWrapper, self).__init__()
        self.audioset_classes = audioset_classes
        self.seq_len = seq_len
        self.num_features = embed_dim
        self.n_classes = n_classes
        self.model = AudiosetWrapper(model, audioset_classes, embed_dim, seq_len, use_attention_head=False,
                                     pretrained_name=pretrained_name)

        print(rnn_layers)
        if rnn_type == "BGRU":
            self.rnn = BidirectionalGRU(
                n_in=audioset_classes,
                n_hidden=rnn_dim,
                dropout=0,
                num_layers=rnn_layers,
            )
        else:
            NotImplementedError("Only BGRU supported for now")

        self.sigmoid_dense = nn.Linear(rnn_dim * 2, self.n_classes)
        self.softmax_dense = nn.Linear(rnn_dim * 2, self.n_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.rnn(x)

        strong = torch.sigmoid(self.sigmoid_dense(x))
        sof = torch.softmax(self.softmax_dense(x), dim=-1)
        sof = torch.clamp(sof, min=1e-7, max=1)

        weak = (strong * sof).sum(1) / sof.sum(1)

        return strong.transpose(1, 2), weak


class Task4CRNNEmbeddingsWrapper(nn.Module):
    def __init__(self,
                 as_model,
                 pretrained_name=None,
                 audioset_classes=527,
                 no_wrapper=False,
                 n_in_channel=1,
                 nclass=27,
                 activation="glu",
                 dropout=0.0,
                 attention=True,
                 n_RNN_cell=128,
                 n_layers_RNN=2,
                 dropout_recurrent=0,
                 dropstep_recurrent=0.0,
                 dropstep_recurrent_len=16,
                 embedding_size=768,
                 model_init_id=None,
                 model_init_mode="teacher",
                 kernel_size=[3, 3, 3, 3, 3, 3, 3],
                 padding=[1, 1, 1, 1, 1, 1, 1],
                 stride=[1, 1, 1, 1, 1, 1, 1],
                 nb_filters=[16, 32, 64, 128, 128, 128, 128],
                 pooling=[[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
                 embed_pool="aap",
                 interpolation_mode="nearest"
                 ):
        super(Task4CRNNEmbeddingsWrapper, self).__init__()
        self.dropstep_recurrent = dropstep_recurrent
        self.dropstep_recurrent_len = dropstep_recurrent_len
        self.n_in_channel = n_in_channel
        self.attention = attention
        n_in_cnn = n_in_channel
        self.cnn = CNN(
            n_in_channel=n_in_cnn,
            activation=activation,
            conv_dropout=dropout,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            nb_filters=nb_filters,
            pooling=pooling
        )

        self.embed_pool = embed_pool

        nb_in = self.cnn.nb_filters[-1]
        nb_in = nb_in * n_in_channel
        self.rnn = BidirectionalGRU(
            n_in=n_RNN_cell,
            n_hidden=n_RNN_cell,
            dropout=dropout_recurrent,
            num_layers=n_layers_RNN,
        )

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sigmoid = nn.Sigmoid()

        self.dense_softmax = nn.Linear(n_RNN_cell * 2, nclass)
        self.softmax = nn.Softmax(dim=-1)

        if no_wrapper:
            # use transformer directly, no wrapper
            self.as_model = as_model
            self.as_model_dim = embedding_size
        else:
            # load wrapper from audioset pre-training
            self.as_model = AudiosetWrapper(as_model, audioset_classes, embedding_size, use_attention_head=False,
                                            pretrained_name=pretrained_name)
            self.as_model_dim = audioset_classes

        self.interpolation_mode = interpolation_mode
        if self.embed_pool == "aap":
            # nothing to initialize
            pass
        elif self.embed_pool == "conv":
            self.pool_assumption = lambda cnn_frames, embed_frames: cnn_frames == 156 and embed_frames == 250
            kernel_size = 250 - 156 + 1
            conv = nn.Conv1d(in_channels=self.as_model_dim, out_channels=self.as_model_dim,
                                       kernel_size=kernel_size, padding=0, groups=self.as_model_dim)

            # Define a custom transpose layer
            class TransposeLayer(nn.Module):
                def __init__(self, dim0, dim1):
                    super(TransposeLayer, self).__init__()
                    self.dim0 = dim0
                    self.dim1 = dim1

                def forward(self, x):
                    return x.transpose(self.dim0, self.dim1)

            with torch.no_grad():
                conv.weight.fill_(1.0 / kernel_size)
                if conv.bias is not None:
                    conv.bias.fill_(0.0)

            self.pool_conv = nn.Sequential(
                conv
            )
        elif self.embed_pool == "int":
            # nothing to initialize
            pass
        else:
            raise ValueError("No such embed pooling type: ", self.embed_pool)

        self.cat_tf = torch.nn.Linear(nb_in + self.as_model_dim, n_RNN_cell)
        self.nclass = nclass

        self.as_model.eval()
        for param in self.as_model.parameters():
            param.detach_()

        if model_init_id:
            # for loading the full model including the wrapper (e.g., load S1 model for S2)
            ckpt = os.path.join(PRETRAINED_MODELS, model_init_id + ".ckpt")
            if model_init_mode == "teacher":
                print("Loaded teacher from ckpt: ", ckpt)
                state_dict = torch.load(ckpt, map_location="cpu")["teacher"]
            else:
                print("Loaded student from ckpt: ", ckpt)
                state_dict = torch.load(ckpt, map_location="cpu")["student"]
            self.load_state_dict(state_dict, strict=True)

        self.first = True

    def forward(self, x, pretrain_x, pad_mask=None, classes_mask=(None,), return_strong_logits=False):
        # conv features
        if self.first:
            print("CNN input:", x.size())
        x = x.transpose(2, 3)
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)  # [bs, frames, chan]
        if self.first:
            print("CNN output:", x.size())
            print("Embedding model intput:", pretrain_x.size())

        # rnn features
        embeddings = self.as_model(pretrain_x).transpose(1, 2)
        if self.first:
            print("Embedding model output:", embeddings.size())

        if self.embed_pool == "aap":
            embeddings = torch.nn.functional.adaptive_avg_pool1d(embeddings, frames).transpose(1, 2)  # 156
        elif self.embed_pool == "conv":
            assert self.pool_assumption(frames, embeddings.size(-1))
            embeddings = self.pool_conv(embeddings).transpose(1, 2)  # 156
        elif self.embed_pool == "int":
            embeddings = torch.nn.functional.interpolate(embeddings, size=frames,
                                                         mode=self.interpolation_mode).transpose(1, 2)  # 156
        else:
            raise ValueError("No such embed pooling type: ", self.embed_pool)

        if self.first:
            print("Embedding model pooled:", embeddings.size())
        self.first = False

        # dropstep
        if self.dropstep_recurrent and self.training:
            dropstep = torchaudio.transforms.TimeMasking(
                self.dropstep_recurrent_len, True, self.dropstep_recurrent
            )
            x = dropstep(x.transpose(1, -1)).transpose(1, -1)
            embeddings = dropstep(embeddings.transpose(1, -1)).transpose(1, -1)

        x = self.cat_tf(self.dropout(torch.cat((x, embeddings), -1)))

        x = self.rnn(x)
        x = self.dropout(x)

        if return_strong_logits:
            return self.dense(x)

        with autocast(enabled=False, device_type='cuda'):
            x = x.float()
            if classes_mask is None or torch.is_tensor(classes_mask):
                return self._get_logits(x, pad_mask, classes_mask)

            # list of masks
            out_args = []
            for mask in classes_mask:
                out_args.append(self._get_logits(x, pad_mask, mask))
        return tuple(out_args)

    def _get_logits_one_head(self, x, pad_mask, dense, dense_softmax, classes_mask=None):
        strong = dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if classes_mask is not None:
            classes_mask = ~classes_mask[:, None].expand_as(strong)
        if self.attention in [True, "legacy"]:
            sof = dense_softmax(x)  # [bs, frames, nclass]
            if not pad_mask is None:
                sof = sof.masked_fill(pad_mask.transpose(1, 2), -1e30)  # mask attention

            if classes_mask is not None:
                # mask the invalid classes, cannot attend to these
                sof = sof.masked_fill(classes_mask, -1e30)
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)

        if classes_mask is not None:
            # mask invalid
            strong = strong.masked_fill(classes_mask, 0.0)
            weak = weak.masked_fill(classes_mask[:, 0], 0.0)

        return strong.transpose(1, 2), weak

    def _get_logits(self, x, pad_mask, classes_mask=None):
        out_strong = []
        out_weak = []
        if isinstance(self.nclass, (tuple, list)):
            # instead of masking the softmax we can have multiple heads for each dataset:
            # maestro_synth, maestro_real and desed.
            # not sure which approach is better. We must try.
            for indx, c_classes in enumerate(self.nclass):
                dense_softmax = self.dense_softmax[indx] if hasattr(self, "dense_softmax") else None
                c_strong, c_weak = self._get_logits_one_head(
                    x, pad_mask, self.dense[indx], dense_softmax, classes_mask
                )
                out_strong.append(c_strong)
                out_weak.append(c_weak)

            # concatenate over class dimension
            return torch.cat(out_strong, 1), torch.cat(out_weak, 1)
        else:
            dense_softmax = self.dense_softmax if hasattr(self,
                                                          "dense_softmax") else None
            return self._get_logits_one_head(
                x, pad_mask, self.dense, dense_softmax, classes_mask
            )


