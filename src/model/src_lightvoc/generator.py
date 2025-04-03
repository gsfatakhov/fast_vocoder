import torch
import torch.nn as nn

from src.model.src_lightvoc.conformer import Conformer


from torch.nn.utils import weight_norm, remove_weight_norm

from torch.nn import Conv1d

import torch.nn.functional as F


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

class LightVocGenerator(torch.nn.Module):
    def __init__(self, h):
        super(LightVocGenerator, self).__init__()
        self.h = h
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        self.conv_pre.apply(init_weights)

        self.conformer = Conformer(input_dim=h.upsample_initial_channel, num_heads=8, ffn_dim=256, num_layers=2, depthwise_conv_kernel_size=31, dropout=0.1)

        self.post_n_fft = h.gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(256, self.post_n_fft + 2, 7, 1, padding=3))
        self.conv_post.apply(init_weights)


    def forward(self,  x):

        time = x.shape[-1]
        # constuct length tensor id batch size is x.shape[0]
        length = torch.tensor([time] * x.shape[0], device=x.device)

        # input 1 x 80 x time
        # pre_conv
        x = self.conv_pre(x) # 80 upsample to upsample_initial_channel

        # x = einops.rearrange(x, 'b f t -> b t f')
        x = x.permute(0, 2, 1)

        x, _ = self.conformer(x, length)

        # x = einops.rearrange(x, 'b t f -> b f t') #1 x 512 x time
        x = x.permute(0, 2, 1)

        x = F.leaky_relu(x)

        x = self.conv_post(x)

        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])

        return spec, phase

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

