import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
import einops

from src.model.src_lightvoc.utils import init_weights

from src.utils.mel import MelSpectrogram

from mamba_ssm import Mamba

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class Generator(torch.nn.Module):
    def __init__(self, h, mel_config):
        super(Generator, self).__init__()
        self.h = h
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            _ups = nn.ModuleList()
            for _i, (_u, _k) in enumerate(zip(u, k)):
                # in_channel = h.upsample_initial_channel // (2 ** i)
                # out_channel = h.upsample_initial_channel // (2 ** (i + 1))
                _ups.append(weight_norm(
                    ConvTranspose1d(h.upsample_initial_channel, h.upsample_initial_channel, _k, _u, padding=(_k - _u) // 2)))
            self.ups.append(_ups)

        self.mamba_layers = nn.ModuleList()
        self.conv_post = nn.ModuleList()
        for i in range(self.num_upsamples):
            # ch = h.upsample_initial_channel // (2 ** (i + 1))

            m = Mamba(
                d_model=h.upsample_initial_channel,  # Соответствует input_dim Conformer
                d_state=64,  # Размер скрытого состояния (SSM)
                d_conv=4,  # Соответствует depthwise_conv_kernel_size=31 из Conformer
                expand=2,  # Коэффициент расширения внутренних размерностей
                bias=False,  # Включить смещение для лучшей аппроксимации
                layer_idx=i  # Задаём индекс слоя
            )
            self.mamba_layers.append(m)

            if self.h.projection_filters[i] != 0:
                self.conv_post.append(
                    weight_norm(
                        Conv1d(
                            h.upsample_initial_channel, self.h.projection_filters[i],
                            self.h.projection_kernels[i], 1, padding=self.h.projection_kernels[i] // 2
                        )))
            else:
                self.conv_post.append(torch.nn.Identity())

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        self.mel_extractor = MelSpectrogram(mel_config)

    def forward(self, x, inference_params=None):
        # input 1 x 80 x time
        # pre_conv
        x = self.conv_pre(x)  # 80 upsample to upsample_initial_channel
        # after conv pre 1 x 512 x time
        self.mamba_layers.to(x.device)

        outs = []
        for i, (ups, mamba_layer, conv_post) in enumerate(zip(self.ups, self.mamba_layers, self.conv_post)):
            x = F.leaky_relu(x, LRELU_SLOPE) # in avocodo use 0.2 slope
            for _ups in ups:
                x = _ups(x)

            x = einops.rearrange(x, 'b f t -> b t f')
            x = mamba_layer(x, inference_params=inference_params)
            x = einops.rearrange(x, 'b t f -> b f t')  # 1 x 512 x time

            if inference_params is None:
                _x = F.leaky_relu(x)
                _x = conv_post(_x)

                _spec = torch.exp(_x[:, :self.post_n_fft // 2 + 1, :])
                _phase = torch.sin(_x[:, self.post_n_fft // 2 + 1:, :])

                _audio = self.mel_extractor.inverse(_spec, _phase)

                _audio = torch.tanh(_audio)
                outs.append(_audio)
            else: # only in first layer
                x = conv_post(x) # identity
        return outs

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
