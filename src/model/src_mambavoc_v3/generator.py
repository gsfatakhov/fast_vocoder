import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, LayerNorm, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
import einops

from src.model.src_lightvoc.utils import get_padding

from src.model.src_hifigan.utils import init_weights, get_padding

from mamba_ssm import Mamba

LRELU_SLOPE = 0.1

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class MambaBlocks(torch.nn.Module):
    def __init__(self, length, begin_idx, channels):
        super(MambaBlocks, self).__init__()
        self.mamba_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for local_idx in range(length):
            m = Mamba(
                d_model=channels,  # Соответствует input_dim Conformer
                d_state=64,  # Размер скрытого состояния (SSM)
                d_conv=4,  # Соответствует depthwise_conv_kernel_size=31 из Conformer
                expand=2,  # Коэффициент расширения внутренних размерностей
                bias=False,  # Включить смещение для лучшей аппроксимации
                layer_idx=begin_idx + local_idx  # Задаём индекс слоя
            )
            self.mamba_layers.append(m)
            self.layer_norms.append(LayerNorm(channels))

    def forward(self, x, inference_params=None):
        x = einops.rearrange(x, 'b f t -> b t f')

        for mamba, layer_norm in zip(self.mamba_layers, self.layer_norms):
            y = layer_norm(x)
            y = mamba(y, inference_params=inference_params)
            x = x + y

        x = einops.rearrange(x, 'b t f -> b f t')  # 1 x 512 x time
        return x

    def to(self, device):
        super().to(device)
        self.mamba_layers = self.mamba_layers.to(device)
        self.layer_norms = self.layer_norms.to(device)
        return self

class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        self.mamba_layers = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock1(h, ch, k, d))

            self.mamba_layers.append(MambaBlocks(h.M, i * h.M, ch))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.conv_post.apply(init_weights)

    def to(self, device):
        super().to(device)
        self.mamba_layers = self.mamba_layers.to(device)
        return self

    def forward(self, x, inference_params=None):
        # input 1 x 80 x time
        # pre_conv
        x = self.conv_pre(x)  # 80 upsample to upsample_initial_channel
        # after conv pre 1 x 512 x time

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            x = self.mamba_layers[i](x, inference_params=inference_params)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)

        x = self.conv_post(x)  # 1 x 1 x time
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
