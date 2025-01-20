from typing import List


import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm


from src.model.src_lightvoc.pqmf import PQMF

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)



class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", use_spectral_norm=False):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))),
        ])

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        with torch.no_grad():
            y = y.squeeze(1)
            y = stft(y, self.fft_size, self.shift_size, self.win_length, self.window.to(y.get_device()))
        y = y.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap


class MRSD(torch.nn.Module):

    """From https://github.com/rishikksh20/UnivNet-pytorch/blob/f90c0123a04ed446093e245f164043db06dd8765/discriminator.py#L13"""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window"):

        super(MRSD, self).__init__()
        self.discriminators = nn.ModuleList([
            SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
            SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
            SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window)
            ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



"""CoMBD and SBD from https://github.com/ncsoft/avocodo/tree/main/avocodo/models"""

class CoMBDBlock(torch.nn.Module):
    def __init__(
        self,
        h_u: List[int],
        d_k: List[int],
        d_s: List[int],
        d_d: List[int],
        d_g: List[int],
        d_p: List[int],
        op_f: int,
        op_k: int,
        op_g: int,
        use_spectral_norm=False
    ):
        super(CoMBDBlock, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm

        self.convs = nn.ModuleList()
        filters = [[1, h_u[0]]]
        for i in range(len(h_u) - 1):
            filters.append([h_u[i], h_u[i + 1]])
        for _f, _k, _s, _d, _g, _p in zip(filters, d_k, d_s, d_d, d_g, d_p):
            self.convs.append(norm_f(
                Conv1d(
                    in_channels=_f[0],
                    out_channels=_f[1],
                    kernel_size=_k,
                    stride=_s,
                    dilation=_d,
                    groups=_g,
                    padding=_p
                )
            ))
        self.projection_conv = norm_f(
            Conv1d(
                in_channels=filters[-1][1],
                out_channels=op_f,
                kernel_size=op_k,
                groups=op_g
            )
        )

    def forward(self, x):
        fmap = []
        for block in self.convs:
            x = block(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.projection_conv(x)
        return x, fmap


class CoMBD(torch.nn.Module):
    def __init__(self, h, pqmf_list=None, use_spectral_norm=False):
        super(CoMBD, self).__init__()
        self.h = h
        if pqmf_list is not None:
            self.pqmf = pqmf_list
        else:
            self.pqmf = [
                PQMF(*h["pqmf_config"]["lv2"]),
                PQMF(*h["pqmf_config"]["lv1"])
            ]

        self.blocks = nn.ModuleList()
        for _h_u, _d_k, _d_s, _d_d, _d_g, _d_p, _op_f, _op_k, _op_g in zip(
            h["combd_h_u"],
            h["combd_d_k"],
            h["combd_d_s"],
            h["combd_d_d"],
            h["combd_d_g"],
            h["combd_d_p"],
            h["combd_op_f"],
            h["combd_op_k"],
            h["combd_op_g"],
        ):
            self.blocks.append(CoMBDBlock(
                _h_u,
                _d_k,
                _d_s,
                _d_d,
                _d_g,
                _d_p,
                _op_f,
                _op_k,
                _op_g,
            ))

    def _block_forward(self, input, blocks, outs, f_maps):
        for x, block in zip(input, blocks):
            out, f_map = block(x)
            outs.append(out)
            f_maps.append(f_map)
        return outs, f_maps

    def _pqmf_forward(self, ys, ys_hat):
        # preprocess for multi_scale forward
        multi_scale_inputs = []
        multi_scale_inputs_hat = []

        for pqmf in self.pqmf:
            multi_scale_inputs.append(
                pqmf.to(ys[-1]).analysis(ys[-1])[:, :1, :]
            )
            multi_scale_inputs_hat.append(
                pqmf.to(ys[-1]).analysis(ys_hat[-1])[:, :1, :]
            )

        outs_real = []
        f_maps_real = []
        # real
        # for hierarchical forward
        outs_real, f_maps_real = self._block_forward(
            ys, self.blocks, outs_real, f_maps_real)
        # for multi_scale forward
        outs_real, f_maps_real = self._block_forward(
            multi_scale_inputs, self.blocks[:-1], outs_real, f_maps_real)

        outs_fake = []
        f_maps_fake = []
        # predicted
        # for hierarchical forward
        outs_fake, f_maps_fake = self._block_forward(
            ys_hat, self.blocks, outs_fake, f_maps_fake)
        # for multi_scale forward
        outs_fake, f_maps_fake = self._block_forward(
            multi_scale_inputs_hat, self.blocks[:-1], outs_fake, f_maps_fake)

        return outs_real, outs_fake, f_maps_real, f_maps_fake

    def forward(self, ys, ys_hat):
        outs_real, outs_fake, f_maps_real, f_maps_fake = self._pqmf_forward(
            ys, ys_hat)
        return outs_real, outs_fake, f_maps_real, f_maps_fake

class MDC(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_size,
        dilations,
        use_spectral_norm=False
    ):
        super(MDC, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.d_convs = nn.ModuleList()
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(
                norm_f(Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=_k,
                    dilation=_d,
                    padding=get_padding(_k, _d)
                ))
            )
        self.post_conv = norm_f(Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=strides,
            padding=get_padding(_k, _d)
        ))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        _out = None
        for _l in self.d_convs:
            _x = torch.unsqueeze(_l(x), -1)
            _x = F.leaky_relu(_x, 0.2)
            if _out is None:
                _out = _x
            else:
                _out = torch.cat([_out, _x], axis=-1)
        x = torch.sum(_out, dim=-1)
        x = self.post_conv(x)
        x = F.leaky_relu(x, 0.2)  # @@

        return x


class SBDBlock(torch.nn.Module):
    def __init__(
        self,
        segment_dim,
        strides,
        filters,
        kernel_size,
        dilations,
        use_spectral_norm=False
    ):
        super(SBDBlock, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList()
        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append([filters[i], filters[i + 1]])

        for _s, _f, _k, _d in zip(
            strides,
            filters_in_out,
            kernel_size,
            dilations
        ):
            self.convs.append(MDC(
                in_channels=_f[0],
                out_channels=_f[1],
                strides=_s,
                kernel_size=_k,
                dilations=_d,
                use_spectral_norm=use_spectral_norm
            ))
        self.post_conv = norm_f(Conv1d(
            in_channels=_f[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=3 // 2
        ))  # @@

    def forward(self, x):
        fmap = []
        for _l in self.convs:
            x = _l(x)
            fmap.append(x)
        x = self.post_conv(x)  # @@

        return x, fmap


class MDCDConfig:
    def __init__(self, h):
        self.pqmf_params = h["pqmf_config"]["sbd"]
        self.f_pqmf_params = h["pqmf_config"]["fsbd"]
        self.filters = h["sbd_filters"]
        self.kernel_sizes = h["sbd_kernel_sizes"]
        self.dilations = h["sbd_dilations"]
        self.strides = h["sbd_strides"]
        self.band_ranges = h["sbd_band_ranges"]
        self.transpose = h["sbd_transpose"]
        self.segment_size = h["segment_size"]


class SBD(torch.nn.Module):
    def __init__(self, h, use_spectral_norm=False):
        super(SBD, self).__init__()
        self.config = MDCDConfig(h)
        self.pqmf = PQMF(
            *self.config.pqmf_params
        )
        if True in h["sbd_transpose"]:
            self.f_pqmf = PQMF(
                *self.config.f_pqmf_params
            )
        else:
            self.f_pqmf = None

        self.discriminators = torch.nn.ModuleList()

        for _f, _k, _d, _s, _br, _tr in zip(
            self.config.filters,
            self.config.kernel_sizes,
            self.config.dilations,
            self.config.strides,
            self.config.band_ranges,
            self.config.transpose
        ):
            if _tr:
                segment_dim = self.config.segment_size // _br[1] - _br[0]
            else:
                segment_dim = _br[1] - _br[0]

            self.discriminators.append(SBDBlock(
                segment_dim=segment_dim,
                filters=_f,
                kernel_size=_k,
                dilations=_d,
                strides=_s,
                use_spectral_norm=use_spectral_norm
            ))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        y_in = self.pqmf.analysis(y)
        y_hat_in = self.pqmf.analysis(y_hat)
        if self.f_pqmf is not None:
            y_in_f = self.f_pqmf.analysis(y)
            y_hat_in_f = self.f_pqmf.analysis(y_hat)

        for d, br, tr in zip(
            self.discriminators,
            self.config.band_ranges,
            self.config.transpose
        ):
            if tr:
                _y_in = y_in_f[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in_f[:, br[0]:br[1], :]
                _y_in = torch.transpose(_y_in, 1, 2)
                _y_hat_in = torch.transpose(_y_hat_in, 1, 2)
            else:
                _y_in = y_in[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in[:, br[0]:br[1], :]
            y_d_r, fmap_r = d(_y_in)
            y_d_g, fmap_g = d(_y_hat_in)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class LightVocMultiDiscriminator(nn.Module):
    def __init__(self, combd_params, sbd_params, mrsd_params):
        super().__init__()
        self.combd = CoMBD(**combd_params)
        self.sbd = SBD(**sbd_params)
        self.mrsd = MRSD(**mrsd_params)

    def forward(self, audio):
        # audio: [B, 1, T]
        out_combd, feats_combd = self.combd(audio)
        out_sbd, feats_sbd = self.sbd(audio)
        out_mrsd, feats_mrsd = self.mrsd(audio)
        return {
            "CoMBD": {"output": out_combd, "features": feats_combd},
            "SBD": {"output": out_sbd, "features": feats_sbd},
            "MRSD": {"output": out_mrsd, "features": feats_mrsd},
        }

