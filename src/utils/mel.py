from dataclasses import dataclass

import torch
from torch import nn

import torchaudio

import librosa

import torch.nn.functional as F


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            center=False,
            pad_mode='reflect',
            power=config.power,
            normalized=False,
            onesided=True,
            norm=None  # нормировка базиса перебьём ниже
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: [B, T]
        :return: [B, n_mels, T_out]
        """
        if audio.dim() == 3 and audio.size(1) == 1:
            audio = audio.squeeze(1)

        # 2) Добавляем ручной паддинг как в референсном коде
        pad_amount = (self.config.n_fft - self.config.hop_length) // 2
        # pad = (left, right)
        audio = F.pad(audio.unsqueeze(1),
                      (pad_amount, pad_amount),
                      mode='reflect').squeeze(1)

        # Прямое преобразование
        mel = self.mel_spectrogram(audio) \
            .clamp(min=1e-5) \
            .log()

        return mel

    def inverse(self, spec: torch.Tensor, phase: torch.Tensor, n_fft : int) -> torch.Tensor:
        """
        Reconstruct waveform from provided spectrogram magnitude and phase.
        :param spec: Magnitude spectrogram of shape [B, n_fft//2+1, T]
        :param phase: Phase tensor of shape [B, n_fft//2+1, T] (in radians)
        :return: Reconstructed waveform of shape [B, T_reconstructed]
        """
        # Ensure the inputs are on the correct device.
        spec = spec.to(self.device)
        phase = phase.to(self.device)

        # Combine magnitude and phase to create a complex spectrogram.
        # Casting spec to complex if needed.
        complex_spec = spec.to(torch.complex64) * torch.exp(1j * phase)

        # Create a Hann window.
        window = torch.hann_window(self.config.win_length).to(self.device)

        # Perform the inverse STFT.
        waveform = torch.istft(
            complex_spec,
            n_fft=n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=True,
            normalized=False,
            onesided=True
        )
        return waveform
