import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.mel import MelSpectrogram

class MultiResolutionMelSpectrogramLoss(nn.Module):
    def __init__(self, mel_configs, eps=1e-7, device="cuda"):
        """
        Args:
            mel_configs (list): List of MelSpectrogramConfig objects (one per resolution).
            eps (float): Small epsilon for numerical stability.
            device (torch.device): Device to use.
        """
        super(MultiResolutionMelSpectrogramLoss, self).__init__()
        self.eps = eps
        self.device = torch.device(device)

        # Create a MelSpectrogram extractor for each resolution.
        self.mel_extractors = nn.ModuleList([
            MelSpectrogram(config, device=self.device) for config in mel_configs
        ])

    def forward(self, x, x_hat):
        """
        Computes the multi-resolution mel spectrogram loss.
        Args:
            x (Tensor): Ground truth waveform, shape [B, T].
            x_hat (Tensor): Generated waveform, shape [B, T].
        Returns:
            loss (Tensor): Scalar loss value.
        """
        total_sc_loss = 0.0
        total_mag_loss = 0.0
        num_resolutions = len(self.mel_extractors)

        for mel_extractor in self.mel_extractors:
            # Compute log-mel spectrograms.
            log_mel_x = mel_extractor(x)  # shape: [B, n_mels, T']
            log_mel_x_hat = mel_extractor(x_hat)

            # For spectral convergence, convert log-mel to linear magnitude.
            mel_x = torch.exp(log_mel_x)
            mel_x_hat = torch.exp(log_mel_x_hat)

            # Spectral convergence loss: normalized Frobenius norm difference.
            sc_loss = torch.sqrt(torch.sum((mel_x - mel_x_hat) ** 2, dim=(1, 2))) \
                      / (torch.sqrt(torch.sum(mel_x ** 2, dim=(1, 2))) + self.eps)
            sc_loss = torch.mean(sc_loss)

            # Log-mel L1 loss.
            mag_loss = F.l1_loss(log_mel_x, log_mel_x_hat)

            total_sc_loss += sc_loss
            total_mag_loss += mag_loss

        loss = (total_sc_loss + total_mag_loss) / num_resolutions
        return loss
