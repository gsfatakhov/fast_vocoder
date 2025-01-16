import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def stft(self, x, fft_size, hop_size, win_length):
        return torch.stft(
            x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, return_complex=True
        )

    def forward(self, x_real, x_fake):
        stft_losses = []
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            real_spec = torch.abs(self.stft(x_real, fft_size, hop_size, win_length))
            fake_spec = torch.abs(self.stft(x_fake, fft_size, hop_size, win_length))
            stft_losses.append(F.l1_loss(real_spec, fake_spec))
        return sum(stft_losses) / len(stft_losses)

class LightVocLoss(nn.Module):
    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super().__init__()
        self.stft_loss = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths)

    def forward(self, disc_real_outputs, disc_fake_outputs, x_real, x_fake):
        stft_loss = self.stft_loss(x_real, x_fake)

        disc_loss = 0.0
        for r_out, f_out in zip(disc_real_outputs, disc_fake_outputs):
            disc_loss += torch.mean((r_out - 1)**2) + torch.mean(f_out**2)
        disc_loss /= len(disc_real_outputs)

        adv_loss = 0.0
        for f_out in disc_fake_outputs:
            adv_loss += torch.mean((f_out - 1)**2)
        adv_loss /= len(disc_fake_outputs)

        gen_loss = stft_loss + adv_loss

        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss
        }
