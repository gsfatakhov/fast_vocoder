import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.mel import MelSpectrogram, MelSpectrogramConfig


class HiFiGANLoss(nn.Module):
    def __init__(self, mel_config=None, device="cuda"):
        super().__init__()
        self.mel_extractor = MelSpectrogram(mel_config, device=torch.device(device))
        self.l1 = nn.L1Loss()
        self.model = None

    def forward(self, **batch):
        real_audio = batch["audio"]  # [B, 1, T]
        pred_audio = batch["pred_audio"]  # [B, 1, T']

        with torch.no_grad():
            mel_real = batch["mel"]  # [B, 1, T]
        mel_pred = self.mel_extractor(pred_audio.squeeze(1))

        mel_loss = self.l1(mel_pred, mel_real)

        disc_real = self.model.discriminate(real_audio)
        disc_fake = self.model.discriminate(pred_audio.detach())

        disc_loss = 0.0
        for (f_out, _), (r_out, _) in zip(disc_fake, disc_real):
            # На реальном out ~ 1
            disc_loss += F.mse_loss(r_out, torch.ones_like(r_out))
            # На фейковом out ~ 0
            disc_loss += F.mse_loss(f_out, torch.zeros_like(f_out))
        disc_loss /= len(disc_fake)

        # adversarial лосс (надо f_out ~ 1)
        disc_fake_g = self.model.discriminate(pred_audio)
        adv_loss_gen = 0.0
        fm_loss = 0.0

        for (f_out, f_feats), (r_out, r_feats) in zip(disc_fake_g, disc_real):
            adv_loss_gen += F.mse_loss(f_out, torch.ones_like(f_out))
            for ff, rf in zip(f_feats, r_feats):
                fm_loss += F.l1_loss(ff, rf)

        adv_loss_gen /= len(disc_fake_g)
        fm_loss /= len(disc_fake_g)

        loss_gen = adv_loss_gen + 2.0 * fm_loss + 45.0 * mel_loss

        return {
            "loss_gen": loss_gen,
            "loss_disc": disc_loss,
        }
