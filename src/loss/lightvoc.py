import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.mel import MelSpectrogram, MelSpectrogramConfig

class LightVocLoss(nn.Module):
    def __init__(self, fft_sizes, hop_sizes, win_lengths, mel_config=MelSpectrogramConfig(), device="cuda"):
        super().__init__()
        self.model = None

        self.mel_extractor = MelSpectrogram(mel_config, device=torch.device(device))

    def _feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2

    def _discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

    def _generator_loss(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    def forward(self, compute_generator_loss=True, compute_discriminator_loss=True, **batch):
        real_audio = batch["audio"]  # [B, 1, T]
        pred_audio = batch["pred_audio"]  # [B, 1, T'] (T' обычно может немного отличаться от T)



        discrimators_results = self.model.discriminate(real_audio, pred_audio.detach())

        out = {}
        if compute_generator_loss:
            y_mel = batch["mel"]  # [B, n_mels, T_mel]
            y_g_hat_mel = self.mel_extractor(pred_audio.squeeze(1))

            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_du_hat_r, y_du_hat_g, fmap_u_r, fmap_u_g = discrimators_results["CoMBD"]
            y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = discrimators_results["SBD"]
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = discrimators_results["MRSD"]

            loss_fm_u = self._feature_loss(fmap_u_r, fmap_u_g)
            loss_fm_p = self._feature_loss(fmap_p_r, fmap_p_g)
            loss_fm_s = self._feature_loss(fmap_s_r, fmap_s_g)

            loss_gen_u, losses_gen_u = self._generator_loss(y_du_hat_g)
            loss_gen_p, losses_gen_p = self._generator_loss(y_dp_hat_g)
            loss_gen_s, losses_gen_s = self._generator_loss(y_ds_hat_g)

            out["loss_gen"] = loss_gen_s + loss_gen_p + loss_gen_u + loss_fm_s + loss_fm_u + loss_fm_p + loss_mel



        if compute_discriminator_loss:

            out["loss_disc"] = 0.0

            for result in discrimators_results:
                out["loss_disc"] += self._discriminator_loss(discrimators_results[result]["y_d_rs"],
                                                             discrimators_results[result]["y_d_gs"])[0]

        return out