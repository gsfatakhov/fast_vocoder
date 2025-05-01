import torch
import torch.nn.functional as F

from src.loss.gan_base_loss import GanBaseLoss
from src.utils.mel import MelSpectrogram


class HiFiGANPaperLoss(GanBaseLoss):
    def __init__(self, model, mel_config):
        super().__init__(model)
        self.mel_extractor = MelSpectrogram(mel_config)

    @staticmethod
    def _feature_loss(fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2

    @staticmethod
    def _discriminator_loss(disc_real_outputs, disc_generated_outputs):
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

    @staticmethod
    def _generator_loss(disc_outputs):
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

        # {
        #     "MPD": {"y_df_hat_r": y_df_hat_r_mpd, "y_df_hat_g": y_df_hat_g_mpd, "fmap_f_r": fmap_f_r_mpd,
        #             "fmap_f_g": fmap_f_g_mpd},
        #     "MSD": {"y_df_hat_r": y_ds_hat_r_msd, "y_df_hat_g": y_ds_hat_g_msd, "fmap_f_r": fmap_s_r_msd,
        #             "fmap_f_g": fmap_s_g_msd},
        # }

        # pred_audio already detached in Trainer for discriminator training
        disc_results = self._discriminate(real_audio, pred_audio)

        out = {}
        if compute_generator_loss:
            # L1 Mel-Spectrogram Loss
            # "mel_for_loss" calculates on CPU in Dataset class if passed mel_conf and calc_mel_for_loss params
            batch["pred_mel"] = self.mel_extractor(pred_audio)
            loss_mel = F.l1_loss(batch["mel_for_loss"], batch["pred_mel"]) * 45

            # y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = disc_results["MPD"]["y_df_hat_r"], disc_results["MPD"][
                "y_df_hat_g"], disc_results["MPD"]["fmap_f_r"], disc_results["MPD"]["fmap_f_g"]

            # y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = disc_results["MSD"]["y_ds_hat_r"], disc_results["MSD"][
                "y_ds_hat_g"], disc_results["MSD"]["fmap_s_r"], disc_results["MSD"]["fmap_s_g"]

            loss_fm_f = self._feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = self._feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = self._generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = self._generator_loss(y_ds_hat_g)

            # Resulting loss
            out["loss_gen_adv_mpd"] = loss_gen_f
            out["loss_gen_adv_msd"] = loss_gen_s

            out["loss_gen_fm_mpd"] = loss_fm_f
            out["loss_gen_fm_msd"] = loss_fm_s

            out["loss_gen_mel"] = loss_mel

            out["loss_gen"] = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        if compute_discriminator_loss:
            # MPD
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = self._discriminator_loss(disc_results["MPD"]["y_df_hat_r"],
                                                                                     disc_results["MPD"]["y_df_hat_g"])

            # MSD
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = self._discriminator_loss(disc_results["MSD"]["y_ds_hat_r"],
                                                                                     disc_results["MSD"]["y_ds_hat_g"])

            out["loss_disc"] = loss_disc_s + loss_disc_f
        return out
