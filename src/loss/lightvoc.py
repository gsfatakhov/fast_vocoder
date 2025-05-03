import torch
import torch.nn.functional as F

from src.loss.gan_base_loss import GanBaseLoss
from src.utils.mel import MelSpectrogram


class LightVocLoss(GanBaseLoss):
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

        # return {
        #     "CoMBD": {"y_du_hat_r": y_du_hat_r, "y_du_hat_g": y_du_hat_g, "fmap_u_r": fmap_u_r,
        #               "fmap_u_g": fmap_u_g},
        #     "SBD": {"y_dp_hat_r": y_dp_hat_r, "y_dp_hat_g": y_dp_hat_g, "fmap_p_r": fmap_p_r,
        #             "fmap_p_g": fmap_p_g},
        #     "MSD": {"y_ds_hat_r": y_ds_hat_r, "y_ds_hat_g": y_ds_hat_g, "fmap_s_r": fmap_s_r,
        #             "fmap_s_g": fmap_s_g},
        # }

        # pred_audio already detached in Trainer for discriminator training
        disc_results = self._discriminate(real_audio, pred_audio)

        out = {}
        if compute_generator_loss:
            # L1 Mel-Spectrogram Loss
            # "mel_for_loss" calculates on CPU in Dataset class if passed mel_conf and calc_mel_for_loss params
            batch["pred_mel"] = self.mel_extractor(pred_audio)
            loss_mel = F.l1_loss(batch["mel_for_loss"], batch["pred_mel"]) * 45

            # y_du_hat_r, y_du_hat_g, fmap_u_r, fmap_u_g = combd(ys, ys_hat)
            y_du_hat_r, y_du_hat_g, fmap_u_r, fmap_u_g = disc_results["CoMBD"]["y_du_hat_r"], disc_results["CoMBD"][
                "y_du_hat_g"], disc_results["CoMBD"]["fmap_u_r"], disc_results["CoMBD"]["fmap_u_g"]

            # y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = sbd(y, y_g_hat)
            y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = disc_results["SBD"]["y_dp_hat_r"], disc_results["SBD"][
                "y_dp_hat_g"], disc_results["SBD"]["fmap_p_r"], disc_results["SBD"]["fmap_p_g"]

            # y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = disc_results["MSD"]["y_ds_hat_r"], disc_results["MSD"][
                "y_ds_hat_g"], disc_results["MSD"]["fmap_s_r"], disc_results["MSD"]["fmap_s_g"]

            loss_fm_u = self._feature_loss(fmap_u_r, fmap_u_g)
            loss_fm_p = self._feature_loss(fmap_p_r, fmap_p_g)
            loss_fm_s = self._feature_loss(fmap_s_r, fmap_s_g)

            loss_gen_u, losses_gen_u = self._generator_loss(y_du_hat_g)
            loss_gen_p, losses_gen_p = self._generator_loss(y_dp_hat_g)
            loss_gen_s, losses_gen_s = self._generator_loss(y_ds_hat_g)

            # Resulting loss
            out["loss_gen_adv"] = loss_gen_s + loss_gen_p + loss_gen_u

            out["loss_gen_fm"] = loss_fm_s + loss_fm_u + loss_fm_p

            out["loss_gen_mel"] = loss_mel

            out["loss_gen"] = out["loss_gen_adv"] + out["loss_gen_fm"] + out["loss_gen_mel"]

        if compute_discriminator_loss:
            # combd
            # ys list length 1 contain y shape batch x 1 x segment_size
            # y_g_hat shape batch x 1 x segment_size
            y_du_hat_r, y_du_hat_g = disc_results["CoMBD"]["y_du_hat_r"], disc_results["CoMBD"]["y_du_hat_g"]
            loss_disc_u, losses_disc_u_r, losses_disc_u_g = self._discriminator_loss(y_du_hat_r, y_du_hat_g)

            # sbd
            y_dp_hat_r, y_dp_hat_g = disc_results["SBD"]["y_dp_hat_r"], disc_results["SBD"]["y_dp_hat_g"]
            loss_disc_p, losses_disc_p_r, losses_disc_p_g = self._discriminator_loss(y_dp_hat_r, y_dp_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g = disc_results["MSD"]["y_ds_hat_r"], disc_results["MSD"]["y_ds_hat_g"]
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = self._discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            out["loss_disc"] = loss_disc_s + loss_disc_p + loss_disc_u
        return out
