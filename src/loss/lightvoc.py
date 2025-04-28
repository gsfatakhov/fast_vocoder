import torch

from src.loss.gan_base_loss import GanBaseLoss
import torch.nn.functional as F
from src.utils.mel import MelSpectrogram


class LightVocLoss(GanBaseLoss):
    def __init__(self, gen_stft_loss, model, lambda_adv, lambda_rec, lambda_feature, mel_config=None):
        super().__init__(model)

        self.lambda_adv = lambda_adv
        self.lambda_rec = lambda_rec
        self.lambda_feature = lambda_feature

        self.gen_stft_loss = gen_stft_loss

        self.mel_extractor = None
        if mel_config:
            self.mel_extractor = MelSpectrogram(mel_config, device=model.device)

    def _feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss

    def _discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += r_loss + g_loss

        return loss

    def _gen_adv_loss(self, disc_outputs):
        loss = 0
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            loss += l

        return loss

    def forward(self, compute_generator_loss=True, compute_discriminator_loss=True, **batch):
        real_audio = batch["audio"]  # [B, 1, T]
        pred_audio = batch["pred_audio"]  # [B, 1, T'] (T' обычно может немного отличаться от T)

        discrimators_results = self._discriminate(real_audio, pred_audio)

        # {
        #  "CoMBD": {"y_d_rs": outs_real, "y_d_gs": outs_fake, "fmap_rs": f_maps_real, "fmap_gs": f_maps_fake},
        #  "SBD": {"y_d_rs": y_d_rs_sbd, "y_d_gs": y_d_gs_sbd, "fmap_rs": fmap_rs_sbd, "fmap_gs": fmap_gs_sbd},
        #  "MRSD": {"y_d_rs": y_d_rs_mrsd, "y_d_gs": y_d_gs_mrsd, "fmap_rs": fmap_rs_mrsd, "fmap_gs": fmap_gs_mrsd},
        # }

        out = {}
        if compute_generator_loss:
            # Adversarial loss
            adv_loss = 0.0
            for result in discrimators_results:
                adv_loss += self._gen_adv_loss(discrimators_results[result]["y_d_gs"])
            adv_loss /= len(discrimators_results)

            # Stft loss
            # aux_loss = self.gen_stft_loss(real_audio, pred_audio)

            pred_audio_mel = self.mel_extractor(pred_audio)
            aux_loss = F.l1_loss(batch["mel"], pred_audio_mel)

            # Feature matching loss
            fm_loss = 0.0
            for result in discrimators_results:
                fm_loss += self._feature_loss(discrimators_results[result]["fmap_rs"],
                                              discrimators_results[result]["fmap_gs"])
            fm_loss /= len(discrimators_results)

            # Resulting loss
            out["loss_gen_rec"] = aux_loss
            out["loss_gen_fm"] = fm_loss
            out["loss_gen_adv"] = adv_loss

            out["loss_gen"] = self.lambda_rec * aux_loss + self.lambda_feature * fm_loss + self.lambda_adv * adv_loss

        if compute_discriminator_loss:
            out["loss_disc"] = 0.0

            for result in discrimators_results:
                out["loss_disc"] += self._discriminator_loss(discrimators_results[result]["y_d_rs"],
                                                             discrimators_results[result]["y_d_gs"])
            out["loss_disc"] /= len(discrimators_results)
        return out
