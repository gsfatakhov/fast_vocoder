import torch

from src.loss.gan_base_loss import GanBaseLoss


class HiFiGANPaperLoss(GanBaseLoss):
    def __init__(self, ):
        super().__init__(model)




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
            aux_loss = self.gen_stft_loss(real_audio, pred_audio)

            # Feature matching loss
            fm_loss = 0.0
            for result in discrimators_results:
                fm_loss += self._feature_loss(discrimators_results[result]["fmap_rs"],
                                              discrimators_results[result]["fmap_gs"])
            fm_loss /= len(discrimators_results)

            # Resulting loss
            out["loss_gen"] = aux_loss + fm_loss + self.lambda_adv * adv_loss

        if compute_discriminator_loss:
            out["loss_disc"] = 0.0

            for result in discrimators_results:
                out["loss_disc"] += self._discriminator_loss(discrimators_results[result]["y_d_rs"],
                                                             discrimators_results[result]["y_d_gs"])
            out["loss_disc"] /= len(discrimators_results)
        return out
