import torch

from src.loss.gan_base_loss import GanBaseLoss


class LightVocLoss(GanBaseLoss):
    def __init__(self, gen_stft_loss, model, lambda_adv=4.0):
        super().__init__(model)

        self.lambda_adv = lambda_adv
        self.gen_stft_loss = gen_stft_loss

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
            loss += r_loss + g_loss

        return loss

    def _gen_adv_loss(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            loss += l

        return loss

    def forward(self, compute_generator_loss=True, compute_discriminator_loss=True, **batch):
        real_audio = batch["audio"]  # [B, 1, T]
        pred_audio = batch["pred_audio"]  # [B, 1, T'] (T' обычно может немного отличаться от T)

        discrimators_results = self._discriminate(real_audio, pred_audio)

        out = {}
        if compute_generator_loss:
            adv_loss = 0.0
            for result in discrimators_results:
                adv_loss += self._gen_adv_loss(discrimators_results[result]["y_d_gs"])
            adv_loss /= len(discrimators_results)

            aux_loss = self.gen_stft_loss(real_audio, pred_audio)

            out["loss_gen"] = aux_loss + self.lambda_adv * adv_loss

        if compute_discriminator_loss:
            out["loss_disc"] = 0.0

            for result in discrimators_results:
                out["loss_disc"] += self._discriminator_loss(discrimators_results[result]["y_d_rs"],
                                                             discrimators_results[result]["y_d_gs"])
            out["loss_disc"] /= len(discrimators_results)
        return out
