from src.model.gan_base_model import GanBaseModel

from src.model.src_mambavoc_v2.generator import Generator
from src.model.src_mambavoc_v2.CoMBD import CoMBD
from src.model.src_mambavoc_v2.SBD import SBD
from src.model.src_mambavoc_v2.pqmf import PQMF

import torch.nn as nn


class DiscriminatorModel(nn.Module):
    def __init__(self, pqmf_config, combd_params, sbd_params):
        super().__init__()

        pqmf_lv2 = PQMF(*pqmf_config["lv2"])
        pqmf_lv1 = PQMF(*pqmf_config["lv1"])

        self.combd = CoMBD(pqmf_list=[pqmf_lv2, pqmf_lv1], **combd_params)
        self.sbd = SBD(**sbd_params)

    def forward(self, real_audio, generated_audio):
        # audio: [B, 1, T]

        ys = [
            self.pqmf_lv2.analysis(
                real_audio
            )[:, :self.hparams.generator.projection_filters[1]],
            self.pqmf_lv1.analysis(
                real_audio
            )[:, :self.hparams.generator.projection_filters[2]],
            real_audio
        ]

        ys_hat = generated_audio

        y_du_hat_r, y_du_hat_g, fmap_u_r, fmap_u_g = self.combd(ys, ys_hat)
        y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = self.sbd(real_audio, generated_audio[-1])

        return {
            "CoMBD": {"y_du_hat_r": y_du_hat_r, "y_du_hat_g": y_du_hat_g, "fmap_u_r": fmap_u_r,
                      "fmap_u_g": fmap_u_g},
            "SBD": {"y_dp_hat_r": y_dp_hat_r, "y_dp_hat_g": y_dp_hat_g, "fmap_p_r": fmap_p_r,
                    "fmap_p_g": fmap_p_g},
        }


class LightVocV2(GanBaseModel):
    def __init__(self, generator_params, discriminator_params, ):
        generator = Generator(**generator_params)
        discriminator = DiscriminatorModel(**discriminator_params)

        super().__init__(generator, discriminator)

    def forward(self, **batch):
        x = batch["mel"]

        inference_params = batch.get("inference_params", None)
        out = self.generator(x, inference_params=inference_params)

        batch["ys_hat"] = out
        batch["pred_audio"] = out[-1]

        return batch
