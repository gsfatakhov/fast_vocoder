from src.model.src_mambavoc.models import Generator, CoMBD, SBD, MultiResSpecDiscriminator
from src.utils.mel import MelSpectrogram
from src.model.src_lightvoc.pqmf import PQMF

from src.model.gan_base_model import GanBaseModel

import torch.nn as nn


class DiscriminatorModel(nn.Module):
    def __init__(self, pqmf_config, combd_params, sbd_params):
        super().__init__()

        subbands2, taps2, cutoff_ratio2, beta2 = pqmf_config["lv2"]
        # in reference repo using "lv2""
        subbands1, taps1, cutoff_ratio1, beta1 = pqmf_config["lv1"]

        pqmf_lv2 = PQMF(subbands2, taps2, cutoff_ratio2, beta2)
        pqmf_lv1 = PQMF(subbands1, taps1, cutoff_ratio1, beta1)

        self.combd = CoMBD(pqmf_list=[pqmf_lv2, pqmf_lv1], **combd_params)
        self.sbd = SBD(**sbd_params)
        self.msd = MultiResSpecDiscriminator()

    def forward(self, real_audio, generated_audio):
        # audio: [B, 1, T]

        # if use this, need to add some conv in generator, follow: https://github.com/ncsoft/avocodo/blob/2999557bbd040a6f3eb6f7006a317d89537b78cd/avocodo/models/generator.py#L109
        # in this implement, only last conv is used

        # ys = [
        #     pqmf_lv2.analysis(
        #         y
        #     )[:, :h.projection_filters[1]],
        #     pqmf_lv1.analysis(
        #         y
        #     )[:, :h.projection_filters[2]],
        #     y
        # ]
        ys = [real_audio]
        ys_hat = [generated_audio]

        y_du_hat_r, y_du_hat_g, fmap_u_r, fmap_u_g = self.combd(ys, ys_hat)
        y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = self.sbd(real_audio, generated_audio)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(real_audio, generated_audio)

        return {
            "CoMBD": {"y_du_hat_r": y_du_hat_r, "y_du_hat_g": y_du_hat_g, "fmap_u_r": fmap_u_r,
                      "fmap_u_g": fmap_u_g},
            "SBD": {"y_dp_hat_r": y_dp_hat_r, "y_dp_hat_g": y_dp_hat_g, "fmap_p_r": fmap_p_r,
                    "fmap_p_g": fmap_p_g},
            "MSD": {"y_ds_hat_r": y_ds_hat_r, "y_ds_hat_g": y_ds_hat_g, "fmap_s_r": fmap_s_r,
                    "fmap_s_g": fmap_s_g},
        }


class MambaVoc(GanBaseModel):
    def __init__(self, generator_params, discriminator_params, mel_config, inference_params=None):
        generator = Generator(**generator_params)
        discriminator = DiscriminatorModel(**discriminator_params)

        super().__init__(generator, discriminator)

        self.mel_extractor = MelSpectrogram(mel_config)
        self.inference_params = inference_params

    def forward(self, **batch):
        x = batch["mel"]

        spec, phase = self.generator(x, inference_params=self.inference_params)

        batch["pred_audio"] = self.mel_extractor.inverse(spec, phase).unsqueeze(1)

        return batch
