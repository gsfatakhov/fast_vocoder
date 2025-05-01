from src.model.src_lightvoc.models import Generator, CoMBD, SBD, MultiResSpecDiscriminator

from src.model.gan_base_model import GanBaseModel

import torch.nn as nn


class DiscriminatorModel(nn.Module):
    def __init__(self, combd_params, sbd_params, mrsd_params):
        super().__init__()

        self.combd = CoMBD(**combd_params)
        self.sbd = SBD(**sbd_params)
        self.combd = MultiResSpecDiscriminator(**mrsd_params)

    def forward(self, real_audio, generated_audio):
        # audio: [B, 1, T]
        y_df_hat_r_mpd, y_df_hat_g_mpd, fmap_f_r_mpd, fmap_f_g_mpd = self.mpd(real_audio, generated_audio)
        y_ds_hat_r_msd, y_ds_hat_g_msd, fmap_s_r_msd, fmap_s_g_msd = self.msd(real_audio, generated_audio)

        return {
            "MPD": {"y_df_hat_r": y_df_hat_r_mpd, "y_df_hat_g": y_df_hat_g_mpd, "fmap_f_r": fmap_f_r_mpd,
                    "fmap_f_g": fmap_f_g_mpd},
            "MSD": {"y_ds_hat_r": y_ds_hat_r_msd, "y_ds_hat_g": y_ds_hat_g_msd, "fmap_s_r": fmap_s_r_msd,
                    "fmap_s_g": fmap_s_g_msd},
        }


class HiFiGANPaper(GanBaseModel):
    def __init__(self, generator_params, discriminator_params):
        generator = Generator(**generator_params)
        discriminator = DiscriminatorModel(**discriminator_params)

        super().__init__(generator, discriminator)

    def forward(self, **batch):
        batch["pred_audio"] = self.generator(batch["mel"])
        return batch
