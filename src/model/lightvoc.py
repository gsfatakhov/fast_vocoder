import torch.nn as nn
import torch

from src.utils.mel import MelSpectrogramConfig, MelSpectrogram
from src.model.src_lightvoc.stft import TorchSTFT

from src.model.src_lightvoc.generator import LightVocGenerator
from src.model.src_lightvoc.discriminator import LightVocMultiDiscriminator

import torch.nn.functional as F



class LightVoc(nn.Module):
    def __init__(self, generator_params, discriminator_params, stft_params, calc_mel=False):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = LightVocGenerator(**generator_params)
        self.discriminators = LightVocMultiDiscriminator(discriminator_params['combd_params'],
                                                         discriminator_params['sbd_params'],
                                                         discriminator_params['mrsd_params'])

        self.stft = TorchSTFT(filter_length=stft_params.filter_length,
                              hop_length=stft_params.hop_length,
                              win_length=stft_params.win_length,
                              device=self.device).to(self.device)

        self.calc_mel = calc_mel
        if calc_mel:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mel_config = MelSpectrogramConfig()
            self.mel_extractor = MelSpectrogram(self.mel_config, device=device)

    def forward(self, **batch):
        if self.calc_mel and "mel" not in batch:
            batch["mel"] = self.mel_extractor(batch["audio"]).squeeze(1)

        spec, phase = self.generator(batch["mel"])

        pred_audio = self.stft.inverse(spec, phase)

        batch["pred_audio"] = pred_audio

        if batch["audio"].shape[2] < batch["pred_audio"].shape[2]:
            pad = batch["pred_audio"].shape[2] - batch["audio"].shape[2]
            batch["audio"] = F.pad(batch["audio"], (0, pad), "constant", 0)
        elif batch["audio"].shape[2] > batch["pred_audio"].shape[2]:
            pad = batch["audio"].shape[2] - batch["pred_audio"].shape[2]
            batch["pred_audio"] = F.pad(batch["pred_audio"], (0, pad), "constant", 0)


        return batch

    def discriminate(self, real_audio, generated_audio):
        return self.discriminators(real_audio, generated_audio)

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminators.parameters())
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        info += f"\nGenerator parameters: {gen_params}"
        info += f"\nDiscriminator parameters: {disc_params}"
        return info


