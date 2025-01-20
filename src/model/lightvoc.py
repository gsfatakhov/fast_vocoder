import torch.nn as nn
import torch

from src.utils.mel import MelSpectrogramConfig, MelSpectrogram

from src.model.src_lightvoc.generator import LightVocGenerator
from src.model.src_lightvoc.discriminator import LightVocMultiDiscriminator



class LightVoc(nn.Module):
    def __init__(self, generator_params, discriminator_params, calc_mel=False):
        super().__init__()
        self.generator = LightVocGenerator(**generator_params)
        self.discriminators = LightVocMultiDiscriminator(discriminator_params['combd_params'],
                                                         discriminator_params['sbd_params'],
                                                         discriminator_params['mrsd_params'])

        self.calc_mel = calc_mel
        if calc_mel:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mel_config = MelSpectrogramConfig()
            self.mel_extractor = MelSpectrogram(self.mel_config, device=device)

    def forward(self, **batch):
        if self.calc_mel and "mel" not in batch:
            batch["mel"] = self.mel_extractor(batch["audio"]).squeeze(1)
        pred_audio = self.generator(batch["mel"])
        if "audio" in batch:
            if pred_audio.shape[-1] > batch["audio"].shape[-1]:
                pred_audio = pred_audio[..., :batch["audio"].shape[-1]]
            elif pred_audio.shape[-1] < batch["audio"].shape[-1]:
                raise ValueError("Predicted audio is shorter than original audio")
        batch["pred_audio"] = pred_audio
        return batch

    def discriminate(self, audio):
        return self.discriminators(audio)

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


