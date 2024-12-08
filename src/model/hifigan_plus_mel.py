import torch.nn as nn
import torch

from src.model.hifigan import HiFiGANGenerator, HiFiGANMultiScaleDiscriminator

from src.utils.mel import MelSpectrogramConfig, MelSpectrogram

class HiFiGAN_Plus_Mel(nn.Module):
    def __init__(self, generator_params, discriminator_params):
        super().__init__()
        self.generator = HiFiGANGenerator(**generator_params)
        self.msd = HiFiGANMultiScaleDiscriminator(**discriminator_params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mel_config = MelSpectrogramConfig()
        self.mel_extractor = MelSpectrogram(self.mel_config, device=device)

    def forward(self, audio, **batch):

        mel = self.mel_extractor(audio).squeeze(1)

        pred_audio = self.generator(mel)

        if pred_audio[0].shape[-1] > audio.shape[-1]:
            pred_audio = pred_audio[..., :audio.shape[-1]]
        elif pred_audio[0].shape[-1] < audio.shape[-1]:
            raise ValueError("Predicted audio is shorter than original audio")
        batch["pred_audio"] = pred_audio
        return batch

    def discriminate(self, audio):
        return self.msd(audio)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        generator_parameters = sum([p.numel() for p in self.generator.parameters()])
        discriminator_parameters = sum([p.numel() for p in self.msd.parameters()])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        result_info = result_info + f"\nGenerator parameters: {generator_parameters}"
        result_info = result_info + f"\nDiscriminator parameters: {discriminator_parameters}"

        return result_info