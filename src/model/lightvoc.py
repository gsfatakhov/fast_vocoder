import torch

from src.utils.mel import MelSpectrogram
from src.model.src_lightvoc.stft import TorchSTFT

from src.model.gan_base_model import GanBaseModel

from src.model.src_lightvoc.generator import LightVocGenerator
from src.model.src_lightvoc.discriminator import LightVocMultiDiscriminator


class LightVoc(GanBaseModel):
    def __init__(self, device, generator_params, discriminator_params, stft_params, mel_config=None):
        device = torch.device(device)

        generator = LightVocGenerator(**generator_params)
        discriminator = LightVocMultiDiscriminator(discriminator_params['combd_params'],
                                                         discriminator_params['sbd_params'],
                                                         discriminator_params['mrsd_params'])

        super().__init__(device, generator, discriminator)

        self.stft = TorchSTFT(filter_length=stft_params.filter_length,
                              hop_length=stft_params.hop_length,
                              win_length=stft_params.win_length,
                              device=self.device).to(self.device)

        self.mel_config = mel_config
        if self.mel_config:
            self.mel_extractor = MelSpectrogram(self.mel_config, device=self.device)

    def forward(self, **batch):
        if self.mel_config and "mel" not in batch:
            batch["mel"] = self.mel_extractor(batch["audio"]).squeeze(1).detach()

        spec, phase = self.generator(batch["mel"])

        length = None
        if "audio" in batch:
            length = batch["audio"].shape[2]

        batch["pred_audio"] = self.stft.inverse(spec, phase, length=length)
        return batch
