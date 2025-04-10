import torch
import torch.nn as nn

class GanBaseModel(nn.Module):
    def __init__(self, device: torch.device, generator: nn.Module, discriminator: nn.Module):
        super().__init__()

        self.device = torch.device(device)

        self.generator = generator
        self.discriminator = discriminator

    def discriminate(self, real_audio: torch.Tensor, generated_audio: torch.Tensor):
        # TODO: use discriminator class instead of nn.Module not to mix up audios
        return self.discriminator(real_audio, generated_audio)
