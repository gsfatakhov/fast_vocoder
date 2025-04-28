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

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        info += f"\nGenerator parameters: {gen_params}"
        info += f"\nDiscriminator parameters: {disc_params}"
        return info
