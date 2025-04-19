import torch.nn as nn
import torch
from src.model.gan_base_model import GanBaseModel

class GanBaseLoss(nn.Module):
    def __init__(self, model: GanBaseModel):
        super().__init__()
        self.model = model

    def set_model(self, model: GanBaseModel) -> None:
        self.model = model

    def _discriminate(self, real_audio: torch.Tensor, generated_audio: torch.Tensor):
        return self.model.discriminate(real_audio, generated_audio)
